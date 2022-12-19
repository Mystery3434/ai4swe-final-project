#! -*- coding:utf-8 -*-
# Pattern-Exploiting Training(PET) score:
# Code adapted from https://github.com/bojone/Pattern-Exploiting-Training/blob/master/sentiment.py
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
import ipykernel
from keras.layers import Lambda, Dense
from tqdm import tqdm
from sklearn.metrics import classification_report
import json
import pandas as pd
import os
np.random.seed(42)
num_classes = 4
maxlen = 128
batch_size = 16

path = "/Users/pan/uncased_L-12_H-768_A-12/"               # 预训练bert 根据自己路径进行调整
config_path = path + 'bert_config.json'
checkpoint_path = path + 'bert_model.ckpt'
dict_path = path + 'vocab.txt'

labels = ['useful', 'useless']


def load_data(X_test, y_test):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D=[]
    data = pd.concat([X_test, y_test], axis=1)
   #data = pd.read_csv(filename, usecols=['reviews', 'Judgement'], keep_default_na=False)
    for index, row in data.iterrows():
        text = row['reviews'].lower()
        label = row['Judgement'].iloc[0]
        D.append((text, label))
    return D

def load_train_data(X_train, y_train):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D=[]
    data = pd.concat([X_train, y_train], axis=1)
    for index, row in data.iterrows():
        text = row['reviews'].lower()
        label = row['Judgement'].iloc[0]
        D.append((text, label))
    return D

train_X_original = pd.read_csv("original_data_X_train.csv")
train_y_original = pd.read_csv("original_data_y_train.csv")

augmented_data = pd.read_csv("balanced_augmentation_dataset.csv")

X_test = pd.read_csv("original_data_X_test.csv")
y_test = pd.read_csv("original_data_y_test.csv")

augmented_data=augmented_data[["Reviews", "Useful?"]]
augmented_data.rename(columns={"Reviews":"reviews", "Useful?":"Judgement"}, inplace=True)

X_train_augmented = pd.concat([train_X_original, augmented_data[["reviews"]]], axis=0, ignore_index=True)
y_train_augmented = pd.concat([train_y_original, augmented_data[["Judgement"]]], axis=0, ignore_index=True)

# 加载数据集
train_data = load_train_data(train_X_original, train_y_original)
test_data = load_data(X_test, y_test)


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 对应的任务描述 前缀模板
prefix = 'it is useful. '

mask_idx = 3  # cls 的位置是 1
n_id = tokenizer.token_to_id('useful')
o_id = tokenizer.token_to_id('useless')


token2id = {n_id: 0, o_id: 1}
id2label = {v: labels[v] for v in range(len(labels))}

def random_masking(token_ids):
    """对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:  # 12%
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:  # 1.5%
            source.append(t)
            target.append(t)
        elif r < 0.15:  # 1.5%
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:  # 85%
            source.append(t)
            target.append(0)
    return source, target


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):  # 调用 forfit()时 默认设置 random为True
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label) in self.sample(random):

            text = prefix + text
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            if random:
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]
            if label == 1:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = n_id
            elif label == 0:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = o_id

            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [
                          batch_token_ids, batch_segment_ids, batch_output_ids
                      ], None
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """

    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name='accuracy')
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


# 加载预训练模型
model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True
)

# 训练用模型
y_in = keras.layers.Input(shape=(None,))
outputs = CrossEntropy(1)([y_in, model.output])

train_model = keras.models.Model(model.inputs + [y_in], outputs)
train_model.compile(optimizer=Adam(1e-5))
# train_model.summary()

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(test_data, batch_size)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_f1 = 0.
        # self.flag = 0

    def on_epoch_end(self, epoch, logs=None):
        f1 = evaluate(valid_generator)

        if f1 > self.best_f1:
            self.best_f1 = f1
            model.save_weights('./best_model.weights')
        print(u'epoch: %d, f1: %.5f, best_f1: %.5f\n' % (epoch, f1, self.best_f1))


# 验证评估
def evaluate(data):
    right = 0
    total = 0
    all_preds = []
    all_labels = []
    for x_true, _ in tqdm(data):
        x_true, y_true = x_true[:2], x_true[2]
        y_pred = model.predict(x_true)
        y_pred = y_pred[:, mask_idx, [n_id, o_id]].argmax(axis=1)
        all_preds.append(y_pred)
        y_true = y_true[:, mask_idx]
        y_true = list(map(lambda i: token2id[i], y_true))
        all_labels.append(y_true)
        right += (y_true == y_pred).sum()
        total += len(y_true)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    print(classification_report(all_labels, all_preds, labels=[0, 1]))
    return right / total

# 测试集推理预测
def predict():
    test = pd.read_csv('original_data_X_test.csv', keep_default_na=False)  # , usecols=['Text', 'label_2']
    res = []
    for index, row in test.iterrows():
        text = row['reviews'].lower()

        token_ids, segment_ids = tokenizer.encode(prefix + text, maxlen=maxlen)
        token_ids[mask_idx] = tokenizer._token_mask_id
        token_ids = sequence_padding([token_ids])
        segment_ids = sequence_padding([segment_ids])
        y_pred = model.predict([token_ids, segment_ids])
        y_pred = y_pred[:, mask_idx, [n_id, o_id]].argmax(axis=1)[0]
        res.append(id2label[y_pred])

    test['Judgement'] = res
    test.to_csv('result.csv', index=None)


if __name__ == '__main__':
    evaluator = Evaluator()

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )
    model.load_weights('./best_model.weights')
    predict()

