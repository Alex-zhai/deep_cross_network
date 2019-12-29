# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/3/22 14:26

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from models import feature_columns


def parse_csv(value):
    columns = tf.decode_csv(value, record_defaults=feature_columns._CSV_COLUMN_DEFAULTS)
    features = dict(zip(feature_columns._CSV_COLUMNS, columns))
    labels = tf.reshape(tf.cast(tf.equal(features.pop('income_bracket'), '>50K'), tf.int32), [-1])
    return features, labels


def get_batch(data_file, batch_size=128, training=True):
    dataset = tf.data.TextLineDataset(data_file)
    if training:
        dataset = dataset.shuffle(100)
        dataset = dataset.repeat()
    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels


if __name__ == '__main__':
    features, labels = get_batch("adult.data", batch_size=128, training=True)
    sess = tf.Session()
    for i in range(1):
        print(sess.run([features, labels]))
