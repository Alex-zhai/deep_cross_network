# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/3/22 10:38


from __future__ import print_function, absolute_import, division
import tensorflow as tf

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num',
    'marital_status', 'occupation', 'relationship', 'race', 'gender',
    'capital_gain', 'capital_loss', 'hours_per_week', 'native_country',
    'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

NUMERIC_COLUMNS = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
CATE_COLUMNS_WITH_VOCA = {
    'education': ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
                  'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
                  '5th-6th', '10th', '1st-4th', 'Preschool', '12th'],
    'marital_status': ['Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
                       'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'],
    'relationship': ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'],
    'workclass': ['Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
                  'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked']
}
CATE_COLUMNS_WITH_EMB = ['occupation']


def get_feature_columns():
    feature_columns = []
    for num_feat_name in NUMERIC_COLUMNS:
        feature_columns.append(tf.feature_column.numeric_column(num_feat_name))
    for cate_feat_name, cate_feat_voca in CATE_COLUMNS_WITH_VOCA.items():
        feature_columns.append(tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(cate_feat_name, cate_feat_voca)))
    for emb_feat_name in CATE_COLUMNS_WITH_EMB:
        feature_columns.append(tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_hash_bucket(emb_feat_name, hash_bucket_size=1000), dimension=32))
    return feature_columns
