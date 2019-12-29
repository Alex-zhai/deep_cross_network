# -*- coding:utf-8 -*-
# Author : zhaijianwei
# Date : 2019/3/22 14:01


from __future__ import print_function, absolute_import, division

import shutil

import tensorflow as tf

from dataset import get_batch
from models import dcn
from models import feature_columns

tf.app.flags.DEFINE_string('mode', default="train", help="train, test, serving")
FLAGS = tf.app.flags.FLAGS


def model_fn(features, labels, mode):
    feat_columns = feature_columns.get_feature_columns()
    input_feats = tf.expand_dims(tf.feature_column.input_layer(features, feat_columns), axis=1)
    input_shape = (1, input_feats.shape.as_list()[2],)

    model = dcn.create_model(input_shape=input_shape, out_units=1)
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    logits = model(input_feats, training=training)
    loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=logits))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            "scores": tf.sigmoid(logits),
        }
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.PREDICT, predictions=predictions, export_outputs={
            'predict': tf.estimator.export.PredictOutput(predictions)
        })

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(0.001)
        global_step = tf.train.get_or_create_global_step()
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN, loss=loss,
            train_op=optimizer.minimize(loss, global_step)
        )

    if mode == tf.estimator.ModeKeys.EVAL:
        sig_logits = tf.sigmoid(logits)
        eval_threshold = tf.constant(0.5)
        eval_predict = tf.cast(tf.greater(sig_logits, eval_threshold), tf.float32)
        eval_metric_spec = {
            'accuracy': tf.metrics.accuracy(labels, eval_predict),
            'precision': tf.metrics.precision(labels, eval_predict),
            'recall': tf.metrics.recall(labels, eval_predict),
        }
        return tf.estimator.EstimatorSpec(
            mode=tf.estimator.ModeKeys.EVAL, loss=loss, eval_metric_ops=eval_metric_spec
        )


def train_and_eval(train_file_path, val_file_path, save_model_path):
    shutil.rmtree(save_model_path, ignore_errors=True)
    distribution = None
    model_function = model_fn

    config = tf.estimator.RunConfig(model_dir=save_model_path, save_checkpoints_steps=1000,
                                    train_distribute=distribution)

    es_model = tf.estimator.Estimator(
        model_fn=model_function, config=config
    )

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: get_batch(train_file_path, 128),
                                        max_steps=10000)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: get_batch(val_file_path, 128, training=False),
                                      throttle_secs=60,
                                      start_delay_secs=60)
    tf.estimator.train_and_evaluate(estimator=es_model, train_spec=train_spec, eval_spec=eval_spec)


def test(test_file_path, save_model_path):
    es_model = tf.estimator.Estimator(model_fn=model_fn, model_dir=save_model_path)
    test_results = es_model.evaluate(
        input_fn=lambda: get_batch(test_file_path, batch_size=32, training=False), steps=10
    )
    print(test_results)
    predictions = es_model.predict(
        input_fn=lambda: get_batch(test_file_path, batch_size=32, training=False)
    )
    scores = list([p["scores"][0] for p in predictions])
    print(scores)


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.mode == "train":
        train_and_eval("adult.data", "adult.test", "dcn_model_03_22")
    elif FLAGS.mode == "test":
        test("adult.test", "dcn_model_03_22")


if __name__ == '__main__':
    tf.app.run()
