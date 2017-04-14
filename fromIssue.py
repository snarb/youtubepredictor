import random
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators.dynamic_rnn_estimator import PredictionType
import numpy as np
import pandas as pd



BATCH_SIZE = 32
SEQUENCE_LENGTH = 16

xc = tf.contrib.layers.real_valued_column("")
estimator = tf.contrib.learn.DynamicRnnEstimator(problem_type=constants.ProblemType.LINEAR_REGRESSION,
                                                 prediction_type=PredictionType.SINGLE_VALUE,
                                                 sequence_feature_columns=[xc],
                                                 context_feature_columns=None,
                                                 num_units=5,
                                                 cell_type='lstm',
                                                 optimizer='SGD',
                                                 learning_rate=0.1)


def get_train_inputs():
    x = tf.random_uniform([BATCH_SIZE, SEQUENCE_LENGTH])
    y = tf.reduce_mean(x, axis=1)
    x = tf.expand_dims(x, axis=2)
    return {"": x}, y


estimator.fit(input_fn=get_train_inputs, steps=1000)
