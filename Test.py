from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import urllib
import urllib.request

import numpy as np
import tensorflow as tf

# Data sets
IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

def main():
  # If the training and test sets aren't stored locally, download them.
  if not os.path.exists(IRIS_TRAINING):
      with urllib.request.urlopen(IRIS_TRAINING_URL) as raw:
            st = raw.read().decode('utf8')
            with open(IRIS_TRAINING, "w") as f:
              f.write(st)

  if not os.path.exists(IRIS_TEST):
      with urllib.request.urlopen(IRIS_TRAINING_URL) as raw:
            st = raw.read().decode('utf8')
            with open(IRIS_TEST, "w") as f:
              f.write(st)

  # Load datasets.
  training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
  test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

  # Build 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[10, 20, 10],
                                              n_classes=3,
                                              model_dir="/tmp/iris_model")
  # Define the training inputs
  def get_train_inputs():
    x = tf.constant(training_set.data)
    y = tf.constant(training_set.target)

    return x, y

  # Fit model.
  classifier.fit(input_fn=get_train_inputs, steps=2000)

  # Define the test inputs
  def get_test_inputs():
    x = tf.constant(test_set.data)
    y = tf.constant(test_set.target)

    return x, y

  # Evaluate accuracy.
  accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                       steps=1)["accuracy"]

  print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

  # Classify two new flower samples.
  def new_samples():
    return np.array(
      [[6.4, 3.2, 4.5, 1.5],
       [5.8, 3.1, 5.0, 1.7]], dtype=np.float32)

  predictions = list(classifier.predict(input_fn=new_samples))

  print(
      "New Samples, Class Predictions:    {}\n"
      .format(predictions))

def subsequences(ts, window, lablesShift):
    DATA_SIZE = 8
    inputRowsCount = ts.shape[0]
    columnsCount = ts.shape[1]
    rowsCount = inputRowsCount - window - lablesShift + 1
    shape = (rowsCount, window, columnsCount)
    strides = (DATA_SIZE * columnsCount,  DATA_SIZE * columnsCount,  DATA_SIZE)
    data = np.lib.stride_tricks.as_strided(ts, shape=shape, strides=strides)

    lables = ts[window - 1 + lablesShift::1]
    return data, lables

def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window +1, window)
    strides = a.strides + (a.strides[-1],)
    lablesShape = a.shape[:-1] + (a.shape[-1] - window + 4 + 1, window)

    data =  np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    shape = np.lib.stride_tricks.as_strided(a, shape=lablesShape, strides=strides)
    return data, shape


if __name__ == "__main__":
    x = np.arange(48.0).reshape(12, 4)
    for i in range(12):
        x[i ,0] =  10*i
        x[i, 1] =  10*i +1
        x[i, 2] = 10*i + 2
        x[i, 3] =  10*i + 3

    data, shape = subsequences(x, 4, 2)


    main()

#
# import pandas as pd
# import numpy as np
# import tensorflow as tf
#
# from math import floor
#
# SEQUENCE_LENGTH = 8
# PREDICTION_DELTA = 4
# VIEWS_SCALE_KOEF = 0
# BATCH_SIZE = 20
# COLUMNS_COUNT = 4
# TRAIN_STEPS = 50000
# DISPLAY_STEPS = 200
#
# FILE_NAME = "_s:{}_p:{}".format(SEQUENCE_LENGTH, PREDICTION_DELTA)
# # views column index - 1. ['channel_subscribers', 'views', 'engagements', 'sentiment']
#
# def unison_shuffled_copies(a, b):
#     assert len(a) == len(b)
#     p = np.random.permutation(len(a))
#     return a[p], b[p]
#
# def LoadData(name):
#     global VIEWS_SCALE_KOEF
#
#     all_training_data = np.load('data/' + name + FILE_NAME + "_training_data.npy" )
#     all_lables = np.load('data/' + name + FILE_NAME + "_lables.npy")
#
#     sta = np.vstack(all_training_data)
#     df = pd.DataFrame(sta, columns=['channel_subscribers', 'views', 'engagements', 'sentiment'])
#     df[df < 0] = 0
#     df[df.views == 0] = 1
#     df[df.channel_subscribers == 0] = 1
#     all_lables[all_lables == 0] = 1
#
#     df['views'] = np.log(df['views'])
#     all_lables = np.log(all_lables)
#     df['channel_subscribers'] = np.log(df['channel_subscribers'])
#     df[df.engagements > 1] = 1
#     df[df.sentiment > 1] = 1
#
#
#     all_training_data = df.values
#
#     koef = SEQUENCE_LENGTH * BATCH_SIZE * COLUMNS_COUNT
#     optimizedSize = int(int(all_training_data.size / koef) * koef)
#     trI = int(optimizedSize / COLUMNS_COUNT)
#     lInd = int(trI / SEQUENCE_LENGTH)
#     all_training_data = all_training_data[:trI]
#     all_lables = all_lables[:lInd]
#
#     inputs = np.reshape(all_training_data, (-1, BATCH_SIZE, SEQUENCE_LENGTH, COLUMNS_COUNT))
#     output = np.reshape(all_lables, (-1, BATCH_SIZE, 1))
#
#
#
#     return inputs, output
#
#
# train_inputs, train_lables = LoadData("TRAIN")
#
#
# X = tf.placeholder("float", shape=(BATCH_SIZE, SEQUENCE_LENGTH, COLUMNS_COUNT), name="Inputs") #, COLUMNS_COUNT
# Y = tf.placeholder("float", shape=(BATCH_SIZE, 1), name="Outputs")
#
# #W = tf.Variable(tf.random_uniform([SEQUENCE_LENGTH, BATCH_SIZE], -1.0, 1.0), name="weight") # COLUMNS_COUNT
# W = tf.Variable(tf.ones([BATCH_SIZE, COLUMNS_COUNT, SEQUENCE_LENGTH]), name="weight") # COLUMNS_COUNT
#
# pred = tf.reduce_sum(tf.matmul(X,W))
#
# #loss = tf.reduce_mean(tf.square(pred - Y))
# loss = tf.reduce_mean(tf.square(pred/Y - 1))
#
# grads = tf.gradients(loss, [W])[0]
#
#
# optimizer = tf.train.AdamOptimizer(1e-4)
# train = optimizer.minimize(loss)
#
# # Before starting, initialize the variables.  We will 'run' this first.
# init = tf.initialize_all_variables()
#
# # Launch the graph.
# sess = tf.Session()
# sess.run(init)
#
# # Fit the line.
# for step in range(TRAIN_STEPS):
#     input = train_inputs[step]
#     lable = train_lables[step]
#     sess.run(train, feed_dict={X: input, Y: lable})
#     if step % DISPLAY_STEPS == 0:
#         print(step, sess.run(loss, feed_dict={X: input, Y: lable}))
#
# test_inputs, test_lables = LoadData("TEST")
# test_inputs, test_lables  = unison_shuffled_copies(test_inputs, test_lables )
#
# rseL = []
# mapeL = []
#
# for step in range(len(test_inputs)):
#     input = test_inputs[step]
#     lable = test_lables[step]
#     prediction =  sess.run(pred, feed_dict={X: input})
#
#     originalPredicted = np.exp(np.array(prediction))
#     originalTestOutputs = np.exp(lable)
#
#     rse = ((originalPredicted / originalTestOutputs) - 1) ** 2
#     mapeAr = abs(originalTestOutputs - originalPredicted) / originalTestOutputs
#
#     rseL.append(rse)
#     mapeL.append(mapeAr)
#
#     # print(lable, sess.run(pred, feed_dict={X: input}))
#     # print(step, sess.run(loss, feed_dict={X: input, Y: lable}))
#
# rseA = np.array(rseL)
# mapeA = np.array(mapeL)
#
#
# print("Original RSE stats. Mean: %f, Std: %f Var: %f" % (rseA.mean(), rseA.std(), rseA.var()))
# print("Original MAPE stats. Mean: %f, Std: %f Var: %f" % (mapeA.mean(), mapeA.std(), mapeA.var()))
#
#
#         # print(step, sess.run(W, feed_dict={X: input, Y: lable}))

