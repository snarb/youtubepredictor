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