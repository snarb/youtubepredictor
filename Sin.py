import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.contrib import learn as tflearn
from tensorflow.contrib import layers as tflayers
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators.dynamic_rnn_estimator import PredictionType
from tensorflow.contrib import rnn as contrib_rnn
from tensorflow.contrib.learn.python.learn.estimators.estimator import SKCompat
import random
from sklearn.metrics import mean_squared_error
from math import sqrt
from math import ceil

def x_sin(x):
    return x * np.sin(x)


def sin_cos(x):
    return pd.DataFrame(dict(a=np.sin(x), b=np.cos(x)), index=x)


def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [3, 4, 5]
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float32)


# def split_data(data, val_size=0.1, test_size=0.1):
#     """
#     splits data to training, validation and testing parts
#     """
#     ntest = int(round(len(data) * (1 - test_size)))
#     nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))
#
#     df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]
#
#     return df_train, df_val, df_test
#
#
# def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
#     """
#     Given the number of `time_steps` and some data,
#     prepares training, validation and test data for an lstm cell.
#     """
#     df_train, df_val, df_test = split_data(data, val_size, test_size)
#     return (rnn_data(df_train, time_steps, labels=labels),
#             rnn_data(df_val, time_steps, labels=labels),
#             rnn_data(df_test, time_steps, labels=labels))
#
#
# def load_csvdata(rawdata, time_steps, seperate=False):
#     data = rawdata
#     if not isinstance(data, pd.DataFrame):
#         data = pd.DataFrame(data)
#
#     train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
#     train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
#     return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)


def generate_data(fct, x, time_steps):
    """generates data with based on a function fct"""
    data = pd.DataFrame(fct(x))
    x = rnn_data(data, time_steps, labels=False)
    y = rnn_data(data, time_steps, labels=True)
    return x, y
    # train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    # train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    # return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)


LOG_DIR = './ops_logs/sin'
RNN_LAYERS = [{'num_units': 5}]
DENSE_LAYERS = None
TRAINING_STEPS = 10000
PRINT_STEPS = TRAINING_STEPS / 10
BATCH_SIZE = 1
SEQUENCE_LENGTH = 3


# regressor = tflearn.Estimator(model_fn=lstm_model(TIMESTEPS, RNN_LAYERS, DENSE_LAYERS),
#                             model_dir=LOG_DIR)

X, y = generate_data(np.sin, np.linspace(0, 100, 1000, dtype=np.float32), SEQUENCE_LENGTH)


Xd = np.reshape(X, (-1, BATCH_SIZE, SEQUENCE_LENGTH))
yd = np.reshape(y,(-1, BATCH_SIZE))
# xData = []
# yData = []
# for _ in range(10000):
#     x = random.random()
#     xData.append(x)
#     y = 2 * x
#     # y = np.sin(x)
#     yData.append(y)


xc = tf.contrib.layers.real_valued_column("")
estimator = tf.contrib.learn.DynamicRnnEstimator(problem_type = constants.ProblemType.LINEAR_REGRESSION,
                                                 prediction_type = PredictionType.SINGLE_VALUE,
                                                 sequence_feature_columns = [xc],
                                                 context_feature_columns = None,
                                                 num_units = 5,
                                                 cell_type = 'lstm', #contrib_rnn.lstm
                                                 optimizer = 'SGD',
                                                 learning_rate = 0.1,
                                                 gradient_clipping_norm=5.0,
                                                 model_dir = LOG_DIR,
                                                 config=tf.contrib.learn.RunConfig(save_checkpoints_secs=10))

# linearEstimator = tf.contrib.learn.LinearRegressor(feature_columns=[xc])
# x = tf.constant(X['train'])
# y = tf.constant(y['train'])

# create a lstm instance and validation monitor
# validation_monitor = tflearn.monitors.ValidationMonitor(X['val'], y['val'],
#                                                      every_n_steps=PRINT_STEPS,
#                                                      early_stopping_rounds=1000)
# print(X['train'])
# print(y['train'])


# Define the test inputs

def get_train_inputs():
    # x1 = tf.random_uniform([BATCH_SIZE, SEQUENCE_LENGTH])
    # y1 = tf.reduce_mean(x1, axis=1)
    # x2 = tf.expand_dims(x1, axis=2)

    x = tf.constant(Xd)
    #x = tf.expand_dims(x, axis=2)
    y = tf.constant(yd)
    #yc = tf.expand_dims(y, axis=1)
    return {"": x}, y


def get_test_inputs():
    x = np.random.random_sample((BATCH_SIZE, SEQUENCE_LENGTH))
    y = np.mean(x, axis=1)
    return {"": x}, y


estimator.fit(input_fn=get_train_inputs, steps=1000)

# Evaluate accuracy.
loss_score = estimator.evaluate(input_fn=get_train_inputs, steps=1)["loss"]

print("\nTest loss: {0:f}\n".format(loss_score))

#data = get_test_inputs()

# test_x = data[0]
# test_y = data[1]

test_x = Xd[0].reshape((1, BATCH_SIZE, SEQUENCE_LENGTH))

test_y = np.reshape(yd[0],(1, BATCH_SIZE))

predictions = list(estimator.predict({"": Xd}))

predicted = [prediction['scores'] for prediction in predictions]

rmse = sqrt(mean_squared_error(predicted, yd))
print ("RMSE: %f" % rmse)
plot_predicted, = plt.plot(predicted, label='predicted')

plot_test, = plt.plot(yd, label='test')
plt.legend(handles=[plot_predicted, plot_test])
plt.show()
