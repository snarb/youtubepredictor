import pandas as pd
import numpy as np
import _pickle as cPickle
import tensorflow as tf
import sys
import time
import math

from Constants import *
from tensorflow.contrib.learn.python.learn.estimators import constants
from tensorflow.contrib.learn.python.learn.estimators.dynamic_rnn_estimator import PredictionType
from sklearn.metrics import mean_squared_error
from tensorflow.contrib.layers import real_valued_column
from math import sqrt
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from statsmodels.tsa.arima_model import ARIMA
from math import floor
import matplotlib.mlab as mlab

SEQUENCE_LENGTH = 8
PREDICTION_DELTA = 4
VIEWS_SCALE_KOEF = 0
BATCH_SIZE = 1

FILE_NAME = "_s:{}_p:{}".format(SEQUENCE_LENGTH, PREDICTION_DELTA)
# views column index - 1. ['channel_subscribers', 'views', 'engagements', 'sentiment']

def LoadData(name):
    global VIEWS_SCALE_KOEF

    all_training_data = np.load('data/' + name + FILE_NAME + "_training_data.npy" )
    all_lables = np.load('data/' + name + FILE_NAME + "_lables.npy")



    # difTr = []
    # for i in range(len(all_training_data)):
    #     seq = all_training_data[i]
    #     lastVal = seq[7, 1]
    #     dif = (all_lables[i] - lastVal) / PREDICTION_DELTA
    #     all_lables[i] = dif
    #     seq[1:8, 1:2] = np.diff(seq[..., 1:2], axis=0)
    #     difTr.append(seq[1:8, ])
    #     min = np.min(seq[1:8, 1:2])
    #     if(min < 0):
    #         bad = 5
    #
    # SEQUENCE_LENGTH = 7

    # sta = np.vstack(difTr)

    # del difTr
    sta = np.vstack(all_training_data)
    df = pd.DataFrame(sta, columns=['channel_subscribers', 'views', 'engagements', 'sentiment'])
    df[df < 0] = 0
    df[df.views == 0] = 1
    df[df.channel_subscribers == 0] = 1
    all_lables[all_lables == 0] = 1



    VIEWS_SCALE_KOEF = floor(np.log(df.max()['views']))

    df['views'] = np.log(df['views']) / VIEWS_SCALE_KOEF
    all_lables = np.log(all_lables) / VIEWS_SCALE_KOEF
    df['channel_subscribers'] = np.log(df['channel_subscribers']) / floor(np.log(df.max()['channel_subscribers']))
    df[df.engagements > 1] = 1
    df[df.sentiment > 1] = 1

    # df['views'] = df['views'].diff()
    # df['views'].diff()[0] = 0



    #df = df.drop(['engagements', 'channel_subscribers', 'sentiment'], axis = 1)

    all_training_data = df['views'].values
    columnsCount = np.size(all_training_data, 1)
    inputs = np.reshape(all_training_data, (-1, BATCH_SIZE, SEQUENCE_LENGTH, columnsCount))
    output = np.reshape(all_lables, (-1, BATCH_SIZE))
    # maxDif = 0
    # bad = 0
    # for i in range(len(inputs)):
    #     lastVal = inputs[i][0][10][1]
    #
    #     cols = inputs[i][0][10]
    #     if(cols[1] < 0 or cols[2] < 0 or cols[3] < 0):
    #         bad+= 1
    #     lable = output[i][0]
    #     dif = lable / lastVal
    #     if(dif > maxDif):
    #         maxDif = dif


    return inputs, output

def MakeLastValuePrediction(inputs):
    prediction = []
    for input in inputs:
        lasViews = input[0][7][1]
        prediction.append(lasViews)

    return prediction


def MakePolifitPrediction(inputs, n):
    prediction = []
    for input in inputs:
        seq = input[0]
        y = seq[:,1]
        x = np.arange(8)

        # calculate polynomial
        z = np.polyfit(x, y, n)
        f = np.poly1d(z)

        pred = f(8)
        prediction.append(pred)

    return prediction

def MakeInterpolatedUnivariateSplinePrediction(inputs, n):
    prediction = []
    for input in inputs:
        seq = input[0]
        y = seq[:,1]
        x = np.arange(8)

        f = InterpolatedUnivariateSpline(x, y, k=n)

        pred = f(8)
        prediction.append(pred)

    return prediction

def MakeSimpleLinPred(inputs):
    prediction = []
    for input in inputs:
        seq = input[0]
        y = seq[:,1]
        delta = y[7] - y[6]
        prediction.append(y[7] + delta)

    return prediction

def ArimaPred(inputs, order):
    prediction = []
    for input in inputs:
        seq = input[0]
        y = seq[:,1]
        x = np.arange(8)


        model = ARIMA(y, order = order)
        results_AR = model.fit()

        prediction.append(results_AR)



def PrintHist(x, std, mean):
    num_bins = 50
    # the histogram of the data
    n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='green', alpha=0.5)
    # add a 'best fit' line
    y = mlab.normpdf(bins, mean, std)
    plt.plot(bins, y, 'r--')
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title(r'Histogram of IQ: $\mu=100$, $\sigma=15$')

    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()

def Train(num_units, cell_type, optimizer, learning_rate):
    tf.logging.set_verbosity(tf.logging.INFO)

    test_inputs, test_outputs = LoadData("TEST")
    train_inputs, train_outputs = LoadData("TRAIN")

    #simple = MakeSimpleLinPred(test_inputs)

    # polPreds_1 = MakeInterpolatedUnivariateSplinePrediction(test_inputs, 1)
    # polPreds_2 = MakeInterpolatedUnivariateSplinePrediction(test_inputs, 2)
    # polPreds_3 = MakeInterpolatedUnivariateSplinePrediction(test_inputs, 3)

    # model = ArimaPred(test_inputs, (2, 1, 0))
    #
    # polPreds_1 = MakePolifitPrediction(test_inputs, 1)
    # polPreds_2 = MakePolifitPrediction(test_inputs, 2)
    # polPreds_3 = MakePolifitPrediction(test_inputs, 3)
    #


    # lastValPreds = MakeLastValuePrediction(test_inputs)
    #
    # rmse = sqrt(mean_squared_error(polPreds, test_outputs))




    columnsCount =  np.size(test_inputs, 3)

    feature_columns = [tf.contrib.layers.real_valued_column("", dimension = columnsCount)]
    estimator = tf.contrib.learn.DynamicRnnEstimator(problem_type = constants.ProblemType.LINEAR_REGRESSION,
                                                     prediction_type = PredictionType.SINGLE_VALUE,
                                                     sequence_feature_columns = feature_columns,
                                                     context_feature_columns = None,
                                                     num_units = num_units,
                                                     cell_type = cell_type, #contrib_rnn.lstm
                                                     optimizer = optimizer,
                                                     learning_rate = learning_rate,
                                                     gradient_clipping_norm=5.0,
                                                     model_dir = "models2/")


    def get_train_inputs():
        inp = tf.constant(train_inputs)
        target = tf.constant(train_outputs)
        return {"": inp}, target

    def get_test_inputs():
        inp = tf.constant(test_inputs)
        target = tf.constant(test_outputs)
        return {"": inp}, target

    estimator.fit(input_fn=get_train_inputs, steps=0)


    # Evaluate accuracy. , metrics={'MSE': tf.contrib.metrics.streaming_mean_squared_error}
    loss_score = estimator.evaluate(input_fn=get_test_inputs, steps=1)["loss"]

    print("\nTest loss: {0:f}\n".format(loss_score))

    predictions = list(estimator.predict({"" : test_inputs}))

    predicted = [prediction['scores'] for prediction in predictions]


    # rmse = sqrt(mean_squared_error(predicted, test_outputs))
    # print ("Log sqrt of RMSE: %f" % rmse)

    originalPredicted = np.exp(np.array(predicted) * VIEWS_SCALE_KOEF)
    originalTestOutputs = np.exp(test_outputs * VIEWS_SCALE_KOEF)

    originalTestOutputs = np.concatenate(originalTestOutputs, axis=0)

    # rmse = sqrt(mean_squared_error(originalPredicted, originalTestOutputs))
    # print ("Original sqrt RMSE: %f" % rmse)

    #rmseAr = (originalTestOutputs - originalPredicted) ** 2
    mapeAr = abs((originalTestOutputs - originalPredicted) / originalTestOutputs)
    rse = ((originalPredicted / originalTestOutputs) - 1)**2
    # rmseL = []
    # mapeL = []
    #
    # for i in range(len(originalTestOutputs)):
    #     prediction = originalPredicted[i]
    #     test_out = originalTestOutputs[i][0]
    #     rmse = sqrt(mean_squared_error([prediction], [test_out]))
    #     mape = (test_out - prediction) / test_out
    #     rmseL.append(rmse)
    #     mapeL.append(mape)
    #
    #
    # rmseAr = np.array(rmseL)
    # mapeAr = np.array(mapeL)

    print("Original RSE stats. Mean: %f, Std: %f Var: %f" % (rse.mean(), rse.std(), rse.var()))

    # print("Original RMSE stats. Mean: %f, Std: %f Var: %f" % (sqrt(rmseAr.mean()), sqrt(rmseAr.std()), sqrt(rmseAr.var())))
    #PrintHist(rmseAr, sqrt(rmseAr.std()), sqrt(rmseAr.mean()))

    print("Original MAPE stats. Mean: %f, Std: %f Var: %f" % (mapeAr.mean(), mapeAr.std(), mapeAr.var()))
    #PrintHist(mapeAr, mapeAr.std(), mapeAr.mean())


    # plot_predicted, = plt.plot(predicted, label='predicted')
    #
    # plot_test, = plt.plot(test_outputs, label='test')
    # plt.legend(handles=[plot_predicted, plot_test])
    # plt.show()

if __name__ == "__main__":
    start = time.time()

    Train(16, 'lstm', 'RMSProp', 1e-3)

    end = time.time()
    delta = end - start
    print("Elapsed: ", delta)

    #Train([int(sys.argv[0]), sys.argv[1], sys.argv[2], float(sys.argv[3])])
