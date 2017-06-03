import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from math import floor
#import Test
import time
import threading
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import FuncFormatter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from multiprocessing.dummy import Pool as ThreadPool, freeze_support
from itertools import repeat
from collections import namedtuple
from DataProducer import DataProducer
from SETTINGS import *
from sklearn.preprocessing import MinMaxScaler
from queue import *

TO_AVG_COUNT = 10000

# SEQUENCE_LENGTH = 8
# PREDICTION_DELTA = 4
#VIDEO_LEN_LIMIT = 50

#MIN_VIEWS = 100  # 1000 ?

#BATCH_SIZE = 10


#fig, ax = plt.subplots()
#plt.figure(figsize=(12, 5))
#ax.xaxis.set_major_formatter(FormatStrFormatter('%.5f'))


def LoadData(name):
    df = pd.read_csv('data/' + name + '.csv',
                     index_col=['date'],
                     dtype={"video_id": object, "views": np.float64},
                     parse_dates=['date'])
    return df


def DfToList(df):
    listOfVideoViews = list(df.groupby('video_id')['views'].apply(pd.DataFrame.as_matrix))
    return listOfVideoViews


# def ExtractSequenes(listOfVideoViews, seqLen, predDelta, minLen, maxLen, daysToRemove  = 30):
#
#
#     if (not minLen):
#         minLen = 0
#
#     if (not maxLen):
#         maxLen = 10e+10
#
#     datas = []
#     lables = []
#
#     datas2 = []
#     lables2 = []
#
#
#     for i in range(len(listOfVideoViews)):
#         s = listOfVideoViews[i]
#         if (len(s) > minLen and len(s) < maxLen):
#             data, lable = rolling_window(s[daysToRemove:], window=seqLen, lablesShift=predDelta)
#             datas.extend(data)
#             lables.extend(lable)
#
#             data2, lable2 = rolling_window(s[daysToRemove + seqLen - 1:], window=1, lablesShift=predDelta)
#             datas2.extend(data2)
#             lables2.extend(lable2)
#
#     datas_res = np.log(datas)
#     lables_res = np.log(lables)
#
#     datas_res2 = np.log(datas2)
#     lables_res2 = np.log(lables2)
#
#     datas_res = np.clip(datas_res, a_min=STABLE_DELTA, a_max=99999999999)
#     lables_res = np.clip(lables_res, a_min=STABLE_DELTA, a_max=99999999999)
#
#     datas_res2 = np.clip(datas_res2, a_min=STABLE_DELTA, a_max=99999999999)
#     lables_res2 = np.clip(lables_res2, a_min=STABLE_DELTA, a_max=99999999999)
#
#     return datas_res, lables_res, datas_res2, lables_res2
#

# def ExtractSequenes(listOfVideoViews, seqLen, predDelta, minLen, maxLen):
#     if (not minLen):
#         minLen = 0
#
#     if (not maxLen):
#         maxLen = 10e+10
#
#     koefs = []
#     datas = []
#     lables = []
#     maxV = 0
#     maxI = 0
#
#     for i in range(len(listOfVideoViews)):
#         s = listOfVideoViews[i]
#         if (len(s) > minLen and len(s) < maxLen):
#
#             # if(i == 297553):
#             #     print("target")
#             daysToRemove = 0
#             passed = False
#
#             for j in range(len(s)):
#                 if (s[j] > 20):
#                     daysToRemove = j
#                     passed = True
#                     break
#
#             if(daysToRemove > 30):
#                 print("fun")
#
#             if (passed and (len(s) - daysToRemove > minLen)):
#                 realDaysToRemove = max(daysToRemove, 30)
#                 data, lable = rolling_window(s[realDaysToRemove:], window=seqLen, lablesShift=predDelta)
#                 # if(data.min() < 20):
#                 #     print("bad")
#
#                 datas.extend(data)
#                 lables.extend(lable)
#
#                 #                 dataMean = np.mean(data)
#                 #                 koef = np.log(lable) / np.log(dataMean)
#                 #                 koefs.append(koef)
#
#     datas_res = np.log(datas)
#     lables_res = np.log(lables)
#
#     datas_res = np.clip(datas_res, a_min=STABLE_DELTA, a_max=99999999999)
#     lables_res = np.clip(lables_res, a_min=STABLE_DELTA, a_max=99999999999)
#
#     #     print(len(koefs))
#     #     print(np.mean(koefs))
#     #     plt.plot(koefs)
#     #     plt.show()
#     #     for i in range(len(datas_res)):
#     #         data = datas_res[i]
#     #         lable = lables_res[i]
#
#     #         koef = lable / data
#     #         curMax = koef.max()
#     #         if (curMax > maxV):
#     #             maxV = curMax
#     #             maxI = i
#
#     #         if (koef.max() > 10000):
#     #             print("Bad")
#
#     return datas_res, lables_res


# def ApproxRollingAverage (avg, new_sample, toAvgCount):
#
#     avg -= avg / toAvgCount;
#     avg += new_sample / toAvgCount;
#
#     return avg;


def TrainModel(trainVideos, testVideos, seqLen, minPredDelta, maxPredDelta,  minLen, maxLen):
    trainVideos = shuffle(trainVideos)
    testVideos = shuffle(testVideos)

    trainDataProducer = DataProducer(trainVideos, seqLen = seqLen, deltasToExtractCount = TRAIN_DELTAS_TO_EXTRACT, minPredDelta = minPredDelta, maxPredDelta = maxPredDelta,  minLen = minLen, maxLen = maxLen)
    testDataProducer = DataProducer(testVideos, seqLen = seqLen, deltasToExtractCount = TEST_DELTAS_TO_EXTRACT, minPredDelta = minPredDelta, maxPredDelta = maxPredDelta,  minLen = minLen, maxLen = maxLen)


    lableShifts = tf.placeholder("float", shape=(BATCH_SIZE, 1), name="lablesShift")
    X = tf.placeholder("float", shape=(BATCH_SIZE, SEQUENCE_LENGTH), name="viewsHistoryInput")
    Y = tf.placeholder("float", shape=(BATCH_SIZE, 1), name="Outputs")

    W = tf.Variable(tf.ones([SEQUENCE_LENGTH]), name="weight")
    lsW = tf.Variable(tf.ones([SEQUENCE_LENGTH]), name="lableShiftsWeights")
    #B = tf.Variable(tf.zeros([1]), name="bias")

    predDeltaKoefs = lsW  * lableShifts
    #predDeltaKoefs = lsW * lableShift

    combinedWeights = predDeltaKoefs * W
    #res = tf.matmul(X, combinedWeights)
    res = X * combinedWeights
    pred = tf.reduce_sum(res, axis=1)
    loss = tf.reduce_mean(tf.square(pred / Y - 1))
    #grads = tf.gradients(loss, [W])[0]

    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    #avgLoss = 0
    #q = Queue(maxsize=5000)
    q = []

    for step in range(TRAIN_STEPS):
        data, lables, lableDeltas = zip(*trainDataProducer.GetNextBatch())
        inputs = np.reshape(data, (BATCH_SIZE, SEQUENCE_LENGTH))
        lableDeltas = np.reshape(lableDeltas, (BATCH_SIZE, 1)) / maxPredDelta
        output = np.reshape(lables, (BATCH_SIZE, 1))

        #cp = sess.run(pred, feed_dict={X: inputs, lableShifts: lableDeltas})
        _, curLoss = sess.run([train, loss], feed_dict={X: inputs, Y: output, lableShifts: lableDeltas})
        q.append(curLoss)
        # for i in range(10):
        #     _, curLoss_i = sess.run([train, loss], feed_dict={X: inputs, Y: output, lableShifts: lableDeltas})

        #avgLoss = ApproxRollingAverage(avgLoss, curLoss, TO_AVG_COUNT)
        if (step % 5000 == 0):
            print("AVG loss: ", np.array(q).mean())
            q.clear()
            #print("AVG loss: ", np.array(list(q.queue)).mean())
            #print("W = ", sess.run(W))

    mapeArs = []

    for step in range(TEST_STEPS):
        data, lables, lableDeltas = zip(*testDataProducer.GetNextBatch())
        inputs = np.reshape(data, (BATCH_SIZE, SEQUENCE_LENGTH))
        lableDeltas = np.reshape(lableDeltas, (BATCH_SIZE, 1))
        predictedLables = sess.run(pred, feed_dict={X: inputs, lableShifts: lableDeltas})
        mapeAr = abs(1 - np.exp(predictedLables) / np.exp(lables))
        mapeArs.append(mapeAr)

    mapeArs = np.array(mapeArs)

    return mapeArs.mean(), mapeArs.var()

def GetRegrouppedViews(trainDf, testDf):
    return DfToList(trainDf), DfToList(testDf)


def AddTrend(x, y):
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--")


def Plot(dataL, step=1):
    rseMean, rseVar, mapeMean, mapeVar = zip(*dataL)
    # calc the trendline
    x = np.arange(start=0, stop=len(mapeMean), step=step)
    y = mapeMean
    plt.title('MAPE mean')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    AddTrend(x, y)
    plt.plot(x, y)
    plt.show()

    x = np.arange(len(mapeVar))
    y = mapeVar
    plt.title('MAPE var')
    AddTrend(x, y)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.plot(x, y)
    plt.show()

    minInd = np.argmin(mapeMean)

    print("Min at index: {}; val: {}%".format(minInd, x[minInd] * 100.0))






