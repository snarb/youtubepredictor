from matplotlib import pyplot as plt
plt.plot()

import pandas as pd
import numpy as np
import tensorflow as tf

# import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation
# import time
# import threading

# from matplotlib.ticker import FormatStrFormatter
# from matplotlib.ticker import FuncFormatter
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
# from sklearn.cluster import KMeans
# from multiprocessing.dummy import Pool as ThreadPool, freeze_support
# from itertools import repeat
# from collections import namedtuple
from DataProducer import DataProducer
from SETTINGS import *
# from sklearn.preprocessing import MinMaxScaler
# from queue import *
from collections import Counter

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

def TestW(initW, testDataProducer, maxPredDelta, minPredDelta):
    X = tf.placeholder("float", shape=(None, SEQUENCE_LENGTH), name="Inputs")  # , COLUMNS_COUNT
    Y = tf.placeholder("float", shape=(None, 1), name="Outputs")

    #W = tf.Variable(tf.ones([SEQUENCE_LENGTH, 1]), name="weight")  # COLUMNS_COUNT


    initW = np.array(initW, dtype='float32')
    W = tf.Variable(initial_value=initW.reshape((SEQUENCE_LENGTH, 1)), name="weight")
    # initW = np.array([-0.02386997, 0.0325182, 0.08833821, 0.14380269, 0.19896919, 0.25399742, 0.30848914], dtype='float32')
    # W = tf.Variable(initial_value =  initW.reshape((7 , 1)), name="weight")



    pred = tf.reduce_sum(tf.matmul(X, W), axis=1)

    #ape = (pred / Y - 1)
    #loss = tf.cond(ape < 1.0, lambda: tf.abs(ape), lambda: tf.square(ape))

    loss = tf.reduce_mean(tf.abs(pred / Y - 1))
    grads = tf.gradients(loss, [W])[0]

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    mapeArs =  TestPerfomance(TEST_STEPS, X, None, maxPredDelta, minPredDelta, pred, sess, testDataProducer)
    mn = np.mean(mapeArs)
    return mn


def TrainModel_2(trainVideos, testVideos, minPredDelta, maxPredDelta,  minLen, maxLen, daysToremove):

    #mnkoef = 1.02






    # Conf median Int:  [1.0378694542222604, 1.039592767081895]
    # mean median conf int 1.03873111065
    # Conf Data Int:  [1.004664572080036, 1.3090827022428926]
    # mean data conf 1.15687363716
    # mean  1.06778123217
    # median  1.03877626202

    #re1 = TestKoef(testDataProducer, 1.01472223)

    # Best stats for mean median conf: (0.3466564213913722, 0.26809115608045392)


    # r5 = TestW([1.01670313], testDataProducer, maxPredDelta, minPredDelta)
    # r6 = TestW([1.01414716], testDataProducer, maxPredDelta, minPredDelta)

    X = tf.placeholder("float", shape=(None, SEQUENCE_LENGTH), name="Inputs")  # , COLUMNS_COUNT
    Y = tf.placeholder("float", shape=(None, 1), name="Outputs")

    #W = tf.Variable(tf.ones([SEQUENCE_LENGTH, 1]), name="weight")  # COLUMNS_COUNT


    # initW = np.array([1.01414716], dtype='float32')
    # W = tf.Variable(initial_value=initW.reshape((1, 1)), name="weight")

    # initW = np.array(
    #     [[-0.87001145],
    #      [-0.51743203],
    #      [-0.19237655],
    #      [0.14398399],
    #      [0.47978213],
    #      [0.81472701],
    #      [1.14944446]], dtype='float32')

    initW = np.array([[-0.24137977],
     [-0.33408758],
     [-0.56187832],
     [-0.64038277],
     [-0.26425102],
     [ 0.62445462],
     [ 2.41232538]], dtype='float32')


    W = tf.Variable(initial_value =  initW.reshape((7 , 1)), name="weight", trainable=False)
    pred = tf.reduce_sum(tf.matmul(X, W), axis=1)

    n_hidden_1 = 32
    n_hidden_2 = 32

    W2 = tf.Variable(tf.random_normal([SEQUENCE_LENGTH, n_hidden_1]), name="w2")
    B2 = tf.Variable(tf.random_normal([n_hidden_1]), name="b2")
    layer_1 = tf.add(tf.matmul(X, W2), B2)
    layer_1 = tf.nn.relu(layer_1)

    W3 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name="w3")
    B3 = tf.Variable(tf.random_normal([n_hidden_2]), name="b3")

    layer_2 = tf.add(tf.matmul(layer_1, W3), B3)
    layer_2 = tf.nn.relu(layer_2)

    w_out = tf.Variable(tf.random_normal([n_hidden_2, 1]), name="w_out")
    b_out = tf.Variable(tf.random_normal([n_hidden_2]), name="b_out")

    out_layer = tf.matmul(layer_2, w_out) + b_out
    loss = tf.reduce_mean(tf.square(out_layer / Y - 1))


    #ape = (pred / Y - 1)
    #loss = tf.cond(ape < 1.0, lambda: tf.abs(ape), lambda: tf.square(ape))

    #loss = tf.reduce_mean(tf.square(pred/ Y - 1))
    #loss = tf.reduce_mean(tf.abs(tf.exp(pred) / tf.exp(Y) - 1))

    #loss = tf.reduce_mean(tf.abs(pred/ Y - 1))
    #grads = tf.gradients(loss, [W])[0]

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    #optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    #mapeArs =  TestClassPerfomance(X, None, maxPredDelta, minPredDelta, pred, sess, trainDataProducer, testDataProducer)


    losses = []
    deltas = Counter()

    trainVideos = shuffle(trainVideos)
    testVideos = shuffle(testVideos)

    trainDataProducer = DataProducer(trainVideos, seqLen = SEQUENCE_LENGTH, deltasToExtractCount = TRAIN_DELTAS_TO_EXTRACT, minPredDelta = minPredDelta,
                                     maxPredDelta = maxPredDelta, minVideoLen= minLen, maxLenToUse= maxLen, daysToRemove = daysToremove)
    testDataProducer = DataProducer(testVideos, seqLen = SEQUENCE_LENGTH, deltasToExtractCount = TEST_DELTAS_TO_EXTRACT, minPredDelta = minPredDelta,
                                    maxPredDelta = maxPredDelta, minVideoLen= minLen, maxLenToUse= maxLen, daysToRemove = daysToremove)



    mapeArs =  TestPerfomance(TEST_STEPS, X, None, maxPredDelta, minPredDelta, pred, sess, testDataProducer)
    print("Init perf: Mean: Median")
    minMean = np.mean(mapeArs)
    minMedian = np.median(mapeArs)
    minWmean = 0
    minWmedian = 0
    print(minMean, minMedian)

    for step in range(TRAIN_STEPS):
        data, lables, lableDeltas = zip(*trainDataProducer.GetNextBatch())
        #deltas.update(lableDeltas)
        inputs = np.reshape(data, (BATCH_SIZE, SEQUENCE_LENGTH))
        output = np.reshape(lables, (BATCH_SIZE, 1))
        _, curLoss = sess.run([train, loss], feed_dict={X: inputs, Y: output})
        losses.append(curLoss)


        if (step % 15000 == 0):
            Wval = sess.run(W, feed_dict={X: inputs, Y: output})
            #print(mnkoef, Wval)
            cp = sess.run(pred, feed_dict={X: inputs})
           # pr2 = inputs * mnkoef
           # print(cp, pr2)
            print("AVG loss: ", np.array(losses).mean())
            losses.clear()

            mapeArs = TestPerfomance(TEST_STEPS, X, None, maxPredDelta, minPredDelta, pred, sess, testDataProducer)
            mean = mapeArs.mean()
            median = np.median(mapeArs)
            print(mean, median)

            pritW = False
            if(mean < minMean):
                pritW = True
                minMean = mean
                minWmean = sess.run(W, feed_dict={X: inputs, Y: output})

            if(median < minMedian):
                pritW = True
                minMedian = median
                minWmedian = sess.run(W, feed_dict={X: inputs, Y: output})

            if(pritW):
                print(sess.run(W, feed_dict={X: inputs, Y: output}))


    mapeArs =  TestPerfomance(TEST_STEPS, X, None, maxPredDelta, minPredDelta, pred, sess, testDataProducer)
    mn = np.mean(mapeArs)
    print(mn)
    resW = sess.run(W, feed_dict={X: inputs, Y: output})
    print(resW)
    print("Done")
    #mnkoef = np.array(koefs).mean()




def TestKoef(testDataProducer, mnkoef):
    mapeArsTr2 = []

    for step in range(TEST_STEPS):
        data, lables, lableDeltas = zip(*testDataProducer.GetNextBatch())
        predictedLables = data[0] * mnkoef
        mapeAr = abs(1 - np.exp(predictedLables) / np.exp(lables))
        #mapeAr = abs(1 - predictedLables / lables)
        mapeArsTr2.append(mapeAr.mean())

    return np.mean(mapeArsTr2), np.median(mapeArsTr2)


def TrainModel(trainVideos, testVideos, seqLen, minPredDelta, maxPredDelta,  minLen, maxLen, daysToremove):
    trainVideos = shuffle(trainVideos)
    testVideos = shuffle(testVideos)

    trainDataProducer = DataProducer(trainVideos, seqLen = SEQUENCE_LENGTH, deltasToExtractCount = TRAIN_DELTAS_TO_EXTRACT, minPredDelta = minPredDelta,
                                     maxPredDelta = maxPredDelta, minVideoLen= minLen, maxLenToUse= maxLen, daysToRemove = daysToremove)
    testDataProducer = DataProducer(testVideos, seqLen = SEQUENCE_LENGTH, deltasToExtractCount = TEST_DELTAS_TO_EXTRACT, minPredDelta = minPredDelta,
                                    maxPredDelta = maxPredDelta, minVideoLen= minLen, maxLenToUse= maxLen, daysToRemove = daysToremove)

    lableShifts = tf.placeholder("float", shape=(BATCH_SIZE, 1), name="lablesShift")
    X = tf.placeholder("float", shape=(BATCH_SIZE, SEQUENCE_LENGTH), name="viewsHistoryInput")
    Y = tf.placeholder("float", shape=(BATCH_SIZE, 1), name="Outputs")

    initW = np.array([ 0.15289065,  0.19918267,  0.2341571 ,  0.26491246,  0.29034972,
        0.31426078,  0.33612537], dtype='float32')

    W = tf.Variable(initial_value = initW, name="weight")

    inLsw = np.array([0.15289067, 0.19918275, 0.2341571, 0.2649124, 0.29034981,
           0.31426069, 0.33612537], dtype='float32')

    lsW = tf.Variable(initial_value = inLsw.reshape(SEQUENCE_LENGTH), name="lableShiftsWeights")

    initBw = np.array([ 1.7117914], dtype='float32')

    B = tf.Variable(tf.ones(initial_value = initBw), name="bias")
    predDeltaKoefs = (B  + lableShifts) * lsW
    combinedWeights = predDeltaKoefs * W
    res = X * combinedWeights
    pred = tf.reduce_sum(res, axis=1)
    loss = tf.reduce_mean(tf.square(pred / Y - 1))
    #grads = tf.gradients(loss, [W])[0]

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
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
        inputLableDeltas = np.reshape(lableDeltas, (BATCH_SIZE, 1))
        inputLableDeltas = ((inputLableDeltas - minPredDelta) / (maxPredDelta -  minPredDelta))

        #inputLableDeltas = inputLableDeltas / (maxPredDelta - 1)

        output = np.reshape(lables, (BATCH_SIZE, 1))

        predictedLables = sess.run(pred, feed_dict={X: inputs, lableShifts: inputLableDeltas})
        lsw_v = sess.run(lsW, feed_dict={X: inputs, Y: output})
        b_v = sess.run(B, feed_dict={X: inputs, Y: output})

        _, curLoss = sess.run([train, loss], feed_dict={X: inputs, Y: output, lableShifts: inputLableDeltas})
        q.append(curLoss)
        # for i in range(10):
        #     _, curLoss_i = sess.run([train, loss], feed_dict={X: inputs, Y: output, lableShifts: lableDeltas})

        #avgLoss = ApproxRollingAverage(avgLoss, curLoss, TO_AVG_COUNT)
        if (step % 5000 == 0):
            mapeAr = abs(1 - np.exp(predictedLables.ravel()) / np.exp(lables))
            print("Training mapeAr: ", mapeAr.mean())
            print("AVG loss: ", np.array(q).mean())
            q.clear()
            #print("AVG loss: ", np.array(list(q.queue)).mean())
            #print("W = ", sess.run(W))

    mapeArs =  TestPerfomance(TEST_STEPS, X, lableShifts, maxPredDelta, minPredDelta, pred, sess, testDataProducer)

    mapeArs = np.array(mapeArs)

    saver = tf.train.Saver()
    saver.save(sess, 'predDeltaM')

    return mapeArs.mean(), mapeArs.var()

def TrainKerasModel(trainVideos, testVideos, seqLen, minPredDelta, maxPredDelta,  minLen, maxLen, daysToremove):

    model = Sequential()
    #model.add(Dense(7, activation='linear', input_dim=SEQUENCE_LENGTH))
    model.add(Dense(32, activation='relu', input_dim=SEQUENCE_LENGTH))
    # model.add(Dense(32, activation='relu'))
    # model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='nadam', loss='mean_absolute_percentage_error')



    trainVideos = shuffle(trainVideos)
    testVideos = shuffle(testVideos)

    trainDataProducer = DataProducer(trainVideos, seqLen = SEQUENCE_LENGTH, deltasToExtractCount = TRAIN_DELTAS_TO_EXTRACT, minPredDelta = minPredDelta,
                                     maxPredDelta = maxPredDelta, minVideoLen= minLen, maxLenToUse= maxLen, daysToRemove = daysToremove)
    testDataProducer = DataProducer(testVideos, seqLen = SEQUENCE_LENGTH, deltasToExtractCount = TEST_DELTAS_TO_EXTRACT, minPredDelta = minPredDelta,
                                    maxPredDelta = maxPredDelta, minVideoLen= minLen, maxLenToUse= maxLen, daysToRemove = daysToremove)

    datas_l = []
    lables_l = []
    for step in range(10000):
        data, lables, lableDeltas = zip(*trainDataProducer.GetNextBatch())
        datas_l.append(data[0] / 18)
        lables_l.append(lables[0] / 18)

    model.fit(np.array(datas_l), np.array(lables_l), epochs=50, batch_size=40)
    predictedLables = model.predict(np.array(datas_l))
    mapeAr = abs(1 - np.exp(predictedLables.ravel() * 18) / np.exp(np.array(lables_l) * 18))
    n = mapeAr.mean()

    test_datas_l = []
    test_lables_l = []
    for step in range(10000):
        data, lables, lableDeltas = zip(*testDataProducer.GetNextBatch())
        test_datas_l.append(data[0])
        test_lables_l.append(lables[0])

    print("Done")

        # predictedLables = model.predict()
        # mapeAr = abs(1 - np.exp(predictedLables.ravel()) / np.exp(lables))
        # mapeArs.append(mapeAr.mean())


def TestClassPerfomance(X, lableShifts, maxPredDelta, minPredDelta, pred, sess, trainDataProducer, testProd):
    mapeArs = []

    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=SEQUENCE_LENGTH))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

    #model.add(Dense(1, activation='sigmoid'))
    # model.compile(optimizer='adam',
    #               loss='binary_crossentropy',
    #               metrics=['accuracy'])

    model.compile(optimizer='adam',
                  loss='mean_absolute_percentage_error')

    datas = []
    lables = []

    data_reg = []
    lables_reg = []
    datas1 = []
    datas2 = []
    lables1 = []
    lables2 = []

    for step in range(15000):
        data, lables, lableDeltas = zip(*testProd.GetNextBatch())
        inputs = np.reshape(data, (BATCH_SIZE, SEQUENCE_LENGTH))
        inputLableDeltas = np.reshape(lableDeltas, (BATCH_SIZE, 1))
        inputLableDeltas = ((inputLableDeltas - minPredDelta) / (maxPredDelta - minPredDelta))

        if(lableShifts is not None):
            predictedLables = sess.run(pred, feed_dict={X: inputs, lableShifts: inputLableDeltas})
        else:
            predictedLables = sess.run(pred, feed_dict={X: inputs})

        mapeAr = abs(1 - np.exp(predictedLables.ravel()) / np.exp(lables))
        mapeArs.append(mapeAr.mean())


        data_reg.append(data[0])
        lables_reg.append(lables[0])


        if mapeAr < 0.1:
            lables1.append(0)
            datas1.append(data[0])
        else:
            lables2.append(1)
            datas2.append(data[0])

    targetLen = min(len(lables1), len(lables2))

    joinedData = datas1[:targetLen] + datas2[:targetLen]
    joinedLables = lables1[:targetLen] + lables2[:targetLen]

    # model.fit(np.array(joinedData), np.array(joinedLables), epochs=20, batch_size=32)
    #model.fit(np.array(data_reg), np.array(lables_reg), epochs=500, batch_size=10)

    for step in range(7000):
        data, lables, lableDeltas = zip(*testProd.GetNextBatch())
        inputs = np.reshape(data, (BATCH_SIZE, SEQUENCE_LENGTH))
        inputLableDeltas = np.reshape(lableDeltas, (BATCH_SIZE, 1))
        inputLableDeltas = ((inputLableDeltas - minPredDelta) / (maxPredDelta - minPredDelta))

        if(lableShifts is not None):
            predictedLables = sess.run(pred, feed_dict={X: inputs, lableShifts: inputLableDeltas})
        else:
            predictedLables = sess.run(pred, feed_dict={X: inputs})

        mapeAr = abs(1 - np.exp(predictedLables.ravel()) / np.exp(lables))
        mapeArs.append(mapeAr.mean())




        if mapeAr < 0.1:
            lables1.append(0)
            datas1.append(data[0])
        else:
            lables2.append(1)
            datas2.append(data[0])

    targetLen = min(len(lables1), len(lables2))

    joinedData = datas1[:targetLen] + datas2[:targetLen]
    joinedLables = lables1[:targetLen] + lables2[:targetLen]
    gg = model.evaluate(np.array(joinedData), np.array(joinedLables))

    return mapeArs


def TestPerfomance(testSteps, X, lableShifts, maxPredDelta, minPredDelta, pred, sess, testDataProducer):
    mapeArs = []

    for step in range(testSteps):
        data, lables, lableDeltas = zip(*testDataProducer.GetNextBatch())
        inputs = np.reshape(data, (BATCH_SIZE, SEQUENCE_LENGTH))
        inputLableDeltas = np.reshape(lableDeltas, (BATCH_SIZE, 1))
        inputLableDeltas = ((inputLableDeltas - minPredDelta) / (maxPredDelta - minPredDelta))

        if(lableShifts is not None):
            predictedLables = sess.run(pred, feed_dict={X: inputs, lableShifts: inputLableDeltas})
        else:
            predictedLables = sess.run(pred, feed_dict={X: inputs})

        mapeAr = abs(1 - np.exp(predictedLables.ravel()) / np.exp(lables))
        mapeArs.append(mapeAr.mean())

    return np.array(mapeArs)


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






