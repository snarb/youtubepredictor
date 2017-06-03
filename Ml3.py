import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from math import floor
import Test
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# SEQUENCE_LENGTH = 8
# PREDICTION_DELTA = 4
VIDEO_LEN_LIMIT = 50

BATCH_SIZE = 1
TRAIN_STEPS = 90000
LEARNING_RATE = 1e-4
STABLE_DELTA = 1e-8

fig, ax = plt.subplots()
plt.figure(figsize=(12, 5))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.5f'))


def rolling_window(a, window, lablesShift):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)

    data = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    lables = a[window - 1 + lablesShift::1]
    return data[:-lablesShift], lables


def LoadData(name):
    df = pd.read_csv('data/' + name + '.csv',
                     index_col=['datetime'],
                     dtype={"video_id": object, "views": np.float64},
                     parse_dates=['datetime'],
                     )
    return df


def GetNextBatch(inputs, lables, batchId):
    batchId = batchId % len(inputs)
    startIndex = batchId * BATCH_SIZE
    endIndex = (batchId + 1) * BATCH_SIZE
    if (len(inputs) < endIndex):
        startIndex = 0
        endIndex = BATCH_SIZE

    return (inputs[startIndex:endIndex], lables[startIndex:endIndex])


def Regroup(df, timeGrouperIndex):
    resDf = df.groupby(['video_id', pd.TimeGrouper(timeGrouperIndex)]).mean()
    resDf = resDf.reset_index().set_index('datetime')
    listOfVideoViews = list(resDf.groupby('video_id')['views'].apply(pd.DataFrame.as_matrix))
    return listOfVideoViews


def ExtractSequenes(listOfVideoViews, seqLen, predDelta, minLen, maxLen):
    if(not minLen):
        minLen = 0

    if (not maxLen):
        maxLen = 10e+10

    datas = []
    lables = []
    for s in listOfVideoViews:
        if (len(s) > minLen and len(s) < maxLen):
            data, lable = rolling_window(s, window=seqLen, lablesShift=predDelta)
            datas.extend(data)
            lables.extend(lable)

    datas = np.log(datas) + STABLE_DELTA
    lables = np.log(lables) + STABLE_DELTA
    return datas, lables


# def GetDataAndlables(name, timeGrouperIndex, seqLength, lableShift, limit):
#     df = LoadData(name)
#     listOfVideoViews = Regroup(df, timeGrouperIndex)
#     return ExtractSequenes(listOfVideoViews, seqLength, lableShift, limit)

def TrainModel(train_inputs, train_lables, test_inputs, test_lables, SEQUENCE_LENGTH):
    X = tf.placeholder("float", shape=(None, SEQUENCE_LENGTH), name="Inputs")  # , COLUMNS_COUNT
    Y = tf.placeholder("float", shape=(None, 1), name="Outputs")

    W = tf.Variable(tf.ones([SEQUENCE_LENGTH, 1]), name="weight")  # COLUMNS_COUNT

    pred = tf.reduce_sum(tf.matmul(X, W), axis=1)
    loss = tf.reduce_mean(tf.square(pred / Y - 1))
    grads = tf.gradients(loss, [W])[0]

    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    for step in range(TRAIN_STEPS):
        training_data, lables = GetNextBatch(train_inputs, train_lables, step)
        inputs = np.reshape(training_data, (BATCH_SIZE, SEQUENCE_LENGTH))
        output = np.reshape(lables, (BATCH_SIZE, 1))

        sess.run(train, feed_dict={X: inputs, Y: output})

    test_lables = np.reshape(test_lables, (len(test_lables), 1))
    predictTest = sess.run(pred, feed_dict={X: test_inputs})

    originalPredicted = np.exp(np.array(predictTest))
    originalTestOutputs = np.exp(test_lables.ravel())
    rse = ((originalPredicted / originalTestOutputs) - 1) ** 2
    mapeAr = abs(1 - originalPredicted / originalTestOutputs)

    return rse.mean(), rse.var(), mapeAr.mean(), mapeAr.var()


def TestParams(listOfTrainVideoViews, listOfTestVideoViews, seqLength, lableShift, minLimit, maxLimit):
    train_inputs, train_lables = ExtractSequenes(listOfTrainVideoViews, seqLength, lableShift, minLimit, maxLimit)
    test_inputs, test_lables = ExtractSequenes(listOfTestVideoViews, seqLength, lableShift, minLimit, maxLimit)
    rseMean, rseVar, mapeMean, mapeVar = TrainModel(train_inputs, train_lables, test_inputs, test_lables, seqLength)
    return rseMean, rseVar, mapeMean, mapeVar


def GetRegrouppedViews(trainDf, testDf, timeGrouperIndex):
    return Regroup(trainDf, timeGrouperIndex), Regroup(testDf, timeGrouperIndex)


def Plot(dataL):
    rseMean, rseVar, mapeMean, mapeVar = zip(*dataL)
    plt.title('RSE mean')
    plt.plot(rseMean, np.arange(len(rseMean)))
    plt.show()

    plt.title('RSE var')
    plt.plot(rseVar, np.arange(len(rseVar)))
    plt.show()

    plt.title('MAPE mean')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.plot(mapeMean, np.arange(len(mapeMean)))
    plt.show()

    plt.title('MAPE var')
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.plot(mapeVar, np.arange(len(mapeVar)))

    plt.show()


def TestSeqLenPerf(listOfTrainVideoViews, listOfTestVideoViews, predDelta, maxSeqLen, seqLenStep=1, minLimit=None,
                   maxLimit=None):
    dataL = []

    for seqLen in range(1, maxSeqLen + 1, seqLenStep):
        print("{} from {}".format(seqLen, maxSeqLen + 1), end="\r")
        statsTuple = TestParams(listOfTrainVideoViews, listOfTestVideoViews, seqLen, predDelta, minLimit, maxLimit)
        dataL.append(statsTuple)

    Plot(dataL)


def TestSeqPredDeltaPerf(listOfTrainVideoViews, listOfTestVideoViews, seqLen, maxPredDelta, predDeltaStep=1,
                         minLimit=None, maxLimit=None):
    dataL = []

    for predDelta in range(1, maxPredDelta + 1, predDeltaStep):
        print("{} from {}".format(predDelta, maxPredDelta + 1), end="\r")
        statsTuple = TestParams(listOfTrainVideoViews, listOfTestVideoViews, seqLen, predDelta, minLimit, maxLimit)
        dataL.append(statsTuple)

    Plot(dataL)


trainDf = LoadData('train')
testDf = LoadData('validation')

listOfTrainVideoViewsW, listOfTestVideoViewsW = GetRegrouppedViews(trainDf, testDf, 'W')
TestSeqLenPerf(listOfTrainVideoViewsW, listOfTestVideoViewsW, predDelta = 51, maxSeqLen = 18, seqLenStep = 1, minLimit = 50, maxLimit = None)