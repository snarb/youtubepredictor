import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from math import floor
import Test
from matplotlib import pyplot as plt

SEQUENCE_LENGTH = 8
PREDICTION_DELTA = 4
VIDEO_LEN_LIMIT = 30

BATCH_SIZE = 1
TRAIN_STEPS = 10000
DISPLAY_STEPS = 200
STABLE_DELTA = 1e-8


# TRAIN_STEPS = 90000
# optimizer = tf.train.AdamOptimizer(2e-3)
#
# Original RSE stats. Mean: 0.010167, Std: 0.394944 Var: 0.155980
# Original MAPE stats. Mean: 0.044979, Std: 0.090243 Var: 0.008144

# views column index - 1. ['channel_subscribers', 'views', 'engagements', 'sentiment']

def rolling_window(a, window, lablesShift):
    shape = a.shape[:-1] + (a.shape[-1] - window +1, window)
    strides = a.strides + (a.strides[-1],)

    data =  np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    lables = a[window - 1 + lablesShift::1]
    return data[:-lablesShift], lables


def LoadData(name):
    df = pd.read_csv('data/' + name + '.csv',
                     index_col=['datetime'],
                     dtype = {"video_id": object, "views": np.float64},
                     parse_dates=['datetime'],
                     )
    return df



def GetNextBatch(inputs, lables, batchId):
    batchId = batchId % len(inputs)
    startIndex = batchId * BATCH_SIZE
    endIndex = (batchId + 1) * BATCH_SIZE
    if(len(inputs) < endIndex):
        startIndex = 0
        endIndex = BATCH_SIZE

    return (inputs[startIndex:endIndex], lables[startIndex:endIndex])

def Regroup(df, timeGrouperIndex):

    # target = df[df.video_id == '---jcia5ufM']
    for group in df.groupby(['video_id']):
        videoDf = group[1]
        videoId = group[0]
        if(len(videoDf['views']) > VIDEO_LEN_LIMIT):
         assert(videoDf['views'].sum() != len(videoDf['views']))
        # minV = videoDf['views'].diff().min()
        # if (minV < 0):
        #     print("bad")

    resDf = df.groupby(['video_id', pd.TimeGrouper(timeGrouperIndex)]).mean()
    resDf = resDf.reset_index().set_index('datetime')

    # for group in resDf.groupby('video_id'):
    #     videoDf = group[1]
    #     videoId = group[0]
    #
    #     assert (len(np.where(videoDf['views'] == 668854)[0]) == 0)

    listOfVideoViews = list(resDf.groupby('video_id')['views'].apply(pd.DataFrame.as_matrix))
    return listOfVideoViews

def ExtractSequenes(listOfVideoViews, seqLen, predDelta, minLen):
    datas = []
    lables = []
    for s in listOfVideoViews:
        if(len(s) > minLen):
            data, lable = rolling_window(s, window = seqLen, lablesShift = predDelta)

            assert(all(lable[i] <= lable[i + 1] for i in range(len(lable) - 1)))

            datas.extend(data)
            lables.extend(lable)
            assert (len(np.where(lable == 1)[0]) == 0)

    datas = np.log(datas) + STABLE_DELTA
    lables = np.log(lables) + STABLE_DELTA

    assert (len(lables) - np.count_nonzero(lables) == 0)

    return datas, lables


def GetDataAndlables(name, timeGrouperIndex, seqLength, lableShift, limit):
    df = LoadData(name)
    listOfVideoViews = Regroup(df, timeGrouperIndex)
    return ExtractSequenes(listOfVideoViews, seqLength, lableShift, limit)

train_inputs, train_lables = GetDataAndlables("train", 'W', seqLength = SEQUENCE_LENGTH, lableShift = PREDICTION_DELTA, limit = VIDEO_LEN_LIMIT)
test_inputs, test_lables = GetDataAndlables("validation", 'W', seqLength = SEQUENCE_LENGTH, lableShift = PREDICTION_DELTA, limit = VIDEO_LEN_LIMIT)

X = tf.placeholder("float", shape=(None, SEQUENCE_LENGTH), name="Inputs") #, COLUMNS_COUNT
Y = tf.placeholder("float", shape=(None, 1), name="Outputs")

W = tf.Variable(tf.ones([SEQUENCE_LENGTH, 1]), name="weight") # COLUMNS_COUNT

check = tf.check_numerics(W, "bad")

pred = tf.reduce_sum(tf.matmul(X,W), axis=1)

loss = tf.reduce_mean(tf.square(pred/Y - 1))

grads = tf.gradients(loss, [W])[0]


optimizer = tf.train.AdamOptimizer(2e-3)
train = optimizer.minimize(loss)

# Before starting, initialize the variables.  We will 'run' this first.
init = tf.initialize_all_variables()

# Launch the graph.
sess = tf.Session()
sess.run(init)

# Fit the line.
for step in range(TRAIN_STEPS):
    training_data, lables = GetNextBatch(train_inputs, train_lables, step)
    inputs = np.reshape(training_data, (BATCH_SIZE, SEQUENCE_LENGTH))
    output = np.reshape(lables, (BATCH_SIZE, 1))
    assert(lables[0] > 0)

    sess.run(train, feed_dict={X: inputs, Y: output})
    sess.run(check)
    if step % DISPLAY_STEPS == 0:
        print(step, sess.run(loss, feed_dict={X: inputs, Y: output}))

test_lables = np.reshape(test_lables, (len(test_lables), 1))
predictTest =  sess.run(pred, feed_dict={X: test_inputs})

originalPredicted = np.exp(np.array(predictTest))
originalTestOutputs = np.exp(test_lables.ravel())
rse = ((originalPredicted / originalTestOutputs) - 1) ** 2
mapeAr = abs(1 - originalPredicted / originalTestOutputs)

print("Original RSE stats. Mean: %f, Std: %f Var: %f" % (rse.mean(), rse.std(), rse.var()))
print("Original MAPE stats. Mean: %f, Std: %f Var: %f" % (mapeAr.mean(), mapeAr.std(), mapeAr.var()))
print("Done")


