import pandas as pd
import numpy as np
import tensorflow as tf

from math import floor

SEQUENCE_LENGTH = 8
PREDICTION_DELTA = 4
VIEWS_SCALE_KOEF = 0
BATCH_SIZE = 1
TRAIN_STEPS = 30000
DISPLAY_STEPS = 200

FILE_NAME = "_s:{}_p:{}".format(SEQUENCE_LENGTH, PREDICTION_DELTA)
# views column index - 1. ['channel_subscribers', 'views', 'engagements', 'sentiment']

def LoadCleanData(name):
    global VIEWS_SCALE_KOEF

    training_data = np.load('data/' + name + FILE_NAME + "_training_data.npy" )
    lables = np.load('data/' + name + FILE_NAME + "_lables.npy")

    training_data = training_data[..., 1] # use only 'views'

    training_data[training_data == 0] = 1
    lables[lables == 0] = 1

    training_data =  np.log(training_data)
    lables = np.log(lables)

    return training_data, lables

def GetNextBatch(inputs, lables, batchId):
    batchId = batchId % len(inputs)
    startIndex = batchId * BATCH_SIZE
    endIndex = (batchId + 1) * BATCH_SIZE
    if(len(inputs) < endIndex):
        startIndex = 0
        endIndex = BATCH_SIZE

    return (inputs[startIndex:endIndex], lables[startIndex:endIndex])


train_inputs, train_lables = LoadCleanData("TRAIN")


X = tf.placeholder("float", shape=(None, SEQUENCE_LENGTH), name="Inputs") #, COLUMNS_COUNT
Y = tf.placeholder("float", shape=(None, 1), name="Outputs")

#W = tf.Variable(tf.random_uniform([SEQUENCE_LENGTH, BATCH_SIZE], -1.0, 1.0), name="weight") # COLUMNS_COUNT
W = tf.Variable(tf.ones([SEQUENCE_LENGTH, 1]), name="weight") # COLUMNS_COUNT

pred = tf.reduce_sum(tf.matmul(X,W), axis=1)

#loss = tf.reduce_mean(tf.square(pred - Y))
loss = tf.reduce_mean(tf.square(pred/Y - 1))

grads = tf.gradients(loss, [W])[0]


optimizer = tf.train.AdamOptimizer(1e-4)
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

    sess.run(train, feed_dict={X: inputs, Y: output})
    if step % DISPLAY_STEPS == 0:
        print(step, sess.run(loss, feed_dict={X: inputs, Y: output}))

    if(step > 0 and step % 2000 == 0):
        predictTrain = sess.run(pred, feed_dict={X: train_inputs})
        originalTrainPredicted = np.exp(np.array(predictTrain)) #np.exp(np.array(predictTrain))
        originalTrainOutputs = np.exp(train_lables.ravel()) #np.exp(train_lables.ravel())
        rse = ((originalTrainPredicted / originalTrainOutputs) - 1) ** 2
        mapeAr = abs(originalTrainOutputs - originalTrainPredicted) / originalTrainOutputs
        print("Original train RSE stats. Mean: %f, Std: %f Var: %f" % (rse.mean(), rse.std(), rse.var()))
        print("Original train MAPE stats. Mean: %f, Std: %f Var: %f" % (mapeAr.mean(), mapeAr.std(), mapeAr.var()))


test_inputs, test_lables = LoadCleanData("TEST")

test_lables = np.reshape(test_lables, (len(test_lables), 1))
predictTest =  sess.run(pred, feed_dict={X: test_inputs})

originalPredicted = np.exp(np.array(predictTest))
originalTestOutputs = np.exp(test_lables.ravel())
rse = ((originalPredicted / originalTestOutputs) - 1) ** 2
mapeAr = abs(originalTestOutputs - originalPredicted) / originalTestOutputs

print("Original RSE stats. Mean: %f, Std: %f Var: %f" % (rse.mean(), rse.std(), rse.var()))
print("Original MAPE stats. Mean: %f, Std: %f Var: %f" % (mapeAr.mean(), mapeAr.std(), mapeAr.var()))
print("Done")


