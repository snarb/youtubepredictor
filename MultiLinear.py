import pandas as pd
import numpy as np
import tensorflow as tf

from math import floor

SEQUENCE_LENGTH = 8
PREDICTION_DELTA = 4
VIEWS_SCALE_KOEF = 0
BATCH_SIZE = 1
COLUMNS_COUNT = 1
TRAIN_STEPS = 30000
DISPLAY_STEPS = 200

FILE_NAME = "_s:{}_p:{}".format(SEQUENCE_LENGTH, PREDICTION_DELTA)
# views column index - 1. ['channel_subscribers', 'views', 'engagements', 'sentiment']

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def LoadData(name):
    global VIEWS_SCALE_KOEF

    all_training_data = np.load('data/' + name + FILE_NAME + "_training_data.npy" )
    all_lables = np.load('data/' + name + FILE_NAME + "_lables.npy")

    sta = np.vstack(all_training_data)
    df = pd.DataFrame(sta, columns=['channel_subscribers', 'views', 'engagements', 'sentiment'])
    df[df < 0] = 0
    df[df.views == 0] = 1
    df[df.channel_subscribers == 0] = 1
    all_lables[all_lables == 0] = 1

    df['views'] = np.log(df['views'])
    all_lables = np.log(all_lables)
    df['channel_subscribers'] = np.log(df['channel_subscribers'])
    df[df.engagements > 1] = 1
    df[df.sentiment > 1] = 1


    all_training_data = df['views'].values
    inputs = np.reshape(all_training_data, (-1, BATCH_SIZE, SEQUENCE_LENGTH, COLUMNS_COUNT))
    output = np.reshape(all_lables, (-1, BATCH_SIZE, 1))

    return inputs, output


train_inputs, train_lables = LoadData("TRAIN")


X = tf.placeholder("float", shape=(BATCH_SIZE, SEQUENCE_LENGTH, COLUMNS_COUNT), name="Inputs") #, COLUMNS_COUNT
Y = tf.placeholder("float", shape=(BATCH_SIZE, 1), name="Outputs")

#W = tf.Variable(tf.random_uniform([SEQUENCE_LENGTH, BATCH_SIZE], -1.0, 1.0), name="weight") # COLUMNS_COUNT
W = tf.Variable(tf.ones([BATCH_SIZE, COLUMNS_COUNT, SEQUENCE_LENGTH]), name="weight") # COLUMNS_COUNT

pred = tf.reduce_sum(tf.matmul(X,W))

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
    input = train_inputs[step]
    lable = train_lables[step]
    sess.run(train, feed_dict={X: input, Y: lable})
    if step % DISPLAY_STEPS == 0:
        print(step, sess.run(loss, feed_dict={X: input, Y: lable}))

test_inputs, test_lables = LoadData("TEST")
test_inputs, test_lables  = unison_shuffled_copies(test_inputs, test_lables )

rseL = []
mapeL = []

for step in range(len(test_inputs)):
    input = test_inputs[step]
    lable = test_lables[step]
    prediction =  sess.run(pred, feed_dict={X: input})

    originalPredicted = np.exp(np.array(prediction))
    originalTestOutputs = np.exp(lable)

    rse = ((originalPredicted / originalTestOutputs) - 1) ** 2
    mapeAr = abs(originalTestOutputs - originalPredicted) / originalTestOutputs

    rseL.append(rse)
    mapeL.append(mapeAr)

    # print(lable, sess.run(pred, feed_dict={X: input}))
    # print(step, sess.run(loss, feed_dict={X: input, Y: lable}))

rseA = np.array(rseL)
mapeA = np.array(mapeL)


print("Original RSE stats. Mean: %f, Std: %f Var: %f" % (rseA.mean(), rseA.std(), rseA.var()))
print("Original MAPE stats. Mean: %f, Std: %f Var: %f" % (mapeA.mean(), mapeA.std(), mapeA.var()))


        # print(step, sess.run(W, feed_dict={X: input, Y: lable}))

