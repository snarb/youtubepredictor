from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from keras.models import Sequential
from keras.layers.merge import add
import keras


x1 = np.random.uniform(low=0., high=1, size=(5000))
x2 = np.random.uniform(low=0., high=1, size=(5000))
#x = np.vstack((x1, x2)).T.reshape(-1, 2)

#y = np.log(x1 * x2).reshape(-1, 1)
y = (x1 * x2).reshape(-1, 1)


# This returns a tensor
input1 = Input(shape=(1,))
x1_l = Dense(16, activation='relu')(input1)
#x1_l = Dense(16, activation='relu')(x1_l)


input2 = Input(shape=(1,))
x2_l = Dense(16, activation='relu')(input2)
#x2_l = Dense(16, activation='relu')(x2_l)

#merged_vector = Add([x1_l, x2_l])
merged_vector = keras.layers.add([x1_l, x2_l])

out = Dense(16, activation='relu')(merged_vector)
predictions = Dense(1, activation='linear')(out)
model = Model(inputs=[input1, input2], outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='mean_squared_error')

model.fit([x1, x2], y, epochs=50, batch_size=50)
print(model.predict([np.array([0.3]), np.array([0.4])]))

# x = Dense(64, activation='relu')
#
#
# predictions = Dense(10, activation='softmax')(x)
#
# # This creates a model that includes
# # the Input layer and three Dense layers
# model = Model(inputs=[input1, input2], outputs=predictions)
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit([x1, x2], [y])  # starts training
#
# left = Sequential()
# left.add(Dense(16, input_shape=(1,)))
# left.add(Dense(16, activation='relu'))
#
# right = Sequential()
# right.add(Dense(16, input_shape=(1,)))
# right.add(Dense(16, activation='relu'))
#
#
# model = Sequential()
# model.add(add([left, right]))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1, activation='softmax'))
# model.compile(optimizer='rmsprop',
#              loss='mean_squared_error')
# model.fit([x1, x2], y, epochs=10, batch_size=50)