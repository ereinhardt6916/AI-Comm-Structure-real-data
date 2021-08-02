from __future__ import print_function

# tag::mcts_go_cnn_preprocessing[]
import os
from keras.backend import conv2d
from keras.layers.convolutional import ZeroPadding2D
from keras.layers.core import Activation
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

np.random.seed(123)
X = np.load('features.npy')
Y = np.load('labels.npy')

samples = X.shape[0]
size = 9
input_shape = (size, size, 1)

X = X.reshape(samples, size, size, 1)

train_samples = int(0.9 * samples)
X_train, X_test = X[:train_samples], X[train_samples:]
Y_train, Y_test = Y[:train_samples], Y[train_samples:]
# end::mcts_go_cnn_preprocessing[]

# tag::mcts_go_cnn_model[]
model = Sequential()
model.add(ZeroPadding2D(padding=3,input_shape=input_shape))
model.add(Conv2D(48,(3,3)))
model.add(Activation('relu'))

model.add(ZeroPadding2D(padding=2,input_shape=input_shape))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))

model.add(ZeroPadding2D(padding=2,input_shape=input_shape))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))

model.add(ZeroPadding2D(padding=2,input_shape=input_shape))
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(size * size, activation='softmax'))
#model.add(Activation('relu'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
# end::mcts_go_cnn_model[]

#*********************************************************
checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,verbose=1)

# tag::mcts_go_cnn_eval[]
model.fit(X_train, Y_train,
          batch_size=64,
          epochs=5,
          verbose=1,
          validation_data=(X_test, Y_test),
          callbacks=[cp_callback])

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# end::mcts_go_cnn_eval[]

test_board = np.array([[[
    0, 0,  0,  0,  0, 0, 0, 0, 0,
    0, 0,  0,  0,  0, 0, 0, 0, 0,
    0, 0,  0,  0,  0, 0, 0, 0, 0,
    0, 0, 0,  0, 0, 0, 0, 0, 0,
    0, 0, 0,  0, 0, 0, 0, 0, 0,
    0, 0,  0, 0,  0, 0, 0, 0, 0,
    0, 0,  0,  0,  0, 0, 0, 0, 0,
    0, 0,  0,  0,  0, 0, 0, 0, 0,
    0, 0,  0,  0,  0, 0, 0, 0, 0,
]]]).reshape(1, 9, 9, 1)
move_probs = model.predict(test_board)[0]
i = 0
for row in range(9):
    row_formatted = []
    for col in range(9):
        row_formatted.append('{:.3f}'.format(move_probs[i]))
        i += 1
    print(' '.join(row_formatted))