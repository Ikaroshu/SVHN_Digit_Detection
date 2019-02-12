import numpy as np
from scipy import io
from os import listdir
import cv2
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten, MaxPooling2D, Conv2D, BatchNormalization, Activation, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.applications.vgg16 import VGG16


tr = io.loadmat('./data/train_32x32.mat')
ts = io.loadmat('./data/test_32x32.mat')
tr_X = np.transpose(tr['X'], (3, 0, 1, 2))
tr_y = np.squeeze(tr['y'])
tr_y[tr_y == 10] = 0
ts_X = np.transpose(ts['X'], (3, 0, 1, 2))
ts_y = np.squeeze(ts['y'])
ts_y[ts_y == 10] = 0
extra_train = []
for f in listdir('./data/train/'):
    if f[-3:] == 'png':
        img = cv2.imread('./data/train/'+f)
        if img.shape[0] > 32 and img.shape[1] > 32:
            extra_train.append(img[:32, :32, :])
tr_X = np.concatenate((tr_X, np.asarray(extra_train, dtype=np.uint8)))
tr_y = np.concatenate((tr_y, 10*np.ones(len(extra_train), dtype=np.uint8)))
tr_y = keras.utils.to_categorical(tr_y, 11)
ts_y = keras.utils.to_categorical(ts_y, 11)

input_shape = tr_X[0].shape


# Home made CNN model

kernel_size = 3

def svhn_layer(model, filters, strides):
    model.add(Conv2D(filters, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(strides, strides)))
    model.add(Dropout(0.2))

    return model


model = Sequential()

# first layer
model.add(Conv2D(48, (kernel_size, kernel_size),  padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.2))

# six hidden layers
svhn_layer(model, 48, 1)
svhn_layer(model, 48, 1)
svhn_layer(model, 64, 2)
svhn_layer(model, 64, 1)
svhn_layer(model, 64, 1)
svhn_layer(model, 128, 2)

model.add(Flatten())
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(11, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(tr_X, tr_y, batch_size=64, epochs=10, verbose=1, validation_data=(ts_X, ts_y))

model.save("./data/my_model.h5")


# VGG model

vgg = VGG16(include_top=False, weights=None)

inputs = Input(shape=(32, 32, 3))
layer = vgg(inputs)
layer = Flatten()(layer)
layer = Dense(512, activation='relu')(layer)
layer = Dense(512, activation='relu')(layer)
layer = Dropout(0.3)(layer)
out = Dense(11, activation='softmax')(layer)

model = Model(inputs, out)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(tr_X, tr_y, batch_size=64, epochs=10, verbose=1, validation_data=(ts_X, ts_y))

model.save("./data/vgg_model.h5")


# VGG pre-trained

vgg_pre = VGG16(include_top=False, weights='imagenet')

inputs = Input(shape=(32, 32, 3))
layer = vgg_pre(inputs)
layer = Flatten()(layer)
layer = Dense(1024, activation='relu')(layer)
layer = Dense(1024, activation='relu')(layer)
out = Dense(11, activation='softmax')(layer)

model = Model(inputs, out)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(tr_X, tr_y, batch_size=64, epochs=10, verbose=1, validation_data=(ts_X, ts_y))

model.save("./data/vgg_pre_model.h5")
