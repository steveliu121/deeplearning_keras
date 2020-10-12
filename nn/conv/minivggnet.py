#!/usr/bin/env python
# coding: utf-8

# # MiniVGGNet
# ## net architecture:
# [CONV-->RELU-->BN-->CON-->RELU-->BN-->POOL-->DO(dropout)]-->[CONV-->RELU-->BN-->CON-->RELU-->BN-->POOL-->DO]-->[FC-->RELU-->BN-->DO-->FC-->SOFTMAX]

# In[1]:


from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Dropout
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


# In[2]:


class MiniVGGNet:
    def build(width, height, depth, classes):
        model = Sequential()

        inputShape = (height, width, depth)
        # the batch-notmalization operates over the "channel dimesion" for every output of each layer
        # e.g.
        # (height, width, channel)
        chanDim = -1

        if K.image_data_format == "channels_first":
            inputShape = (depth, height, width)
            # while the book said "chanDim" should be "1" here, but i think this is a mistake
            chanDim = 0

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

