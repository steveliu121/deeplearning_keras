#!/usr/bin/env python
# coding: utf-8

# # shallownet模型类
# 构建一个简单的CNN模型类
# INPUT-->CONV-->RELU-->FC

# In[1]:


from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend


# In[ ]:


class ShallowNet:
    def build(width, height, depth, classes):
        model = Sequential()
        # 默认情况下image_data_format是“channels_last”
        # 注意“高宽”对应“行列”
        inputShape = (height, width, depth)
        
        if backend.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            
        model.add(Conv2D(32, (3,3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        return model

