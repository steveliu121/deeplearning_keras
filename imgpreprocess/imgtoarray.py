#!/usr/bin/env python
# coding: utf-8

# # 将图片转换为numpy矩阵
# 可以设定channel这一dimension在numpy矩阵中是哪一个维度，譬如
# + tensorflow的numpy.shape为(row, colum, channel)，也即"channels_last"（kera默认配置是channels_last，因为keras默认采用tensorflow作为backend）
# + theano的numpy.shape为(channel, row, colum)，也即"channels_first"

# In[1]:


from keras.preprocessing.image import img_to_array

class ImgToArray:
    '''
    dataFormat="channels_first"(theano) / "channels_last"(tensorflow)
    '''
    def __init__(self, dataFormat=None):
        self.dataFormat=dataFormat

    def preprocess(self, image):
        return img_to_array(image, data_format=self.dataFormat)

