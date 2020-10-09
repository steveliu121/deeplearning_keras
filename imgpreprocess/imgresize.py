#!/usr/bin/env python
# coding: utf-8

# ## 对图片执行resize操作
# 将他们resize到指定尺寸上
# interpolation（插值方法）采用INTER_AREA方法

# In[ ]:


import cv2 

class ImgResize:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
        
    def preprocess(self, image):
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)

