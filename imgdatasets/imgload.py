#!/usr/bin/env python
# coding: utf-8

# ## 加载数据集（图片）
# 对所有的图片执行指定的预处理，可以指定一个预处理队列，对图片依次执行处理

# In[ ]:


import numpy as np
import cv2
import os

class ImgLoad:
    def __init__(self, preprocessors=None):
        self.preprocessors = preprocessors
        
        if self.preprocessors is None:
            self.preprocessors = []
            
    def load(self, imagepaths, verbose=-1):
        data = []
        labels = []
        
        for (i, imagepath) in enumerate(imagepaths):
            # 从硬盘加载图片，从路径中获取label
            # 图片路径形如“/path/to/dataset/{class}/{image}.jpg”
            image = cv2.imread(imagepath)
            label = imagepath.split(os.path.sep)[-2]
            
            # 按序对该图片执行前处理
            if self.preprocessors is not None:
                for p in self.preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

        if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
            print("[INFO] processed {}/{}".format(i + 1, len(imagepaths)))

        return (np.array(data), np.array(labels))

