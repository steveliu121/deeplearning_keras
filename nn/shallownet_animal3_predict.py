#!/usr/bin/env python
# coding: utf-8

# # 加载模型，对图片进行预测
# 从disk中加载已经训练好的模型，对输入图片进行分类预测

# In[1]:


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from imgpreprocess.imgtoarray import ImgToArray
from imgpreprocess.imgresize import ImgResize
from imgdatasets.imgload import ImgLoad
from keras.models import load_model

import numpy as np
from imutils import paths
import argparse
import cv2


# In[2]:


'''
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
               help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
               help="path to pre-trained model")
args = vars(ap.parse_args())
'''


# In[3]:


classLabels = ["dog", "horse", "elephant"]


# In[14]:


print("INFO sampling images...")
#imagePaths = np.array(list(paths.list_images(args["dataset"])))
'''
imagePaths = np.array(list(paths.list_images("../imgdatasets/animals3")))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]
'''
imagePaths = np.array(list(paths.list_images("./predict_dataset/animals3")))


# In[15]:


print("[INFO] image preprocess...")
imgresize = ImgResize(32, 32)
imgtoarray = ImgToArray()

imgload = ImgLoad(preprocessors=[imgresize, imgtoarray])
(data, labels) = imgload.load(imagePaths)
data = data.astype("float") / 255.0


# In[16]:


print("[INFO] load pre-trained network...")
#model = load_model(args["model"])
model = load_model("./shallownet_animal3_weights.hdf5")


# In[17]:


print("[INFO] predicting...")
preds = model.predict(data, batch_size=32).argmax(axis=1)


# In[19]:


for (i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

