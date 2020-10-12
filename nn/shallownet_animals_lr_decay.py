#!/usr/bin/env python
# coding: utf-8

# # learning rate decay 
# 网络结构由之前我们定义的类ShallowNet构建
# 为了评估learning rate decay对网络过拟合带来的改进，网络中添加了learning rate decay模块

# In[1]:


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imgpreprocess.imgtoarray import ImgToArray
from imgpreprocess.imgresize import ImgResize
from imgdatasets.imgload import ImgLoad
from nn.conv.shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

from keras.callbacks import LearningRateScheduler


# 这里定义我们的learning rate decay调整函数，并将其作为模型训练时的回调函数使用

# In[2]:


def lr_step_decay(epoch):
    initAlpha = 0.005
    factor = 0.9
    dropEvery = 5
    
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
    print("#epoch:{}".format(epoch))
    
    return float(alpha)


# In[3]:


'''
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
               help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
               help="path to output model")
args = vars(ap.parse_args())
'''


# In[4]:


print("[INFO] loading images...")
#imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = list(paths.list_images("../imgdatasets/animals3"))
print("[INFO] load images dnoe...")


# In[5]:


print("[INFO] images preprocessing...")
resize = ImgResize(32, 32)
imgtoarray = ImgToArray()

imgload = ImgLoad(preprocessors=[resize, imgtoarray])
(data, labels) = imgload.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0
print("[INFO] images preprocess done")


# In[6]:


print("[INFO] train test dataset split")
(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.25, random_state=42, stratify=labels)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)


# In[7]:


print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])


# In[8]:


print("[INFO] trainning network...")
callbacks = [LearningRateScheduler(lr_step_decay)]

H = model.fit(trainX, trainY, validation_data=(testX, testY),
    batch_size=32, epochs=50, callbacks=callbacks, verbose=1)


# In[9]:


print("[INFO] serializing network...")
#model.save(args["model"])
model.save("shallownet_animal3_weights.hdf5")


# In[10]:


print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=[str(x) for x in lb.classes_]))


# In[11]:


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 50), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 50), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 50), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 50), H.history["val_accuracy"], label="val_accuracy")
plt.title("Trainning Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("./results/shallownet_animals3_trainning_result_lr_decay")


# In[ ]:




