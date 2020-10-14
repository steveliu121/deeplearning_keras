#!/usr/bin/env python
# coding: utf-8

# # 保存训练过程的metrics数据
# 保存每个epoch得到的"train_loss/accuracy"和"validation_loss/accuracy"，并实时绘制图形，方便我们根据图形来判断训练的状态，做出及时的调整 譬如发生overfitting(可能由于learning rate太大，网络太冗余导致)，判断何时停止training。

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

from callbacks.trainingmonitor import TrainingMonitor 


# In[2]:


'''
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
               help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
               help="path to output model")
args = vars(ap.parse_args())
'''


# In[3]:


print("[INFO] loading images...")
#imagePaths = list(paths.list_images(args["dataset"]))
imagePaths = list(paths.list_images("../imgdatasets/animals3"))
print("[INFO] load images dnoe...")


# In[4]:


print("[INFO] images preprocessing...")
resize = ImgResize(32, 32)
imgtoarray = ImgToArray()

imgload = ImgLoad(preprocessors=[resize, imgtoarray])
(data, labels) = imgload.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0
print("[INFO] images preprocess done")


# In[5]:


print("[INFO] train test dataset split")
(trainX, testX, trainY, testY) = train_test_split(data, labels,
    test_size=0.25, random_state=42, stratify=labels)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)


# In[6]:


print("[INFO] compiling model...")
opt = SGD(lr=0.001, momentum=0.9, nesterov=True)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])


# In[7]:


print("[INFO] trainning network...")
#figPath = os.path.sep.join([args["output"], "{}.png"].format(os.getpid()))
#jsonPath = os.path.sep.join([args["output"], "{}.json"].format(os.getpid()))
figPath = os.path.sep.join(["./results", "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join(["./results", "{}.json".format(os.getpid())])
callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

H = model.fit(trainX, trainY, validation_data=(testX, testY),
    batch_size=32, epochs=100, callbacks=callbacks, verbose=1)


# In[8]:


print("[INFO] serializing network...")
#model.save(args["model"])
model.save("shallownet_animal3_weights.hdf5")


# In[9]:


print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=[str(x) for x in lb.classes_]))


# In[10]:


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_accuracy")
plt.title("Trainning Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("./results/shallownet_animals3_trainning_result_lr_decay")


# In[ ]:




