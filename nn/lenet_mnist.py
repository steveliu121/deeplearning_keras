#!/usr/bin/env python
# coding: utf-8

# # LeNet网络训练

# In[1]:


import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../")))

from keras import backend as K
from sklearn import datasets
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from keras.optimizers import SGD
from nn.conv.lenet import LeNet

import matplotlib.pyplot as plt
import numpy as np


# In[2]:


print("[INFO] loading MNIST dataset...")
dataset = datasets.fetch_openml(name="mnist_784",
                                data_home=os.path.abspath(
                                    os.path.join(os.getcwd(),
                                                 "../imgdatasets")))
print("[INFO] load MNIST dataset done")


# In[3]:


print("[INFO] dataset preprocess...")
data = dataset.data
if K.image_data_format == "channels_first":
    data = data.reshape((data.shape[0], 1, 28, 28))
else:
    data = data.reshape((data.shape[0], 28, 28, 1))

data = data.astype("float") / 255.0
dataset.target.astype("int")

(trainX, testX, trainY, testY) = train_test_split(data,
                                                  dataset.target,
                                                  test_size=0.25,
                                                  random_state=42,
                                                  stratify=dataset.target)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)


# In[4]:


print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])


# In[5]:


print("[INFO] training network...")
H = model.fit(trainX,
              trainY,
              validation_data=(testX, testY),
              batch_size=128,
              epochs=20,
              verbose=1)


# In[7]:


print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(
    classification_report(testY.argmax(axis=1),
                          predictions.argmax(axis=1),
                          target_names=[str(x) for x in lb.classes_]))


# In[8]:


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

