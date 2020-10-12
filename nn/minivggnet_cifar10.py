#!/usr/bin/env python
# coding: utf-8

# # MiniVGGNet
# 基于CIFAR-10数据集的一个MiniVGGNet网络训练
# ## 网络结构:
# [CONV-->RELU-->BN-->CON-->RELU-->BN-->POOL-->DO(dropout)]-->[CONV-->RELU-->BN-->CON-->RELU-->BN-->POOL-->DO]-->[FC-->RELU-->BN-->DO-->FC-->SOFTMAX]

# In[ ]:


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from nn.conv.minivggnet import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse


# In[ ]:


'''
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
               help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())
'''


# In[ ]:


print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
trainY = trainY.astype("int")
testY = testY.astype("int")

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

lableNames = [
    "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse",
    "ship", "truck"
]
print("[INFO] load CIFAR-10 done")


# In[ ]:


print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy",
              optimizer=opt,
              metrics=["accuracy"])


# In[ ]:


print("[INFO] training network...")
H = model.fit(trainX,
              trainY,
              validation_data=(testX, testY),
              batch_size=64,
              epochs=40,
              verbose=1)


# In[ ]:


print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lableNames))


# In[ ]:


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

