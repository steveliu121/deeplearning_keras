#!/usr/bin/env python
# coding: utf-8

# # 采用keras库搭建简单神经网络对CIFAR-10数据集分类
# 代码流程同"keras_mnist.ipynb"，只是数据集不同，激活函数和网络神经元数量不同。

# In[1]:


from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
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


# In[2]:


print("[INFO] loading CIFAR-10 data...")
'''
默认存储路径~/.keras/datasets/
'''
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0
trainX = trainX.reshape((trainX.shape[0], 3072))
testX = testX.reshape((testX.shape[0], 3072))
print("[INFO] load CIFAR-10 dataset ok...")


# In[3]:


lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"]


# In[5]:


model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation = "relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(10, activation="softmax"))


# In[7]:


print("[INFO] trainning network...")
sgd = SGD(0.01)
model.compile(loss="categorical_crossentropy", optimizer=sgd,
    metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    epochs=40, batch_size=32)


# In[9]:


print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                           predictions.argmax(axis=1), target_names=labelNames))


# In[11]:


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 40), H.history["val_accuracy"], label="val_accuracy")
plt.title("Trainning Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("./results/cifar10_trainning_result")


# In[ ]:




