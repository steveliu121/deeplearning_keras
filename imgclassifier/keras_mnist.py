#!/usr/bin/env python
# coding: utf-8

# # 采用keras库搭建简单神经网络对MNIST数据集分类
# 
# ## 代码流程
# + 加载MNIST数据集（首次使用MNIST数据集需要下载）
# + 量化数据，raw pixel [0, 255] --> [0, 1.0]
# + 划分训练集和测试集
# + 将label转换为one-hot格式
# + 构建一个三层全连接神经网络(Dense)(不包括输入层)，两个隐层层采用sigmod激活，输出层采用softmax进行10分类
# 
# ## 注意
# + 输入是灰度图像，只有一个通道，分辨率是"28 x 28"
# + 为了简单示范起见，我们将数据集划分为"trainning set"和"testing set"，并将"testing set"当做"validation set"使用。正式使用中需要从"trainning set"再划分出"validation set"
# + 优化算法采用"mini-batch SGD"，"learning_rate=0.01"，loss采用"categorical cross entropy"，"epoch=100"，　"batch_size=128"
# 

# In[2]:


from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


'''
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
            help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())
'''

'''
加载"mnist_784"数据集，若"data_home"目录下不存在需要下载
'''
print("[INFO] loading MNIST dataset...")
dataset = datasets.fetch_openml(name="mnist_784", 
                                data_home=os.path.abspath(os.path.join(os.getcwd(), "../imgdatasets")))
print("[INFO] load MNIST dataset done")


# In[3]:


print("[INFO] split trainning&testing dataset...")
'''
将数据raw pixel[0, 255]量化[0, 1.0]，并划分"trainning set"和"testing set"
random_state用来保证每次随机划分"dataset"时用的随机数是同一组一样的
stratify表示"trainning set"和"testing set"的数据分布都同整个dataset相同（也可以设置成同）
'''
data = dataset.data.astype("float") / 255.0
(trainX, testX, trainY, testY) = train_test_split(data,
    dataset.target, test_size=0.25, random_state = 42, stratify=dataset.target)

print("[INFO] transform labels to one-hot...")
'''
将标签转换为one-hot格式
'''
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

print("[INFO] structure a Dense network architecture...")
model = Sequential()
'''
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="softmax"))
'''
model.add(Dense(256, input_shape=(784,), activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))
'''
加载同一组权重，保证不同optimization的初始权重相同，具有可比较意义
'''
model.load_weights("./weights/keras_mnist_weights.hd5")

print("[INFO] trainning network...")
sgd = SGD(0.01)
'''
metrics表示训练时同时衡量它的"accuracy"（此处）指标，metrics只参与衡量，不参与优化计算
'''
model.compile(loss="categorical_crossentropy", optimizer=sgd,
             metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
             epochs=20, batch_size=128)

#model.save_weights("./keras_mnist_weights.hd5")


# In[6]:


print("[INFO] evaluating network...")
'''
testY.shape=[17500, 10]，共有17500张图片，每张图片的标签是一个dimension=10的one-hot向量
testY.argmax(axis=1)返回的是每个one-hot向量中最大的数的index，one-hot只有一个元素是1，其余都是0
其实就是返回的这个数字的原始label，譬如[0,0,0,0,0,0,1,0,0,0,0]--->6
predictions同上
target_names是数据集的原始label
'''
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1),
     predictions.argmax(axis=1),
     target_names=[str(x) for x in lb.classes_]))
print(testY.shape, predictions.shape)
print(testY.shape[0])
print(testY.shape[1])


# In[ ]:


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, 20), H.history["val_accuracy"], label="val_accuracy")
plt.title("Trainning Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("./results/mnist_trainning_result")

