#!/usr/bin/env python
# coding: utf-8

# ## KNN分类器
# K-Nearest Neighbor classifier
# 注意：
# 无需学习的一个简单非线性分类器，适合于数据集小，维度少的数据分类
# K值的选取会影响分类的准确度
# 原理：
# 每次对一个数据分类时，需要计算该数据到训练集（其实不训练，或者叫带标签的先验集）中所有数据的距离（欧氏距离（推荐），曼哈顿距离等），找最近的k个数据，这k个数据就代表了该数据的所属类别，譬如K个数据中有x个A类数据，y个B类数据（x + y = k），那么该数据属于A类的概率x/k，属于B类的概率y/k。

# In[1]:


'''导入自定义模块'''
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from sklearn.neighbors import KNeighborsClassifier as KNNC
from sklearn.preprocessing import LabelEncoder as LE
from sklearn.model_selection import train_test_split as TTS
from sklearn.metrics import classification_report as CR
from imgdatasets.imgload import ImgLoad as IL
from imgpreprocess.imgresize import ImgResize as IR
from imutils import paths
import argparse


'''
匹配输入参数
'''
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
               help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
               help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
               help="# of jobs for k-NN distance (-1 uses all acailable cores)")
args = vars(ap.parse_args())

print("[INFO] loading images...")
imagepaths = list(paths.list_images(args["dataset"]))


'''
图片序列前处理，resize到指定尺寸，并reshape成flatten格式
'''
ir = IR(32, 32)
il = IL(preprocessors=[ir])
(data, labels) = il.load(imagepaths, verbose=500)


'''将每一个datapoint（每张图片的array[32, 32, 3]
reshape成flatten模式array[3072]，数据集中共3000个datapoint
'''
data = data.reshape((data.shape[0], 3072))#data.shape = [3000, 32, 32, 3]

print("[INFO] features matrix: {:.1f}MB".format(
    data.nbytes / (1024 * 1000.0)))


'''
将字符串labels转换成整型 
'''
le = LE()
labels = le.fit_transform(labels)


'''
划分数据集，75%trainning set 25% testing set
stratify=labels保证trainning set和testing set中数据分布与整个dataset(labels)一致
'''
(trainX, testX, trainY, testY) = TTS(data, labels,
        test_size=0.25, random_state=42, stratify=labels)


print("[INFO] evaluating K-NN classifier...")
model = KNNC(n_neighbors=args["neighbors"],
            n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(CR(testY, model.predict(testX), target_names=le.classes_))

