#!/usr/bin/env python
# coding: utf-8

# # 回调函数
# 每完成一次epoch回调一次，主要完成记录"train_loss/accuracy"和"validation_loss/accuracy"以及实时绘制他们的变化图工作。
# 方便我们根据图形来判断训练的状态，做出及时的调整
# 譬如发生overfitting(可能由于learning rate太大，网络太冗余导致)，判断何时停止training。

# In[1]:


from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
import numpy as np
import json
import os


# In[ ]:


class TrainingMonitor(BaseLogger):
    def __init__(self, figPath, jsonPath=None, startAt=0):
        # 集成父类"BaseLogger"的属性
        super(TrainingMonitor, self).__init__()
        self.figPath = figPath
        self.jsonPath = jsonPath
        self.startAt = startAt

    '''
    在训练开始时回调
    '''

    def on_train_begin(self, logs={}):
        self.H = {}

        if self.jsonPath is not None:
            if os.path.exists(self.jsonPath):
                self.H = json.loads(open(self.jsonPath).read())

                if self.startAt > 0:
                    for k in self.H.keys():
                        self.H[k] = self.H[k][:self.startAt]

    '''
    在每一次训练结束时回调
    记录历史信息："train_loss,train_accuracy,test_loss,test_accuracy"
    '''

    def on_epoch_end(self, epoch, logs={}):
        for (k, v) in logs.items():
            l = self.H.get(k, [])
            l.append(v)
            self.H[k] = l
        if self.jsonPath is not None:
            f = open(self.jsonPath, "w")
            f.write(json.dumps(self.H))
            f.close()
        if len(self.H["loss"]) > 1:
            N = np.arange(0, len(self.H["loss"]))
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(N, self.H["loss"], label="train_loss")
            plt.plot(N, self.H["val_loss"], label="val_loss")
            plt.plot(N, self.H["accuracy"], label="train_accuracy")
            plt.plot(N, self.H["val_accuracy"], label="val_accuracy")
            plt.title("Training Loss an Accuracy [Epoch {}]".format(
                len(self.H["loss"])))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()

            plt.savefig(self.figPath)
            plt.close()

