# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 13:33:45 2019

@author: hardyliu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,precision_recall_curve
data = pd.read_csv('./creditcard.csv')

#数据探索
print(data.describe())

#print(data.head())

#设置中文支持字体
plt.rcParams['font.sans-serif']=['SimHei']
plt.figure()
#画类别计数图
sns.countplot(x='Class',data=data)
plt.title('类别分布')
plt.show()

num_total = len(data)
num_fraud = len(data[data['Class']==1])

print('总交易笔数：',num_total)
print('诈骗交易笔数：',num_fraud)
print('诈骗交易比例：{:.6f}'.format(num_fraud/num_total))

#诈骗和正常交易可视化
f,(axFraud,axNormal)=plt.subplots(2,1,sharex=True,figsize=(15,8))

bins=50

axFraud.set_title('诈骗交易')

axFraud.hist(data.Time[data.Class==1],bins=bins,color='deeppink')
axNormal.set_title('正常交易')
axNormal.hist(data.Time[data.Class==0],bins=bins,color='deepskyblue')
plt.xlabel('时间')
plt.ylabel('交易次数')
plt.show()

#对Amount进行数据规范化
data['Amount_Norm']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

y=np.array(data.Class.tolist())
data = data.drop(['Time','Amount','Class'],axis=1)
X=np.array(data.values)
#准备测试集合训练集
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.1,random_state=33)
#采用逻辑回归分类
clf = LogisticRegression()

clf.fit(train_x,train_y)

predict_y=clf.predict(test_x)
#预测样本的置信分数
score_y=clf.decision_function(test_x)

#计算混淆矩阵并显示
cm = confusion_matrix(test_y,predict_y)
class_names=[0,1]

#混淆矩阵的可视化
def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmap = plt.cm.Blues):
    plt.figure()
    plt.imshow(cm,interpolation='nearest',cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks,classes,rotation=0)
    plt.yticks(tick_marks,classes)
    
    thresh = cm.max()/2.

    for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
        print(i,j,cm[i,j])#这里注意CM的值和坐标的对应关系
        plt.text(j,i,cm[i,j],horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
 
plot_confusion_matrix(cm,classes=class_names,title='逻辑回归 混淆矩阵')
#显示模型评估分数
# 显示模型评估结果
#TP：预测为正，判断正确；
#FP:预测为正，判断错误
#TN:预测为负，判断正确
#FN:预测为负，判断错误
def show_metrics():
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]
    print('精确率: {:.3f}'.format(tp/(tp+fp)))
    print('召回率: {:.3f}'.format(tp/(tp+fn)))
    print('F1 值: {:.3f}'.format(2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn))))))
show_metrics()

precision,recall,thresholds = precision_recall_curve(test_y,score_y)
# 绘制精确率 - 召回率曲线
def plot_precision_recall():
    plt.step(recall, precision, color = 'b', alpha = 0.2, where = 'post')
    plt.fill_between(recall, precision, step ='post', alpha = 0.2, color = 'b')
    plt.plot(recall, precision, linewidth=2)
    plt.xlim([0.0,1])
    plt.ylim([0.0,1.05])
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率 - 召回率 曲线')
    plt.show();
plot_precision_recall()


    
    
    


