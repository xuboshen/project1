import time
from sklearn.metrics import roc_curve, auc, roc_auc_score
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

data = sio.loadmat('ADNI_90_120_fMRI.mat')
x_positive = data['NC']
x_negative = data['AD']
x_total = np.concatenate((x_positive, x_negative), axis=0).astype(np.float)
y_positive = np.ones(x_positive.shape[0])
y_negative = np.zeros(x_negative.shape[0])
y_total = np.concatenate((y_positive, y_negative), axis=0).astype(np.float)
print('没有加伪样本的时候有{}个正样本，有{}个负样本，一共有{}个样本'.format(y_positive.shape[0], y_negative.shape[0], x_total.shape[0]))
num_x_pos = x_positive.shape[0]
num_x_neg = x_negative.shape[0]
dim_input = np.prod(x_positive.shape[1]**2)
num_total = x_total.shape[0]
print(x_positive.shape, x_negative.shape, x_total.shape, y_positive.shape, num_x_neg, num_x_pos, dim_input)
meta_size = int(0.1 * x_total.shape[0])


def corrcoef(input):
    """传入一个tensor格式的矩阵x(x.shape(m,n))，输出其相关系数矩阵"""
    output = []
    for index, x in enumerate(input):
        f = (x.shape[0] - 1) / x.shape[0]      # 方差调整系数
        x_reducemean = x - np.mean(x, axis=0)
        numerator = np.matmul(x_reducemean.T, x_reducemean) / x.shape[0]
        var_ = x.var(axis=0).reshape(x.shape[1], 1)
        denominator = np.sqrt(np.matmul(var_, var_.T)) * f
        output.append(numerator / denominator)
    return output
# pearson = corrcoef(x_total.reshape(181, 200, 90))
print(x_total.shape)
pearson = corrcoef(x_total.reshape(107, 120, 90))

pearson = np.array(pearson)
plt.imshow(pearson[0])
pearson_cutted = np.zeros((107, 45*89))
dim_pearson = 90
t = 0
for k in range(pearson.shape[0]):
    for i in range(dim_pearson):
        for j in range(dim_pearson):
            if i < j:
                pearson_cutted[k, t] = pearson[k, i, j]
                t = t + 1
        
    t = 0
pearson_cutted = np.array(pearson_cutted)
print(pearson_cutted.shape)

batch_size = 2
kfold = 10
dim_input = 4005

meta_size = int(0.1*x_total.shape[0])
x_total = pearson_cutted


def evaluate_svm(x_val, net):
    score = clf.score(x_val, y_val)
    pred = clf.predict(x_val)
    FN = 0
    TN = 0
    TP = 0
    FP = 0
    for i in range(pred.shape[0]):
        if pred[i] == y_val[i, 0] and y_val[i, 0] == 1:
            TP += 1
        elif pred[i] == y_val[i, 0] and y_val[i, 0] == 0:
            TN += 1
        elif pred[i] != y_val[i, 0] and y_val[i, 0] == 1:
            FN += 1
        else:
            FP += 1
    if (TP + FN) == 0:
        sensitivity = 0
    else:
        sensitivity = TP / (TP + FN)
    if (FP+TN) == 0:
        specificity = 0
    else:
        specificity = TN / (FP + TN)

    return score, sensitivity, specificity


avg_sen = 0
avg_spe = 0
avg_score = 0
roc_auc = dict()
fpr = dict()
tpr = dict()
for j in range(100):  # 算100次，减小随机性
    random.seed(j + 1)
    mixed = np.concatenate((x_total, y_total.reshape((-1, 1))), axis=1)
    random.shuffle(mixed)
    kfold = 10
    GTlist = None
    Problist = None
    #     scores = []
    #     avg_score = 0
    for i in range(kfold):  # Kfold交叉验证，其中meta_size为数据集大小的0.1
        clf = RandomForestClassifier(n_estimators=100,criterion='gini',bootstrap=True)
        x_val = mixed[meta_size * i:meta_size * (i + 1), :dim_input]
        y_val = mixed[meta_size * i:meta_size * (i + 1), dim_input:]
        x_train = np.concatenate((mixed[:meta_size * i, :dim_input], mixed[meta_size * (i + 1):, :dim_input]), axis=0)
        y_train = np.concatenate((mixed[:meta_size * i, dim_input:], mixed[meta_size * (i + 1):, dim_input:]), axis=0)
        clf.fit(x_train, y_train)
        #     print('predict:{}, true:{}'.format(clf.predict(x_val), y_val))

        score, sen, spe = evaluate_svm(x_val, clf)
        #         score = clf.score(x_val, y_val)
        #         scores.append(score)
        avg_score += score
        avg_spe += spe
        avg_sen += sen
        if j is 1:
            if GTlist is None:
                GTlist = y_val
            else:
                GTlist = np.concatenate((GTlist, y_val), axis=0)
            if Problist is None:
                Problist = clf.predict(x_val)
            else:
                Problist = np.concatenate((Problist, clf.predict_proba(x_val)[:, 1]), axis=0)
                
    if j is 1:  # 随便取一次实验，绘制ROC曲线。
        print(GTlist.shape, Problist.shape)
        fpr, tpr, thresholds = metrics.roc_curve(GTlist.reshape(100), Problist, pos_label=1)
        print(fpr.shape)
        roc_auc = metrics.auc(fpr, tpr)  # auc为Roc曲线下的面积
        print(roc_auc_score(GTlist, Problist))
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        # plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([-0.1, 1.1])
        plt.ylim([-0.1, 1.1])
        plt.xlabel('False Positive Rate')  # 横坐标是fpr
        plt.ylabel('True Positive Rate')  # 纵坐标是tpr
        plt.title('Receiver operating characteristic example')
        plt.savefig('roc_curve.png')
        plt.show()
        
#         fpr, tpr, _ = roc_curve(y_val, clf.predict(x_val), pos_label=2)
#         roc_auc = auc(fpr, tpr)
#     print(scores)
print('score:{}'.format(avg_score / (kfold * 100)))
print('specificity:{}'.format(avg_spe / (kfold * 100)))
print('sensitivity:{}'.format(avg_sen / (kfold * 100)))
print(clf.predict(x_val), y_val)
