# coding=UTF-8
import scipy.io as sio
import numpy as np

from sklearn import metrics
import matplotlib.pylab as pylab

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import warnings
import torch
import torch
import torch.utils.data as Data
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn.functional as F
# import torch
from torch.autograd import Variable
import random
import learn2learn as l2l

warnings.filterwarnings("ignore")

data = sio.loadmat('ADNI.mat')
# print(data)

x_positive = data['NC']  # 正常
x_negative = data['AD']  # 患病
# x_fake_positive = data['EMCI']
# x_fake_negative = data['LMCI']
x_total = np.concatenate((x_positive, x_negative), axis=0).astype(np.float)
y_positive = np.ones(x_positive.shape[0])
# y_fake_positive = np.ones(x_fake_positive.shape[0])
# y_fake_negative = np.zeros(x_negative.shape[0])
y_negative = np.zeros(x_negative.shape[0])
y_total = np.concatenate((y_positive, y_negative), axis=0).astype(np.float)
# print('没有加伪样本的时候有{}个正样本，有{}个负样本，一共有{}个样本'.format(y_positive.shape[0], y_negative.shape[0], x_total.shape[0]))
num_x_pos = x_positive.shape[0]
num_x_neg = x_negative.shape[0]
dim_input = x_positive.shape[1]
# num_x_fake_pos = x_fake_positive.shape[0]
# num_x_fake_neg = x_fake_negative.shape[0]
# x_total_faked = np.concatenate((x_positive, x_fake_positive, x_fake_negative, x_negative), axis=0).astype(np.float)
# y_total_faked = np.concatenate((y_positive, y_fake_positive, y_fake_negative, y_negative), axis=0).astype(np.float)
# num_fake_total = x_total_faked.shape[0]
num_total = x_total.shape[0]
# print('加了伪样本之后有{}个正样本，有{}个负样本，一共有{}个样本'.format(num_x_fake_pos+num_x_pos, num_x_fake_neg+num_x_neg, num_fake_total))
print(x_positive.shape, x_negative.shape, x_total.shape, y_positive.shape, num_x_neg, num_x_pos, dim_input)

'''
def corrcoef(input):
    """传入一个tensor格式的矩阵x(x.shape(m,n))，输出其相关系数矩阵"""
    output = []
    for index, x in enumerate(input):
        f = (x.shape[0] - 1) / x.shape[0]  # 方差调整系数
        x_reducemean = x - np.mean(x, axis=0)
        numerator = np.matmul(x_reducemean.T, x_reducemean) / x.shape[0]
        var_ = x.var(axis=0).reshape(x.shape[1], 1)
        denominator = np.sqrt(np.matmul(var_, var_.T)) * f
        output.append(numerator / denominator)
    return output


# print(x_total[:, :, 19:179].shape)
# 裁剪
pearson = corrcoef(x_total[:, :, 20:180].reshape(82, 160, 90))
# pearson = corrcoef(x_total.reshape(82, 200, 90))
pearson = np.array(pearson)
pearson_cutted = np.zeros((82, 45 * 89))
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
'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 1
kfold = 10
dim_input = 186
meta_size = int(0.1 * x_total.shape[0])


def evaluate_accuracy(loader, net):
    net.eval()
    sum_all = 0
    sum_acc = 0
    FN = 0
    TN = 0
    TP = 0
    FP = 0
    labels = []
    preds = []

    for index, (data, label) in enumerate(loader):
        with torch.no_grad():
            data = Variable(data, requires_grad=False).to(device)
            label = Variable(label, requires_grad=False).to(device)
        outputs = net(data).argmax(1).reshape(data.shape[0])
        #         print(outputs.shape, label.shape)
        sum_acc += torch.sum(outputs == label.reshape(data.shape[0]))
        label = label.reshape(data.shape[0])
        preds.append(outputs.detach().cpu())
        labels.append(label.detach().cpu())
        for i in range(outputs.shape[0]):
            if outputs[i] == label[i] and label[i] == 1:
                TP += 1
            elif outputs[i] == label[i] and label[i] == 0:
                TN += 1
            elif outputs[i] != label[i] and label[i] == 1:
                FN += 1
            else:
                FP += 1
        sum_all += data.shape[0]
        # print('\t\t\ttotal validation set data:{}, numbers of accurately labeled data:{}'.format(sum_all,
        # sum_acc.detach()))
    acc = float(sum_acc) / float(sum_all)
    net.train()
    if (TP + FN) == 0:
        sensitivity = 0
    else:
        sensitivity = TP / (TP + FN)
    if (FP + TN) == 0:
        specificity = 0
    else:
        specificity = TN / (FP + TN)
    GTlist = labels
    Problist = preds
    fpr, tpr, thresholds = metrics.roc_curve(GTlist, Problist, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)  # auc为Roc曲线下的面积
    return acc, sensitivity, specificity, roc_auc, preds, labels


'''
class generator(nn.Module):
    def __init__(self, n_input, n_feature):
        super(generator, self)
        self.linear1 = nn.Linear(n_input, n_features[0])
        self.linear2 = nn.Linear(n_features[0], n_features[1])
        self.linear3 = nn.Linear(n_features[1], 4095)

    def forward(x):
        x = nn.LeakyReLU(self.linear1(x), inplace=True)
        x = nn.LeakyReLU(self.linear2(x), inplace=True)
        x = self.linear3(x)
        return x
'''


class MLP(nn.Module):
    def __init__(self, n_feature):
        super(MLP, self).__init__()
        #         nn.conv1 =
        resnet18 = torchvision.models.resnet18(pretrained=True)
        resnet18.fc2 = nn.Linear(100, 2)
        resnet18.fc1 = nn.Linear(n_feature, 100)
        # self.linear1 = nn.Linear(n_feature, 100)
        self.linear1 = resnet18.fc1
        # self.linear3 = nn.Linear(1000, 100)
        #         self.linear4 = resnet18.fc
        self.linear2 = resnet18.fc2
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        self.linear1.weight.data.uniform_(-1, 1)
        self.linear1.bias.data.fill_(0)
        #         self.linear2.weight.data.uniform_(-1, 1)
        #         self.linear2.bias.data.fill_(0)
        self.linear2.weight.data.uniform_(-1, 1)
        self.linear2.bias.data.fill_(0)
        # self.linear4.weight.data.uniform_(-1, 1)
        # self.linear4.bias.data.fill_(0)

    def forward(self, x):
        out = F.dropout(F.relu((self.linear1(x))), 0.4)
        out = self.linear2(out)
        #         x = self.linear4(x)
        return out


losses = []
accuracy = []
sensitivity = []
specificity = []
preds = None
labels = None
avg_accuracy = []
auc_total = []
print_every = 200
print_loss_total = 0
mean_accuracy = 0
mean_taccuracy = 0
num_seed = 10
for seed in range(num_seed):
    print(seed)
    random.seed(seed)
    mixed = np.concatenate((x_total, y_total.reshape((-1, 1))), axis=1)
    random.shuffle(mixed)
    kfold = 10
    GTlist = None
    Problist = None
    for i in range(kfold):
        num_epochs = 20  # 训练的epoch
        dim_input = 186
        xx = MLP(dim_input).to(device)
        learning_rate = 0.0001
        learning_rate_maml = 0.0001
        xx = l2l.algorithms.MAML(xx, lr=learning_rate_maml).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        # optimizer = optim.SGD(xx.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.6)
        optimizer = optim.Adam(xx.parameters(), lr=learning_rate)  # , weight_decay=5e-4, momentum=0.9)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
        print('fold:{}'.format(i))
        x_val = mixed[meta_size * i:meta_size * (i + 1), :dim_input]
        y_val = mixed[meta_size * i:meta_size * (i + 1), dim_input:]
        x_train = np.concatenate((mixed[:meta_size * i, :dim_input], mixed[meta_size * (i + 1):, :dim_input]), axis=0)
        y_train = np.concatenate((mixed[:meta_size * i, dim_input:], mixed[meta_size * (i + 1):, dim_input:]), axis=0)
        train_data = Data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        val_data = Data.TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
        train_data = l2l.data.MetaDataset(train_data)
        val_data = l2l.data.MetaDataset(val_data)
        #     test_data = Data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = Data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, num_workers=0)
        #     test_loader = Data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle=True, num_workers=1)

        for epoch in range(num_epochs):
            xx.train()
            print('\tEpoch:{}'.format(epoch+1))
            for index, (data1, label1) in enumerate(train_loader):
                optimizer.zero_grad()
                data = Variable(data1, requires_grad=True).to(device)
                label = Variable(label1, requires_grad=True).to(device)
                clone_model = xx.clone()
                error = criterion(clone_model(data), label.detach().reshape(label.shape[0]).long())
                clone_model.adapt(error)
                #                 print(output.shape, label.shape)
                #                 label = label.squeeze()
                #         label = label.squeeze()
                output = clone_model(data)
                loss = criterion(output, label.detach().reshape(label.shape[0]).long())

                print_loss_total += loss

                loss.backward()
                optimizer.step()
                # scheduler.step()
                if index % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    # print('\t\tloss:{}'.format(print_loss_avg))
                    losses.append(print_loss_avg)
                    print_loss_total = 0
            # acc, sen, spe, auc, preds, labels = evaluate_accuracy(val_loader, xx)
            # accuracy.append(acc)
            # if epoch % 10 == 0: print('\tEpoch:{}, Accuracy:{}, Sensitivity:{}, Specificity:{}, auc:{}'.format(
            # epoch + 1, acc, sen, spe, auc))
            tacc, _, _, _, _, _ = evaluate_accuracy(train_loader, xx)
            # print('training accuracy:{}'.format(tacc))
        acc, sen, spe, auc, preds, labels = evaluate_accuracy(val_loader, xx)
        sensitivity.append(sen)
        specificity.append(spe)
        avg_accuracy.append(acc)

        auc_total.append(auc)
        mean_accuracy += acc
        mean_taccuracy += tacc
print('average_accuracy:{}'.format(mean_accuracy / (kfold * num_seed)))
print('train_accuracy:{}'.format(mean_taccuracy / (kfold * num_seed)))
total_loader = Data.DataLoader(
    dataset=Data.TensorDataset(torch.Tensor(mixed[:, :dim_input]), torch.Tensor(mixed[:, dim_input:])),
    batch_size=batch_size, shuffle=True, num_workers=0)
tmp, _, _, total_auc, preds, labels = evaluate_accuracy(total_loader, xx)
print('total acc:{}, total_auc:{}'.format(tmp, total_auc))
avg_sen = 0
avg_spe = 0
avg_auc = 0
for i, (sen, spe, auc) in enumerate(zip(sensitivity, specificity, auc_total)):
    avg_sen += sen
    avg_spe += spe
    avg_auc += auc
print('Sensitivity:{}, Specificity:{}, auc:{}'.format(avg_sen / (kfold * num_seed), avg_spe / (kfold * num_seed),
                                                      avg_auc / (kfold * num_seed)))

GTlist = labels
Problist = preds
fpr, tpr, thresholds = metrics.roc_curve(GTlist, Problist, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)  # auc为Roc曲线下的面积
print(roc_auc)
pylab.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
pylab.legend(loc='lower right')
pylab.xlim([-0.1, 1.1])
pylab.ylim([-0.1, 1.1])
pylab.xlabel('False Positive Rate')  # 横坐标是fpr
pylab.ylabel('True Positive Rate')  # 纵坐标是tpr
pylab.title('Receiver operating characteristic example')
pylab.show()
pylab.savefig('auc.png', dpi=200, bbox_inches='tight')
