import random

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
from sklearn import metrics

from torch.autograd import Variable

data = sio.loadmat('ADNI.mat')
# print(data)

x_positive = data['NC']  # 正常
x_negative = data['AD']  # 患病
# 归一化
# x_positive_max = 0
# x_positive_min = 1000
# x_negative_max = 0
# x_negative_min = 1000
# for i in range(x_positive.shape[0]):
#     for j in range(x_positive.shape[1]):
#         if x_positive_max < x_positive[i][j]:
#             x_positive_max = x_positive[i][j]
#         if x_positive_min > x_positive[i][j]:
#             x_positive_min = x_positive[i][j]
# x_positive = x_positive.astype(np.float64)
# for i in range(x_positive.shape[0]):
#     for j in range(x_positive.shape[1]):
#         x_positive[i][j] = float(x_positive[i][j] - x_positive_min) / float(x_positive_max - x_positive_min)
#
# for i in range(x_negative.shape[0]):
#     for j in range(x_negative.shape[1]):
#         if x_negative_max < x_negative[i][j]:
#             x_negative_max = x_negative[i][j]
#         if x_negative_min > x_negative[i][j]:
#             x_negative_min = x_negative[i][j]
# x_negative = x_negative.astype(np.float64)
# for i in range(x_negative.shape[0]):
#     for j in range(x_negative.shape[1]):
#         x_negative[i][j] = float(x_negative[i][j] - x_negative_min) / float(x_negative_max - x_negative_min)
# print('data normalized!')

x_total = np.concatenate((x_positive, x_negative), axis=0).astype(np.float)
y_positive = np.ones(x_positive.shape[0])
y_negative = np.zeros(x_negative.shape[0])
y_total = np.concatenate((y_positive, y_negative), axis=0).astype(np.float)
print('有{}个正样本，有{}个负样本，一共有{}个样本'.format(y_positive.shape[0], y_negative.shape[0], x_total.shape[0]))
num_x_pos = x_positive.shape[0]
num_x_neg = x_negative.shape[0]
dim_input = x_positive.shape[1]
# print('x原本是只有一个特征的，为294维')
print(x_positive.shape, x_negative.shape, x_total.shape, y_positive.shape, num_x_neg, num_x_pos, dim_input)
meta_size = int(0.1 * x_total.shape[0])

# 载入数据
random.seed(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device: %s' % device)
batch_size = 1

train_size = int(0.8 * x_total.shape[0])
test_size = int(0.9 * x_total.shape[0])

mixed = np.concatenate((x_total, y_total.reshape((-1, 1))), axis=1)
random.shuffle(mixed)
# print(mixed)
x_train = mixed[:train_size, :dim_input]
y_train = mixed[:train_size, dim_input:]
print(x_train.shape, y_train.shape)
x_val = mixed[train_size:test_size, :dim_input]
y_val = mixed[train_size:test_size, dim_input:]
print(x_val.shape, y_val.shape)
x_test = mixed[test_size:, :dim_input]
y_test = mixed[test_size:, dim_input:]
print(x_test.shape, y_test.shape)

train_data = Data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
val_data = Data.TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
test_data = Data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = Data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=0)
print(len(train_data), len(val_data), len(test_data))


#  评估
def evaluate_accuracy(loader, net):
    net.eval()
    sum_all = 0
    sum_acc = 0
    FN = 0
    TN = 0
    TP = 0
    FP = 0

    for index, (data, label) in enumerate(loader):
        with torch.no_grad():
            data = Variable(data, requires_grad=False).to(device)
            label = Variable(label, requires_grad=False).to(device)
        outputs = net(data).argmax(1).reshape(data.shape[0])
        #         print(outputs.shape, label.shape)
        sum_acc += torch.sum(outputs == label.reshape(data.shape[0]))
        label = label.reshape(data.shape[0])
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
    # print('total validation set data:{}, numbers of accurately labeled data:{}'.format(sum_all, sum_acc.detach()))
    acc = sum_acc.cpu().numpy() / sum_all
    # print(TP, TN, FP, FN)
    net.train()
    if (TP + FN) == 0:
        sensitivity = 0
    else:
        sensitivity = TP / (TP + FN)
    if (FP + TN) == 0:
        specificity = 0
    else:
        specificity = TN / (FP + TN)
    return acc, sensitivity, specificity


# Logistic Classification


#  GAN
class Discriminator(nn.Module):
    def __init__(self, n_feature):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(n_feature, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model(x)
        output = F.softmax(output)
        return output


class Generator(nn.Module):
    def __init__(self, n_feature):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, n_feature),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)

        return output


# MLP
class MLP(nn.Module):
    def __init__(self, n_feature):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_feature, 100)
        self.fc2 = nn.Linear(100, 2)
        # self.fc3 = nn.Linear(50, 10)
        # self.fc4 = nn.Linear(10, 2)

        self.init_weights()

    def init_weights(self):
        self.fc1.weight.data.uniform_(-1, 1)
        self.fc1.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-1, 1)
        self.fc2.bias.data.fill_(0)
        # self.fc3.weight.data.uniform_(-1, 1)
        # self.fc3.bias.data.fill_(0)

    def forward(self, x):
        output1 = F.relu(self.fc1(x))
        output1 = F.dropout(output1, 0.4)
        output2 = self.fc2(output1)
        # output2 = F.dropout(output2, 0.4)
        # output3 = F.relu(self.fc3(output2))
        # output3 = F.dropout(output3, 0.4)
        # output4 = self.fc4(output3)
        return output2


#  混合方法
class xxx(nn.Module):
    def __init__(self, n_feature):
        super(xxx, self).__init__()
        self.linear = nn.Linear(n_feature, 1)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        self.linear.weight.data.uniform_(-1, 1)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        y = self.linear(x)
        return torch.sigmoid(y)


class LabelSmoothSoftmaxCEV1(nn.Module):
    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float()  # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss


losses = []
accuracy = []
sensitivity = []
specificity = []
preds = []
labels = []
avg_accuracy = []

print_every = 100
print_loss_total = 0
for seed in range(1):
    random.seed(seed)
    mixed = np.concatenate((x_total, y_total.reshape((-1, 1))), axis=1)
    random.shuffle(mixed)
    kfold = 10
    GTlist = None
    Problist = None
    for i in range(kfold):
        num_epochs = 25  # 训练的epoch
        model = MLP(dim_input).to(device)
        criterion = nn.CrossEntropyLoss().to(device)  # 损失函数
        learning_rate = 0.0005  # 学习率
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5e-4, momentum=0.8)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
        print('fold:{}'.format(i))
        x_val = mixed[meta_size * i:meta_size * (i + 1), :dim_input]
        y_val = mixed[meta_size * i:meta_size * (i + 1), dim_input:]
        x_train = np.concatenate((mixed[:meta_size * i, :dim_input], mixed[meta_size * (i + 1):, :dim_input]), axis=0)
        y_train = np.concatenate((mixed[:meta_size * i, dim_input:], mixed[meta_size * (i + 1):, dim_input:]), axis=0)
        train_data = Data.TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        # print(train_data[0][0].dtype)
        val_data = Data.TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
        #     test_data = Data.TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
        train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = Data.DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, num_workers=0)
        #     test_loader = Data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle=True, num_workers=1)

        for epoch in range(num_epochs):
            model.train()
            #             print('\tEpoch:{}'.format(epoch+1))
            for index, (data1, label1) in enumerate(train_loader):
                data = Variable(data1, requires_grad=True).to(device)
                label = Variable(label1, requires_grad=True).to(device)
                output = model(data)
                #                 print(output.shape, label.shape)
                #                 label = label.squeeze()
                #         label = label.squeeze()
                loss = criterion(output, label.detach().reshape(label.shape[0]).long())
                print_loss_total += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                for i in range(data.shape[0]):
                    preds.append(output[i, int(np.array(label[i].detach().cpu())[0])].detach().cpu())
                    labels.append(label[i].detach().cpu())
                if index % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print('\t\tloss:{}'.format(print_loss_avg))
                    losses.append(print_loss_avg)
                    print_loss_total = 0
            acc, sen, spe = evaluate_accuracy(val_loader, model)
            accuracy.append(acc)
            sensitivity.append(sen)
            specificity.append(spe)
            #             if epoch % 10 == 0:
            print('Epoch:{}, Accuracy:{}, Sensitivity:{}, Specificity:{}'.format(epoch + 1, acc, sen, spe))
            print('\n')
            tacc, _, _ = evaluate_accuracy(train_loader, model)
            print('training accuracy:{}'.format(tacc))
        acc, sen, spe = evaluate_accuracy(val_loader, model)
        avg_accuracy.append(acc)
    mean_accuracy = 0
    for i in avg_accuracy:
        mean_accuracy += i
    print('average_accuracy:{}'.format(mean_accuracy / kfold))
total_loader = Data.DataLoader(
    dataset=Data.TensorDataset(torch.Tensor(mixed[:, :dim_input]), torch.Tensor(mixed[:, :dim_input])),
    batch_size=batch_size, shuffle=True, num_workers=0)

# tmp, _, _ = evaluate_accuracy(total_loader, model)
# print('total acc:', tmp)

avg_sen = 0
avg_spe = 0
for i, (sen, spe) in enumerate(zip(sensitivity, specificity)):
    avg_sen += sen
    avg_spe += spe
print('average_sensitivity:{}, average_specificity:{}'.format(avg_sen / len(sensitivity), avg_spe / len(specificity)))

# 可视化
plt_list = [losses, accuracy, sensitivity, specificity]
plt.figure(figsize=(10, 10))
for i in range(4):
    plt.subplot(4, 1, i + 1)
    plt.plot(plt_list[i])
plt.show()

# ROC曲线
# 这个GTlist是真实标签
GTlist = labels
# 这个是预测值，
Problist = preds

# print(GTlist, Problist)
fpr, tpr, thresholds = metrics.roc_curve(GTlist, Problist, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)  # auc为Roc曲线下的面积
print('auc:{}'.format(roc_auc))

pylab.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
pylab.legend(loc='lower right')
# plt.plot([0, 1], [0, 1], 'r--')
pylab.xlim([-0.1, 1.1])
pylab.ylim([-0.1, 1.1])
pylab.xlabel('False Positive Rate')  # 横坐标是fpr
pylab.ylabel('True Positive Rate')  # 纵坐标是tpr
pylab.title('Receiver operating characteristic example')
pylab.show()

acc, sen, spe = evaluate_accuracy(test_loader, model)
print('\n')
print('Accuracy on test set:{}, \tSensitivity on test set:{}, \tSpecificity on test set:{}'.format(acc, sen, spe))
