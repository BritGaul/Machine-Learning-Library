import numpy as np
from numpy.linalg import inv

BANK_TRAINING = "./bank-note/train.csv"
BANK_TESTING = "./bank-note/test.csv"

with open(BANK_TRAINING, mode='r') as f:
    train_bank = []
    for line in f:
        terms = line.strip().split(',')
        train_bank.append(terms)
with open(BANK_TESTING, mode='r') as f:
    test_bank = []
    for line in f:
        terms = line.strip().split(',')
        test_bank.append(terms)


def convert_to_float(data):
    for row in data:
        for j in range(len(data[0])):
            row[j] = float(row[j])
    return data


def constant_feature(data):
    label = [row[-1] for row in data]
    temp = data
    for i in range(len(data)):
        temp[i][-1] = 1.0
    for i in range(len(data)):
        temp[i].append(label[i])
    return temp


def polar_label(data):
    temp = data
    for i in range(len(data)):
        temp[i][-1] = 2 * data[i][-1] - 1
    return temp


bank_train = convert_to_float(train_bank)
bank_test = convert_to_float(test_bank)

train_data = constant_feature(polar_label(bank_train))
test_data = constant_feature(polar_label(bank_test))

train_len = len(train_data)
test_len = len(test_data)


def rate_with_d(x, gamma_0, d):
    return gamma_0 / (1 + gamma_0 * x / d)


def rate_without_d(x, gamma_0):
    return gamma_0 / (1 + x)


def sub_gradient(curr_wt, sample, iter_cnt, rate_flag, C, gamma_0, d):
    weight = list(np.zeros(len(sample) - 1))
    w_0 = curr_wt[0:len(curr_wt) - 1];
    w_0.append(0)
    w_00 = w_0
    if rate_flag == 1:
        temp_1 = 1 - rate_with_d(iter_cnt, gamma_0, d)
        temp_2 = rate_with_d(iter_cnt, gamma_0, d)
        temp_3 = temp_2 * C * train_len * sample[-1]
        if sample[-1] * np.inner(sample[0:len(sample) - 1], curr_wt) <= 1:
            weight_1 = [x * temp_1 for x in w_00]
            weight_2 = [x * temp_3 for x in sample[0:len(sample) - 1]]
            weight = [weight_1[i] + weight_2[i] for i in range(len(weight_1))]
        else:
            weight = [x * temp_1 for x in w_00]
    if rate_flag == 2:
        temp_1 = 1 - rate_without_d(iter_cnt, gamma_0)
        temp_2 = rate_without_d(iter_cnt, gamma_0)
        temp_3 = temp_2 * C * train_len * sample[-1]
        if sample[-1] * np.inner(sample[0:len(sample) - 1], curr_wt) <= 1:
            weight_1 = [x * temp_1 for x in w_00]
            weight_2 = [x * temp_3 for x in sample[0:len(sample) - 1]]
            weight = [weight_1[i] + weight_2[i] for i in range(len(weight_1))]
        else:
            weight = [x * temp_1 for x in w_00]
    return weight


def svm_single(weight, iter_cnt, permu, train_data, C, rate_flag, gamma_0, d):
    loss_ = [];
    for i in range(train_len):
        weight = sub_gradient(weight, train_data[permu[i]], iter_cnt, rate_flag, C, gamma_0, d)
        loss_.append(loss_func(weight, C, train_data))
        iter_cnt = iter_cnt + 1
    return [weight, iter_cnt, loss_]


def svm_epoch(weight, T, train_data, C, rate_flag, gamma_0, d):
    iter_cnt = 1
    loss = []
    for i in range(T):
        permu = np.random.permutation(train_len)
        [weight, iter_cnt, loss_] = svm_single(weight, iter_cnt, permu, train_data, C, rate_flag, gamma_0, d)
        loss.extend(loss_)
    return [weight, loss]


def sign_func(x):
    y = 0
    if x > 0:
        y = 1
    else:
        y = -1
    return y


def calculate_error(xx, yy):
    cnt = 0
    length = len(xx)
    for i in range(length):
        if xx[i] != yy[i]:
            cnt = cnt + 1
    return cnt / length


def predict(weight, data):
    pred_seq = [];
    for i in range(len(data)):
        pred_seq.append(sign_func(np.inner(data[i][0:len(data[0]) - 1], weight)))
    label = [row[-1] for row in data]
    return calculate_error(pred_seq, label)


def loss_func(weight, C, train_data):
    temp = [];
    for i in range(train_len):
        temp.append(max(0, 1 - train_data[i][-1] * np.inner(weight, train_data[i][0:len(train_data[0]) - 1])))
    val = 0.5 * np.linalg.norm(weight) ** 2 + C * sum(temp)
    return val


def svm(rate_flag, T, gamma_0, d):
    C_global = [x / 873 for x in [100, 500, 700]]
    for C_glo in C_global:
        wt = list(np.zeros(len(train_data[0]) - 1))
        [ww, loss_val] = svm_epoch(wt, T, train_data, C_glo, rate_flag, gamma_0, d)
        print('LEARNED WEIGHT:', ww)
        err_train = predict(ww, train_data)
        err_test = predict(ww, test_data)
        print('TRAIN ERROR:', err_train)
        print('TEST ERROR:', err_test)


rate_flag = 2
T = 100
gamma_0 = 2.3
d = 1
svm(rate_flag, T, gamma_0, d)
