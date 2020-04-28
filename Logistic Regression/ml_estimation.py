import math
import numpy as np

BANK_TRAINING = "./bank-note/train.csv"
BANK_TESTING = "./bank-note/test.csv"

with open(BANK_TRAINING,mode='r') as f:
    bank_train=[]
    for line in f:
        terms=line.strip().split(',')
        bank_train.append(terms)
with open(BANK_TESTING,mode='r') as f:
    bank_test = []
    for line in f:
        terms=line.strip().split(',')
        bank_test.append(terms)


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


bank_train = convert_to_float(bank_train)
bank_test = convert_to_float(bank_test)

train_data = constant_feature(polar_label(bank_train))
test_data = constant_feature(polar_label(bank_test))

train_len = len(train_data)
test_len = len(test_data)
dim_s = len(train_data[0]) - 1


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


def predict(wt, data):
    pred_seq = [];
    for i in range(len(data)):
        pred_seq.append(sign_func(np.inner(data[i][0:len(data[0]) - 1], wt)))
    label = [row[-1] for row in data]
    return calculate_error(pred_seq, label)


def sigmoid(x):
    # avoid overflow
    if x < -100:
        temp = 0
    else:
        temp = 1 / (1 + math.e ** (-x))
    return temp


def calculate_loss(weight, data):
    seq = []
    for row in data:
        temp = -row[-1] * np.inner(weight, row[0:dim_s])
        if temp > 100:
            t2 = temp
        else:
            t2 = math.log(1 + math.e ** (temp))
        seq.append(t2)
    return sum(seq)


def stochastic_gradient(weight, sample):
    cc = train_len * sample[-1] * (1 - sigmoid(sample[-1] * np.inner(weight, sample[0:dim_s])))
    return np.asarray([-cc * sample[i] for i in range(dim_s)])


def gamma(t, gamma_0, d):
    return gamma_0 / (1 + (gamma_0 / d) * t)


def sgd_single(weight, perm, iter_cnt, gamma_0, d):
    weight = np.asarray(weight)
    loss_seq = []
    for i in range(train_len):
        weight = weight - gamma(iter_cnt, gamma_0, d) * stochastic_gradient(weight, train_data[perm[i]])
        loss_seq.append(calculate_loss(weight, train_data))
        iter_cnt = iter_cnt + 1
    return [weight, loss_seq, iter_cnt]


def sgd_epoch(weight, T, gamma_0, d):
    iter_cnt = 1
    loss = []
    for i in range(T):
        perm = np.random.permutation(train_len)
        [weight, loss_seq, iter_cnt] = sgd_single(weight, perm, iter_cnt, gamma_0, d)
        loss.extend(loss_seq)
        print('epochs=', i)
    return [weight, loss, iter_cnt]


def ml_estimation(VV, TT):
    gamma_0 = 1
    d = 2.0
    train_err = []
    test_err = []
    for var in VV:
        w = np.zeros(5)
        [wt, loss, cnt] = sgd_epoch(w, TT ,gamma_0, d)
        train_err.append(predict(wt, train_data))
        test_err.append(predict(wt, test_data))
    return [train_err, test_err]


V = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]
T = 100
[train_error, test_error] = ml_estimation(V, T)
print('train_error =', train_error)
print('test_error =', test_error)
