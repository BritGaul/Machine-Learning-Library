import math
import numpy as np
from scipy.optimize import minimize

BANK_TRAINING = "./bank-note/train.csv"
BANK_TESTING = "./bank-note/test.csv"

with open(BANK_TRAINING, mode='r') as f:
    train_bank = []
    for line in f:
        terms = line.strip().split(',')  # 7*N matrix
        train_bank.append(terms)
with open(BANK_TESTING, mode='r') as f:
    test_bank = []
    for line in f:
        terms = line.strip().split(',')  # 7*N matrix
        test_bank.append(terms)


def convert_to_float(data):
    for row in data:
        for j in range(len(data[0])):
            row[j] = float(row[j])
    return data


def constant_feature(data):
    label = [row[-1] for row in data]
    temp = data;
    for i in range(len(data)):
        temp[i][-1] = 1.0
    for i in range(len(data)):
        temp[i].append(label[i])
    return temp


def polar_label(data):
    temp = data;
    for i in range(len(data)):
        temp[i][-1] = 2 * data[i][-1] - 1
    return temp


bank_train = convert_to_float(train_bank)
bank_test = convert_to_float(test_bank)

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


def predict(weight, data):
    pred_seq = [];
    for i in range(len(data)):
        pred_seq.append(sign_func(np.inner(data[i][0:len(data[0]) - 1], weight)))
    label = [row[-1] for row in data]
    return calculate_error(pred_seq, label)


def gaussian_kernel(s_1, s_2, gamma):
    dim = len(s_1) - 1
    s_11 = s_1[0:dim]
    s_22 = s_2[0:dim]
    diff = [s_11[i] - s_22[i] for i in range(dim)]
    kernel = math.e ** (-np.linalg.norm(diff) ** 2 / gamma)
    return kernel


def calculate_matrix():
    k_hat_t = np.ndarray([train_len, train_len])
    for i in range(train_len):
        for j in range(train_len):
            k_hat_t[i, j] = (train_data[i][-1]) * (train_data[j][-1]) * np.inner(train_data[i][0:dim_s],
                                                                                 train_data[j][0:dim_s])
    return k_hat_t


def objective_function(x):
    tp1 = x.dot(K_hat_)
    tp2 = tp1.dot(x)
    tp3 = -1 * sum(x)
    return 0.5 * tp2 + tp3


def constraint(x):
    return np.inner(x, np.asarray(label_))


def svm_dual(C):
    bd = (0, C)
    bds = tuple([bd for i in range(train_len)])
    x0 = np.zeros(train_len)
    cons = {'type': 'eq', 'fun': constraint}
    sol = minimize(objective_function, x0, method='SLSQP', bounds=bds, constraints=cons)
    return [sol.fun, sol.x]


def recover_weights(dual_x):
    lenn = len(dual_x)
    ll = []
    for i in range(lenn):
        ll.append(dual_x[i] * train_data[i][-1] * np.asarray(train_data[i][0: dim_s]))
    return sum(ll)


def svm_dual_main(C):
    [sol_f, sol_x] = svm_dual(C)
    weight = recover_weights(sol_x)
    train_error = predict(weight, train_data)
    test_error = predict(weight, test_data)
    print('weight=', weight)
    print('train err=', train_error)
    print('test err=', test_error)


K_hat_ = calculate_matrix()
label_ = [row[-1] for row in train_data]
CC = [100 / 873, 500 / 873, 700 / 873]
for C_ in CC:
    svm_dual_main(C_)