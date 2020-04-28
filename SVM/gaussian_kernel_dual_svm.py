import math
import numpy as np
from scipy.optimize import minimize

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
    temp = data;
    for i in range(len(data)):
        temp[i][-1] = 1.0
    for i in range(len(data)):
        temp[i].append(label[i])
    return temp


def polar_label(data):
    temp = data;
    for i in range(len(data)):
        temp[i][-1] = 2 * data[i][-1] - 1;
    return temp


bank_train = convert_to_float(train_bank)
bank_test = convert_to_float(test_bank)

train_data = constant_feature(polar_label(bank_train))
test_data = constant_feature(polar_label(bank_test))

train_len = len(train_data)  # NO. of samples
test_len = len(test_data)
dim_s = len(train_data[0]) - 1  # sample dim


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


def kernel_predict(dual_x, data, gamma):
    true_label = [row[-1] for row in data]
    pred_seq = [];
    for row in data:
        ll = []
        for i in range(len(dual_x)):
            ll.append(dual_x[i] * train_data[i][-1] * gaussian_kernel(train_data[i][0:dim_s], row[0:dim_s], gamma))
        pred = sign_func(sum(ll))
        pred_seq.append(pred)
    return calculate_error(pred_seq, true_label)


def gaussian_kernel(s_1, s_2, gamma):
    s_1_ = np.asarray(s_1)
    s_2_ = np.asarray(s_2)
    return math.e ** (-np.linalg.norm(s_1_ - s_2_) ** 2 / gamma)


def calculate_kernel_matrix(gamma):
    K_hat_t = np.ndarray([train_len, train_len])
    for i in range(train_len):
        for j in range(train_len):
            K_hat_t[i, j] = gaussian_kernel(train_data[i][0:dim_s], train_data[j][0:dim_s], gamma)
    return K_hat_t


def objective_function(x):
    tp1 = x.dot(K_mat_)
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
    sol = minimize(objective_function, x0, method='SLSQP', bounds=bds, constraints=cons)  # MINIMIZER
    return [sol.fun, sol.x]


def count_support_vectors(dual_x):
    ll = []
    for i in range(len(dual_x)):
        if dual_x[i] != 0.0:
            ll.append(i)
    return [np.count_nonzero(dual_x), set(ll)]


def svm_main(C):
    [sol_f, sol_x] = svm_dual(C)
    [cnt, gg] = count_support_vectors(sol_x)
    #err_1 = predict_ker(sol_x, train_data, gamma)
    #err_2 = predict_ker(sol_x, test_data, gamma)
    #print('train err=', err_1)
    #print('test err=', err_2)
    return [cnt, gg]


label_ = [row[-1] for row in train_data]
CC_ = [100 / 873, 500 / 873, 700 / 873]
Gamma_ = [0.1, 0.5, 1, 5, 100]
for C in CC_:
    for gamma in Gamma_:
        print('C=',C, 'gamma=', gamma)
        K_mat_ = calculate_kernel_matrix(gamma)
        svm_main(C)


C = 5/873
gamma = 10
K_mat_ = calculate_kernel_matrix(gamma)
svm_main(C)