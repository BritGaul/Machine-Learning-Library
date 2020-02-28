import math
import statistics
import numpy as np
import matplotlib.pyplot as plot
from numpy.linalg import inv

CONCRETE_TRAINING = "./datasets/concrete/train.csv"
CONCRETE_TESTING = "./datasets/concrete/test.csv"

with open(CONCRETE_TRAINING,mode='r') as f:
    concrete_train=[]
    for line in f:
        terms=line.strip().split(',') 
        concrete_train.append(terms)
with open(CONCRETE_TESTING,mode='r') as f:
    concrete_test = []
    for line in f:
        terms=line.strip().split(',') 
        concrete_test.append(terms)
        
def convert_to_float(data):
    for row in data:
        for j in range(len(data[0])):
            row[j] = float(row[j])
    return data

m = len(concrete_train)   
d = len(concrete_train[0]) - 1

concrete_train = convert_to_float(concrete_train)  
concrete_test = convert_to_float(concrete_test) 


def calculate_cost(w, dataset):
    loss = 0.5*sum([ (row[-1]-np.inner(w,row[0:7]))**2 for row in dataset ])
    return loss

def gradient(w, dataset):
    gradient = []
    for j in range(d):
        gradient.append(-sum([ (row[-1]-np.inner(w, row[0:7]))*row[j] for row in dataset]))
    return gradient

def batch_gradient(eps, rate, w, dataset):
    cost =[]
    while np.linalg.norm(gradient(w, dataset)) >= eps:
        cost.append(calculate_cost(w, dataset))
        w = w - [rate*x for x in gradient(w, dataset)]       
    return [w, cost]


def sgd_single(eps, rate, w, dataset, pi):
    flag = 0
    loss_vec =[]
    for x in pi:
        if np.linalg.norm(sgd_gradient(w, pi[x], dataset)) <= eps:
            flag = 1
            return [w, loss_vec, flag]
        loss_vec.append(calculate_cost(w, dataset))
        w = w - [rate*x for x in sgd_gradient(w, pi[x] ,dataset)]     
    return [w, loss_vec, flag]


def sgd_gradient(w, sample_idx, dataset):
    s_grad = []
    for j in range(d):
        s_grad.append(-(dataset[sample_idx][-1]-np.inner(w, dataset[sample_idx][0:7]) )*dataset[sample_idx][j])
    return s_grad

def sgd_random(eps, rate, w, dataset, N_epoch ):
    loss_all =[]
    for i in range(N_epoch):
        pi = np.random.permutation(m)
        [w, loss_vec, flag] = sgd_single(eps, rate, w, dataset, pi)
        if flag == 1:
            return [w, loss_all]
        loss_all = loss_all + loss_vec
    return [w, loss_all]


# Test Batch
[ww, loss_v] = batch_gradient(0.0001, 0.01, np.zeros(d), concrete_train)
print(ww)
print(calculate_cost(ww, concrete_train))
print(calculate_cost(ww, concrete_test))
plot.plot(loss_v)
plot.ylabel('cost function value')
plot.xlabel('steps')
plot.title('Batch Gradient Descent')
plot.show()

# Test Stochastic
[ww, all_cost] = sgd_random(0.000001, 0.001, np.zeros(d), concrete_train, 20000)
print(ww)
print(calculate_cost(ww, concrete_train))
print(calculate_cost(ww, concrete_test))
plot.plot(all_cost)
plot.ylabel('cost function value')
plot.xlabel('steps')
plot.title('Stochastic Gradient Descent')
plot.show()

# Test Analytical
data_list = [row[0:7] for row in concrete_train]
label_list = [row[-1] for row in concrete_test]
data_mat = np.array(data_list)
label_mat = np.array(label_list)
X = data_mat.transpose()
a = inv(np.matmul(X, X.transpose()))
b = np.matmul(a, X)
c =np.matmul(b, label_mat)
print(c)