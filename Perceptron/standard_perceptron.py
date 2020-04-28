# perceptron algorithm
import math
import statistics
import numpy as np

BANK_TRAINING = "./bank-note/train.csv"
BANK_TESTING = "./bank-note/test.csv"

with open(BANK_TRAINING,mode='r') as f:
    bank_train=[]
    for line in f:
        terms=line.strip().split(',') 
        bank_train.append(terms)
with open(BANK_TESTING,mode='r') as f:
    bank_test=[]
    for line in f:
        terms=line.strip().split(',') 
        bank_test.append(terms)
      
def convert_to_float(bank_list):  
    temp_list = bank_list
    for k in range(len(temp_list)):
        for i in range(len(temp_list[0])):
            temp_list[k][i] = float(bank_list[k][i])
    return temp_list  

def polar_label(bank_list): 
    temp_list = bank_list
    for i in range(len(bank_list)):
        temp_list[i][-1] = 2*bank_list[i][-1]-1
    return temp_list      


Num_attr = len(bank_train[0])-1 
train_bank = convert_to_float(bank_train)
test_bank = convert_to_float(bank_test)      
train_polar = polar_label(train_bank)
test_polar = polar_label(test_bank)    

print(train_bank)
print("############################################################################################################")
print(train_polar)

def perceptron_single(bank_list,pi, w, b, rate):  
    for i in range(len(bank_list)):
        if (bank_list[pi[i]][-1])*(np.inner(w, bank_list[pi[i]][0:Num_attr])+b) <=0:
            w = w + [rate*(bank_list[pi[i]][-1])*x for x in bank_list[pi[i]][0:Num_attr] ]    
            b = b + rate*bank_list[pi[i]][-1]*1
    return [w,b]


def predict_error(test_data, w,b):   
    num_test =len(test_data)
    count = 0
    for i in range(num_test):
        if (test_data[i][-1])*(np.inner(w, test_data[i][0:Num_attr])+b) <=0:
            count +=1
    return count/num_test
        
       
def epoch_perceptron(train_data, w, b,rate,T):
    for t in range(T):
        pi = np.random.permutation(len(train_data))
        [w, b] = perceptron_single(train_data, pi, w, b, rate )
    return [w,b]

rate = 1     
T =10
def standard_perceptron(train_p, test_p, w , b, rate, T):
    [ww,bb] = epoch_perceptron(train_p, w,b, rate, T)
    err = predict_error(test_polar, ww, bb)
    return [ww, bb, err]  


# for t in range (1, T+1):
#     print("T = ", t)
#     print(standard_perceptron(train_polar, test_polar, np.zeros(Num_attr), 0 ,rate,t))