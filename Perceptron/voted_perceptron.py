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



def null_remove(ll): 
    temp =[]
    for i in range(len(ll)):
        if ll[i]!= 0:
            temp.append(ll[i])
    return temp
              
 
def voted_perceptron_train(train_data, w, b, rate, T):
    data_len = len(train_data)
    permu = []  
    for i in range(T):
        pi= np.random.permutation(data_len).tolist()
        permu= permu + pi
    C_last_col =np.zeros(T*data_len).tolist()
    CC= []
    m =0
    for i in range(T*data_len):
        if (train_data[permu[i]][-1])*( np.inner(w, train_data[permu[i]][0:Num_attr])+b ) <= 0:
            w = w + [rate *(train_data[permu[i]][-1])*x for x in train_data[permu[i]][0:Num_attr]] 
            b= b + rate*train_data[permu[i]][-1]   
            m +=1
            row = np.append(w,b)
            CC.append(row)
            C_last_col[m]=1                
        if (train_data[permu[i]][-1])*( np.inner(w, train_data[permu[i]][0:Num_attr])+b ) > 0:
            C_last_col[m] += 1
    return [CC, null_remove(C_last_col)]


def sign_func(x):
    a=0
    if x >0:
        a=1
    else:
        a=-1
    return a


def _merge_(CC,c):
    temp =[]
    for i in range(len(CC)):
        tt = CC[i].tolist() + [c[i]]
        temp.append(tt)
    return temp


def voted_perceptron_predict(test_data, CC,c):
    CC_c =_merge_(CC,c)
    pred_seq =[]
    for i in range(len(test_data)):
       pred_seq.append(sign_func(sum( [(row[-1])*sign_func(np.inner(test_data[i][0:Num_attr], row[0:Num_attr] )+row[Num_attr]) for row in CC_c])))
    count =0
    for i in range(len(test_data)):
        if pred_seq[i] != test_data[i][-1]:
            count+=1
    return count/len(test_data)

# warning: initial w shuld be array type 
def voted_perceptron(train_data, test_data, w, b, rate, T):
    [CC,c] = voted_perceptron_train(train_data, w, b, rate, T)
    print(_merge_(CC,c))
    err = voted_perceptron_predict(test_data, CC,c)
    print(err)


rate = 1    
T = 10
w= np.zeros(Num_attr)
b=0
voted_perceptron(train_polar, test_polar, w, b, rate, T) 