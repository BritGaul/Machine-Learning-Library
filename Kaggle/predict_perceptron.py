import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats import pointbiserialr, spearmanr
from sklearn.model_selection import cross_val_score


TRAINING_DATA = "./datasets/train_final.csv"
TESTING_DATA = "./datasets/test_final.csv"

def load_data():
    data = pd.read_csv(TRAINING_DATA)
    
    category_col =['workclass', 'race', 'education','marital.status', 'occupation',
               'relationship', 'sex', 'native.country', 'income>50K']

    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True) 
        data[col] = c

    col_names = data.columns

    param=[]
    correlation=[]
    abs_corr=[]

    for c in col_names:
        #Check if binary or continuous
        if c != "income>50K":
            if len(data[c].unique()) <= 2:
                corr = spearmanr(data['income>50K'],data[c])[0]
            else:
                corr = pointbiserialr(data['income>50K'],data[c])[0]
            param.append(c)
            correlation.append(corr)
            abs_corr.append(abs(corr))
        # make the dataset linearly separable
        data = data[:100]
        data[9] = np.where(data.iloc[:, -1]=='income>50K', 0, 1)
        data = np.asarray(data, dtype = 'float64')
        return data


def load_test_data():
    test_data = pd.read_csv(TESTING_DATA)
    
    category_col =['workclass', 'race', 'education','marital.status', 'occupation',
               'relationship', 'sex', 'native.country', 'ID']

    for col in category_col:
        b, c = np.unique(test_data[col], return_inverse=True) 
        test_data[col] = c

    
     # make the dataset linearly separable
    test_data = test_data[:100]
    test_data[9] = np.where(test_data.iloc[:, -1]=='ID', 0, 1)
    test_data = np.asarray(test_data, dtype = 'float64')
    return test_data

data = load_data()
test_data = load_test_data()

def polar_label(input_list): 
    temp_list = input_list
    for i in range(len(input_list)):
        temp_list[i][-1] = 2*input_list[i][-1]-1
    return temp_list



Num_attr = len(data[0]) 
train_data = data
test_data = test_data     
train_polar = polar_label(train_data)
test_polar = polar_label(test_data)    


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
    print(pred_seq)
    for i in range(len(test_data)):
        if pred_seq[i] != test_data[i][-1]:
            count+=1
    return count/len(test_data)

# warning: initial w shuld be array type 
def voted_perceptron(train_data, test_data, w, b, rate, T):
    [CC,c] = voted_perceptron_train(train_data, w, b, rate, T)
    err = voted_perceptron_predict(test_data, CC,c)
    print(err)


rate = 1    
T = 10
w= np.zeros(Num_attr)
b=0
voted_perceptron(train_polar, test_polar, w, b, rate, T) 