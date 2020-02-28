import math
import statistics
import numpy as np
import matplotlib.pyplot as plot

BANK_TRAINING = "./datasets/bank/train.csv"
BANK_TESTING = "./datasets/bank/test.csv"

with open(BANK_TRAINING,mode='r') as f:
    train_bank=[]
    for line in f:
        terms=line.strip().split(',') 
        train_bank.append(terms)

num_set={0,5,9,11,12,13,14}  

def convert_to_float(mylist):
    temp_list = mylist
    for k in range(len(temp_list)):
        for i in {0,5,9,11,12,13,14}:
            temp_list[k][i] = float(mylist[k][i])
    return temp_list

train_bank = convert_to_float(train_bank)

obj={0:0,5:5,9:9,11:11,12:12,13:13,14:14}
for i in obj:
    obj[i] = statistics.median([row[i] for row in train_bank])
    
for row in train_bank:
    for i in obj:
        if row[i] >= obj[i]:
            row[i] = 'yes'
        else:
            row[i] = 'no'                     

with open(BANK_TESTING,mode='r') as f:
    test_bank=[]
    for line in f:
        terms=line.strip().split(',') 
        test_bank.append(terms)

test_bank = convert_to_float(test_bank)
for i in obj:
    obj[i] = statistics.median([row[i] for row in test_bank])
for row in test_bank:
    for i in obj:
        if row[i] >= obj[i]:
            row[i] = 'yes'
        else:
            row[i] = 'no'  

bank_attributes ={'age':['yes','no'],
             'job':['admin.','unknown','unemployed','management',
                    'housemaid','entrepreneur','student','blue-collar',
                    'self-employed','retired','technician','services'],
                    'martial':['married','divorced','single'],
                    'education':['unknown','secondary','primary','tertiary'],
                     'default':['yes','no'],
                     'balance':['yes','no'],
                     'housing':['yes','no'],
                     'loan':['yes','no'],
                     'contact':['unknown','telephone','cellular'],
                     'day':['yes','no'],
                     'month':['jan', 'feb', 'mar', 'apr','may','jun','jul','aug','sep','oct', 'nov', 'dec'],
                     'duration': ['yes','no'],
                     'campaign':['yes','no'],
                     'pdays':['yes','no'],
                     'previous':['yes','no'],
                     'poutcome':[ 'unknown','other','failure','success']}

attribute_set = set(key for key in bank_attributes)   

def pos(attr):
    pos=0
    if attr=='age':
        pos=0
    if attr=='job':
        pos=1
    if attr=='martial':
        pos=2
    if attr=='education':
        pos=3
    if attr=='default':
        pos=4
    if attr=='balance':
        pos=5
    if attr=='housing':
        pos=6
    if attr=='loan':
        pos=7
    if attr=='contact':
        pos=8
    if attr=='day':
        pos=9
    if attr=='month':
        pos=10
    if attr=='duration':
        pos=11
    if attr=='campaign':
        pos=12
    if attr=='pdays':
        pos=13
    if attr=='previous':
        pos=14
    if attr=='poutcome':
        pos=15
    if attr=='y':
        pos=16
    return pos        
 
 
def create_list(attr):
    obj={}
    for attr_val in bank_attributes[attr]:
        obj[attr_val]=[]
    return obj   

def build_empty_list(attr):
    obj={}
    for attr_val in attr:
        obj[attr_val]=0
    return obj    

    
def information_gain(groups, classes):
    Q = 0.0   
    tp =0.0
    for attr_val in groups:
        tp = sum([row[-1] for row in groups[attr_val]])
        Q = Q + tp        
    exp_ent = 0.0
    for attr_val in groups:
        size = float(len(groups[attr_val]))
        if size == 0:
            continue    
        score = 0
        q = sum([row[-1] for row in groups[attr_val]])
        for class_val in classes:          
            p = sum([row[-1] for row in groups[attr_val] if row[-2] == class_val])/q  
            if p==0:
                temp=0
            else:
                temp=p*math.log2(1/p)
            score +=temp 
        exp_ent += score* sum([row[-1] for row in groups[attr_val]])/Q 
    return exp_ent          


def data_split(attr, dataset):
    branch_obj=create_list(attr)  
    for row in dataset:
        for attr_val in bank_attributes[attr]:
           if row[pos(attr)] == attr_val:
               branch_obj[attr_val].append(row)
    return branch_obj


def find_best_split(dataset):
    if dataset==[]:
        return 
    label_values = list(set(row[-2] for row in dataset)) 
    metric_obj = build_empty_list(bank_attributes)
    for attr in bank_attributes:
        groups = data_split(attr, dataset)
        metric_obj[attr] = information_gain(groups, label_values)             
    best_attr = min(metric_obj, key=metric_obj.get)
    best_groups = data_split(best_attr, dataset)  
    return {'best_attr':best_attr, 'best_groups':best_groups}


def leaf_node_label(group):
    majority_labels = [row[-2] for row in group]    
    return max(set(majority_labels), key=majority_labels.count)

def if_node_divisible(branch_obj):
    non_empty_indices=[key for key in branch_obj if not (not branch_obj[key])]
    if len(non_empty_indices)==1:
        return False
    else:
        return True

def child_node(node, max_depth, curr_depth):
    if curr_depth >= max_depth:
        for key in node['best_groups']:
            if  node['best_groups'][key]!= []: 
                node[key] = leaf_node_label(node['best_groups'][key])   
            else:
                node[key] = leaf_node_label(sum(node['best_groups'].values(),[]))
        return  
    for key in node['best_groups']:
        if node['best_groups'][key]!= []: 
            node[key] = find_best_split(node['best_groups'][key]) 
            child_node(node[key], max_depth, curr_depth + 1)
        else:
            node[key] = leaf_node_label(sum(node['best_groups'].values(),[]))  

    
def tree_build(train_data, max_depth):
	root = find_best_split(train_data)              
	child_node(root, max_depth, 1)
	return root


def label_predict(node, inst):
    if isinstance(node[inst[pos(node['best_attr'])]],dict):
        return label_predict(node[inst[pos(node['best_attr'])]],inst)
    else:
        return node[inst[pos(node['best_attr'])]]   


def sign_func(val):
    if val > 0:
        return 1.0
    else:
        return -1.0 


def label_return(dataset,tree):
    true_label = []
    pred_seq = []   
    for row in dataset:
        true_label.append(row[-2])    
        pre = label_predict(tree, row)
        pred_seq.append(pre)
    return [true_label, pred_seq]
    

def list_obj(n):
    obj={}
    for i in range(n):
        obj[i] = []
    return obj

def to_binary(llist):
    bin_list =[]
    for i in range(len(llist)):
        if llist[i] == 'yes':
            bin_list.append(1.0)
        else:
            bin_list.append(-1.0)
    return bin_list


def weight_update(curr_wt, vote, bin_true, bin_pred):  
    next_wt=[]  
    for i in range(len(bin_true)):
        next_wt.append(curr_wt[i]*math.e**(- vote*bin_true[i]*bin_pred[i]))
    next_weight = [x/sum(next_wt) for x in next_wt]
    return next_weight


def weight_append(mylist, weights):
    for i in range(len(mylist)):
        mylist[i].append(weights[i]) 
    return mylist 


def weight_update_data(data, weight):
    for i in range(len(data)):
        data[i][-1] = weight[i]
    return data


def fin_dec(indiv_pred, vote, data_len, _T):
    fin_pred = []
    for j in range(data_len):
        score = sum([indiv_pred[i][0][j]*vote[i] for i in range(_T)])
        fin_pred.append(sign_func(score))
    return fin_pred


def wt_error(true_label, predicted, weights):
    count = 0  
    for i in range(len(true_label)):
        if true_label[i] != predicted[i]:
            count += weights[i]
    return count


def _error(_true_lb, _pred_lb): 
    count = 0
    size = len(_true_lb)
    for i in range(size):
        if _true_lb[i] != _pred_lb[i]:
            count += 1
    return count/size 


delta = 1e-8  
def ada_boost(_T, _delta, train_data):
    pred_result = list_obj(_T)   
    vote_say = []
    weights = [row[-1] for row in train_data]
    for i in range(_T):
        tree = tree_build(train_data, 1)    
        print(tree['best_attr'])
        [pp_true, qq_pred] = label_return(train_data, tree)   
        pred_result[i].append(to_binary(qq_pred))
        err = wt_error(pp_true, qq_pred, weights)  
        print(err)   
        print(weights[0])
        vote_say.append( 0.5*math.log((1-err)/err ))   
        weights = weight_update(weights, 0.5*math.log((1-err)/err ), to_binary(pp_true), to_binary(qq_pred))
        train_data = weight_update_data(train_data, weights) 
    return [pred_result, vote_say, weights]


W_1 = np.ones(len(train_bank))/len(train_bank)   
train_bank = weight_append(train_bank, W_1) 
true_label_bin = to_binary([row[-2] for row in train_bank]) 


def iteration_error(T_max):
   ERR =[]
   for t in range(1,T_max):
       [aa_pred, bb_vote, weights] = ada_boost(t, .001, train_bank)
       fin_pred = fin_dec(aa_pred, bb_vote, len(train_bank), t)
       ERR.append(_error(true_label_bin, fin_pred))
   return ERR
 
Err = iteration_error(10)       
        
plot.plot(Err)
plot.ylabel('cost function value')
plot.xlabel('steps')
plot.title('Error Plot')
plot.show()


# W_1 = np.ones(len(train_bank))/len(train_bank)   
# train_bank = weight_append(train_bank, W_1)      
# [aa_pred, bb_vote, weights] = ada_boost(50, delta, train_bank)
# train_bank = weight_update_data(train_bank, weights) 
# tree = tree_build(train_bank, 1)
# [pp, qq] =label_return(train_bank, tree)


# def compare(x,y):
#     count =0
#     for i in range(len(x)):
#         if x[i] != y[i]:
#             count += 1
#     return count
# print(compare(pp,qq))

# print(wt_error(to_binary(pp), to_binary(qq), weights))


# fin_pred = fin_dec(aa_pred, bb_vote, len(train_bank), 50)
# true_label =to_binary([row[-2] for row in train_bank])  
# print(_error(true_label, fin_pred))

        

