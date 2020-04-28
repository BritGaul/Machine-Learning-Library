import math
import statistics
import numpy as np
import csv

TRAINING_DATA = "./datasets/train_final.csv"
TESTING_DATA = "./datasets/test_final.csv"

with open(TRAINING_DATA, mode='r') as f:
    train_data = []
    next(f)
    for line in f:
        terms = line.strip().split(',')
        train_data.append(terms)

num_set = {0, 2, 4, 10, 11, 12}

def convert_to_float(mylist):
    temp_list = mylist
    for k in range(len(temp_list)):
        for i in {0, 2, 4, 10, 11, 12}:
            temp_list[k][i] = float(mylist[k][i])
    return temp_list


train_data = convert_to_float(train_data)

obj = {0: 0, 2: 2, 4: 4, 10: 10, 11: 11, 12: 12}
for i in obj:
    obj[i] = statistics.median([row[i] for row in train_data])

for row in train_data:
    for i in obj:
        if row[i] >= obj[i]:
            row[i] = 'yes'
        else:
            row[i] = 'no'

# Replace unknown in training data
major_label = []
for i in range(13):
    majority_labels = [row[i] for row in train_data if row[i] != '?']
    lb = max(set(majority_labels), key=majority_labels.count)
    major_label.append(lb)

for i in range(len(train_data)):
    for j in range(13):
        if train_data[i][j] == '?':
            train_data[i][j] = major_label[j]



with open(TESTING_DATA, mode='r') as f:
    test_bank = []
    next(f)
    for line in f:
        terms = line.strip().split(',')
        terms.pop(0)
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

# Replace unknown
major_label_test = []
for i in range(13):
    majority_labels = [row[i] for row in test_bank if row[i] != '?']
    lb = max(set(majority_labels), key=majority_labels.count)
    major_label_test.append(lb)

for i in range(len(test_bank)):
    for j in range(13):
        if test_bank[i][j] == '?':
            test_bank[i][j] = major_label_test[j]

bank_attributes = {
             'age': ['yes', 'no'],
             'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?'],
             'fnwlwgt': ['yes', 'no'],
             'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th',
                            'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool', '?'],
             'education-num': ['yes', 'no'],
             'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse', '?'],
             'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct',
                            'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?'],
             'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried', '?'],
             'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black', '?'],
             'sex': ['Female', 'Male', '?'],
             'capital-gain': ['yes', 'no'],
             'capital-loss': ['yes', 'no'],
             'hours-per-week': ['yes', 'no'],
             'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan',
                                'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal',
                                'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua',
                                'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?'],
            }

def pos(attribute):
    pos = 0
    if attribute == 'age':
        pos = 0
    if attribute == 'workclass':
        pos = 1
    if attribute == 'fnlwgt':
        pos = 2
    if attribute == 'education':
        pos = 3
    if attribute == 'education-num':
        pos = 4
    if attribute == 'marital-status':
        pos = 5
    if attribute == 'occupation':
        pos = 6
    if attribute == 'relationship':
        pos = 7
    if attribute == 'race':
        pos = 8
    if attribute == 'sex':
        pos = 9
    if attribute == 'capital-gain':
        pos = 10
    if attribute == 'capital-loss':
        pos = 11
    if attribute == 'hours-per-week':
        pos = 12
    if attribute == 'native-country':
        pos = 13
    return pos




def create_list(attr):
    obj = {}
    for attr_val in bank_attributes[attr]:
        obj[attr_val] = []
    return obj


def build_empty_list(attr):
    obj = {}
    for attr_val in attr:
        obj[attr_val] = 0
    return obj


def gini_index(groups, classes):
    n = float(sum([len(groups[attr_val]) for attr_val in groups]))  # attr_val--str type
    gini = 0.0
    for attribute_value in groups:
        size = float(len(groups[attribute_value]))
        if size == 0:
            continue
        score = 0.0
        for value in classes:
            p = [row[-1] for row in groups[attribute_value]].count(value) / size
            score += p * p
        gini += (1.0 - score) * (size / n)
    return gini


def information_gain(groups, classes):
    Q = 0.0
    tp = 0.0
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
            p = sum([row[-1] for row in groups[attr_val] if row[-2] == class_val]) / q
            if p == 0:
                temp = 0
            else:
                temp = p * math.log2(1 / p)
            score += temp
        exp_ent += score * sum([row[-1] for row in groups[attr_val]]) / Q
    return exp_ent

def majority_error(groups, classes):
    n = float(sum([len(groups[attr_val]) for attr_val in groups]))
    majority_error = 0.0
    for attr_val in groups:
        size = float(len(groups[attr_val]))
        if size == 0:
            continue
        score = 0.0
        temp = 0
        for class_val in classes:
            p = [row[-1] for row in groups[attr_val]].count(class_val) / size
            temp = max(temp, p)
            score = 1 - temp
        majority_error += score * (size / n)
    return majority_error

def data_split(attr, dataset):
    branch_obj = create_list(attr)
    for row in dataset:
        for attr_val in bank_attributes[attr]:
            if row[pos(attr)] == attr_val:
                branch_obj[attr_val].append(row)
    return branch_obj


def find_best_split(dataset):
    if dataset == []:
        return
    label_values = list(set(row[-2] for row in dataset))
    metric_obj = build_empty_list(bank_attributes)
    for attr in bank_attributes:
        groups = data_split(attr, dataset)
        metric_obj[attr] = gini_index(groups, label_values)
    best_attr = min(metric_obj, key=metric_obj.get)
    best_groups = data_split(best_attr, dataset)
    return {'best_attr': best_attr, 'best_groups': best_groups}


def leaf_node_label(group):
    majority_labels = [row[-2] for row in group]
    return max(set(majority_labels), key=majority_labels.count)


def if_node_divisible(branch_obj):
    non_empty_indices = [key for key in branch_obj if not (not branch_obj[key])]
    if len(non_empty_indices) == 1:
        return False
    else:
        return True


def child_node(node, max_depth, curr_depth):
    if curr_depth >= max_depth:
        for key in node['best_groups']:
            if node['best_groups'][key] != []:
                node[key] = leaf_node_label(node['best_groups'][key])
            else:
                node[key] = leaf_node_label(sum(node['best_groups'].values(), []))
        return
    for key in node['best_groups']:
        if node['best_groups'][key] != []:
            node[key] = find_best_split(node['best_groups'][key])
            child_node(node[key], max_depth, curr_depth + 1)
        else:
            node[key] = leaf_node_label(sum(node['best_groups'].values(), []))


def tree_build(train_data, max_depth):
    root = find_best_split(train_data)
    child_node(root, max_depth, 1)
    return root


def label_predict(node, inst):
    if isinstance(node[inst[pos(node['best_attr'])]], dict):
        return label_predict(node[inst[pos(node['best_attr'])]], inst)
    else:
        return node[inst[pos(node['best_attr'])]]


def sign_func(val):
    if val > 0:
        return 1.0
    else:
        return -1.0


def label_return(dataset, tree):
    true_label = []
    pred_seq = []
    for row in dataset:
        true_label.append(row[-2])
        pre = label_predict(tree, row)
        pred_seq.append(pre)
    return [true_label, pred_seq]


def list_obj(n):
    obj = {}
    for i in range(n):
        obj[i] = []
    return obj


def to_binary(llist):
    bin_list = []
    for i in range(len(llist)):
        if llist[i] == 'yes':
            bin_list.append(1.0)
        else:
            bin_list.append(-1.0)
    return bin_list


def weight_update(curr_wt, vote, bin_true, bin_pred):
    next_wt = []
    for i in range(len(bin_true)):
        next_wt.append(curr_wt[i] * math.e ** (- vote * bin_true[i] * bin_pred[i]))
    next_weight = [x / sum(next_wt) for x in next_wt]
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
        score = sum([indiv_pred[i][0][j] * vote[i] for i in range(_T)])
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
    return count / size


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
        vote_say.append(0.5 * math.log((1 - err) / err))
        weights = weight_update(weights, 0.5 * math.log((1 - err) / err), to_binary(pp_true), to_binary(qq_pred))
        train_data = weight_update_data(train_data, weights)
    return [pred_result, vote_say, weights]


W_1 = np.ones(len(train_data)) / len(train_data)
train_bank = weight_append(train_data, W_1)
true_label_bin = to_binary([row[-2] for row in train_bank])


def iteration_error(T_max):
    ERR = []
    for t in range(1, T_max):
        [aa_pred, bb_vote, weights] = ada_boost(t, .001, train_bank)
        fin_pred = fin_dec(aa_pred, bb_vote, len(train_bank), t)
        ERR.append(_error(true_label_bin, fin_pred))
    return ERR

def calc_prediction_error_testing_bank(tree):
    true_label = []
    pred_seq = []
    id = 0
    with open('adaboost_output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID'] + ['Prediction'])
        for row in test_bank:
            id += 1
            true_label.append(row[-1])
            pre = label_predict(tree, row)
            pred_seq.append(pre)
            writer.writerow([str(id)] + [str(pre)])
    return iteration_error(10)

def calc_prediction_error_training_bank(tree):
    true_label = []
    pred_seq = []
    for row in train_data:
        true_label.append(row[-1])
        pre = label_predict(tree, row)
        #print("prediction: ", pre)
        pred_seq.append(pre)
    return iteration_error(10)

tree = tree_build(train_data, 13)

print(calc_prediction_error_training_bank(tree))
print(calc_prediction_error_testing_bank(tree))
