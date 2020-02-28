import math
import statistics

BANK_TRAINING = "./datasets/bank/train.csv"
BANK_TESTING = "./datasets/bank/test.csv"

with open(BANK_TRAINING, mode='r') as f:
    train_bank = []
    for line in f:
        terms = line.strip().split(',')
        train_bank.append(terms)

num_set = {0, 5, 9, 11, 12, 13, 14}


def convert_to_float(mylist):
    temp_list = mylist
    for k in range(len(temp_list)):
        for i in {0, 5, 9, 11, 12, 13, 14}:
            temp_list[k][i] = float(mylist[k][i])
    return temp_list


train_bank = convert_to_float(train_bank)

obj = {0: 0, 5: 5, 9: 9, 11: 11, 12: 12, 13: 13, 14: 14}
for i in obj:
    obj[i] = statistics.median([row[i] for row in train_bank])

for row in train_bank:
    for i in obj:
        if row[i] >= obj[i]:
            row[i] = 'yes'
        else:
            row[i] = 'no'

# Replace unknown in training data
major_label = []
for i in range(16):
    majority_labels = [row[i] for row in train_bank if row[i] != 'unknown']
    lb = max(set(majority_labels), key=majority_labels.count)
    major_label.append(lb)

for i in range(len(train_bank)):
    for j in range(16):
        if train_bank[i][j] == 'unknown':
            train_bank[i][j] = major_label[j]


# Read test data for bank
with open(BANK_TESTING, mode='r') as f:
    test_bank = []
    for line in f:
        terms = line.strip().split(',')  # 7*N matrix
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
for i in range(16):
    majority_labels = [row[i] for row in test_bank if row[i] != 'unknown']
    lb = max(set(majority_labels), key=majority_labels.count)
    major_label_test.append(lb)

for i in range(len(test_bank)):
    for j in range(16):
        if test_bank[i][j] == 'unknown':
            test_bank[i][j] = major_label_test[j]

bank_attributes = {
             'age': ['yes', 'no'],
             'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student',
                     'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
             'martial': ['married', 'divorced', 'single'],
             'education': ['unknown', 'secondary', 'primary', 'tertiary'],
             'default': ['yes', 'no'],
             'balance': ['yes', 'no'],
             'housing': ['yes', 'no'],
             'loan': ['yes', 'no'],
             'contact': ['unknown', 'telephone', 'cellular'],
             'day': ['yes', 'no'],
                   'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
                   'duration': ['yes', 'no'],
                   'campaign': ['yes', 'no'],
                   'pdays': ['yes', 'no'],
                   'previous': ['yes', 'no'],
                   'poutcome': ['unknown', 'other', 'failure', 'success']
            }


def bank_pos(attribute):
    pos = 0
    if attribute == 'age':
        pos = 0
    if attribute == 'job':
        pos = 1
    if attribute == 'martial':
        pos = 2
    if attribute == 'education':
        pos = 3
    if attribute == 'default':
        pos = 4
    if attribute == 'balance':
        pos = 5
    if attribute == 'housing':
        pos = 6
    if attribute == 'loan':
        pos = 7
    if attribute == 'contact':
        pos = 8
    if attribute == 'day':
        pos = 9
    if attribute == 'month':
        pos = 10
    if attribute == 'duration':
        pos = 11
    if attribute == 'campaign':
        pos = 12
    if attribute == 'pdays':
        pos = 13
    if attribute == 'previous':
        pos = 14
    if attribute == 'poutcome':
        pos = 15
    if attribute == 'y':
        pos = 16
    return pos


# create obj with multipe empty lists
def create_list_bank(attribute):
    obj = {}
    for attr_val in bank_attributes[attribute]:
        obj[attr_val] = []
    return obj


# create obj with multipe zero elements
def build_empty_list(attribute):
    obj = {}
    for attr_val in attribute:
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
    n = float(sum([len(groups[attr_val]) for attr_val in groups]))
    exp_ent = 0.0
    for attribute_value in groups:
        size = float(len(groups[attribute_value]))
        if size == 0:
            continue
        score = 0.0
        for value in classes:
            p = [row[-1] for row in groups[attribute_value]].count(value) / size
            if p == 0:
                temp = 0
            else:
                temp = p * math.log2(1 / p)
            score += temp
        # weight the group score by its relative size
        exp_ent += score * (size / n)
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


def data_split_bank(attributes, dataset):
    branch_obj = create_list_bank(attributes)
    for row in dataset:
        for attr_val in bank_attributes[attributes]:
            if row[bank_pos(attributes)] == attr_val:
                branch_obj[attr_val].append(row)
    return branch_obj


def find_best_split(dataset):
    label_values = list(set(row[-1] for row in dataset))
    metric_obj = build_empty_list(bank_attributes)
    for attr in bank_attributes:
        groups = data_split_bank(attr, dataset)
        metric_obj[attr] = information_gain(groups, label_values)  # change metric here
    best_attr = min(metric_obj, key=metric_obj.get)
    best_groups = data_split_bank(best_attr, dataset)
    return {'best_attr': best_attr, 'best_groups': best_groups}


# returns the majority label within 'group'
def leaf_node_label(group):
    majority_labels = [row[-1] for row in group]
    return max(set(majority_labels), key=majority_labels.count)


# if there is only one non-empty branch, then return False
def if_node_divisible(branch_obj):
    non_empty_indices = [key for key in branch_obj if not (not branch_obj[key])]
    if len(non_empty_indices) == 1:
        return False
    else:
        return True


def child_node(node, max_depth, curr_depth):
    if not if_node_divisible(node['best_groups']):
        for key in node['best_groups']:
            if node['best_groups'][key]:
                node[key] = leaf_node_label(node['best_groups'][key])
            else:
                node[key] = leaf_node_label(sum(node['best_groups'].values(), []))
        return
    if curr_depth >= max_depth:
        for key in node['best_groups']:
            if node['best_groups'][key]:
                node[key] = leaf_node_label(node['best_groups'][key])
            else:
                node[key] = leaf_node_label(sum(node['best_groups'].values(), []))
        return
    for key in node['best_groups']:
        if node['best_groups'][key]:
            node[key] = find_best_split(node['best_groups'][key])
            child_node(node[key], max_depth, curr_depth + 1)
        else:
            node[key] = leaf_node_label(sum(node['best_groups'].values(), []))


def tree_build_bank(train, max_depth):
    root = find_best_split(train)
    child_node(root, max_depth, 1)
    return root


# test if an instance belongs to a node recursively
def label_predict_bank(node, inst):
    if isinstance(node[inst[bank_pos(node['best_attr'])]], dict):
        return label_predict_bank(node[inst[bank_pos(node['best_attr'])]], inst)
    else:
        return node[inst[bank_pos(node['best_attr'])]]  # leaf node


def error(true_label, predicted):
    count = 0
    for i in range(len(true_label)):
        if true_label[i] != predicted[i]:
            count += 1
    return count / float(len(true_label)) * 100.0


def calc_prediction_error_training_bank(tree):
    true_label = []
    pred_seq = []
    for row in train_bank:
        true_label.append(row[-1])
        pre = label_predict_bank(tree, row)
        pred_seq.append(pre)
    return error(true_label, pred_seq)


def calc_prediction_error_testing_bank(tree):
    true_label = []
    pred_seq = []
    for row in test_bank:
        true_label.append(row[-1])
        pre = label_predict_bank(tree, row)
        pred_seq.append(pre)
    return error(true_label, pred_seq)


tree_depth_1 = tree_build_bank(train_bank, 1)
tree_depth_2 = tree_build_bank(train_bank, 2)
tree_depth_3 = tree_build_bank(train_bank, 3)
tree_depth_4 = tree_build_bank(train_bank, 4)
tree_depth_5 = tree_build_bank(train_bank, 5)
tree_depth_6 = tree_build_bank(train_bank, 6)
tree_depth_7 = tree_build_bank(train_bank, 7)
tree_depth_8 = tree_build_bank(train_bank, 8)
tree_depth_9 = tree_build_bank(train_bank, 9)
tree_depth_10 = tree_build_bank(train_bank, 10)
tree_depth_11 = tree_build_bank(train_bank, 11)
tree_depth_12 = tree_build_bank(train_bank, 12)
tree_depth_13 = tree_build_bank(train_bank, 13)
tree_depth_14 = tree_build_bank(train_bank, 14)
tree_depth_15 = tree_build_bank(train_bank, 15)
tree_depth_16 = tree_build_bank(train_bank, 16)
print("******Prediction Error in Training Data Set for Bank******")
print(calc_prediction_error_training_bank(tree_depth_1))
print(calc_prediction_error_training_bank(tree_depth_2))
print(calc_prediction_error_training_bank(tree_depth_3))
print(calc_prediction_error_training_bank(tree_depth_4))
print(calc_prediction_error_training_bank(tree_depth_5))
print(calc_prediction_error_training_bank(tree_depth_6))
print(calc_prediction_error_training_bank(tree_depth_7))
print(calc_prediction_error_training_bank(tree_depth_8))
print(calc_prediction_error_training_bank(tree_depth_9))
print(calc_prediction_error_training_bank(tree_depth_10))
print(calc_prediction_error_training_bank(tree_depth_11))
print(calc_prediction_error_training_bank(tree_depth_12))
print(calc_prediction_error_training_bank(tree_depth_13))
print(calc_prediction_error_training_bank(tree_depth_14))
print(calc_prediction_error_training_bank(tree_depth_15))
print(calc_prediction_error_training_bank(tree_depth_16))
print("******Prediction Error in Testing Data Set for Bank******")
print(calc_prediction_error_testing_bank(tree_depth_1))
print(calc_prediction_error_testing_bank(tree_depth_2))
print(calc_prediction_error_testing_bank(tree_depth_3))
print(calc_prediction_error_testing_bank(tree_depth_4))
print(calc_prediction_error_testing_bank(tree_depth_5))
print(calc_prediction_error_testing_bank(tree_depth_6))
print(calc_prediction_error_testing_bank(tree_depth_7))
print(calc_prediction_error_testing_bank(tree_depth_8))
print(calc_prediction_error_testing_bank(tree_depth_9))
print(calc_prediction_error_testing_bank(tree_depth_10))
print(calc_prediction_error_testing_bank(tree_depth_11))
print(calc_prediction_error_testing_bank(tree_depth_12))
print(calc_prediction_error_testing_bank(tree_depth_13))
print(calc_prediction_error_testing_bank(tree_depth_14))
print(calc_prediction_error_testing_bank(tree_depth_15))
print(calc_prediction_error_testing_bank(tree_depth_16))