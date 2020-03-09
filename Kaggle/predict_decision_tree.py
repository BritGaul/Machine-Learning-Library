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

def bank_pos(attribute):
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


def find_best_split(dataset, attribute):
    if dataset == []:
        return
    if attribute == "bank":
        label_values = list(set(row[-1] for row in dataset))
        metric_obj = build_empty_list(bank_attributes)
        for attr in bank_attributes:
            groups = data_split_bank(attr, dataset)
            metric_obj[attr] = gini_index(groups, label_values)  # change metric here
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


def child_node(node, max_depth, curr_depth, dataset):
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
        if dataset == "car":
            if node['best_groups'][key]:
                node[key] = find_best_split(node['best_groups'][key], dataset)
                child_node(node[key], max_depth, curr_depth + 1, dataset)
            else:
                node[key] = leaf_node_label(sum(node['best_groups'].values(), []))
        if dataset == "bank":
            if node['best_groups'][key]:
                node[key] = find_best_split(node['best_groups'][key], dataset)
                child_node(node[key], max_depth, curr_depth + 1, dataset)
            else:
                node[key] = leaf_node_label(sum(node['best_groups'].values(), []))

            

def tree_build_bank(train, max_depth):
    root = find_best_split(train, "bank")
    child_node(root, max_depth, 1, "bank")
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
    for row in train_data:
        true_label.append(row[-1])
        pre = label_predict_bank(tree, row)
        #print("prediction: ", pre)
        pred_seq.append(pre)
    return error(true_label, pred_seq)


def calc_prediction_error_testing_bank(tree):
    true_label = []
    pred_seq = []
    id = 0
    with open('decisiontree_output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['ID'] + ['Prediction'])
        for row in test_bank:
            id += 1
            true_label.append(row[-1])
            pre = label_predict_bank(tree, row)
            pred_seq.append(pre)
            writer.writerow([str(id)] + [str(pre)])
    return error(true_label, pred_seq)



tree_depth_13 = tree_build_bank(train_data, 13)

print("******Prediction Error in Training Data Set for Bank******")
print(calc_prediction_error_training_bank(tree_depth_13))
print("******Prediction Error in Testing Data Set for Bank******")
print(calc_prediction_error_testing_bank(tree_depth_13))
