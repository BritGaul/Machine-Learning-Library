import numpy as np
import statistics
TRAINING_DATA = "./datasets/train_final_no_label_row.csv"
TESTING_DATA = "./datasets/test_final_no_label_row.csv"

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


def constant_feature(data):
    label = [row[-1] for row in data]
    temp = data
    for i in range(len(data)):
        temp[i][-1] = 1.0
    for i in range(len(data)):
        temp[i].append(label[i])
    return temp



test_data = test_bank
train_len = len(train_data)  # NO. of samples
test_len = len(test_data)
dim_s = len(train_data[0]) - 1  # sample dimension = 5 (including constant feature)


def sign_func(x):
    y = 0
    if x > 0:
        y = 1
    else:
        y = -1
    return y


def calculate_error(xx, yy):
    count = 0
    length = len(xx)
    for i in range(length):
        if xx[i] != yy[i]:
            count = count + 1
    return count / length


@np.vectorize
def sigmoid(x):
    if x < -100:
        temp = 0
    else:
        temp = 1 / (1 + np.e ** (-x))
    return temp


@np.vectorize
def lu(x):
    return np.maximum(0.0, x)


@np.vectorize
def phi(x):
    return x * (1.0 - x)


def gamma(t, gamma_0, d):
    return gamma_0 / (1 + (gamma_0 / d) * t)


def get_random_weight(shape):
    mean = 0
    std_dev = 1
    return np.random.normal(mean, std_dev, shape)


def get_zero_weight(shape):
    return np.zeros(shape)


generate_weights = get_zero_weight


class neural_network:
    def __init__(self,
                 input_nodes,
                 output_nodes,
                 hidden_nodes_1,
                 hidden_nodes_2):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_nodes = hidden_nodes_1
        self.hidden_nodes_2 = hidden_nodes_2
        self.initialize_weight_matrix()
        self.stepsize = gamma

    def initialize_weight_matrix(self):
        self.weights_layer_0 = generate_weights([self.hidden_nodes - 1, self.input_nodes])
        self.weights_layer_1 = generate_weights([self.hidden_nodes_2 - 1, self.hidden_nodes])
        self.weights_layer_3 = generate_weights([self.output_nodes, self.hidden_nodes_2])

    def forward_pass(self, input_vec):
        input_vec = np.array(input_vec, ndmin=2).T
        interm_val_vec_1 = np.dot(self.weights_layer_0, input_vec)
        interm_val_vec_1 = sigmoid(interm_val_vec_1)  # z values in layer 1
        interm_val_vec_2 = np.dot(self.weights_layer_1, np.concatenate((interm_val_vec_1, [[1]]), axis=0))
        interm_val_vec_2 = sigmoid(interm_val_vec_2)  # z values in layer 2
        output = np.dot(self.weights_layer_3, np.concatenate((interm_val_vec_2, [[1]]), axis=0))
        return sign_func(output)

    def train(self, input_vector, true_label, iteration, gamma_0, d):
        input_vector = np.array(input_vector, ndmin=2).T
        interm_val_vec_1 = sigmoid(np.dot(self.weights_layer_0, input_vector))
        interm_val_vec_2 = sigmoid(np.dot(self.weights_layer_1, np.concatenate((interm_val_vec_1, [[1]]), axis=0)))
        output = np.dot(self.weights_layer_3, np.concatenate((interm_val_vec_2, [[1]]), axis=0))
        # calculate the partial derivative matrix
        # gradient W_2
        output_error = output - true_label
        grad_w_2 = output_error * (np.concatenate((interm_val_vec_2, [[1]]), axis=0)).T
        # gradient W_1
        hidden_error_vec_2 = output_error * (self.weights_layer_3[0, :][:-1])
        temp = hidden_error_vec_2 * (interm_val_vec_2.T) * (1 - (interm_val_vec_2.T))
        tt = np.concatenate((interm_val_vec_1, [[1]]), axis=0)
        grad_w_1 = np.dot(tt, temp).T  # matrix
        # gradient W_0
        alpha_vec = self.weights_layer_3[0, :][:-1]
        beta_vec = phi(interm_val_vec_2.T)
        ab = alpha_vec * beta_vec
        tpp = np.zeros((self.hidden_nodes - 1, 1))
        for i in range(self.hidden_nodes - 1):
            tpp[i, 0] = output_error * np.inner(ab, self.weights_layer_1[:, i].T) * phi(interm_val_vec_1.T)[0, i]
        grad_w_0 = np.dot(tpp, input_vector.T)
        # update weights
        self.weights_layer_3 = self.weights_layer_3 - gamma(iteration, gamma_0, d) * grad_w_2
        self.weights_layer_1 = self.weights_layer_1 - gamma(iteration, gamma_0, d) * grad_w_1
        self.weights_layer_0 = self.weights_layer_0 - gamma(iteration, gamma_0, d) * grad_w_0
        iteration = iteration + 1
        return iteration


width = 10
gamma_0 = 0.02
d = 2
network = neural_network(input_nodes=15,
                         output_nodes=1,
                         hidden_nodes_1=width,
                         hidden_nodes_2=width)

# train NN
count = 1
for i in range(train_len):
    count = network.train(train_data[i][0:dim_s], train_data[i][-1], count, gamma_0, d)
    print(network.weights_layer_0)

# prediction on training data
pred_seq = []
for i in range(train_len):
    true_labels = [row[-1] for row in train_data]
    pred_seq.append(network.forward_pass(train_data[i][0:dim_s]))
print('train error = ', calculate_error(pred_seq, true_labels))

# prediction on test data
pred_seq_test = []
for i in range(test_len):
    true_labels_test = [row[-1] for row in test_data]
    pred_seq_test.append(network.forward_pass(test_data[i][0:dim_s]))
print('test error = ', calculate_error(pred_seq_test, true_labels_test))
print(pred_seq)