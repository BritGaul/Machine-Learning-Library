import numpy as np

BANK_TRAINING = "./bank-note/train.csv"
BANK_TESTING = "./bank-note/test.csv"

with open(BANK_TRAINING,mode='r') as f:
    bank_train=[]
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
        bank_train.append(terms)
with open(BANK_TESTING,mode='r') as f:
    bank_test = []
    for line in f:
        terms=line.strip().split(',') # 7*N matrix
        bank_test.append(terms)


def str_2_flo(data):
    for row in data:
        for j in range(len(data[0])):
            row[j] = float(row[j])
    return data


def add_cons_feature(data):
    label = [row[-1] for row in data]
    temp = data
    for i in range(len(data)):
        temp[i][-1] = 1.0
    for i in range(len(data)):
        temp[i].append(label[i])
    return temp


def polar_label(data):
    temp = data
    for i in range(len(data)):
        temp[i][-1] = 2 * data[i][-1] - 1
    return temp


train_bank = str_2_flo(bank_train)  # convert to float  types data
test_bank = str_2_flo(bank_test)
train_data = add_cons_feature(polar_label(train_bank))
test_data = add_cons_feature(polar_label(test_bank))
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


width = 100
gamma_0 = 0.02
d = 2
network = neural_network(input_nodes=5,
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