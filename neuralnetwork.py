import numpy as np
import warnings
#warnings.filterwarnings("error")
network = []
cost = []
epsilon = 1e-8
v_dw = []
v_db = []
s_dw = []
s_db = []
# hyper parameters
alpha = 0.005  # 50 w/o adam, 1e-2 w/ adam
l2_reg_lambda = 0.  # 0 disables l2 regularization
dropout_keep_prob = 1.0  # 1.0 disables dropout regularization
momentum_beta = 0.9  # 0 disables momentum
momentum_bias_corr = True
rms_prop_beta = 0.99  # 0 disables rms prop
rms_prop_bias_corr = True


def activation(z, activation_func):
    if activation_func == "sigmoid":
        if z.any() < -88.7:
            print(np.max(-z))
        a = 1 / (1 + np.exp(-z))
        return a
    elif activation_func == "relu":
        return np.maximum(0, z)
    elif activation_func == "softmax":
        e_x = np.exp(z - np.max(z, axis=0))
        # axis=0 is the vertical axis
        return e_x / e_x.sum(axis=0)
    else:
        print("unrecognized activation function: " + activation_func)


def activation_prime(z, activation_func):
    if activation_func == "sigmoid":
        sigmoid = activation(z, "sigmoid")
        return sigmoid * (1 - sigmoid)
    elif activation_func == "relu":
        return (z > 0) * 1.0
    elif activation_func == "softmax":
        pass  # I don't know how to implement this
    else:
        print("unrecognized activation function: " + activation_func)


class DenseLayer:
    def __init__(self, num_neurons, num_inputs, activation_func, initialization):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        self.activation_func = activation_func
        if initialization == "rand":
            self.w = np.random.randn(num_neurons, num_inputs) * 0.01
            self.b = np.random.randn(num_neurons, 1) * 0.01
        elif initialization == "kaiming":
            self.w = np.random.randn(num_neurons, num_inputs) * np.sqrt(2 / num_inputs)
            self.b = np.random.randn(num_neurons, 1) * np.sqrt(2 / num_inputs)
        else:
            print("unrecognized initialization method: " + initialization)

    def forward_propagation(self, x):
        z = np.dot(self.w, x) + self.b
        a = activation(z, self.activation_func)
        return z, a


# initialize network
n_inputs = 784
n_neurons = [96, 80, 10]
activation_f = ["relu", "relu", "sigmoid"]
init = ["kaiming", "kaiming", "rand"]
for i in range(len(n_neurons)):
    if i == 0:
        network.append(DenseLayer(n_neurons[i], n_inputs, activation_f[i], init[i]))
    else:
        network.append(DenseLayer(n_neurons[i], n_neurons[i - 1], activation_f[i], init[i]))


def test(x, y):
    for i, layer in enumerate(network):
        if i == 0:
            z, a = layer.forward_propagation(x)
        else:
            z, a = layer.forward_propagation(a)

    answers = np.argmax(a, axis=0)
    results = (answers == y).astype(int)
    correct_times = np.sum(results)
    all_times = x.shape[1]
    print("test result: " + str(correct_times) + " out of " + str(all_times) + " times (" +
          str(correct_times / all_times) + ")")


def learn(x, y, t):
    global v_dw, v_db, s_dw, s_db
    z_cache, a_cache, d_cache = [], [], []
    m = x.shape[1]
    c = 0
    if t == 0:
        # cannot do [[0] * len(network)] * 4 as v_dw = v_db = s_dw = s_db
        v_dw, v_db, s_dw, s_db = [0] * len(network), [0] * len(network), [0] * len(network), [0] * len(network)
    for i, layer in enumerate(network):
        if i == 0:
            z, a = layer.forward_propagation(x)
        else:
            z, a = layer.forward_propagation(a)
        if i == len(network) - 1:
            d = np.ones(a.shape)
        else:
            d = np.random.rand(a.shape[0], a.shape[1])
            d = (d < dropout_keep_prob).astype(int)
            a = a * d
            a = a / dropout_keep_prob
        d_cache.append(d)
        z_cache.append(z)
        a_cache.append(a)
        c += l2_reg_lambda / 2 / m * np.sum(np.square(layer.w))
    global cost
    c += -1 / m * np.sum(y * np.log(a + epsilon) + (1 - y) * np.log(1 - a + epsilon))
    cost.append(c)

    # place x at end of list so that a_cache[-1] will refer to x, x will be removed after gd
    a_cache.append(x)
    dw_cache, db_cache = [], []
    for l in reversed(range(len(network))):
        if l == len(network) - 1:
            da = -1 / m * (np.divide(y, a + epsilon) - np.divide(1 - y,
                                                                 1 - a + epsilon))  # epsilon to prevent division by 0
        da = da * d_cache[l]
        da = da / dropout_keep_prob
        dz = da * activation_prime(z_cache[l], network[l].activation_func)
        dw = 1 / m * np.dot(dz, a_cache[l - 1].T)
        db = 1 / m * np.sum(dz, axis=1, keepdims=True)
        dw = dw + l2_reg_lambda / m * network[l].w
        dw_cache.insert(0, dw)
        db_cache.insert(0, db)
        da = np.dot(network[l].w.T, dz)
    del a_cache[-1]

    for l in range(len(dw_cache)):
        v_dw[l] = momentum_beta * v_dw[l] + (1 - momentum_beta) * dw_cache[l]
        v_db[l] = momentum_beta * v_db[l] + (1 - momentum_beta) * db_cache[l]
        s_dw[l] = rms_prop_beta * s_dw[l] + (1 - rms_prop_beta) * np.square(dw_cache[l])
        s_db[l] = rms_prop_beta * s_db[l] + (1 - rms_prop_beta) * np.square(db_cache[l])
        v_dw_corr, v_db_corr, s_dw_corr, s_db_corr = np.copy(v_dw[l]), np.copy(v_db[l]), np.copy(s_dw[l]),\
                                                     np.copy(s_db[l])
        if momentum_bias_corr:
            v_dw_corr = v_dw_corr / (1 - np.power(momentum_beta, t + 1))
            v_db_corr = v_db_corr / (1 - np.power(momentum_beta, t + 1))
        if rms_prop_bias_corr:
            s_dw_corr = s_dw_corr / (1 - np.power(rms_prop_beta, t + 1))
            s_db_corr = s_db_corr / (1 - np.power(rms_prop_beta, t + 1))
        if rms_prop_beta == 0:
            network[l].w = network[l].w - alpha * v_dw_corr
            network[l].b = network[l].b - alpha * v_db_corr
        else:
            network[l].w = network[l].w - alpha * v_dw_corr / (np.sqrt(s_dw_corr) + epsilon)
            network[l].b = network[l].b - alpha * v_db_corr / (np.sqrt(s_db_corr) + epsilon)
