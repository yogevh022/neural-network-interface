import signal
import numpy as np
from collections import defaultdict
import random
import json
import time


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1/(1 + np.exp(-z))

def calc_wsum(f, w, b):
    return np.dot(f, w.T) + b


class Activation:
    def __init__(self, activation_class):
        self._activation_func = activation_class
        self._activation_v = np.vectorize(self._activation_func.activate)
        self._derive_v = np.vectorize(self._activation_func.derive)

    def activate_layer(self, layer):
        """ layer: numpy matrix """
        return self._activation_v(layer)

    def derive_layer(self, layer):
        """ layer: numpy matrix """
        return self._derive_v(layer)

    def activate(self, wsum):
        return self._activation_func.activate(wsum)

    def derive(self, wsum):
        return self._activation_func.derive(wsum)


class SigmoidFunction:
    @staticmethod
    def activate(wsum):
        return sigmoid(wsum)

    @staticmethod
    def derive(wsum):
        t = sigmoid(wsum)
        return t * (1 - t)


class ReLuFunction:
    @staticmethod
    def activate(wsum):
        return max(0, wsum)

    @staticmethod
    def derive(wsum):
        return 0 if wsum < 0 else 1


class Cost:
    def __init__(self, error_func) -> None:
        self.err_f = error_func
    
    def error(self, target, output):
        return self.err_f.error(target, output)

    def derive(self, target, output):
        return self.err_f.derive(target, output)

    def error_sum(self, target, output):
        sm = 0
        for ti, oi in zip(target, output):
            sm += self.err_f.error(ti, oi)
        return sm

    def derive_sum(self, target, output):
        sm = 0
        for ti, oi in zip(target, output):
            sm += self.err_f.derive(ti, oi)
        return sm


class SquareErrorFunction:
    @staticmethod
    def error(t, o):
        return (t - o) ** 2

    @staticmethod
    def derive(t, o):
        return 2 * (o - t)


class NeuralNetwork:
    _weighted_layers = []
    _biases = []

    def __init__(self, inputs, outputs, hidden_layers, activation, cost, weights=None, biases=None):
        self._input_layer_size = inputs
        self._output_layer_size = outputs
        self._hidden_layers_size = hidden_layers
        self._init_activation(activation)
        self._init_cost(cost)
        if weights is None:
            self._init_neural_network(inputs, outputs, hidden_layers)
            self._init_random_weights()
            self._init_random_biases()
        else:
            self._weighted_layers = weights
            self._biases = biases
        self._init_backprop()

    def __repr__(self) -> str:
        outp_str = ""
        for i in range(len(self._weighted_layers)):
            title_str = f"HL {i}" if i < len(self._weighted_layers)-1 else f"Output:"
            outp_str += f"{title_str}:\n{np.array_repr(self._weighted_layers[i], precision=3)}\n {np.array_repr(self._biases[i], precision=3)}\n"
        return outp_str

    def _init_activation(self, act):
        activations = {
            "sigmoid": SigmoidFunction,
            "relu": ReLuFunction
        }
        if act not in activations.keys():
            raise Exception(f"Activation function '{act}' does not exist")
        self._activation_function = act
        self._activation = Activation(activations[act]())

    def _init_cost(self, cost):
        costs = {
            "sqerr": SquareErrorFunction
        }
        if cost not in costs.keys():
            raise Exception(f"Cost function '{cost}' does not exist")
        self._cost_function = cost
        self._cost = Cost(costs[cost]())

    def _init_neural_network(self, inputs_n, outputs_n, hidden_layers_inp):
        weight_num = inputs_n
        weighted_layers = hidden_layers_inp + [outputs_n]
        for hl_size in weighted_layers:
            hl = np.zeros((hl_size, weight_num))
            weight_num = hl_size
            self._weighted_layers.append(hl)

    def _init_random_weights(self):
        # TODO this needs to be a separate class/function
        # random range is 0.0 - 1.0
        for lay in self._weighted_layers:
            lay += random.uniform(-0.3, 0.3) #np.random.random(lay.shape)

    def _init_random_biases(self):
        self._biases = []
        for lay in self._weighted_layers:
            lb = np.array([0.1 * random.random() for _ in range(len(lay))])
            self._biases.append(lb)

    def _calc_layer(self, inp, weights, bias):
        # generates weighted sum
        wsum = calc_wsum(inp, weights, bias)
        return wsum

    def _backprop(self, layer_index, inp, wsum) -> None:
        self._wsum_derivative_memo[layer_index] += inp
        self._bias_derivative_memo += 1 # this will be broadcasted

    def _init_backprop(self) -> None:
        # clears backprop before calculating a new batch
        self._wsum_derivative_memo = [np.zeros(i.shape) for i in self._weighted_layers]
        self._bias_derivative_memo = 0      # will simply broadcast this when needed
        self._neuron_to_cost_deriv_memo = defaultdict(lambda: 0.0)

    def _update_parameters(self, learnrate):
        for current_layer in range(len(self._weighted_layers)):
            for current_neuron in range(len(self._weighted_layers[current_layer])):
                for current_weight in range(len(self._weighted_layers[current_layer][current_neuron])):
                    self._weighted_layers[current_layer][current_neuron][current_weight] -= learnrate * self._wsum_derivative_memo[current_layer][current_neuron][current_weight] * self._neuron_to_cost_deriv_memo[current_layer][current_neuron]
                self._biases[current_layer][current_neuron] -= learnrate * self._neuron_to_cost_deriv_memo[current_layer][current_neuron]
    
    def _learn(self, batch, learnrate=1):
        """ updates weights and biases according to given batch and learnrate, returns mean error """
        # clear and setup backprop memo dictionaries before predicting a new batch
        self._init_backprop()

        sample_count = 0
        cost = 0.0
        cost_deriv = 0
        for sample_ in batch:
            sample_count += 1       # do this another way
            
            feats = sample_[0]
            label = sample_[1]

            prediction_output = self._learning_prediction(feats, label)     # also sums up derivatives
            cost += self._cost.error_sum(label, prediction_output)
        
        # converting sum to mean
        for k in self._neuron_to_cost_deriv_memo.keys():
            self._neuron_to_cost_deriv_memo[k] /= sample_count
            self._wsum_derivative_memo[k] /= sample_count
        self._update_parameters(learnrate)
        return cost / sample_count

    def _validate_input(self, inputs) -> bool:
        """ raises exception if incorrect number of features was given,
            returns False if array is not a numpy array. """
        if len(inputs) != self._input_layer_size:
            raise Exception("incorrect number of features")
        if not isinstance(inputs, np.ndarray):
            return False

    def _predict(self, inputs, repr_str=False):
        """ used only for predicting and not calculating derivatives for learning. """
        if not self._validate_input(inputs):
            inputs = np.array(inputs)

        # clear and setup backprop memo dictionaries before predicting a new batch
        self._init_backprop()
        
        last_layer_outp = inputs

        for layer_index in range(len(self._weighted_layers)):
            wsum = self._calc_layer(last_layer_outp, self._weighted_layers[layer_index], self._biases[layer_index])
            last_layer_outp = self._activation.activate_layer(wsum)

        if repr_str:
            return np.array_repr(last_layer_outp, precision=5)[6:-1]
        return last_layer_outp

    def _learning_prediction(self, inputs, label):
        """ used for calculating derivatives for learning. """
        if not self._validate_input(inputs):
            inputs = np.array(inputs)
        
        last_layer_outp = inputs
        act_memo = {}
        wsum_memo = {}
        wsum_deriv_memo = {}
        for layer_index in range(len(self._weighted_layers)):
            # calculating weighted sum for current layer
            wsum = self._calc_layer(last_layer_outp, self._weighted_layers[layer_index], self._biases[layer_index])
            # calculating backprop derivatives for current layer
            self._backprop(layer_index, last_layer_outp, wsum)
            # activating and updating current layer outputs
            last_layer_outp = self._activation.activate_layer(wsum)
            # memoizing current layer's activation
            act_memo[layer_index] = last_layer_outp
            wsum_memo[layer_index] = wsum
            wsum_deriv_memo[layer_index] = self._activation.derive_layer(wsum)
        act_keys = list(act_memo.keys())

        # fist calculating the output layer's derivative in respect to the cost function
        k_layer = []
        for neur_i in range(len(act_memo[act_keys[-1]])):
            act_to_cost = self._cost.derive(label[neur_i], act_memo[act_keys[-1]][neur_i])
            # wsum_to_act = self._activation.derive(act_memo[act_keys[-1]][neur_i])
            k_layer.append(act_to_cost)
        self._neuron_to_cost_deriv_memo[act_keys[-1]] += np.array(k_layer)

        # calculating the rest of the chain derivatives
        for k in act_keys[:-1][::-1]:  # from end to start, excluding last (output) layer
            this_layer = act_memo[k]
            k_layer = []
            for neur_i in range(len(this_layer)):
                k_layer_item = 0

                for next_neur_i in range(len(self._neuron_to_cost_deriv_memo[k+1])):
                    act_to_next_wsum = self._weighted_layers[k+1][next_neur_i][neur_i]
                    next_wsum_to_next_act = self._activation.derive(wsum_memo[k+1][next_neur_i])
                    k_layer_item += act_to_next_wsum * next_wsum_to_next_act * self._neuron_to_cost_deriv_memo[k+1][next_neur_i]

                k_layer.append(k_layer_item)

            self._neuron_to_cost_deriv_memo[k] += np.array(k_layer)
        return last_layer_outp

    @staticmethod
    def _split_into_batches(list, batch_size):
        res = []
        for i in range(0, len(list), batch_size):
            res.append(list[i:i+batch_size])
        return res

    def Predict(self, inputs):
        return self._predict(inputs)

    def Train(self, dataset, epochs=1, batchsize=1, learnrate=1.0, print_interval=0):
        batches = self._split_into_batches(dataset, batchsize)
        for e in range(1, epochs+1):
            current_epoch_error = 0
            c = 0
            for b in batches:
                c += 1
                current_epoch_error += self._learn(b, learnrate)
            if print_interval != 0 and e % print_interval == 0:
                print(f"Training epoch: {e}, Gen mean error: {current_epoch_error / c}")
            if interrupted:
                print(f"stopping training at epoch {e}, user interrupt.")
                break


class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def New(inputs, outputs, hidden_layers, activation, cost) -> NeuralNetwork:
    return NeuralNetwork(inputs, outputs, hidden_layers, activation, cost)


def Save(nn: NeuralNetwork, f) -> None:
    """ Saves a -'s dimensions and parameters to a file, can be loaded back using Load(file) """
    # f = open("bruh", "w")
    nn_dict = {}
    nn_dict["inp_size"] = nn._input_layer_size
    nn_dict["outp_size"] = nn._output_layer_size
    nn_dict["hidden_layers"] = nn._hidden_layers_size
    nn_dict["weights"] = nn._weighted_layers
    nn_dict["biases"] = nn._biases
    nn_dict["activation"] = nn._activation_function
    nn_dict["cost"] = nn._cost_function
    json.dump(nn_dict, f, cls=NumpyJsonEncoder)


def Load(f) -> NeuralNetwork:
    """ Loads a previously saved NeuralNetwork ( using Save(file) ) """
    # f = open("bruh", "r")
    nn_dict = json.load(f)
    inputs = nn_dict["inp_size"]
    outputs = nn_dict["outp_size"]
    weights = [np.array(ar) for ar in nn_dict["weights"]]
    biases = [np.array(ar) for ar in nn_dict["biases"]]
    hidden_layers = nn_dict["hidden_layers"]
    activation = nn_dict["activation"]
    cost = nn_dict["cost"]
    return NeuralNetwork(inputs, outputs, hidden_layers, activation, cost, weights=weights, biases=biases)


def signal_handler(signal, frame):
    global interrupted
    interrupted = True

signal.signal(signal.SIGINT, signal_handler)
interrupted = False

if __name__ == "__main__":
    inputs = 2
    outputs = 1
    hidden_layers = [4, 3, 2]
    activation_function = "relu"
    cost_function = "sqerr"
    
    # creating a new neural network
    nn = New(inputs, outputs, hidden_layers, activation_function, cost_function)

    # training the neural network with given dataset
    dataset = [[[1, 1], [1]]]
    nn.Train(dataset, epochs=10000, batchsize=1, learnrate=0.01)

    # getting a prediction using the model
    prediction = nn.Predict([1, 1])

    with open("saved_neural_network.sav", "w") as f:
        # saving the neural network's dimensions, weights, biases, activation function, and cost function into a json file
        Save(nn, f)

        # loading and assembling a neural network from the data in the json file
        nn = Load(f)
