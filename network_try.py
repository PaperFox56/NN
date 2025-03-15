import numpy as np
import sys

class SimpleForward:
    def __init__(self, layers, size):
        self.iterations = 10000
        self.precision = .01
        self.layers = layers
        self.size = size
        self.bias = [None]
        self.weights = [None]
        for i in range(layers-1):
            self.bias.append(np.random.random(size)*10-5)
            self.weights.append(np.random.random((size, size))*10-5)

    def predicte(self, x):
        weightedInputs = [None]
        activations = [x.copy()]
        for i in range(1, self.layers):
            weightedInputs.append(self.weights[i].dot(activations[i-1]) + self.bias[i])
            activations.append(self.activationFunction(weightedInputs[-1]))
        return activations

    def train(self, trainingSet, Lambda):
        prevCost = 1000000
        tempCost = 0
        for i in range(self.iterations):
            np.random.shuffle(trainingSet)
            for set in trainingSet:
                g = []
                output = self.predicte(np.array(set[0]))
                cost = self.cost(output[-1], set[1])
                tempCost += cost
                # Computing each layer error gradient
                g.append(self.derivative(output[-1])*(output[-1] - set[1]))
                #print(output[-1])
                for j in range(-2, -self.layers, -1):
                    d = self.derivative(output[j])
                    g.insert(0, np.transpose(self.weights[j+1]).dot(d*g[-1]))
                self.gradient(g, output, set[1], Lambda)
            tempCost /= len(trainingSet)
            print(i, tempCost)
            if tempCost-prevCost > 1 or tempCost < self.precision:
                return tempCost
                #for i in range(1, self.layers):
                #    self.weights[i] += g[i-1]
            prevCost = tempCost
            tempCost = 0
        return prevCost

    def gradient(self, gradient, outputs, target, Lambda):
        """ for each layer, we compute the gradient
            of each neurone """
        for i in range(1, self.layers):
            #print(self.mul(g, outputs[i-1]))
            self.weights[i] -= Lambda * self.mul(gradient[i-1], outputs[i-1])
            self.bias[i] -= Lambda * gradient[i-1]

    def mul(self, a, b):
        c = np.zeros((len(a), len(a)))
        for i in range(len(a)):
            c[i] = (b*a[i])
        #print(c)
        return c

    def cost(self, output, target):
        return ((target - output)**2).sum()

    def activationFunction(self, x):
        return 1 / (1 + np.e**(-x))
        #return np.tanh(x)
        #return x

    def derivative(self, x):
        return x*(1-x)
        #return 1 - x**2
        #return 1

def trainNetwork(Type, size, trainingSet, Lambda, iterations, precision):
    cost = 1000000
    network = None
    while cost > precision[0]:
        network = Type(size[0], size[1])
        network.iterations = iterations
        network.precision = precision[1]
        cost = network.train(trainingSet, Lambda)
        #print(cost)
    return network

if __name__ == "__main__":
    trainingSet = []
    for i in range(10):
        a = np.random.random((2))
        b = None
        if i % 2 == 0 or np.abs([1] - a[0]*2) < .03:
            a[1] = a[0]*2
            b = [0, 1]
        else:
            b = [1, 0]
        trainingSet.append([a, b])
    network = trainNetwork(SimpleForward, (2, 2), trainingSet, .05, 50000, [.2, .1])
    entry=" "
    print(network.bias)
    print(network.weights)
    while entry != "":
        entry = input("> ").split(" ")
        result = network.predicte(np.array([float(entry[0]), float(entry[1])]))[-1]
        print(result)
