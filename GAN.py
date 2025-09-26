from network import *
from ones import ones
from images.circles import circles
from arrayViewer import view

import numpy.random as rd

#First attempt at a concurency based neural network
# The generator try to create good images while the discriminer try to tell whether an image is generated or not

class Discriminer(SimpleForward):
    def __init__(self, layers, size):
        SimpleForward.__init__(self, layers, size)

    def predicte(self, x):
        weightedInputs = [None]
        activations = [x.copy()]
        for i in range(1, self.layers):
            weightedInputs.append(self.weights[i].dot(activations[i-1]) + self.bias[i])
            activations.append(self.activationFunction(weightedInputs[-1]))
        return activations

    def train(self, trainingSet, Lambda, i):
        tempCost = 0
        output = None
        g = []
        for set in trainingSet:
            g = []
            output = self.predicte(np.array(set[0]))
            target = output[-1].copy()
            target[int(self.size / 2)] = set[1]
            cost = self.cost(output[-1], target)
            tempCost += cost
            print(output[-1][18])
            
            # Computing each layer's error gradient
            g.append(self.derivative(output[-1])*(output[-1] - target))
            for j in range(-2, -self.layers-1, -1):
                g.insert(0, self.derivative(output[j])*np.transpose(self.weights[j+1]).dot(g[0]))
            self.gradient(g, output, Lambda)
        tempCost /= len(trainingSet)
        print(i, "Discriminer -> ", tempCost)
        return output

class Generator(SimpleForward):
    def __init__(self, layers, size):
        SimpleForward.__init__(self, layers, size)

    def update(self, output, o, Lambda):
        g = []
        # Computing each layer error gradient
        g.append(o)
        for j in range(-2, -self.layers-1, -1):
            g.insert(0, self.derivative(output[j])*np.transpose(self.weights[j+1]).dot(g[0]))
        self.gradient(g, output, Lambda)


def train(data, shape, Lambda):
    size = shape[0] * shape[1]
    prevCost = 10000000
    generator = Generator(4, size)
    discriminer = Discriminer(4, size)
    for i in range(10000):
        generated = generator.predicte(rd.random(size))
        trainingSet = [[data[rd.randint(len(data))], 1], [generated[-1], 0]]
        #trainingSet = [[ones[0], 1], [generated[-1], 0]]
        o = discriminer.train(trainingSet, Lambda, i)
        g = []
        target = o[-1].copy()
        target[int(size/2)] = 1
        g.append(discriminer.derivative(o[-1])*(o[-1] - target))
        for j in range(-2, -discriminer.layers-1, -1):
            g.insert(0, discriminer.derivative(o[j])*np.transpose(discriminer.weights[j+1]).dot(g[0]))
        c = 1-o[-1][int(size/2)]
        generator.update(generated, g[0], Lambda*c)
        print(i, "Generator -> ", c)
        #print(g, discriminer.derivative(o[-1]))
        #if cost-prevCost > 1 or cost < .9:
        #    break
    return generator, discriminer


shape = (10, 10)
size = shape[0] * shape[1]
generator, discriminer = train(circles, shape, .1)
generator.save("./saves/circlesGenerator")
result = generator.predicte(rd.random(size))[-1]
print(discriminer.predicte(result)[-1][int(size/2)])
result.resize(shape)
view(result, 30)
