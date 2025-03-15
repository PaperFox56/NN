from network import *
from ones import ones
from arrayViewer import view


class NoiseRemover(SimpleForward):
    def __init__(self, layers, size):
        SimpleForward.__init__(self, layers, size)

    def activationFunction(self, x):
        return 1 / (1 + np.e**(-x))

    def derivative(self, x):
        return x*(1-x)

def noise(x, percent):
    return x*(1-percent) + np.random.random(x.shape)*percent

if __name__ == "__main__":
    trainingSet = []

    for one in ones:
        for i in range(5):
            p = np.random.random(1)[0] * .6
            trainingSet.append([noise(one, p+.1), noise(one, p)])
    network = trainNetwork(SimpleForward, (3, 36), trainingSet, .05, 50000, [.5, .5])
    a = noise(np.array([
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 1, 1, 0,
        0, 0, 1, 0, 1, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 1, 0
    ]), .2)
    result = network.predicte(a)[-1]
    result.resize(6, 6)
    a.resize(6, 6)
    #print(network.weights)
    network.save("./saves/wo")
    view(a, 50)
    view(result, 50)
