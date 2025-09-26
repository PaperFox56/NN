from saves.circlesGenerator import *
from NoiseRemover import *

## Load the last saved model
network = SimpleForward(layers, size)
network.weights = weights
network.bias = bias

# generate an output from a random image
a = np.random.random(size)
result = network.predicte(a)[-1]
s = 10
result.resize(s, s)
a.resize(s, s)

#view(a, 30)
view(result, 30)
