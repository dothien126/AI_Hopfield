from random import randint

import numpy as np
from matplotlib import pyplot as plt

from hopfieldnet.net import HopfieldNetwork
from hopfieldnet.trainers import hebbian_training

# Create the training patterns
e_pattern = np.array([[1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1]])

f_pattern = np.array([[1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 1, 1, 1, 1],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0]])



e_pattern *= 2
e_pattern -= 1

f_pattern *= 2
f_pattern -= 1


input_patterns = np.array([e_pattern.flatten(),
                           f_pattern.flatten()])

# Create the neural network and train it using the training patterns
network = HopfieldNetwork(35)

hebbian_training(network, input_patterns)

# Create the test patterns by using the training patterns and adding some noise to them
# and use the neural network to denoise them
e_test = e_pattern.flatten()

for i in range(2):
    p = randint(0, 34)
    e_test[p] *= -1

e_result = network.run(e_test)

e_result.shape = (7, 5)
e_test.shape = (7, 5)

f_test = f_pattern.flatten()

for i in range(2):
    p = randint(0, 34)
    f_test[p] *= -1

f_result = network.run(f_test)

f_result.shape = (7, 5)
f_test.shape = (7, 5)

t_test = t_pattern.flatten()


# Show the results
plt.subplot(4, 2, 1)
plt.imshow(e_test, interpolation="nearest")
plt.subplot(4, 2, 2)
plt.imshow(e_result, interpolation="nearest")

plt.subplot(4, 2, 3)
plt.imshow(f_test, interpolation="nearest")
plt.subplot(4, 2, 4)
plt.imshow(f_result, interpolation="nearest")


plt.show()
