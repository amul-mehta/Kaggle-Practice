import Network
import MNIST_Loader
import numpy as np
training_data, validation_data, test_data = MNIST_Loader.load_data_wrapper()

print np.shape(training_data)
print np.shape(validation_data)
print np.shape(test_data)



net = Network.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
