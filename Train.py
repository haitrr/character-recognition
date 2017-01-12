import Data
from NeuralNetwork import *

file = "network16.nn"
sizes = (Data.input_size[0] * Data.input_size[1], 100, Data.output_size)
network = NeuralNetwork(sizes)
training_data = Data.get_training_data()
testing_data = Data.get_testing_data()
network = load(file)
network.train(training_data, 5, 66, 0.0001, 5.0, testing_data, False, True,
              False, True)
network.save(file)
