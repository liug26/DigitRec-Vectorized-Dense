# Digit Recognition w/ Dense Layers from Scratch

## Summary
The contained python files provide a simple framework to implement and train a dense-layer neural network that recognizes hand-written digits. The framework is written from scratch (no TensorFlow, only NumPy). I incorporated gradient descent, l2 regularization, dropout regularization, adam optimization into the framework. All parameters are vectorized, so training is decently fast. My own implementation results in a roughly 97% dev set accuracy network. 

## Structure
main.py -- where the main method is stored and where training/testing is called. 
neuralnetwork.py -- the network module containing the class DenseLayer, responsible for forward and backward propagation of the network
networkio.py -- stores and reads network and training information in files via pickle
training.csv -- the dataset
network.pickle -- the pickle file that can be read through networkio.py, stores the network I trained myself.
cost_map.png -- image generated by matplotlib that maps the movement of cost function value during training

## Data source
The dataset (training.csv after unzipping training.zip) comes from: https://www.kaggle.com/c/digit-recognizer

## Example Network Output
training set accuracy:
test result: 30513 out of 30720 times (0.99326171875)
dev set accuracy:
test result: 5519 out of 5640 times (0.9785460992907802)
test set accuracy:
test result: 5482 out of 5640 times (0.9719858156028369)
