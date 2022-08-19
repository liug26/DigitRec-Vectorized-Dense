# Digit Recognition w/ Dense Layers from Scratch

## Summary
The contained python files provide a framework to implement a dense-layer neural network that recognizes hand-written digits. I incorporated gradient descent, l2 regularization, dropout regularization, adam optimization into the framework. All parameters are vectorized, so training is decently fast. My own implementation results in a roughly 97% dev set accuracy network. 

## Structure
main.py -- where the main method is stored and where training/testing is called. 
neuralnetwork.py -- the network module containing the class DenseLayer, responsible for forward and backward propagation of the network
networkio.py -- stores and reads network and training information in files via pickle
training.csv -- the dataset
network.pickle -- the pickle file that can be read through networkio.py, stores the network I trained myself.
cost_map.png -- image generated by matplotlib that maps the movement of cost function value during training

## Data source
The dataset (training.csv after unzipping training.zip) comes from: https://www.kaggle.com/c/digit-recognizer

## Example Network
The network stored in network.pickle has 3 dense layers, each having 96, 80, 10 neurons. There will be 784 inputs in range of [0, 1) inputted into the network, and 10 outputs in range of (0, 1) that represents how confident the network thinks the inputs are an image of each digit. The output with the largest value will be selected as the network's answer, and if that matches the label of the image, then the network gets it correct. 
Kaiming initialization and adam optimization are implemented to speed up training, and l2 regularization and dropout implemented to minimize variance in training and dev set accuracy. The network is trained on a training set with 30,720 samples, a batch size of 512 and for 50 epochs. Training lasted for roughly 55s on my PC, and the following test results are printed:
  training set accuracy:
  test result: 30513 out of 30720 times (0.99326171875)
  dev set accuracy:
  test result: 5519 out of 5640 times (0.9785460992907802)
  test set accuracy:
  test result: 5482 out of 5640 times (0.9719858156028369)

## Contact
Any critique on my code is welcome. My email is liug22@hotmail.com
