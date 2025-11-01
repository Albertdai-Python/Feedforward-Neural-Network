# Feedforward-Neural-Network
## High Accuracy Feedforward Neural Network with Customizable Layer and Neuron Number
**Programmer:** Yu-Cheng Dai

---
### Project Introduction & Motivation
This project initially stemmed from my [Convolution-Based-Cellular-Automata](https://github.com/Albertdai-Python/Convolution-Based-Cellular-Automata) project, where I initially struggled with finding a kernel and initial pattern to generate stable automata. 

This calculus-based model is built entirely from scratch, without the assistance of modules such as **PyTorch or Tensorflow**. It supports multiple hidden layers, where the layer size can be customized. It also allows different learning rates via in-script adjustments. 

By default, this model consists of 1 input layer, 3 hidden layers, and an output layer. The input and hidden layers are connected through weights, biases, and use $tanh(x)$ as the activation function, whereas the output layer uses a softmax function to determine one-hot data selection. Note that this model also utilizes backpropagation to shift weights and biases according to the training data.

---
### Achieved Results
After failing multiple times due to activation function mismatch and inappropriate learning rate, I found the best learning rate for this model and the MNIST dataset (around 0.001). At 3 epochs, the model has a 94% accuracy, whereas at 10 epochs, a 97% accuracy can be observed.

---
### Future Directions
- Integrate with the cellular automata project
- Introduce probability to inter-neuron connections
- Design a multi-channel neural network that processes multiple sensory inputs (digital brain)
- Build UI for training visualization

---
### Python Module Prerequisites
- keras.datasets (only used for the dataset, not for the neural network structure)
- numpy
- matplotlib

---
### Detailed Functions of Each Script
- `model.py`
  - ***Model & Layer Class Storage and Key Algorithms***
  - Passes data forward through matrix multiplication (dot products and element-wise multiplication)
  - Initiates backpropagation through application of the Chain Rule and the partial derivative of the loss function to each weight and bias
---
- `main.py`
  - ***Training and Testing Initiation***
  - Loads MNIST dataset from keras.datasets
  - Converts expected results to one-hot
  - Split data into batches and record accuracy whilst training

---
### References
[Backpropagation, intuitively | Deep Learning Chapter 3](https://www.youtube.com/watch?v=Ilg3gGewQ5U&t=571s)
