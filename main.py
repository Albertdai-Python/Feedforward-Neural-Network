import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
import model

def one_hot(y, num_classes=10):
    return np.eye(num_classes)[y].T

def test():
    outputs = model.forward_propagate(x_test_shuffled)
    predicted_labels = np.argmax(outputs, axis=0)
    true_labels = np.argmax(y_test_shuffled, axis=0)
    matches = (predicted_labels == true_labels)
    num_correct = np.sum(matches)
    total = matches.size
    accuracy = num_correct / total
    # print(f'{num_correct} out of {total} correct, accuracy: {accuracy:.4f}')
    return accuracy

# Data loading, converting to y values to one-hot format
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], -1).T / 255.0
x_test = x_test.reshape(x_test.shape[0], -1).T / 255.0
y_train_one_hot = np.eye(10)[y_train].T
y_test_one_hot = np.eye(10)[y_test].T
num_train = x_train.shape[1]
indices = np.arange(num_train)
num_test = x_test.shape[1]

# Test data shuffling
test_indices = np.arange(num_test)
np.random.shuffle(test_indices)
x_test_shuffled = x_test[:, test_indices]
y_test_shuffled = y_test_one_hot[:, test_indices]

eta = 0.1
model = model.Model(structure=[784, 256, 256, 10], eta=eta)
batch_size = 128
accuracy_vals = []
for epoch in range(3):
    # Shuffling the training data at the start of each epoch
    np.random.shuffle(indices)
    x_train_shuffled = x_train[:, indices]
    y_train_shuffled = y_train_one_hot[:, indices]
    for i in range(0, num_train, batch_size):
        end = min(i + batch_size, num_train)
        batch_x = x_train_shuffled[:, i:end]
        batch_y = y_train_shuffled[:, i:end]
        model.forward_propagate(batch_x)
        model.back_propagate(batch_y)
        accuracy_vals.append(test())
    print(f'Finished epoch {epoch}, accuracy: {accuracy_vals[-1]}.')

plt.plot(accuracy_vals)
plt.xlabel('Batch Number')
plt.ylabel('Accuracy')
plt.title(f'Accuracy Over Time, Learning Rate = {eta}')
plt.show()
print(accuracy_vals[-1])