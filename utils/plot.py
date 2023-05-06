from __future__ import annotations
import matplotlib.pyplot as plt


def plot_train_test_data(train_data: list[float], test_data: list[float], label):
    plt.clf()
    print('Saving training/validation losses plot')
    epochs = range(1, len(train_data) + 1)
    plt.plot(epochs, train_data, 'b-', label='Training ' + label)
    plt.plot(epochs, test_data, 'r-', label='Validation ' + label)
    plt.title('Training and validation ' + label)
    plt.xlabel('Epochs')
    plt.ylabel(label)
    plt.legend()
    plt.savefig(label + '_plot')
