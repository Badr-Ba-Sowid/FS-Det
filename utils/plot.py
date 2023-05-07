from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib as cm

from sklearn.manifold import TSNE
import numpy as np

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


def plot_embeddings(embeddings, labels):
    tsne = TSNE(n_components=2, verbose=1)
    embeddings_2d = tsne.fit_transform(embeddings)
    # plot and label them based on the pred labels
    fig, ax = plt.subplots(figsize=(8, 8))
    num_classes = len(np.unique(labels))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))
    for i in range(num_classes):
        plt.scatter(embeddings_2d[labels == i, 0],
                    embeddings_2d[labels == i, 1],
                    color=colors[i], label=str(i),
                    alpha=0.5)
    ax.legend(fontsize='large', markerscale=2)
    plt.savefig("pcd_embedings.png")
