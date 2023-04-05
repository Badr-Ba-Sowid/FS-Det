
import matplotlib.pyplot as plt
import numpy as np


def plot(point_cloud, title=""):
    x = point_cloud[0, :]
    y = point_cloud[1, :]
    z = point_cloud[2, :]
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(projection='3d')
    img = ax.scatter(x, y, z, cmap=plt.hot())
    fig.colorbar(img)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title)   
    plt.tight_layout()
    plt.show()