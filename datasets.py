import numpy as np
from tensorflow.keras.datasets import cifar10


# CIFAR-10データセットをロード
def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


# データセットを保存する関数
def save(data, labels, filename):
    np.savez(filename, data=data, labels=labels)


# 保存されたデータセットのロード
def load_saved_data(filename):
    data = np.load(filename)
    return data["data"], data["labels"]
