# This function generates with a specific kernel. The argument int_effect
# represents the strength of interaction relative to the main effect since all
# sampled functions have been standardized to have unit norm.

import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn.metrics.pairwise
from sklearn.preprocessing import scale

def generate_data(n=100, method="linear", rho=0, snr=2, p=20):
    # define features, 3 groups with 8 features in total
    cov1 = [[1, rho],
            [rho, 1]]
    cov2 = [[1, rho, rho],
            [rho, 1, rho],
            [rho, rho, 1]]
    cov3 = np.eye(p-8)
    X1 = np.random.multivariate_normal(np.zeros(2), cov1, 2*n)
    X2 = np.random.multivariate_normal(np.zeros(3), cov2, 2*n)
    X3 = np.random.multivariate_normal(np.zeros(3), cov2, 2*n)
    X4 = np.random.multivariate_normal(np.zeros(p-8), cov3, 2*n)

    w1 = np.random.normal(size=2*n)
    w2 = w1
    w3 = w1

    # generate y=f(x1, x4, x7) according to given kernel
    x1 = X1[:, 0].reshape((-1, 1))
    x4 = X2[:, 1].reshape((-1, 1))
    x7 = X3[:, 1].reshape((-1, 1))

    if method == "polynomial":
        K1 = sklearn.metrics.pairwise.polynomial_kernel(x1)
        K2 = sklearn.metrics.pairwise.polynomial_kernel(x4)
        K3 = sklearn.metrics.pairwise.polynomial_kernel(x7)
    elif method == "rbf":
        K1 = sklearn.metrics.pairwise.rbf_kernel(x1)
        K2 = sklearn.metrics.pairwise.rbf_kernel(x4)
        K3 = sklearn.metrics.pairwise.rbf_kernel(x7)
    else:
        K1 = sklearn.metrics.pairwise.linear_kernel(x1)
        K2 = sklearn.metrics.pairwise.linear_kernel(x4)
        K3 = sklearn.metrics.pairwise.linear_kernel(x7)

    K1 = K1 / np.trace(K1)
    K2 = K2 / np.trace(K2)
    K3 = K3 / np.trace(K3)

    h0 = np.dot(w1, K1) + np.dot(w2, K2) + np.dot(w3, K3)
    h0 = h0 / np.sqrt(np.sum(h0**2))

    y = h0 + np.random.normal(size=1)
    y = y.reshape((-1, 1))

    noise = np.random.normal(scale=1, size=2*n)
    k = np.sqrt(np.var(y) / (snr*np.var(noise)))
    y = y + k*noise.reshape((-1, 1))

    raw_data = scale(np.c_[y, X1, X2, X3, X4])

    columns = ["y"]
    for i in range(p):
        columns.append("x" + str(i + 1))

    df_total = pd.DataFrame(raw_data, columns=columns)
    df_total = df_total.astype(np.float32)

    df_train = df_total.head(int(.75*n))
    df_test = df_total.tail(int(.25*n))

    return df_train, df_test


def load_data(y_name="y", n=1000, method="linear",
              rho=0, snr=6, p=20):
    train, test = generate_data(n=n, method=method,
                                rho=rho, snr=snr, p=p)

    train_x, train_y = train, train.pop(y_name)

    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

