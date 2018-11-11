import numpy as np

def norm(a: np.ndarray, base=2, dim=-1):
    a_base = np.power(a, base)
    if dim is not None:
        reduce = np.sum(a_base, axis=dim)
    else:
        reduce = a_base
    a = np.power(reduce, 1/base)
    return a

def l2_dist(expectation: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    assert len(expectation.shape) == len(prediction.shape)
    assert len(expectation.shape) == 2
    batch_size = expectation.shape[0]
    prediction = np.expand_dims(prediction, 1)
    prediction = np.repeat(prediction, batch_size,1)
    expectation = np.expand_dims(expectation, 0)
    return norm(prediction - expectation, base=2, dim=-1)


def sink_horn(expectation: np.ndarray,
              prediction: np.ndarray,
              iter: int=5,
              dist=l2_dist,
              lamb: float = 0.1) -> float:

    assert len(expectation.shape) == len(prediction.shape)
    assert len(expectation.shape) == 2

    k = expectation.shape[1]
    n = expectation.shape[0]
    M = dist(expectation=expectation, prediction=prediction)
    K = np.exp(-lamb*M)
    KM = K * M
    u = np.ones([n,k]) / float(k)
    for _ in range(iter):
        u = prediction / np.dot(K, expectation / np.dot(K, u))
    v = expectation / np.dot(K, u)
    loss = u * np.dot(KM, v)
    return loss.sum() / n

if __name__ == '__main__':
    y = np.array([[1,2,3,7,3,2,1], [1/7]*7])
    y = y / y.sum(axis=-1, keepdims=True)
    x = np.array([1/7]*7)
    x = np.expand_dims(x, 0).repeat(2, axis=0)
    for i in range(1, 51, 5):
        print("Sinkhorn Loss - iter: " + str(i), sink_horn(expectation=y, prediction=x, iter=i, lamb=0.1,
                                                           dist=l2_dist))

    print("Norm2: ", norm(y - x, base=2, dim=-1).mean())

