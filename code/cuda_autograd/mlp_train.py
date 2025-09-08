import time
import random
import math
import numpy as np
from basictensor_cuda import BasicTensor
from module import Layer, Module, MLP

np.random.seed(42)
random.seed(42)

def cross_entropy_loss(yhat, y):
    """
    yhat: BasicTensor with shape (batch, num_classes)
    y: numpy array of integer labels with shape (batch,)
    """
    # shift for numerical stability
    shifted = yhat.value - np.max(yhat.value, axis=1, keepdims=True)
    exps = np.exp(shifted)
    probs = exps / np.sum(exps, axis=1, keepdims=True)

    # convert true labels into one-hot
    batch_size, num_classes = probs.shape
    one_hot = np.zeros((batch_size, num_classes), dtype=np.float32)
    one_hot[np.arange(batch_size), y] = 1.0

    # average cross-entropy
    loss_value = -np.sum(one_hot * np.log(probs + 1e-9)) / batch_size

    # wrap in BasicTensor for autograd
    out = BasicTensor(loss_value, _children=(yhat,), _what="cross_entropy")

    def _backward():
        # dL/dyhat = probs - one_hot, averaged
        grad_yhat = (probs - one_hot) / batch_size
        yhat.grad += grad_yhat

    out._backward = _backward
    return out

def train_batch(model, X_train, Y_train, lr=0.01, batch_size=32):
    num_examples = len(X_train)
    total_loss = 0.0
    for start in range(0, num_examples, batch_size):
        end = min(start + batch_size, num_examples)
        X_batch = X_train[start:end]
        Y_batch = Y_train[start:end]
        
        # Wrap batch as BasicTensor (batch_size, features)
        x = BasicTensor(np.stack(X_batch))  # Stack to (batch_size, 784)
        y = np.array(Y_batch)  # Labels as array
        
        model.zero_grad()
        yhat = model(x)  # (batch_size, 10)
        loss = cross_entropy_loss(yhat, y)  # Updated to handle batch
        loss.backward()
        
        # Average gradients (divide by batch size)
        batch_len = len(X_batch)
        for p in model.parameters():
            p.grad /= batch_len
            if np.linalg.norm(p.grad) > 1.0:
                p.grad *= (1.0 / np.linalg.norm(p.grad))
            p.value -= lr * p.grad
        
        total_loss += loss.value * batch_len  # For average loss reporting
    
    return total_loss / num_examples

def evaluate(model, X_test, Y_test, batch_size=128):
    num_examples = len(X_test)
    correct = 0
    total = 0

    for start in range(0, num_examples, batch_size):
        end = min(start + batch_size, num_examples)
        X_batch = X_test[start:end]
        Y_batch = Y_test[start:end]

        # Wrap batch
        x = BasicTensor(np.stack(X_batch))
        y = np.array(Y_batch)

        # Forward pass
        yhat = model(x)  # logits shape: (batch_size, num_classes)

        # Predicted labels
        preds = np.argmax(yhat.value, axis=1)

        correct += np.sum(preds == y)
        total += len(y)

    return correct / total

def load_data(num_examples):
    assert num_examples >= 1, "Use atleast one example"
    # Data generation process
    path = '/home/ndev/ml/datasets/mnist.npz'
    with np.load(path,allow_pickle=True) as mnistData:
      x_train,y_train = mnistData['x_train'],mnistData['y_train']
      x_test,y_test = mnistData['x_test'],mnistData['y_test']

    x_train = x_train[0:num_examples]
    y_train = y_train[0:num_examples]

    x_test = x_test[0:num_examples]
    y_test = y_test[0:num_examples]

    return x_train,y_train,x_test,y_test 

def flatten_data(x_train,x_test) :
    num_train_examples = len(x_train)
    x_train = x_train[0:num_train_examples].reshape(num_train_examples,28*28)/255
    num_test_examples = len(x_test)
    x_test = x_test[0:num_test_examples].reshape(num_test_examples,28*28)/255
    return x_train,x_test

if __name__ == "__main__" :
    model = MLP(28*28,[100, 10])
    x_train_orig,y_train,x_test_orig,y_test = load_data(10000)
    x_train,x_test = flatten_data(x_train_orig,x_test_orig)

    start = time.perf_counter()
    for epoch in range(100):
        loss = train_batch(model, x_train, y_train, lr = 0.05)
        acc = evaluate(model, x_test, y_test)
        print(f"Epoch {epoch}: Loss = {loss : .4f}, Test Accuracy = {acc : .4f}")
    stop = time.perf_counter()
    print(f"Elapsed: {stop - start:.6f} seconds")
