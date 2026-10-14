import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy(probs, y):
    n = y.shape[0]
    return -np.log(probs[np.arange(n), y] + 1e-12).mean()


def accuracy(probs, y):
    return (probs.argmax(axis=1) == y).mean()


class MLP:
    def __init__(self, sizes, seed=42):
        rng = np.random.default_rng(seed)
        self.W = []
        self.b = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            # инициализация He под relu
            self.W.append(rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out)))
            self.b.append(np.zeros(n_out))

    def forward(self, x):
        self.zs = []
        self.acts = [x]
        a = x
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            self.zs.append(z)
            if i < len(self.W) - 1:
                a = relu(z)
            else:
                a = softmax(z)
            self.acts.append(a)
        return a

    def backward(self, y, lr):
        n = y.shape[0]
        # градиент кроссэнтропии по логитам после softmax: p - y_onehot
        delta = self.acts[-1].copy()
        delta[np.arange(n), y] -= 1
        delta /= n

        for i in reversed(range(len(self.W))):
            dW = self.acts[i].T @ delta
            db = delta.sum(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * relu_grad(self.zs[i - 1])
            self.W[i] -= lr * dW
            self.b[i] -= lr * db

    def predict_proba(self, x):
        return self.forward(x)


def train(model, X_train, y_train, X_test, y_test, epochs=60, batch_size=32, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    history = {"train_loss": [], "test_loss": [], "train_acc": [], "test_acc": []}

    for epoch in range(1, epochs + 1):
        idx = rng.permutation(len(X_train))
        for start in range(0, len(idx), batch_size):
            batch = idx[start:start + batch_size]
            model.forward(X_train[batch])
            model.backward(y_train[batch], lr)

        p_train = model.predict_proba(X_train)
        p_test = model.predict_proba(X_test)
        history["train_loss"].append(cross_entropy(p_train, y_train))
        history["test_loss"].append(cross_entropy(p_test, y_test))
        history["train_acc"].append(accuracy(p_train, y_train))
        history["test_acc"].append(accuracy(p_test, y_test))

        if epoch % 5 == 0 or epoch == 1:
            print(f"epoch {epoch:3d} | "
                  f"train loss {history['train_loss'][-1]:.4f}, acc {history['train_acc'][-1]:.4f} | "
                  f"test loss {history['test_loss'][-1]:.4f}, acc {history['test_acc'][-1]:.4f}")

    return history


def plot_history(history, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["test_loss"], label="test")
    ax1.set_title("Loss (cross-entropy)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["test_acc"], label="test")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.grid(alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    path = os.path.join(out_dir, "loss_accuracy.png")
    fig.savefig(path, dpi=120)
    print(f"графики сохранены в {path}")
    plt.show()


def main():
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 64 входа (картинка 8x8), два скрытых слоя, 10 классов
    model = MLP([64, 128, 64, 10])
    history = train(model, X_train, y_train, X_test, y_test)

    print(f"\nитоговая точность на тесте: {history['test_acc'][-1]:.4f}")
    plot_history(history, os.path.join(os.path.dirname(__file__), "images"))


if __name__ == "__main__":
    main()
