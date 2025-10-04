from tqdm import tqdm
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import os


## 데이터셋 다운로드
import torchvision.datasets as datasets
import torchvision.transforms as transforms

train_dataset = datasets.MNIST(
    root="./data",                      # 저장위치
    train=True,                         # 학습용 데이터
    transform=transforms.ToTensor(),    # 형태: 텐서
    download=True                       # 없으면 자동 다운로드
    )                                   
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transforms.ToTensor(),
    download=True
    )

## 이미지 시각화

def show_tensor_images(dataset, n_rows=5, n_cols=10):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 2 * n_rows))
    for i in range(n_rows):
        indices = random.sample(range(len(dataset)), n_cols)
        for j, idx in enumerate(indices):
            img, label = dataset[idx]

            ax = axes[i, j]

            ax.imshow(img.squeeze(0).numpy(), cmap="gray")
            ax.set_title(str(label))
            ax.axis("off")
    
    plt.tight_layout()  
    plt.show()

# show_tensor_images(train_dataset)


## 화이트박스 딥러닝 모델

# Layer 클래스

class Layer:
    def forward(self, x):
        return
    def backward(self, grad_output, lr):
        return

class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        self.W = np.random.rand(in_dim, out_dim) * 0.01
        self.b = np.zeros((1, out_dim))
        self.x = None
        self.dW = None
        self.db = None
    def forward(self, x):
        """
        Args:
            x (ndarray (N, in_dim))
        Returns:
            z (ndarray (N, out_dim)) = (N, in_dim) * (in_dim, out_dim) + (out_dim, )
        """
        self.x = x
        return x.dot(self.W) + self.b
    def backward(self, grad_output, lr=0.01):
        """
        Args:
            grad_output (ndarray (N, out_dim))
            lr (scalar) : learning rate = 0.01
        Returns:
            dx (ndarray (N, in_dim))
        Attributes:
            self.dW (ndarray (in_dim, out_dim))
            self.db (ndarray (out_dim, ))
        """
        self.dW = self.x.T @ grad_output
        self.db = np.sum(grad_output, axis=0, keepdims=True)
        dx = grad_output @ self.W.T
        return dx

class Leaky_ReLU(Layer):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.x = None
    def forward(self, x):
        self.x = x
        return np.where(x>0, x, self.alpha * x)
    def backward(self, grad_output, lr=0.01):
        grad = grad_output.copy()
        grad[self.x <= 0] *= self.alpha
        return grad
class Sigmoid(Layer):
    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    def backward(self, grad_output, lr=0.01):
        return grad_output * self.out * (1 - self.out)
class Tanh(Layer):
    def forward(self, x):
        self.out = np.tanh(x)
        return self.out
    def backward(self, grad_output, lr=0.01):
        return grad_output * (1 - self.out ** 2)
class ReLU(Layer):
    def forward(self, x):
        self.out = np.maximum(0, x)
        return self.out
    def backward(self, grad_output, lr=0.01):
        return grad_output * (self.out > 0)

class MyLoss:
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    def forward(self, x, y_true, delta=1e-9):
        """
        Args:
            x (ndarray (N, out_dim))
            y_true (ndarray (N, Answers)) : one-hot
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.y_pred = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.y_true = y_true    # 왜 여깄음?

        loss = -np.mean(np.sum(y_true * np.log(self.y_pred + delta), axis=1))
        return loss
    def backward(self):
        return (self.y_pred - self.y_true) / self.y_true.shape[0]

class SGDMomentum:
    def __init__(self, params, lr, momentum=0.9):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}    # 각 레이어별 속도
    def step(self):
        for idx, layer in enumerate(self.params):
            if hasattr(layer, "W"):
                if idx not in self.velocities:
                    self.velocities[idx] = {
                        "W": np.zeros_like(layer.W),
                        "b": np.zeros_like(layer.b)
                    }
                vW = self.velocities[idx]["W"]
                vb = self.velocities[idx]["b"]

                vW = self.momentum * vW - self.lr * layer.dW
                vb = self.momentum * vb - self.lr * layer.db
                
                layer.W += vW
                layer.b += vb

                self.velocities[idx]["W"] = vW
                self.velocities[idx]["b"] = vb
                """
                기존:
                seta = seta -  lr * d Loss / d seta
                """

class Sequential:
    def __init__(self, layers, loss_fn, optimizer):
        self.layers = layers
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.log = []
    def forward(self, x, y_true):
        for layer in self.layers:
            x = layer.forward(x)
        loss = self.loss_fn.forward(x, y_true)
        return x, loss
    def backward(self, lr=0.01):
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
    def fit(self, X, y, epoch=10, batch_size=32, lr=0.01, scheduler=None, X_val=None, y_val=None):
        n_samples = X.shape[0]
        for i in range(epoch):
            idx = np.random.permutation(n_samples)
            X, y = X[idx], y[idx]
            batch_losses = []
            with tqdm(range(0, n_samples, batch_size), desc=f"Epoch {i+1}/{epoch}") as pbar:
                for j in pbar:
                    X_batch = X[j : j + batch_size]
                    y_batch = y[j : j + batch_size]
                    logits, loss = self.forward(X_batch, y_batch)
                    self.backward(lr)
                    self.optimizer.step()
                    batch_losses.append(loss)
                iter_per_sec = pbar.n / pbar.format_dict['elapsed'] if pbar.format_dict['elapsed'] > 0 else 0
                avg_loss = np.mean(batch_losses)
                train_acc = self.evaluate(X, y)
                msg = f"Epoch {i+1}/{epoch}, Loss: {avg_loss:.4f}"

                if X_val is not None and y_val is not None:
                    acc = self.evaluate(X_val, y_val)
                    msg += f", Val acc:{acc:.2f}"
                
                self.log.append({
                    "epoch": i+1,
                    "loss": avg_loss,
                    "train_acc": train_acc,
                    "val_acc": acc,
                    "iter_per_sec": iter_per_sec
                })
            
            print(msg)
            if scheduler is not None:
                scheduler.step(i)
    def predict(self, X, batch_size=128):
        preds = []
        for i in range(0, X.shape[0], batch_size):
            x = X[i : i + batch_size]
            for layer in self.layers:
                x = layer.forward(x)
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
            preds.append(np.argmax(probs, axis=1))
        
        return np.concatenate(preds, axis=0)
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        y_true = np.argmax(y, axis=1)       # 정답 레이블이 원-핫 인코딩되어 있으므로, 1의 위치 찾는 용도, axis는 1
        acc = np.mean(y_pred == y_true) * 100
        return acc
    def predict_proba(self, X, batch_size=128):
        probs = []
        for i in range(0, X.shape[0], batch_size):
            x = X[i : i + batch_size]
            for layer in self.layers:
                x = layer.forward(x)
            exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            prob = exp_x / np.sum(exp_x, axis=1, keepdims=True)
            probs.append(prob)
        return np.vstack(probs)
    def train_test_split(X, y, test_size=0.2, shuffle=True, seed=None):
        n_samples = X.shape[0]
        if shuffle:
            if seed is not None:
                np.random.seed(seed)
            indices = np.random.permutation(n_samples)
        else:
            indices = np.arange(n_samples)
        test_size = int(n_samples * test_size)
        test_idx = indices[ : test_size]
        train_idx = indices[test_size : ]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# Scheduler

class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.base_lr = optimizer.lr
    def step(self, epoch):          # 자식 코드에 step이 반드시 오버라이딩 되어야한다.
        raise NotImplementedError
class StepLR(LRScheduler):
    def __init__(self, optimizer, step_size=10, gamma=0.1):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
    def step(self, epoch):
        if epoch % self.step_size == 0 and epoch > 0:
            self.optimizer.lr *= self.gamma

# 데이터 전처리
sample_num = len(train_dataset)
class_num = 10
X = []
y = []
for idx in range(sample_num):
    color, label = train_dataset[idx]
    color = np.array(color[0])
    color = color.reshape(-1)
    X.append(color)
    y.append(np.eye(class_num)[label])
    

X = np.array(X)
y = np.array(y)

# K-Fold Cross Validation

def k_fold_split(n_samples, k=5, shuffle=True, seed=None):
    indices = np.arange(n_samples)
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(indices)
    fold_size = n_samples // k
    folds = []
    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n_samples
        val_idx = indices[val_start : val_end]
        train_idx = np.concatenate([indices[: val_start], indices[val_end :]])
        folds.append((train_idx, val_idx))
    return folds


## 실행

def main_loop(lr, momentum):
    """
    X (N, 784)
    y (N, 10)
    """
    activations = [Sigmoid(), Tanh(), ReLU(), Leaky_ReLU()]
    titles = ["Sigmoid", "Tanh", "ReLU", "Leaky_ReLU"]
    folds = k_fold_split(X.shape[0], k=len(activations), shuffle=True, seed=42)
    fold_models = []
    all_logs = [] # 전체 fold 로그를 저장할 리스트
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"Fold {fold+1} : {titles[fold]}")
        print("Trainsize:", len(train_idx), "Val size:", len(val_idx))
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        layers = [
            Linear(784, 128),
            activations[fold],
            Linear(128, 10)
        ]
        loss_fn = MyLoss()
        optimizer = SGDMomentum(layers, lr, momentum)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        model = Sequential(layers, loss_fn, optimizer)

        model.fit(X_train, y_train, epoch=5, batch_size=32, scheduler=scheduler, X_val=X_val, y_val=y_val)

        fold_models.append(model)
        df_log = pd.DataFrame(model.log)
        df_log["Activation"] = titles[fold]
        df_log["Fold"] = fold + 1
        df_log["learning_rate"] = lr
        df_log["momentum"] = momentum
        all_logs.append(df_log)

        acc = model.evaluate(X_val, y_val)
        print(f"Fold {fold+1} Accuracy: {acc:.2f}%\n")

    final_log = pd.concat(all_logs, ignore_index=True)
    csv_path = "all_folds_results.csv"
    if not os.path.exists(csv_path):
        final_log.to_csv(csv_path, mode="w", header=True, index=False, encoding="utf-8-sig")
        print("첫 저장 (header 포함)")
    else:
        final_log.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8-sig")
        print("이어쓰기 (header 제외)")

interval = 0.05
for i in range(5):
    main_loop(0.04, 0.9 - (i + 1) * interval)


test_num = len(test_dataset)
class_num = 10
X_test = []
y_test = []
for idx in range(test_num):
    color, label = train_dataset[idx]
    color = np.array(color[0])
    color = color.reshape(-1)
    X_test.append(color)
    y_test.append(np.eye(class_num)[label])
X_test = np.array(X_test)
y_test = np.array(y_test)

