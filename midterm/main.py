import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Fashion MNIST dataset
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

try:
    with np.load(parent_dir + "/midterm/fashion_mnist_data.npz") as data:
        train_imgs = data["train_imgs"]
        train_labels = data["train_labels"]
        test_imgs = data["test_imgs"]
        test_labels = data["test_labels"]

except FileNotFoundError:
    print(f"'fashion_mnist_data.npz' 파일을 찾을 수 없습니다. 원본 데이터셋을 로드하고 저장합니다...")
    (train_imgs, train_labels), (test_imgs, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    np.savez_compressed(
        parent_dir + '/midterm/fashion_mnist_data.npz',
        train_imgs = train_imgs, 
        train_labels = train_labels, 
        test_imgs = test_imgs, 
        test_labels = test_labels
        )
    
except Exception as e:
    # 기타 로드 오류 (예: 키 오류, 파일 손상 등) 처리
    print(f"데이터 로드 중 예상치 못한 오류 발생: {e}")

# ===============================

import os
import csv
save_dir = "saved"
if not os.path.exists(parent_dir + "/midterm/" + save_dir):
    os.makedirs(parent_dir + "/midterm/" + save_dir)
    print(f"'{save_dir}' 폴더를 생성했습니다.")

log_path = os.path.join(parent_dir + "/midterm/" + save_dir, "training_log.csv")
model_save_path = os.path.join(parent_dir + "/midterm/" + save_dir, "best_model.pth")
with open(log_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # CSV header
    writer.writerow(["Epoch", "Train Loss", "Test Loss", "Accuracy"])

# ===============================


# 사용할 처리 장치
device = "mps" if torch.mps.is_available() else 'cpu'
print(f'using {device} device')

# Preprocessing
x_train = torch.from_numpy(train_imgs).float() / 255.0      # 데이터값의 분포를 0과 1로 맞춘다.
y_train = torch.from_numpy(train_labels).long()

x_test = torch.from_numpy(test_imgs).float() / 255.0
y_test = torch.from_numpy(test_labels).long()

train_dataset = TensorDataset(x_train, y_train)
test_dataset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Model
class FashionDNN(nn.Module):
    def __init__(self):
        super(FashionDNN, self).__init__()
        self.flatten = nn.Flatten()     # 28x28 이미지 평탄화
        self.network = nn.Sequential(
            nn.Linear(28*28, 256),      # 완전연결계층
            nn.ReLU(),                  # 활성함수
            nn.Dropout(0.2),            # 과적합 방지를 위해 0.2 랜덤값으로 노드간 연결 해제
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )
    
    def forward(self, x):
        x = self.flatten(x)     
        out = self.network(x)
        return out
    
    ## Autograd
    # 역전파는 완전연결계층과 ReLU, Dropout 각각에 적용시키는 함수만 있으면 된다.
    # 완전연결계층과 ReLU는 수업시간에 배운 내용, Dropout의 경우 순전파시에 끊어진 노드를 역전파에도 똑같이 끊고 
    # 나머지는 그대로 흘려주면 된다.

    
model = FashionDNN().to(device)     # 모델을 GPU공간으로 이동


# trainner
loss = nn.CrossEntropyLoss()        # 소프트맥스 포함
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_loop(dataloader, model, loss, optimizer):
    model.train()   # 학습모드
    
    pbar = tqdm(dataloader, desc="Training")
    running_loss = 0.0

    for x, y in pbar:
        x, y = x.to(device), y.to(device)   # GPU공간의 데이터를 매 배치마다 갱신

        # forward
        pred = model(x)
        l = loss(pred, y)

        # backward
        optimizer.zero_grad()
        l.backward()
        optimizer.step() 

        running_loss += l.item()

        pbar.set_postfix({'loss': f'{l.item():.4f}'})
    
    return running_loss / len(dataloader)


def test_loop(dataloader, model, loss):
    model.eval()    # 평가 모드
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():       # 역전파를 위한 캐싱작업 제거, 메모리 효율
        pbar = tqdm(dataloader, desc="Testing")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            pred = model(x)

            test_loss += loss(pred, y).item()   # cpu로 이동, type=float
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            # 예측 확률의 최대값과 정답 비교 후 불값을 실수값으로 치환, 전부 더한 후 숫자만 남긴다.
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct

# train
EPOCHS = 13
best_acc = 0.0

for t in range(EPOCHS):
    print(f'Epoch {t+1}/{EPOCHS}')
    train_loss = train_loop(train_loader, model, loss, optimizer)
    val_loss, val_acc = test_loop(test_loader, model, loss)

    with open(log_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([t+1, train_loss, val_loss, val_acc])
    
    if val_acc > best_acc:
        best_acc = val_acc
        
        torch.save(model.state_dict(), model_save_path)
    else:
        print("")

print("finish")

def predict_numpy_image(index):
    model.eval()
    # NumPy 배열 하나를 가져옴
    image_np = test_imgs[index]
    label_np = test_labels[index]
    
    # 모델에 넣기 위해 전처리 (Tensor변환 -> 차원추가 -> device이동)
    # 이미지: (28, 28) -> (1, 28, 28)
    image_tensor = torch.from_numpy(image_np).float().div(255.0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(image_tensor)
        predicted_idx = pred.argmax(1).item()
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    plt.imshow(image_np, cmap='gray')
    plt.title(f"Actual: {class_names[label_np]}, Predicted: {class_names[predicted_idx]}")
    plt.axis('off')
    plt.show()

predict_numpy_image(0)
