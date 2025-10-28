import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

# # cmd + / 주석처리

# # shape of data
# print("훈련 이미지 크기:", train_imgs.shape)        # (Number of samples, row, col)
# print("훈련 라벨 크기:", train_labels.shape)
# print("테스트 이미지 크기:", test_imgs.shape)
# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# # visualize images of distinct label
# indices = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# n=0
# for idx, label in enumerate(train_labels):
#     if indices[label] == 0:
#         indices[label] += idx
#         n += 1
#     if n == 11:
#         break
# print(indices)
    
# img = train_imgs[indices]

# fig, axes = plt.subplots(1, 10, figsize = (15, 2))

# for idx in range(10):
#     ax = axes[idx]
#     ax.imshow(img[idx], cmap=plt.cm.binary)

#     ax.set_title(f"{class_names[idx]}", fontsize=10)   

#     ax.axis('off')

# plt.tight_layout()

# plt.show()

# preprocessing

def one_hot_encode(labels, num_classes=10):
    """
    Args:
        labels (N, 1)       [1, 0, 2, 0, ...]
    Returns:
        (N, num_classes)    [[0, 1, 0, ...], [1, 0, ...], ...]
    """
    return np.eye(num_classes)[labels]

y_train = one_hot_encode(train_labels)
print(y_train.shape)
print(y_train[:5], train_labels[:5])