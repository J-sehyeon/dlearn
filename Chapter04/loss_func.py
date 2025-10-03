import tensorflow as tf
import numpy as np


# Loss Function - 손실함수 : 신경망의 성능을 수치적으로 나타내는 지표


# Sum of squares of error (SSE) - 오차제곱합

def sum_squares_error(y_true, y_pred):
    return 0.5 * np.sum((y_true - y_pred) ** 2)


# Cross entropy error (CEE) - 교차 엔트로피 오차

def cross_entropy_error(y_true, y_pred):
    delta = 1e-7        # 발산 방지 변수
    y_pred = np.clip(y_pred, delta, 1. - delta)
    return -np.sum(y_true * np.log(y_pred))


# mini-batch - 미니배치
## 모든 데이터를 학습하기 어려울 때 사용하는 방법, 전체 데이터의 일부만을 골라 학습시켜 신경망의 가중치와 편향의 근사치를 구한다.

def create_dataset_pipeline(X, y, batch_size, is_training=True):
    """
    Tensorflow 데이터셋 파이프라인을 생성하는 표준 함수
    
    Args:
        X: input (feature)
        y: true (label)
        batch_size (int): 배치 크기
        is_training (bool): 훈련용 데이터셋 여부, True라면 셔플링 적용
    Returns:
        tf.data.Dataset: 데이터셋 파이프라인
    """
    # 1. 기본적인 Dataset 객체 생성
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    # 2. 훈련용 데이터셋에만 셔플링 적용
    if is_training:
        buffer_size = X.shape[0]
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
    
    # 3. 데이터를 배치단위로 묶음
    dataset = dataset.batch(batch_size)

    # 4. GPU가 학습하는 동안 CPU가 다음 데이터를 준비하도록 prefetch 적용
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
