import numpy as np

# Loss Function - 손실함수 : 신경망의 성능을 수치적으로 나타내는 지표


# Sum of squares of error (SSE) - 오차제곱합

def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# Cross entropy error (CEE) - 교차 엔트로피 오차

def cross_entropy_error(y, t):
    delta = 1e-7        # 발산 방지 변수
    return -np.sum(t * np.log(y + delta))
