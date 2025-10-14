import sys
import os

# 현재 파일의 절대 경로를 파악하여 프로젝트 루트를 sys.path에 추가
# os.path.abspath(__file__) -> 현재 파일의 절대 경로
# os.path.dirname(...) -> 해당 경로의 디렉토리 (즉, 현재 파일이 있는 폴더)
# '..'을 사용하여 상위 폴더로 이동, 여기서는 my_project 폴더를 가리킴
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# -----------------------------------------

from Chapter03 import network_func as nf
import numpy as np

# 수치미분 : 해석학적 미분과는 다르게 어느정도의 오차를 포함하는 미분이다. 컴퓨터가 소수를 다루는 방식에 의한 한계를 포함한다.

def numerical_diff(f, x, h=1e-4):
    # (f(x + h) - f(x)) / h
    # 위 식은 오차가 O(h)인 반면 아래는 O(h**2)으로 아래 식의 오차가 더 작다.
    # 2차항의 오차가 사라진다.
    return (f(x + h) - f(x - h)) / (2 * h)


# gradient (기울기) : 수치 미분 대신 사용되는 방법, 해석학적 미분을 구현

def numerical_grad(f, X, h=1e-4):
    """
    Args:
        f : activation function
        X (ndarray (m, n)) : input variabl
    Returns:
        df/dX (ndarray (m, n))
    """

    return (f(X + h) - f(X - h)) / (2*h)

a = np.random.rand(2, 3)
print(nf.sigmoid(a))

