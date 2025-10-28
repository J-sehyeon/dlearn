import sys, os
sys.path.append(os.pardir)
import numpy as np
# 현재 파일의 절대 경로를 파악하여 프로젝트 루트를 sys.path에 추가
# os.path.abspath(__file__) -> 현재 파일의 절대 경로
# os.path.dirname(...) -> 해당 경로의 디렉토리 (즉, 현재 파일이 있는 폴더)
# '..'을 사용하여 상위 폴더로 이동, 여기서는 my_project 폴더를 가리킴
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
# -----------------------------------------

from Chapter03 import network_func as nf
import diff
import loss_func as lf

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)
    
    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, y_true):
        z = self.predict(x)
        y_pred = nf.softmax(z)
        loss = lf.cross_entropy_error(y_pred, y_true)
        return loss

net = simpleNet()

x = np.array([0.6, 0.9])
p = net.predict(x)
print(f"predict: {p}")