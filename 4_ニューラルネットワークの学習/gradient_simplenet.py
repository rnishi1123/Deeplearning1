import sys,os
sys.path.append(os.pardir)
import numpy as np
from common.function import softmax,cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)#ガウス分布で初期化

    def predict(self ,x):
        return np.dot(x,self.W)#重みと積を取る
    
    def loss(self,x,t):
        z=self.predict(x)
        y=softmax(z)#出てきた値をsoftmax
        loss = cross_entropy_error(y,t)#損失関数の値を求める

        return loss

net=simpleNet()
print(net.W)

x=np.array([0.6,0.9])
p = net.predict(x)
print(p)

np.argmax(p)

t=np.array([0,0,1])
print(net.loss(x,t))

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)
