import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def identity_function(x):
    return x

#1層目
X=np.array([1,0.5])#1層目ニューロン
W1=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])#重み1
B1=np.array([0.1,0.2,0.3])#バイアス1

print("W1=",W1.shape)
print("X=",X.shape)
print("B1=",B1.shape)

A1=np.dot(X,W1)+B1#ニューロンと重み積＋バイアス
Z1=sigmoid(A1)#0~1にする

print("A1=",A1)
print("Z1=",Z1)

#2層目
W2 = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])#重み2
B2 = np.array([0.1,0.2])#バイアス2

print()
print("Z1=",Z1.shape)#Z1が2層目ニューロンになる
print("W2=",W2.shape)
print("B2=",B2.shape)

A2=np.dot(Z1,W2)+B2
Z2=sigmoid(A2)

print("A2=",A2)
print("Z2=",Z2)

#出力層へ
W3 = np.array([[0.1,0.3],[0.2,0.4]])#重み3
B3 = np.array([0.1,0.2])#バイアス3

print()
print("Z2=",Z2.shape)#Z2が出力層ニューロンになる
print("W3=",W3.shape)
print("B3=",B3.shape)

A3 = np.dot(Z2,W3)+B3
Y = identity_function(A3)

print("A3=",A3)
print("Y=",Y)