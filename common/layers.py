import numpy as np
from common.function import *
from common.util import *

class Relu:
    def __init__(self):
        self.mask=None
    
    def forward(self,x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0

        return out
    
    def backward(self,dout):
        dout[self.mask] = 0
        dx = dout

        return dx
    
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self,x):
        out = 1 /(1+np.exp(-x))
        self.out = out

        return out
    
    def backward(self, dout):
        dx = dout * (1 - self.out) * self.out

        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W#重み
        self.b = b#バイアス
        
        self.x = None
        self.original_x_shape = None
        # 重み・バイアスパラメータの微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # テンソル対応
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b#出力結果

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)#xと重みの転置の積をとる
        self.dW = np.dot(self.x.T, dout)#xの転置と重みの積を取る
        self.db = np.sum(dout, axis=0)#バイアスとの足し算
        
        dx = dx.reshape(*self.original_x_shape)  # 入力データの形状に戻す（テンソル対応）
        return dx
    
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self,x,t):
        self.t = t
        self.y = softmax(x) # type: ignore
        self.loss = cross_entropy_error(self.y,self.t) # pyright: ignore[reportUndefinedVariable]

        return self.loss
    
    def backward(self,dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx
    
class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W #(FN, C, FH, FW)
        self.b = b #バイアス
        self.stride = stride #スライドの数
        self.pad = pad #パンティング　画像の外に空のやつ入れるあれ
        
        # 中間データ（backward時に使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 重み・バイアスパラメータの勾配
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape #フィルター
        N, C, H, W = x.shape #入力データ
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride) #出力サイズの計算
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad) # im2colによる入力データの2次元化
        col_W = self.W.reshape(FN, -1).T #フィルタの2次元化

        out = np.dot(col, col_W) + self.b #行列積による畳み込み演算
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2) #形状を4次元に戻す

        #中間データの保存
        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN) #同じ形状にそろえる

        self.db = np.sum(dout, axis=0) #バイアスの計算
        self.dW = np.dot(self.col.T, dout) #重みを出す計算
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW) #形状を戻す

        dcol = np.dot(dout, self.col_W.T) #元画像に戻す処理
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad) #形状を戻す

        return dx
    
class Pooling:
    def __init__(self,pool_h,pool_w,stride=1,pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self,x):
        N,C,H,W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        #最大値の探索
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        self.x = x
        self.arg_max = arg_max

        return out
    
    def backward(self,dout):
        dout = dout.transpose(0,2,3,1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx