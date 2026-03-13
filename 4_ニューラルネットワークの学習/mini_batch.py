import sys,os
import numpy as np
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.abspath(os.path.join(current_dir, "../.."))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(base_dir)
from dataset.mnist import load_mnist

(x_train,t_train),(x_test,t_test)=load_mnist( normalize=True,one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

#ミニバッチ処理
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size,batch_size)#train_size内からbatch_size分だけ無作為に選ぶ

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
print(x_batch.shape, t_batch.shape)


print(x_batch)
print(t_batch)