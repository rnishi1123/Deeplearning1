from layer_naive import *

apple = 100
apple_num = 2
tax = 1.1

#layer
mul_apple_layer = MulLayer() # type: ignore
mul_tax_layer = MulLayer() # type: ignore

apple_price = mul_apple_layer.forward(apple,apple_num)#初めの掛け算
price = mul_tax_layer.forward(apple_price,tax)#2つ目の掛け算

print(price)#出力結果

#backward
dprice = 1
dapple_price,dtax = mul_tax_layer.backward(dprice)#2つ目の掛け算を分解
dapple,dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)

