# 关于张量操作
## 运算机制

```py
x = torch.randn([3,3])
y = torch.arange(1,10).reshape([3,3])
z = torch.cat((x,y))
print(z)
z = torch.cat((x,y),dim= 1)
print(z)

z = (x == y)
z = x.sum()
```

torch.arange是生成一个步长默认为1的tensor，左闭右开

torch.cat拼接张量，第一个参数指定拼接对象，先拼的放在前面，dim指定的是轴向

z = x==y 如果x y处处相同，才会返回1，否则返回0

z.sum返回的是一个张量所有元素的和，
## 广播机制
在上面的部分中，我们看到了如何在相同形状的两个张量上执行按元素操作。 在某些情况下，即使形状不同，我们仍然可以通过调用 广播机制（broadcasting mechanism）来执行按元素操作。

## 节省内存
在python中 如果我们使用x = x + y来实现求和赋值给x，这样是浪费内存的，他会丢弃掉原来指向的内存，但不销毁，而再选一块新的内存进行存储

所以我们采用原地操作，使用**切片表示法**
```py
#使用id（）函数可以查到变量的地址
X = Y + X#这种方法最费内存

#以下都是原地操作
X[:] = X + Y 
X += Y
```

## 转换
将深度学习框架定义的张量转换为NumPy张量（ndarray）很容易，反之也同样容易。 torch张量和numpy数组将共享它们的底层内存，就地操作更改一个张量也会同时更改另一个张量。
```py
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
```

在 PyTorch 中，可以使用 .item() 方法从 单元素张量（Scalar Tensor） 中提取 Python 数值（int 或 float）。

针对文件操作，使用with语句更安全
```py
with xxx as yxxx:
    yxxx.do_something()

```
在文件操作的时候经常会open 和 close ，但都需要手动调用，即使使用try 和 finally也是麻烦的，而with语句是上下文管理器，避免内存泄漏，它可以在执行完with语句自动调用xxx的exit方法，避免内存泄漏