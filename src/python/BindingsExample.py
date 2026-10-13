import MyMLPy
print(f".pyd path: {MyMLPy.__file__}")
print(f"MyMLPy dir: {dir(MyMLPy)}")


t1 = MyMLPy.Tensor([1.0, 2.0, 3.0, 4.0], [4])
t2 = MyMLPy.Tensor([1.0, 2.0, 3.0, 4.0], [4])
result = MyMLPy.dot(t1, t2)

print(f"Result: {result}")