def f(a, **batch):
    print(a)
    batch.update({"aboba": 1})


d = {"a": 100, "x": 1}
f(**d)
print(d)
