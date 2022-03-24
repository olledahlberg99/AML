import itertools
olle = []
dal = []
a = range(0,4)
b = range(0,5)
olle.append(range(0,4))
olle.append(range(0,5))
print(olle)
for i in itertools.product(*olle):
    dal.append(i)
    print(olle)
    print(dal)

    