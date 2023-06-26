import numpy as np
import time as tm

# Normal Append

init = tm.time()

a = []

for i in range(10000):

    a.append(i)

a = np.array(a)

end = tm.time()

print("Normal Append: ", end - init)

# Numpy Append

init = tm.time()

b = np.empty((1,), dtype=np.int16)
for i in range(10000):
    b = np.append(b, i)

end = tm.time()

print("Numpy Append: ", end - init)
