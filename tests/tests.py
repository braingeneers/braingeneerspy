import numpy as np
import matplotlib.pyplot as plt

test = np.arange(23)
test1 = np.array([[[1, 2, 3, 4], [3, 5, 2, 6]], [[3, 4, 6, 2], [9, 8, 6, 7]]])
test2 = [np.array([11, 21, 31, 41]), np.array([31, 51, 21, 61])]
test3 = [[17, 27, 37, 47], [37, 57, 27, 67], [4, 6, 2, 1]]

print(len(test), len(test1), len(test2))
print(np.hstack(test))
print(np.hstack(test1))
print(np.hstack(test2))
print(np.hstack(test3))

bin_size = 1
rec_length = 10
bin_num = int(rec_length // bin_size) + 1
bins = np.linspace(0, rec_length, bin_num)
fr, b = np.histogram(np.hstack(test1), bins)
print(bins, fr, b)


trains = np.array([3, 4, 5, 6, 7, 2])
if isinstance(trains, (list, np.ndarray)) and not isinstance(trains[0], (list, np.ndarray)):
    N = 1
else:
    N = len(trains)

print(N)
print(isinstance(trains, np.ndarray))
print(type(trains))
print(type(trains[0]))

def addNumbers(self, x, y):
    return x + y

class Mathematics:
    addNumbers = addNumbers


math = Mathematics()
res = math.addNumbers(4, 8)
print(res)

res1 = addNumbers(1, 2)