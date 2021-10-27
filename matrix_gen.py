import random
import numpy

DIM = 5000

matrix = numpy.zeros((DIM, DIM))
vector = numpy.zeros((DIM))

for i in range(0, DIM):
    for j in range(0, DIM):
        rand = random.randint(-9, 10)
        if rand < -9:
            rand = 0
        matrix[i][j] = float(rand)

for i in range(0, DIM):
    rand = random.randint(-9, 10)
    if rand < -9:
        rand = 0
    vector[i] = float(rand)
    matrix[i][i] = 1.0

right_side = matrix.dot(vector)

numpy.savetxt('matrix.txt', matrix, fmt='%.1f')
numpy.savetxt('vector.txt', right_side, fmt='%.1f')

print("Solution: ", vector)
print("Right Side: ", right_side)
