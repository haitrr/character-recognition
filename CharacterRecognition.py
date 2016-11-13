import numpy as np
from PIL import Image

BLACK = (0,0,0)

# sigmoid function
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))
def get_input_from_memory():
    input_layer = []

    return False
def get_pixels(image):
    im = Image.open(image)
    rgb_im = im.convert('RGB')
    input_vector = []
    for i in range(0,128):
        for j in range(0,128):
            pixel = rgb_im.getpixel((i, j))
            if pixel == BLACK:
                input_vector.append(1)
            else:
                input_vector.append(0)
    return input_vector
im = get_pixels("testim.png")
print(im)
# input dataset
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])

# output dataset
Y = np.array([[0, 1, 1, 0]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1

for j in range(60000):
    # forward propagation
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))

    # how much did we miss?
    l2_error = Y - l2
    if (j % 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * sigmoid(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)
    # multiply how much we missed by the
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * sigmoid(l1, True)

    # update weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print("Output After Training:")
print(l1)
print(l2)
