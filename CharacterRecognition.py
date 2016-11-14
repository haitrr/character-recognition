import numpy as np
from PIL import Image
import tkinter

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
size = 16, 16


# sigmoid function
def save_weights(syn):
    f = open("weights.txt", 'w')
    f.write("SYN0\n")
    for i in syn[0]:
        for j in i:
            f.write(str(j) + "\n")
    f.write("SYN1\n")
    for i in syn[1]:
        for j in i:
            f.write(str(j) + "\n")
    f.close()


def load_weight():
    f = open("weights.txt", 'r')
    f.readline()
    t = []
    t2 = []
    t3 = []
    for i in range(0,256):
        t = []
        for i in range(0,90):
            t.append(float(f.readline()))
        t2.append(t)
    f.readline()
    for i in range(0, 90):
        t = []
        for i in range(0, 62):
            t.append(float(f.readline()))
        t3.append(t)
    return t2,t3
def sigmoid(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


def get_input_from_samples():
    input_layer = []
    for i in range(1, 63):
        print("Loading sample " + str(i))
        for j in range(1, 21):
            input_layer.append((get_pixels(
                "Samples" + "\Sample" + str(i).zfill(3) + "\img" + str(i).zfill(3) + "-" + str(j).zfill(5) + ".png")))
    return np.array(input_layer)


def get_pixels(image):
    im = Image.open(image)
    im.thumbnail(size)
    rgb_im = im.convert('RGB')
    input_vector = []
    for i in range(0, 16):
        for j in range(0, 16):
            pixel = rgb_im.getpixel((i, j))
            if pixel != WHITE:
                input_vector.append(1)
            else:
                input_vector.append(0)
    return np.array(input_vector)


def get_output_layer():
    output_layer = []
    output = []
    for i in range(1, 63):
        output = []
        for k in range(1, 63):
            if k == i:
                output.append(1)
            else:
                output.append(0)
        for j in range(1, 21):
            output_layer.append(output)
    return np.array(output_layer)


output_layer = get_output_layer()
input_layer = get_input_from_samples()


def train(input, output, times):
    # seed random numbers
    np.random.seed(1)
    # initialize weights randomly with mean 0
    syn0 = 2 * np.random.random((256, 90)) - 1
    syn1 = 2 * np.random.random((90, 62)) - 1

    for j in range(times):
        # forward propagation
        l0 = input_layer
        l1 = sigmoid(np.dot(l0, syn0))
        l2 = sigmoid(np.dot(l1, syn1))

        # how much did we miss?
        l2_error = output_layer - l2
        if (j % 100) == 0:
            print("Error:" + str(np.mean(np.abs(l2_error))))
        l2_delta = l2_error  # * sigmoid(l2, deriv=True)

        l1_error = l2_delta.dot(syn1.T)
        # multiply how much we missed by the
        # slope of the sigmoid at the values in l1
        l1_delta = l1_error  # * sigmoid(l1, True)

        # update weights
        syn1 += l1.T.dot(l2_delta)
        syn0 += l0.T.dot(l1_delta)
    return syn0, syn1


def sample(i, j):
    return "Samples" + "\Sample" + str(i).zfill(3) + "\img" + str(i).zfill(3) + "-" + str(j).zfill(5) + ".png"


def reconize(image, syn0, syn1):
    l0 = get_pixels(image)
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    for i in range(0, 62):
        print(str(i + 1) + "    " + str(l2[i]))
    print(l2.argmax())

def accurate(syn0, syn1):
    count = 0
    l0 = input_layer
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    for t in range(0, 62):
        for l in range(0,20):
            maxs = output_layer[t][l].argmax()
            maxo = l2[t][l].argmax()
            if maxs == maxo:
                count = count + 1
    print("Accuracy rate : " + str(count) + " " + str(count * 100 / 1240))


# print("Output After Training:")
# print(l2)
#syn = train(input_layer, output_layer, 20000)
syn = load_weight()
#save_weights(syn)
accurate(syn[0], syn[1])
