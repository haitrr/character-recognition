from PIL import Image
import numpy as np

from os import listdir

en_char = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd',
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
    't', 'u', 'v', 'w', 'x', 'y', 'z'
]
vn_char = [
    'Ă', 'Â', 'Đ', 'Ê', 'Ô', 'Ơ', 'Ư', 'ă', 'â', 'đ', 'ê', 'ô', 'ơ', 'ư'
]

char = en_char + vn_char
input_size = (16, 16)

output_size = 66

samples = "Samples/Sample"

test = "Tests/Sample"


# generate the output layer
def load_output(sample_size):
    output_layer = get_output_layer(sample_size)


# get input from samples
def get_input_from_samples(data):
    input_layer = []
    output_layer = []
    for i in range(1, output_size + 1):
        print("Loading sample " + str(i))
        for j in listdir(data + str(i).zfill(3)):
            input_layer.append(get_pixels(data + str(i).zfill(3) + "/" + j))
            temp = np.zeros((output_size, 1))
            temp[i - 1] = 1
            output_layer.append(temp)
    return list(zip(input_layer, output_layer))


def scale(val, src, dst):
    """
    Scale the given value from the scale of src to the scale of dst.
    """
    return ((val - src[0]) / (src[1] - src[0])) * (dst[1] - dst[0]) + dst[0]


# get pixel from image
def get_pixels(image):
    im = Image.open(image)
    input_vector = []
    pixels = list(im.getdata())
    for i in pixels:
        input_vector.append(1 - scale(sum(i), (0, 765), (0, 1)))
    return np.reshape(input_vector, (input_size[0] * input_size[1], 1))


# generate input layer
def get_output_layer(sample_size):
    output_layer = []
    for i in range(0, output_size):
        for j in range(0, sample_size):
            temp = np.zeros((output_size, 1))
            temp[i] = 1
            output_layer.append(temp)
    return output_layer


# get training data
def get_training_data():
    return get_input_from_samples(samples)


# get test data
def get_testing_data():
    return get_input_from_samples(test)
