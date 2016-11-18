import numpy as np
from PIL import Image

WHITE = (255, 255, 255)


class CharacterReconition(object):
    def __init__(self):
        self.output_layer = None
        self.input_layer = None
        self.syn = None
        self.input_size = 16, 16
        #size of layer 1
        self.l1_size = 512
        self.output_size = 66
        self.sample_size = 194
        self.char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
                     'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                     'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'Ă', 'Â',
                     'Đ', 'Ê',
                     'Ô', 'Ơ', 'Ư', 'ă', 'â', 'đ', 'ê', 'ô', 'ơ', 'ư']
        return

    # load input and output layer
    def load_input_output(self):
        self.input_layer = self.get_input_from_samples()
        self.output_layer = self.get_output_layer()

    # storage weights to file
    def save_weights(self):
        f = open("weights.txt", 'w')
        f.write("SYN0\n")
        for i in self.syn[0]:
            for j in i:
                f.write(str(j) + "\n")
        f.write("SYN1\n")
        for i in self.syn[1]:
            for j in i:
                f.write(str(j) + "\n")
        f.close()

    # load saved weights
    def load_weight(self):
        f = open("weights.txt", 'r')
        f.readline()
        t2 = []
        t3 = []
        for i in range(0, self.input_size[0] * self.input_size[1]):
            t = []
            for i in range(0, self.l1_size):
                t.append(float(f.readline()))
            t2.append(t)
        f.readline()
        for i in range(0, self.l1_size):
            t = []
            for i in range(0, self.output_size):
                t.append(float(f.readline()))
            t3.append(t)
        self.syn = np.array(t2), np.array(t3)

    # sigmoid function
    def sigmoid(self, x, deriv=False):
        if deriv:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # get input from samples
    def get_input_from_samples(self):
        input_layer = []
        for i in range(1, self.output_size + 1):
            print("Loading sample " + str(i))
            for j in range(1, self.sample_size + 1):
                input_layer.append((self.get_pixels(
                    "Samples" + "\Sample" + str(i).zfill(3) + "\img" + str(i).zfill(3) + "-" + str(j).zfill(
                        5) + ".png")))
        return np.array(input_layer)

    # get pixel from image
    def get_pixels(self, image):
        im = Image.open(image)
        rgb_im = im.convert('RGB')
        input_vector = []
        for i in range(0, self.input_size[1]):
            for j in range(0, self.input_size[0]):
                pixel = rgb_im.getpixel((i, j))
                if pixel != WHITE:
                    input_vector.append(1)
                else:
                    input_vector.append(0)
        return np.array(input_vector)

    # generate input layer
    def get_output_layer(self):
        output_layer = []
        for i in range(1, self.output_size + 1):
            output = []
            for k in range(1, self.output_size + 1):
                if k == i:
                    output.append(1)
                else:
                    output.append(0)
            for j in range(1, self.sample_size + 1):
                output_layer.append(output)
        return np.array(output_layer)

    # train the network
    def train(self, times, resume=False):
        # continue training?
        if resume is True:
            self.load_weight()
            syn0 = self.syn[0]
            syn1 = self.syn[1]
        else:
            # seed random numbers
            np.random.seed(1)
            # initialize weights randomly with mean 0
            syn0 = 2 * np.random.random((self.input_size[0] * self.input_size[1], self.l1_size))-1
            syn1 = 2 * np.random.random((self.l1_size, self.output_size))-1
            self.syn=syn0,syn1

        for j in range(times):
            # forward propagation
            l0 = self.input_layer
            l1 = self.sigmoid(np.dot(l0, syn0))
            l2 = self.sigmoid(np.dot(l1, syn1))

            # how much did we miss?
            l2_error = self.output_layer - l2
            if j % 5 == 0 :
                print(str(j))
                print("Error:" + str(np.mean(np.abs(l2_error))))
                self.accurate()
            l2_delta = l2_error

            l1_error = l2_delta.dot(syn1.T)
            # multiply how much we missed by the
            # slope of the sigmoid at the values in l1
            l1_delta = l1_error

            # update weights
            alpha1 = l1.T.dot(l2_delta)
            alpha0 = l0.T.dot(l1_delta)
            syn1 += alpha1
            syn0 += alpha0
            self.syn = syn0, syn1

    def train2(self, times, resume=False):
        # continue training?
        if resume is True:
            self.load_weight()
            syn0 = self.syn[0]
            syn1 = self.syn[1]
        else:
            # seed random numbers
            np.random.seed(1)
            # initialize weights randomly with mean 0
            syn0 = 2 * np.random.random((self.input_size[0] * self.input_size[1], self.l1_size)) - 1
            syn1 = 2 * np.random.random((self.l1_size, self.output_size)) - 1
            self.syn = syn0, syn1

        for j in range(times):
            # forward propagation
            l0 = self.input_layer
            l1 = self.sigmoid(np.dot(l0, syn0))
            l2 = self.sigmoid(np.dot(l1, syn1))

            # how much did we miss?
            l2_error = self.output_layer - l2
            if j % 5 == 0:
                print(str(j))
                print("Error:" + str(np.mean(np.abs(l2_error))))
                self.accurate()
                # if self.accurate()>30:
                # break
            l2_delta = l2_error  * self.sigmoid(l2, deriv=True)

            l1_error = l2_delta.dot(syn1.T)
            # multiply how much we missed by the
            # slope of the sigmoid at the values in l1
            l1_delta = l1_error  * self.sigmoid(l1, deriv=True)

            # update weights
            alpha1 = l1.T.dot(l2_delta)
            alpha0 = l0.T.dot(l1_delta)
            syn1 += alpha1
            syn0 += alpha0
            self.syn = syn0, syn1
    # get the sample link
    def sample(self, i, j):
        return "Samples" + "\Sample" + str(i).zfill(3) + "\img" + str(i).zfill(3) + "-" + str(j).zfill(5) + ".png"

    # reconize character from a image
    def reconize(self, image):
        l0 = self.get_pixels(image)
        l1 = self.sigmoid(np.dot(l0, self.syn[0]))
        l2 = self.sigmoid(np.dot(l1, self.syn[1]))
        for i in range(0, self.output_size):
            print(str(i + 1) + "    " + str(l2[i]))
        print(str(l2.argmax()+1))
        return self.char[l2.argmax()]

    # test the accuracy of the network
    def accurate(self):
        count = 0
        l0 = self.input_layer
        l1 = self.sigmoid(np.dot(l0, self.syn[0]))
        l2 = self.sigmoid(np.dot(l1, self.syn[1]))
        for t in range(0, self.sample_size * self.output_size):
            maxs = self.output_layer[t].argmax()
            maxo = l2[t].argmax()
            if maxs == maxo:
                count = count + 1
        print("Accuracy rate : " + str(count) + " " + str(count * 100 / (self.sample_size * self.output_size)))
        return count * 100 / (self.sample_size * self.output_size)
#
#
char_rec = CharacterReconition()
char_rec.load_input_output()
for i in range(250):
    char_rec.train2(30, True)
    char_rec.save_weights()
