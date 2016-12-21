from PIL import Image, ImageDraw, ImageFont
from os import listdir, mkdir
import Data
en_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
           'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
           'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
vn_char = ['Ă', 'Â',
           'Đ', 'Ê',
           'Ô', 'Ơ', 'Ư', 'ă', 'â', 'đ', 'ê', 'ô', 'ơ', 'ư']
fontsize = 90
train_fonts = "font/English/train/"
test_fonts = "font/English/test/"
vn_train_fonts = "font/Vietnamese/train/"
vn_test_fonts = "font/Vietnamese/test/"

def create_image(fonts_dir,dir):
    fonts=listdir(fonts_dir)
    for i in en_char:
        for j in fonts:
            image = Image.new("RGB", (128, 128), (255, 255, 255))
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(fonts_dir + j, fontsize)
            w, h = font.getsize(str(i))
            draw.text(((128 - w) / 2, (128 - h) / 2 ), str(i), (0, 0, 0), font=font)
            img_resized = image.resize((Data.input_size[0], Data.input_size[1]), Image.ANTIALIAS)
            if ("Sample" + str(en_char.index(i) + 1).zfill(3)) not in listdir(dir):
                mkdir(dir + "Sample" + str(en_char.index(i) + 1).zfill(3))
            img_resized.save(
                dir + "Sample" + str(en_char.index(i)+1).zfill(3) + "/" + j + ".png",
                "PNG")
def create_image_vn(fonts_dir,dir):
    fonts=listdir(fonts_dir)
    for i in vn_char:
        for j in fonts:
            image = Image.new("RGB", (128, 128), (255, 255, 255))
            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype(fonts_dir + j, fontsize)
            w, h = font.getsize(str(i))
            draw.text(((128 - w) / 2, (128 - h) / 2 ), str(i), (0, 0, 0), font=font)
            img_resized = image.resize((Data.input_size[0], Data.input_size[1]), Image.ANTIALIAS)
            if ("Sample" + str(vn_char.index(i) + 1 + len(en_char)).zfill(3)) not in listdir(dir):
                mkdir(dir + "Sample" + str(vn_char.index(i) + 1 + len(en_char)).zfill(3))
            img_resized.save(
                dir + "Sample" + str(vn_char.index(i) + 1 + len(en_char)).zfill(3) + "/" + j +".png",
                "PNG")
create_image_vn(vn_train_fonts,"Samples/")
create_image_vn(vn_test_fonts,"Tests/")
create_image(train_fonts,"Samples/")
create_image(test_fonts,"Tests/")