from PIL import Image, ImageDraw, ImageFont
from os import listdir, mkdir

char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N',
        'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
        'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'Ă', 'Â', 'Đ', 'Ê',
        'Ô', 'Ơ', 'Ư', 'ă', 'â', 'đ', 'ê', 'ô', 'ơ', 'ư']
fontsize = 90
fonts = listdir("font/");
for i in char:
    for j in fonts:
        image = Image.new("RGB", (128, 128), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("font/" + j, fontsize)
        w, h = font.getsize(str(i))
        draw.text(((128 - w) / 2, (128 - h) / 2 - 20), str(i), (0, 0, 0), font=font)
        img_resized = image.resize((16, 16), Image.ANTIALIAS)
        if ("Sample" + str(char.index(i) + 1).zfill(3)) not in listdir("Samples/"):
            mkdir("Samples/" + "Sample" + str(char.index(i) + 1).zfill(3))
        img_resized.save(
            "Samples/Sample" + str(char.index(i) + 1).zfill(3) + "/img" + str(char.index(i) + 1).zfill(
                3) + "-" + str(
                fonts.index(j) + 1).zfill(5) + ".png",
            "PNG")