from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
from CharacterRecognition import CharacterReconition


class App(object):
    def __init__(self):
        self.root = Tk()
        self.style = ttk.Style()
        self.style.theme_use('winnative')
        self.root.title("Character Reconition")

        # frame
        frm = Frame(self.root)
        frm.pack(expand=True, fill='both')
        self.file_name = ""

        # image
        self.image = None
        self.image_panel = Label(self.root, image=self.image)
        self.image_panel.pack(side="bottom", fill="both", expand="yes")

        # open image button
        self.open_image_button = Button(self.root, text="Choose image", command=self.open_image_button_click)
        self.open_image_button.pack(side="top", fill="both")

        # reconize button
        self.reconize_button = Button(self.root, text="Reconize", command=self.reconize_button_click)
        self.reconize_button.pack(side="left", fill="both")

        # Result lable
        self.resuilt_lable = Label(self.root, text="")
        self.resuilt_lable.pack(side="right", fill="both")

        # Character recognition
        self.character_recognition = CharacterReconition()
        self.character_recognition.load_weight()

    def open_image_button_click(self):
        self.file_name = filedialog.askopenfilename(initialdir= "Samples", #"D:\Download\EnglishFnt\EnglishFnt\English\Fnt",
                                                    title="Choose image",
                                                    filetypes=(("png files", "*.png"), ("all files", "*.*")))
        if self.file_name is not "":
            self.image = ImageTk.PhotoImage(Image.open(self.file_name))
            self.image_panel.config(image=self.image)
            self.image_panel.pack(side="bottom", fill="both", expand="yes")

    def reconize_button_click(self):
        rs = self.character_recognition.reconize(self.file_name)
        self.resuilt_lable.config(text=rs)


app = App()
app.root.mainloop()
