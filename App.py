from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image
import Data
import NeuralNetwork
import numpy as np

class App(object):
    def __init__(self,neural_net):
        # Setup window
        self.root = Tk()
        self.style = ttk.Style()
        self.style.theme_use('winnative')
        self.root.title("Character Reconition")
        self.root.grid()
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Init variables
        self.images = []
        self.image_panels = []
        self.result_lables=[]
        self.files = []
        self.entry_per_row = 7

        # open image button
        self.open_image_button = Button(self.root, text="Choose image", command=self.open_image_button_click)
        self.open_image_button.grid(row=0,column=0,columnspan = self.entry_per_row,sticky=N+S+E+W)

        # reconize button
        self.reconize_button = Button(self.root, text="Reconize", command=self.reconize_button_click)
        self.reconize_button.grid(row=1,column=0,columnspan = self.entry_per_row,sticky=N+S+E+W)

        # Character recognition
        self.character_recognition = neural_net


        # Set default layout
        self.files.append("default.png")
        self.set_layout()
    def open_image_button_click(self):
        files = filedialog.askopenfilenames(initialdir= "Samples",
                                                    title="Choose image",
                                                    filetypes=(("png files", "*.png"), ("all files", "*.*")))

        self.files = list(files)
        self.set_layout()
    def set_layout(self):

        # Clean previous entries
        for panel in self.image_panels:
            panel.destroy()
        for lable in self.result_lables:
            lable.destroy()
        self.image_panels.clear()
        self.result_lables.clear()

        # Show new entries
        for file in self.files:
            # Show images
            image = Image.open(file)
            image = image.resize((150, 150), Image.ANTIALIAS)
            image = ImageTk.PhotoImage(image)
            image_panel = Label(self.root, image=image)
            image_panel.image = image
            r = int((self.files.index(file)) / self.entry_per_row) * 2 + 2
            c = (self.files.index(file)) % self.entry_per_row
            image_panel.grid(row=r, column=c)

            # Store panels
            self.image_panels.append(image_panel)

            # Create result lables
            result_lable = Label(self.root, width=10, text="?", font=("Arial", 16))
            result_lable.grid(row=r + 1, column=c)

            # Create result labels
            self.result_lables.append(result_lable)
    def reconize_button_click(self):

        # Get input from files
        ip= [Data.get_pixels(file) for file in self.files]

        # Recognize the images using neural network
        rs = [Data.char[np.argmax(self.character_recognition.feed_forward(i))] for i in ip]

        # Display the results
        for lable,result in zip(self.result_lables,rs):
            lable.config(text=str(result))


file = "network16.nn"
app = App(NeuralNetwork.load(file))
app.root.mainloop()
