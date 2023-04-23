from PyQt5 import QtWidgets, QtGui
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph.opengl as gl

from transformers import AutoProcessor, AutoTokenizer, CLIPModel
import torch

import numpy as np
import umap

import sys
from PIL import Image
import glob

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")

ART_PATH = 'images/*'
art = []

for filename in glob.iglob(ART_PATH):
  image = Image.open(filename)
  art.append(image)

class GridWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.im = QtGui.QPixmap("topk_grid.jpg")
        self.label = QtWidgets.QLabel()
        self.label.setPixmap(self.im)

        self.grid = QtWidgets.QGridLayout()
        self.grid.addWidget(self.label,1,1)
        self.setLayout(self.grid)

        self.setGeometry(50,50,320,200)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.initVis()
        self.promptView()

    def initVis(self):
        self.image_features = np.load('image_features.npz')['features']
        self.map = umap.UMAP(n_components=3, metric='minkowski', n_epochs=300, random_state=72).fit(self.image_features)
        self.embedding = self.map.embedding_

        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 10
        self.w.show()

        self.g = gl.GLGridItem()
        self.w.addItem(self.g)

        kmeans_group = np.load('kmeans_labels.npz')['features']
        color_pal = np.load('color_palette.npz')['colors']
        self.color = np.array([color_pal[i] for i in kmeans_group])

        self.sp1 = gl.GLScatterPlotItem(pos=self.embedding, size=np.ones(self.embedding.shape[0])*0.1, color=self.color, pxMode=False)
        self.sp1.translate(0,-10,-5)
        self.w.addItem(self.sp1)
    
    def promptView(self):
        self.promptWidget = QtWidgets.QLineEdit()
        self.promptWidget.setPlaceholderText("Enter a description")
        self.promptWidget.returnPressed.connect(self.return_pressed)

        self.resetButton = QtWidgets.QPushButton()
        self.resetButton.setText("reset")
        self.resetButton.clicked.connect(self.resetView)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.promptWidget)
        layout.addWidget(self.resetButton)

        widget = QtWidgets.QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
    
    def resetView(self):
        self.sp1.setData(pos=self.embedding, size=np.ones(self.embedding.shape[0])*0.1, color=self.color)
        self.promptWidget.clear()
    
    def return_pressed(self):
        prompt = self.promptWidget.text()
        print(str(prompt))
        self.updateUMAP(prompt)
    
    def getTextEmbedding(self, prompt):
        inputs = tokenizer([prompt], padding=True, return_tensors="pt").to(device)
        text_features = model.get_text_features(**inputs)
        return text_features
    
    def getTopK(self, text_features):
        k = 10 # set how many images to add to collection
        logit_scale = model.logit_scale.exp()
        similarity = torch.nn.functional.cosine_similarity(text_features, torch.from_numpy(self.image_features)) * logit_scale
        topk = torch.topk(similarity, k)

        cols = 5
        size = (256, 256)
        out_image = Image.new(mode='RGB', size=(size[0]*cols, size[1]*int(k / cols)))
        for i in range(k):
            index = int(topk[1][i].item())
            display_art = art[index]
            display_art.thumbnail(size, resample=3)
            out_image.paste(display_art, ((i%cols)*size[0], i//cols*size[1]))
        out_image.save("topk_grid.jpg")

        return [int(topk[1][i].item()) for i in range(k)]

    def updateUMAP(self, prompt):
        text_features = self.getTextEmbedding(prompt)
        text_UMAP_point = self.map.transform(text_features.detach().numpy())
        topk_indices = self.getTopK(text_features)
        print(topk_indices)

        embedding_new = np.append(self.embedding, text_UMAP_point, axis=0)
        color_new = np.append(self.color, [[1, 0, 0, 1]], axis=0)
        for idx in topk_indices:
            color_new[idx] = [1, 0, 0, 1]
        self.sp1.setData(pos=embedding_new, size=np.ones(embedding_new.shape[0])*0.1, color=color_new)
        self.gridWin = GridWindow()
        self.gridWin.show()

def main():
    app = QtWidgets.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()