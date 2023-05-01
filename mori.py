from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph.opengl as gl

import numpy as np
import umap

image_features = np.load('features.npz')['features']
embedding = umap.UMAP(n_components=3, metric='minkowski', n_epochs=300, random_state=72).fit_transform(image_features)

app = QtWidgets.QApplication([])
w = gl.GLViewWidget()
w.opts['distance'] = 10
w.show()

g = gl.GLGridItem()
w.addItem(g)

color = np.array([(1*x, 0.2+0.5*x, 0.1+0.3*x, 1) for x in np.linspace(0, 1, embedding.shape[0])])

sp1 = gl.GLScatterPlotItem(pos=embedding, size=np.ones(embedding.shape[0])*0.1, color=color, pxMode=False)
sp1.translate(0,-10,-5)
w.addItem(sp1)

# embedding_new = np.append(embedding, [[0, 0, 1]], axis=0)
# color_new = np.append(color, [[1, 0, 0, 1]], axis=0)
# sp1_new = gl.GLScatterPlotItem(pos=embedding_new, size=np.ones(embedding_new.shape[0])*0.1, color=color_new, pxMode=False)
# sp1_new.translate(-5, -5, -5)
# w.addItem(sp1_new)

embedding_new = np.append(embedding, [[0, 0, 1]], axis=0)
color_new = np.append(color, [[1, 0, 0, 1]], axis=0)
sp1.setData(pos=embedding_new, size=np.ones(embedding_new.shape[0])*0.1, color=color_new)

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()