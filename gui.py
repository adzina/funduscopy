import sys

import numpy as np
import pyqtgraph as pg
from PIL import Image
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog
from pyqtgraph.Qt import QtGui
import cv2

import prepare
import stats

FILENAME = ""
LOADED_PIC_SIZE = 300
RESULT_PIC_SIZE = 300
result_img = ""
PLAY_RATE = 5


class App(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        self.title = 'Fundoscopy'
        self.left = 10
        self.top = 10
        self.width = 640
        self.height = 480

        self.initUI()

    def initUI(self):
        global LOADED_PIC_SIZE, RESULT_PIC_SIZE
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.loadButton = QPushButton('Load image', self)
        self.loadButton.setToolTip('Click here to load the image from files')
        self.loadButton.clicked.connect(self.loadClickAction)

        self.imageView = pg.ImageView(self)
        self.imageView.setToolTip('The loaded image will appear here.')

        param_tree = (
            {'name': 'blur coefficient', 'type': 'float', 'value': 0.1},
            {'name': 'sharp coefficient', 'type': 'bool', 'value': True},
            {'name': 'color histogram normalization', 'type': 'bool', 'value': True}
        )
        self.parameters = pg.parametertree.Parameter.create(name='Settings', type='group', children=param_tree)
        self.param_tree = pg.parametertree.ParameterTree()
        self.param_tree.setParameters(self.parameters, showTop=False)
        self.param_tree.setSizePolicy(QtGui.QSizePolicy(QtGui.QSizePolicy.Ignored, QtGui.QSizePolicy.Preferred))

        self.startButton = QPushButton('Start', self)
        self.startButton.setToolTip('Click here to run the program')
        self.startButton.clicked.connect(self.startClickAction)
        self.startButton.setEnabled(False)

        self.applyButton = QPushButton('Apply', self)
        self.applyButton.setToolTip('Click here to apply preliminary image modifications')
        self.applyButton.clicked.connect(self.applyClickAction)
        self.applyButton.setEnabled(False)

        self.resultView = pg.ImageView(self)
        self.resultView.setToolTip('The result will appear here.')

        self.layout = QtGui.QVBoxLayout(self)
        self.inputLayout = QtGui.QHBoxLayout()

        self.layout.addWidget(self.loadButton)
        self.inputLayout.addWidget(self.param_tree)
        self.inputLayout.addWidget(self.imageView)
        self.layout.addLayout(self.inputLayout)
        self.layout.addWidget(self.applyButton)
        self.layout.addWidget(self.resultView)
        self.layout.addWidget(self.startButton)

        self.show()

    @pyqtSlot()
    def loadClickAction(self):
        global FILENAME
        print('load image')
        FILENAME = QFileDialog.getOpenFileName(filter="Images (*.png *.jpg *.ppm)")[0]
        if FILENAME == '': return
        print(FILENAME)

        self.img = Image.open(FILENAME)
        self.to_disp = np.array(self.img)

        self.imageView.setImage(self.to_disp)

        self.startButton.setEnabled(True)
        self.applyButton.setEnabled(True)

    @pyqtSlot()
    def startClickAction(self):
        global result_img

        tryb = 0

        if tryb==0:
            off_x = 0
            off_y = 0
            size_x = len(self.to_disp)
            size_y = len(self.to_disp[0])
            cut = self.to_disp
            cut = np.array(cut).flatten().reshape((size_x,size_y,3))
            binary = stats.contoursApprox(self.to_disp)

        else:
            off_x = 400
            off_y = 300
            size_x = 150
            size_y = size_x
            a = self.to_disp[off_y:off_y+size_y]
            cut=[]
            for i in a:
                cut.append(i[off_x:off_x+size_x])
            cut = np.array(cut).flatten().reshape((size_x,size_y,3))
            binary = stats.generateBinaryImage(cut)


        self.startButton.setEnabled(False)

        self.showImage(binary, off_x, off_y)
        


    def showImage(self, mask, offset_x, offset_y):
        for i in range(len(mask)):
            for j in range(len(mask[0])):
                if(mask[i][j]==255):
                    self.to_disp[i+offset_x][j+offset_y] = np.array(list([255,255,255]))
        self.resultView.setImage(self.to_disp)

    @pyqtSlot()
    def applyClickAction(self):
        self.applyButton.setEnabled(False)
        blur = self.parameters.child('blur coefficient').value()
        sharp = self.parameters.child('sharp coefficient').value()
        norm = self.parameters.child('color histogram normalization').value()
        self.img = prepare.blur(self.img, blur)
        if (sharp):
            self.img = prepare.sharp(self.img)

        if (norm):
            self.img = prepare.norm(self.img)
        self.to_disp = np.array(self.img)
        self.imageView.setImage(self.to_disp)
        self.applyButton.setEnabled(True)


def startApp():
    app = QApplication(sys.argv)
    ex = App()
    ex.resize(800, 500)
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    startApp()
