import sys

import matplotlib.pyplot as plt
import pyqtgraph as pg
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog
from pyqtgraph.Qt import QtGui
import numpy as np
import prepare
import scipy

FILENAME = ""
LOADED_PIC_SIZE = 300
RESULT_PIC_SIZE = 300
result_img = ""
PLAY_RATE = 5


class App(QWidget):

	def __init__(self):
		QWidget.__init__(self)
		self.title = 'Tomograph Simulator'
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
			{'name': 'blur coefficient', 'type': 'float', 'value': 5.0},
			{'name': 'sharp coefficient', 'type': 'float', 'value': 5.0},
			{'name': 'color histogram normalization', 'type' : 'bool', 'value': True}
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
		FILENAME = QFileDialog.getOpenFileName(filter="Images (*.png *.jpg)")[0]
		if FILENAME == '': return
		print(FILENAME)

		self.img = plt.imread(FILENAME)    
		
		if len(self.img.shape) == 3:  # RGB
		    self.img = self.img[:, :, 0]
		assert len(self.img.shape) == 2
		
		self.imageView.setImage(self.img)

		self.startButton.setEnabled(True)
		self.applyButton.setEnabled(True)

	@pyqtSlot()
	def startClickAction(self):
		global result_img
		self.startButton.setEnabled(False)
		
	
	@pyqtSlot()
	def applyClickAction(self):
		self.applyButton.setEnabled(False)
		blur = self.parameters.child('blur coefficient').value()
		sharp = self.parameters.child('sharp coefficient').value()
		norm = self.parameters.child('color histogram normalization').value()
		#self.img = scipy.misc.face(gray=True).astype(float)
		self.img = prepare.blur(self.img,blur)
		self.image = prepare.sharp(self.img,sharp)
		self.imageView.setImage(self.img)
		self.applyButton.setEnabled(True)

def startApp():
	app = QApplication(sys.argv)
	ex = App()
	ex.resize(800, 500)
	ex.show()
	sys.exit(app.exec_())


if __name__ == '__main__':
	startApp()