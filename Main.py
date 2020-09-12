
from PyQt5.QtWidgets import (QApplication, QMessageBox, QMainWindow, QVBoxLayout, QAction, QFileDialog, QDialog, QLineEdit, QPushButton, QLineEdit)
from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon
from PyQt5.uic import loadUiType
from os.path import dirname, realpath, join
from sys import argv
import sys



scriptDir = dirname(realpath(__file__))
FROM_MAIN, _ = loadUiType(join(dirname(__file__), "MainWindow.ui"))

class Main(QMainWindow, FROM_MAIN):
    def __init__(self, parent = FROM_MAIN):
        super(Main, self).__init__()

        QMainWindow.__init__(self)
        self.init_ui()
        self.show()

    
    def init_ui(self):
        
        self.lineEdit.setText()
        
    

def main():
    app = QApplication(argv)
    window = Main()
    # window.showFullScreen() # Start at position full screen
    
    app.exec_()


if __name__ == '__main__':
    main()