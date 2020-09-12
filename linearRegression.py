# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'linearRegression.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!
import os
import sys
import csv


from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from PyQt5 import QtCore, QtGui, QtWidgets

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Ui_lrWindow(object):
        
    def setupUi(self, lrWindow):
        lrWindow.setObjectName("lrWindow")
        lrWindow.resize(628, 641)
        self.centralwidget = QtWidgets.QWidget(lrWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.readCSV = QtWidgets.QPushButton(self.centralwidget)
        self.readCSV.setGeometry(QtCore.QRect(240, 40, 141, 31))
        self.readCSV.setObjectName("readCSV")
        
        
        #self.readCSV.pressed.connect(self.runFile)
        
        
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(160, 120, 111, 31))
        font = QtGui.QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(350, 120, 113, 31))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(210, 410, 201, 51))
        self.pushButton.setObjectName("pushButton")
        
        self.pushButton.pressed.connect(self.runFile)
        
        
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(160, 90, 111, 21))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(350, 90, 111, 21))
        self.label_2.setObjectName("label_2")
        self.lineEdit_3 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_3.setGeometry(QtCore.QRect(160, 190, 111, 31))
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(160, 170, 47, 13))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(30, 130, 101, 16))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(30, 200, 81, 16))
        self.label_5.setObjectName("label_5")
        lrWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(lrWindow)
        QtCore.QMetaObject.connectSlotsByName(lrWindow)
        
        ###############################################################
        ####################### ML CODE################################
        
        
    def runFile(self):
    
        
        # Importing the dataset
        dataset = pd.read_csv('Salary_Data.csv')
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, 1].values

        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

        # Feature Scaling
        """from sklearn.preprocessing import StandardScaler
        sc_X = StandardScaler()
        X_train = sc_X.fit_transform(X_train)
        X_test = sc_X.transform(X_test)
        sc_y = StandardScaler()
        y_train = sc_y.fit_transform(y_train)"""

        # Fitting Simple Linear Regression to the Training set
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = regressor.predict(X_test)

        # Visualising the Training set results
        plt.scatter(X_train, y_train, color = 'red')
        plt.plot(X_train, regressor.predict(X_train), color = 'blue')
        plt.title('Salary vs Experience (Training set)')
        plt.xlabel('Years of Experience')
        plt.ylabel('Salary')
        plt.show()

        # Visualising the Test set results
        #plt.scatter(X_test, y_test, color = 'red')
        #plt.plot(X_train, regressor.predict(X_train), color = 'blue')
        #plt.title('Salary vs Experience (Test set)')
        #plt.xlabel('Years of Experience')
        #plt.ylabel('Salary')
        #plt.show()
        print("Done>>")
    def readFile(self):
        path = QFileDialog.getOpenFileName(self,'Open CSV', os.getenv('Desktop'),'CSV(*.csv)')
        if path[0] != '':
            with open(path[0], newline ='') as csv_file:
                my_file = csv.reader(csv_file, dialect = 'excel')
                
                
                df = pd.DataFrame(my_file)
                print(df)

    def retranslateUi(self, lrWindow):
        _translate = QtCore.QCoreApplication.translate
        lrWindow.setWindowTitle(_translate("lrWindow", "MainWindow"))
        self.readCSV.setText(_translate("lrWindow", "Read CSV"))
        self.pushButton.setText(_translate("lrWindow", "Visualize Result"))
        self.label.setText(_translate("lrWindow", "Independent variable"))
        self.label_2.setText(_translate("lrWindow", "dependent variable"))
        self.label_3.setText(_translate("lrWindow", "Test size"))
        self.label_4.setText(_translate("lrWindow", "Data selection"))
        self.label_5.setText(_translate("lrWindow", "Train/test split"))
