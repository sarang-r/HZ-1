# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'logisticRegression.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_lgrWindow(object):
    def setupUi(self, lgrWindow):
        lgrWindow.setObjectName("lgrWindow")
        lgrWindow.resize(628, 641)
        self.centralwidget = QtWidgets.QWidget(lgrWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.readCSV = QtWidgets.QPushButton(self.centralwidget)
        self.readCSV.setGeometry(QtCore.QRect(240, 40, 141, 31))
        self.readCSV.setObjectName("readCSV")
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
        self.VR = QtWidgets.QPushButton(self.centralwidget)
        self.VR.setGeometry(QtCore.QRect(210, 410, 201, 51))
        self.VR.setObjectName("VR")
        
        self.VR.pressed.connect(self.LogicsticR)
        
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
        lgrWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(lgrWindow)
        QtCore.QMetaObject.connectSlotsByName(lgrWindow)
    def LogicsticR(self):
        # Logistic Regression

        # Importing the libraries
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        # Importing the dataset
        dataset = pd.read_csv('Social_Network_Ads.csv')
        X = dataset.iloc[:, [2, 3]].values
        y = dataset.iloc[:, 4].values

        # Splitting the dataset into the Training set and Test set
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Fitting Logistic Regression to the Training set
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        classifier.fit(X_train, y_train)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)

        # Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)

        # Visualising the Training set results
        from matplotlib.colors import ListedColormap
        X_set, y_set = X_train, y_train
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                            np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                    alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title('Logistic Regression (Training set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()

        # Visualising the Test set results
        from matplotlib.colors import ListedColormap
        X_set, y_set = X_test, y_test
        X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                            np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
        plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                    alpha = 0.75, cmap = ListedColormap(('red', 'green')))
        plt.xlim(X1.min(), X1.max())
        plt.ylim(X2.min(), X2.max())
        for i, j in enumerate(np.unique(y_set)):
            plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                        c = ListedColormap(('red', 'green'))(i), label = j)
        plt.title('Logistic Regression (Test set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()

    def retranslateUi(self, lgrWindow):
        _translate = QtCore.QCoreApplication.translate
        lgrWindow.setWindowTitle(_translate("lgrWindow", "MainWindow"))
        self.readCSV.setText(_translate("lgrWindow", "Read CSV"))
        self.VR.setText(_translate("lgrWindow", "Visualize Result"))
        self.label.setText(_translate("lgrWindow", "Independent variable"))
        self.label_2.setText(_translate("lgrWindow", "dependent variable"))
        self.label_3.setText(_translate("lgrWindow", "Test size"))
        self.label_4.setText(_translate("lgrWindow", "Data selection"))
        self.label_5.setText(_translate("lgrWindow", "Train/test split"))
