
# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'KNN.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_KNNWindow(object):
    def setupUi(self, KNNWindow):
        KNNWindow.setObjectName("KNNWindow")
        KNNWindow.resize(628, 641)
        self.centralwidget = QtWidgets.QWidget(KNNWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.readCSV = QtWidgets.QPushButton(self.centralwidget)
        self.readCSV.setGeometry(QtCore.QRect(160, 50, 111, 31))
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
        self.VR.setGeometry(QtCore.QRect(160, 390, 111, 31))
        self.VR.setObjectName("VR")
        
        self.VR.pressed.connect(self.runFile)
        
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
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(350, 190, 111, 31))
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(350, 170, 47, 13))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.centralwidget)
        self.label_7.setGeometry(QtCore.QRect(30, 250, 81, 16))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(30, 290, 81, 16))
        self.label_8.setObjectName("label_8")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(30, 330, 111, 16))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.centralwidget)
        self.label_10.setGeometry(QtCore.QRect(30, 400, 111, 16))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(30, 60, 101, 16))
        self.label_11.setObjectName("label_11")
        KNNWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(KNNWindow)
        QtCore.QMetaObject.connectSlotsByName(KNNWindow)

    def retranslateUi(self, KNNWindow):
        _translate = QtCore.QCoreApplication.translate
        KNNWindow.setWindowTitle(_translate("KNNWindow", "MainWindow"))
        self.readCSV.setText(_translate("KNNWindow", "Read "))
        self.VR.setText(_translate("KNNWindow", "Visualize"))
        self.label.setText(_translate("KNNWindow", "Independent variable"))
        self.label_2.setText(_translate("KNNWindow", "dependent variable"))
        self.label_3.setText(_translate("KNNWindow", "train size"))
        self.label_4.setText(_translate("KNNWindow", "Data selection"))
        self.label_5.setText(_translate("KNNWindow", "Train/test split"))
        self.label_6.setText(_translate("KNNWindow", "Test size"))
        self.label_7.setText(_translate("KNNWindow", "Features scaling"))
        self.label_8.setText(_translate("KNNWindow", "Confusion Matrix"))
        self.label_9.setText(_translate("KNNWindow", "Classifier Parameter"))
        self.label_10.setText(_translate("KNNWindow", "Visualize result"))
        self.label_11.setText(_translate("KNNWindow", "Read csv files"))

    def runFile(self):
        # K-Nearest Neighbors (K-NN)

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

        # Fitting K-NN to the Training set
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
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
        plt.title('K-NN (Training set)')
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
        plt.title('K-NN (Test set)')
        plt.xlabel('Age')
        plt.ylabel('Estimated Salary')
        plt.legend()
        plt.show()