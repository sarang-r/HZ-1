from PyQt5 import QtWidgets, uic
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def Func():
    v = call.lineEdit.text()
    v1 = call.lineEdit_2.text()
    v2 = call.lineEdit_3.text()
    a1 = call.lineEdit_4.text()
    a2 = call.lineEdit_5.text()
    a = int(v)
    b = int(v1)
    c = int(v2)
    a1 = int(a1)
    a2 = int(a2)
    # Importing the dataset
    dataset = pd.read_csv('Salary_Data.csv')
    X = dataset.iloc[:, :c].values
    y = dataset.iloc[:,a2].values
    
    print("Value of v is" , v)
    print("Value of v1", v1)
    print("Value of v2 is ", v2)
    print("the value of a1", a1)
    print("The value of a2 is", a2)
    
    print(list(X))
    print(list(y))
app = QtWidgets.QApplication([])
call = uic.loadUi("New.ui")
call.pushButton.clicked.connect(Func)
call.show()
app.exec()