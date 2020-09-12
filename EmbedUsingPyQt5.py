from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton
import sys
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        
        title = "Victor test V_1.1"
        top = 400
        left = 400
        width = 1280
        height = 738
        
        self.setWindowTitle(title)
        self.setGeometry(top, left, width, height)
        self.MyUI()
    def MyUI(self):
        canvas = Canvas(self, width=8, height= 4)
        canvas.move(0,0)
        
        button = QPushButton("Click here", self)
        button.move(0,450)
        
        button2 = QPushButton("Click here Two", self)
        button2.move(200,450)
        
class Canvas(FigureCanvas):
    def __init__(self, parent = None, width = 5, height = 5, dpi = 100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        FigureCanvas.__init__(self,fig)
        self.setParent(parent)
        self.plot()
    def plot(self):
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        # Importing the dataset
        dataset = pd.read_csv('Salary_Data.csv')
        X = dataset.iloc[:, :-1].values
        y = dataset.iloc[:, 1].values
        print(X)
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


app  = QApplication(sys.argv)
window = Window() 
window.show()
app.exec()