from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasAgg, NavigationToolbar2Tk

class Root(Tk):
    def __init__(self):
        super(Root, self).__init__()
        self.title("Victor")
        self.minsize(640,400)
        
        self.matplotCanvas()
    def matplotCanvas(self):
        f = Figure(figsize= (5,5), dpi = 100)
        a = f.add_subplot(111)
        a.plot([1,2,3,4,5,6,7,8], [5,6,1,3,8,9,3,5])
        
        canvas = FigureCanvasAgg(f, self)
        canvas.show()
        
        canvas.get_tk_widget().pack(side =BOTTOM, fill = BOTH, expand =True)
        
        

if __name__ == '__main__':
    root = Root()
    root.mainloop()