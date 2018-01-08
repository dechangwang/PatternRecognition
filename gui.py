#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Author wangdechang
Time 2018/1/8
"""

from Tkinter import *
from ttk import *
import matplotlib
import RandomForestClassification as rfc
import numpy as np

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

selected = 'RF'


def reDraw():
    reDraw.f.clf()
    reDraw.a = reDraw.f.add_subplot(111)
    reDraw.a.set_ylabel('cdf')
    reDraw.a.set_yticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
    reDraw.a.set_xlabel('distance error(m)')
    reDraw.a.scatter(reDraw.raw_data[:, 0], reDraw.raw_data[:, 1], c='r', marker='x')
    reDraw.a.plot(reDraw.raw_data[:, 0], reDraw.raw_data[:, 1], linewidth=1)
    reDraw.canvas.show()


def classify():
    if selected == "RF":
        print (selected)
        data = rfc.classify()
        data = np.sort(data)
        size = len(data) - 1
        raw_data = []
        for i in range(1, 11):
            value = data[int(size * i / 10)]
            raw_data.append([value, float(i) / 10])
        reDraw.raw_data = np.array(raw_data)

    reDraw()


def show_list_item(arg):
    global selected
    selected = algorithm_chosen.get()
    print (algorithm_chosen.get())


root = Tk()
root.title("locate")

reDraw.f = Figure(figsize=(5, 4), dpi=150)  # create canvas
reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
reDraw.canvas.show()
reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Label(root, text="Choose a algorithm").grid(row=1, column=0)
# Label(root, text="tolN").grid(row=1, column=0)
# tolNentry = Entry(root)
# tolNentry.grid(row=1, column=1)
# tolNentry.insert(0, '10')

Button(root, text="classify", command=classify).grid(row=1, column=2, rowspan=3)

algorithms = StringVar()
algorithm_chosen = Combobox(root, width=12, textvariable=algorithms)
algorithm_chosen['values'] = ("KNN", "RF", "Bayes")
algorithm_chosen.grid(row=1, column=1)
algorithm_chosen.current(0)
algorithm_chosen.bind("<<ComboboxSelected>>", show_list_item)
# action = Button(root, text="classify", command=classify)

#
# reDraw.rawDat = mat(regTrees.loadDataSet('sine.txt'))
# reDraw.testDat = arange(min(reDraw.rawDat[:, 0]), max(reDraw.rawDat[:, 0]), 0.01)
# reDraw(1.0, 10)

root.mainloop()
