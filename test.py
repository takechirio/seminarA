from matplotlib import pyplot as pyp
date = [49,1]
labels = ['expected', 'error']

pyp.title("error rate[hand:x<300(pixel/s)]")
pyp.pie(date, startangle=90, counterclock=False,  autopct='%.1f%%', pctdistance=0.8, labels=labels)
pyp.show()

date = [31,19]
labels = ['expected', 'error']

pyp.title("error rate[hand:1000<x(pixel/s)]")
pyp.pie(date, startangle=90, counterclock=False,  autopct='%.1f%%', pctdistance=0.8, labels=labels)
pyp.show()