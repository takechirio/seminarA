from matplotlib import pyplot as pyp
date = [44,6,0]
labels = ['expected', 'speed error', 'touch error']

pyp.title("error rate[hand:slow]")
pyp.pie(date, startangle=90, counterclock=False,  autopct='%.1f%%', pctdistance=0.8, labels=labels)
pyp.show()

date = [29,8,13]
labels = ['expected', 'speed error', 'touch error']

pyp.title("error rate[hand:quick]")
pyp.pie(date, startangle=90, counterclock=False,  autopct='%.1f%%', pctdistance=0.8, labels=labels)
pyp.show()