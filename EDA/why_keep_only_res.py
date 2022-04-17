import matplotlib.pyplot as plt
import csv

first = {'SS':0, '':0, 'ESS':0, 'R':0}
with open('resolutions.csv') as inp, open('filtered_resolutions.csv', 'w') as out:
    reader = csv.DictReader(inp)
    for row in reader:
        e = row["resolution"].split("/")
        first[e[0]] += 1

first["Special Session"] = first.pop("SS")
first["Emergency SS"] = first.pop("ESS")
first["Resolution"] = first.pop("R")
first["*empty*"] = first.pop("")

fig, ax = plt.subplots()
bar_plot = plt.bar(*zip(*first.items()))
bar_label = list(first.values())

def autolabel(rects):
    for idx,rect in enumerate(bar_plot):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                bar_label[idx],
                ha='center', va='bottom', rotation=0)
autolabel(bar_plot)
plt.title("# of documents in Kaggle data")
plt.show()