import matplotlib.pyplot as plt
import csv

assembly_list = []
with open("../filtered_resolutions.csv") as f:
    for row in csv.DictReader(f):
        assembly_list.append(row["assembly_session"])

plt.hist(assembly_list, [i for i in range(1, 71)])
plt.show()