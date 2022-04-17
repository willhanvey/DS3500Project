from os import listdir, remove
from os.path import isfile, join
import re
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
"""
mypath = "/Users/alex/2021/spring/35P/ds3500_project/res_text"
files_in_dir = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# keep ENG
r = re.compile("^(?![\.\w-]*-EN.txt).*")
to_remove = list(filter(r.match, files_in_dir))
for f in to_remove:
    remove(mypath + "/" + f)
"""

num_docs, num_rows = 0,0
mypath = "./res_text"
files_in_dir = [f for f in listdir(mypath) if isfile(join(mypath, f))]
"""
No B's in files
q = re.compile(".*B.*")
bees = list(filter(q.match, files_in_dir))
if len(bees)>0:
    print(bees)
"""
with open('resolutions.csv') as inp, open('filtered_resolutions.csv', 'w') as out:
    reader = csv.DictReader(inp)
    writer = csv.DictWriter(out, reader.fieldnames)
    writer.writeheader()
    for row in tqdm(reader):
        e = row["resolution"].split("/")
        if e[0]=="R" and len(e)==3:
            r = re.compile("A[\.\w-]*_{}[\.\w-]*{}.*\.txt".format(e[1], e[2]))
            available = list(filter(r.match, files_in_dir))
            if len(available) > 0:
                num_docs += len(available)
                num_rows += 1
                writer.writerow(row)

print("docs:",num_docs)
print("rows:",num_rows)
