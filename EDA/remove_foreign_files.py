from os import listdir, remove
from os.path import isfile, join
import re
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

mypath = "./res_text"
files_in_dir = [f for f in listdir(mypath) if isfile(join(mypath, f))]

r = re.compile("^(?![\.\w-]*-EN.txt).*")
to_remove = list(filter(r.match, files_in_dir))
for f in to_remove:
    remove(mypath + "/" + f)