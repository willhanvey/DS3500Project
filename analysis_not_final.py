# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 02:13:54 2022

@author: danie
text analysis of UN resolutions
"""

import os
from pdf_reader import write_text

def read_text(filename):
    with open(filename) as file:
        return file.read()

def main():
    print('Hello there!')
    # get list of files in folder
    all_files = os.listdir("res_text/")
    # clean list - remove non-english and addendums
    all_f = [file for file in all_files if file[-5:-8:-1]=='NE-'] # keep english only
    all_f = [file for file in all_f if file[-6:-9:-1]!='ddA'] # remove addendums
    
    print(len(all_f))
    # loop through all files
    for file in all_f:
        # open clean etc
        txt = read_text(filename="res_text/"+file)
        
        txt = "".join(u for u in txt.lower() if u not in ("?", ".", ";", ":",  "!",'"', ',','{', '}'
                                                          '~','/',"\\", "'", '(',')', 'xx', '·'))
        txt_split = [word for word in txt.split() if len(word)>4]
        #print(txt_split)
        
        # write text to txt file
        write_text(",".join(txt_split), filename=file, path="clean_text/")
        

if __name__ == "__main__":
    main()

