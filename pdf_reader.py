# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:30:18 2022

@author: danie
DS3500 Project
"""

# imports
from tika import parser
from pathlib import Path
from pdfminer.high_level import extract_text


def read_pdf(filename):
    """ 
    Read pdf file using tika parser. Returns resolution text and filename
    """
    try:
        raw = parser.from_file('DraftResolutions/'+filename) #pdf reader
        return raw['content'], filename
    # if name is not in directory, return none object
    except FileNotFoundError:
        return None, filename
    
def read_pdf_2(filename):
    """ Read pdf using pdfminersix"""
    content = extract_text(filename)
    return repr(content), filename

def write_text(content, filename, path='res_text/'):
    """
    writes the resolution to a text file
    param: content: the string extracted from the pdf
    param: filename: the name of the text file created
    """
    if type(content) == str: # avoids type error
        file = Path(path+filename+'.txt') #user needs to make folder called res_text
        file.touch(exist_ok=True)
        with open(file, 'w') as text_file:
            # try/except to avoid encoding error
            try:
                text_file.write(content)
            except UnicodeEncodeError:
                pass
    else: #if content isn't a string
        pass

def main():  
    
    # open and make list of pdfs to read
    with open('FinalPDFList.txt') as file:
        lines = file.readlines()
    # creates list of resolutions, drops Error values and \n
    res_lst =[line[:-1] for line in lines if line != 'Error\n']
    
    d = {} # define dictionary to store text
    for res in res_lst:
        content, filename = read_pdf(res) # read pdf
        write_text(content, filename=filename[:-4]) # write res to txt file
        d[filename] = content # store filename and content in dict
    
if __name__ == '__main__':
    main()



