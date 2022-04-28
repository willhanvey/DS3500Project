# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 02:13:54 2022

@author: danie
text analysis of UN resolutions
"""

import os
from pdf_reader import write_text
from collections import Counter, defaultdict
import pycountry
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


class Text:
    
    def __init__(self):
        "constructor"
        # A dictionary where title:text will be stored
        self.data = defaultdict(dict)
        # A list to store all the words
        self.words = []
        # A dict where title: length will be stored
        self.size = defaultdict(dict)

    def save(self, name, results):
        "adds new item to dict. Name of resolution: contents of resolution"
        self.data[name] = results
        
    def read_text(self, filename, min_length=4, stop_lst=['united', 'nations', 
                                                          'resolution','committee', 'assembly', 'general'],
                  stemmer="no"):
        """Open file and clean text: remove punctuation, lowecase, short words, stop words
        writes clean data to txt files"""
        with open('res_text/'+filename) as file:
            txt = file.read()
        
        # clean punctuation
        txt = "".join(u for u in txt.lower() if u not in ("?", ".", ";", ":",  "!",'"', ',','{', '}',
                                                          '~',"\ " ,"\\", "[", "]" ,"'", '(',')', '·' ))
        
        self.size[filename] = len(txt.split())
        # get rid of stop words, NLTK library
        country_lst = [list(pycountry.countries)[i].name.lower() for i in range(len(pycountry.countries))]
        stops = list(stopwords.words('english')) # stop words
        stops.extend(stop_lst) # add stop words
        
        # convert to list of words
        txt_split = [word for word in txt.split() if len(word)>min_length or word in country_lst]
        txt_split_2 = [word for word in txt_split if word not in stops]
        
        # add stemming, make nation/nations the same
        if stemmer =="no":
            self.words.extend(txt_split_2) # extend list of words
            self.save(filename, txt_split_2) # save data in dictionary
            self.write(filename, txt_split_2) # write clean text to txt file
        else:
            txt_split_3 = [stemmer.stem(plural) for plural in txt_split_2]
            self.words.extend(txt_split_3)
            self.save(filename, txt_split_3)
            self.write(filename, txt_split_3)
        
    def write(self, file, results):
        """Write results to a txt file in a folder called clean_text"""
        write_text(",".join(results), filename=file, path="clean_text/")
    
    def count(self, country_name):
        """Returns integer: number of times a country is mentioned in all resolutions"""
        return self.words.count(country_name)
    
    def wordcount_viz(self, comm=2):
        """visualize most common words, excluding x most common words, default 2"""
        # make dictionary of most common words
        most_com = dict(Counter(self.words).most_common(10+comm))
        # define variables for visualization
        x = list(most_com.keys())[-10:]
        y = list(most_com.values())[-10:]
        y_pos = np.arange(len(x))
        
        # create bar plot
        plt.bar(y_pos, y, align='center', alpha=0.8)
        plt.xticks(y_pos, x, fontsize=8, rotation=60)
        plt.ylabel("Word Count")
        plt.xlabel("Word")
        plt.title("Number of time word appears accross all resolutions")
        plt.savefig("Common_words_count.png")
        plt.show()

    def country_count(self, UNSC_country ='libya',
                      countries=["israel", "russian", "america", "france", "syria", "kosovo", "libya"],
                      title='Countries_Mentioned.png'):
        "Visualize how often a country is mentioned in UNSC Res and in general"
        # count how often countries are mentioned
        country_count = []
        for country in countries:
            country_count.append(self.count(country))
        
        # plot the countries mentioned in a bar plot
        plt.bar(np.arange(len(countries)), country_count, alpha=0.6, color='red')
        plt.xticks(np.arange(len(countries)), countries, fontsize = 8, rotation=45)
        plt.title(title[0:-4])
        plt.savefig(title)
        plt.show()
        
        # count how often one country is mentioned in the UNSC
        count = []
        spec_count = []
        df = pd.DataFrame(0, index = np.arange(1),columns=range(1995,2022,1))
        # loop to only read UNSC resolutions and occurences of country
        for key, value in self.data.items():
            if key[0:2] == 'S_' and list(value).count(UNSC_country) != 0:
                count.append(key[2:6])
                spec_count.append(list(value).count(UNSC_country))
                # add to relevant year
                df[int(key[2:6])] += [list(value).count(UNSC_country)]
            else:
                pass
        
        # create scatter plot
        sns.scatterplot(data=df.transpose()).set(title="Security Council mentions of: "+UNSC_country)
        plt.savefig("UNSC_"+UNSC_country+".png")
        plt.close()
        
    def hdi_viz(self, hdi_file='HDI.csv'):
        """Visualization function using HDI data for comparison"""
        # read csv
        hdi_df = pd.read_csv(hdi_file, index_col=1)
        # drop HDI Rank
        hdi_df.drop('HDI Rank', axis=1, inplace=True)
        # drop empty unnamed columns
        hdi_df = hdi_df.loc[:, ~hdi_df.columns.str.contains('^Unnamed')]
        # loop through possible columns to eliminate cells with '..'
        for year in range(1990,2020,1):
            hdi_df = hdi_df[hdi_df[str(year)]!='..']
        
        # convert HDI values to floats
        hdi_df = hdi_df.astype(float)
        
        # correlate times country is mentioned and HDI
        # obtain HDI measure
        hdi_2019 = hdi_df['2019']
        # get country list and clean to match rest of data
        countries = [country.lower()[1:] for country in list(hdi_df.index)]
        # define counts variable
        counts = []
        # loop through countries to get each countries' count
        for country in countries:
            counts.append(self.count(country))
        
        # define df with HDI, counts and country names
        df_1 = pd.DataFrame(dict(country= countries, HDI= hdi_2019,count=counts))
        # sort by HDI, smallest to largest
        df_1.sort_values('HDI',inplace=True)
        # only include those with 5 or more counts
        df_1 = df_1[df_1['count']>4]
        # creat seaborn scatter plot
        ax = sns.scatterplot(x='country',y='count',data=df_1, hue='HDI',s=200, 
                             alpha=0.7)
        ax.set(title="Relationship between country word count in the UN and HDI")
        # set xticks to show examples of countries with lower/higher HDI
        ax.set_xticks([])
        plt.savefig("HDI_Count_Correlation.png")
        plt.show()
    
    def scatter_certain_words(self, df_1, x_words=["nuclear", "conflict", "crimes"],
                              title="Occurence of certain words in General Assembly"):
        """Visualize occurences of x words in scatter plots"""
        df_1 = df_1.set_index("index")
        
        for word in x_words:
            df_1[word] = df_1['only_words'].str.count(word)
        
        fig, axes = plt.subplots(1,1)
        sns.scatterplot(data=df_1.drop('word_count', axis=1))
        # limit y axis to avoid low counts
        plt.ylim(5,60)
        # elimate x ticks and title plot
        axes.set_xticks([])
        axes.set(title=title)
        plt.savefig(title+".png")
        plt.show()
        plt.close()
    
    def ga_sc_viz(self, x_words=['american','russian','nuclear', 'conflict', 'crimes'], title="Occurence of certain words in Security Council"):
        """Visualization function"""
        # make dataframe with size data
        df = pd.DataFrame(data=self.size.values(), index=self.size.keys(), columns=['word_count'])
        # add words to the df
        df['words'] = list(self.data.values())
        df['only_words'] = df['words'].apply(lambda x: " ".join(map(str, x)))
        
        # reset index to distinguish GA and SC
        df_1 = df.reset_index()
        df_2 = df_1[~df_1['index'].str.contains("S_")]
        df_3 = df_1[~df_1['index'].str.contains("A_")]

        # Make seaborn plot
        fig, axes = plt.subplots(1,1)
        sns.lineplot(x='index',y='word_count', data=df_2)
        sns.lineplot(x=df_3['index'],y=df_3['word_count'], data=df_2)
        axes.set_xticks([])
        plt.legend(title='Word Count over time', labels=['UNGA', 'UNSC'])
        plt.savefig("word_count_time.png")
        plt.show()
        plt.close()
        
        # make scatter plots for the two data frames: security countil and general assembly
        self.scatter_certain_words(df_3, x_words=x_words, 
                                   title=title)
        self.scatter_certain_words(df_2 , x_words=x_words)

def main():
    
    # initialize class
    texts = Text()
    # get list of files in folder
    all_files = os.listdir("res_text/")
    # clean list - remove non-english and some addendums
    all_f = [file for file in all_files if file[-5:-8:-1]=='NE-'] # keep english only
    all_f = [file for file in all_f if file[-6:-9:-1]!='ddA'] # remove addendums
    
    # read all the texts
    for filename in all_f:
        texts.read_text(filename)
    
    # make the visualizations
    texts.wordcount_viz(comm=20)
    texts.country_count()
    texts.country_count(countries=['sustainable', 'economy', 'product', 'mortality', 'water', 'hunger'], 
                        title="development_words.png")
    texts.hdi_viz()
    texts.ga_sc_viz()

if __name__ == "__main__":
    main()

"""Sources of code:
https://stackoverflow.com/questions/45306988/column-of-lists-convert-list-to-string-as-a-new-column
https://stackoverflow.com/questions/17573814/count-occurrences-of-each-of-certain-words-in-pandas-dataframe

    
"""

