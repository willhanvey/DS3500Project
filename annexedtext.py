import re
import numpy as np
import collections
import seaborn as sns
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import annexedtextparsers as atp
from scipy.stats import pearsonr
from nltk.corpus import stopwords
import plotly.graph_objects as go


class annexedtext:

    def __init__(self):

        self.filecount = 0
        self.committee_list = ('JDC', 'PAL', 'POL', 'SOC', 'SAM', 'ECON', 'ENVR', 'HOS')

        # Initializing variables that will be needed later
        self.stats = {}
        self.all_text, self.file_details = [], []
        self.sentiment_dict = self.word_sentiment_dict()
        sns.set_theme()

    def load_file(self, file, year, committee, parser=None, removestopwords=True):
        # Takes in file name (str), year (int), committee (str) (year and committee acting as label), and parser,
        # and unpacks and opens file while adding information to class variables. Removestopwords is a bool that
        # allows the option of keeping or removing stopwords
        self.file_details.append([year, committee])
        committee_year = str(year) + str(committee)

        if parser is None:
            with open(file, 'r', encoding='UTF-8') as infile:
                # Creating lists to temporarily store data that will be passed to class variables later
                textlist = []
                for line in infile:
                    sentence = line.split()
                    if sentence:
                        # Grabbing words from sentences and culling headings
                        for word in sentence:
                            word = re.sub('[^a-zA-Z]+', '', word).lower()
                            if removestopwords:
                                if word != '' and word not in stopwords.words('english'):
                                    textlist.append(word)
                            else:
                                if word != '':
                                    textlist.append(word)
        else:
            textlist, parser_stats = parser(file, removestopwords)
        if not parser_stats:
            self.initial_stats(committee_year, textlist)
        else:
            self.stats = parser_stats
        self.all_text.append(textlist)
        self.filecount += 1

    def initial_stats(self, committee_year, textlist):
        # Takes in textlist (list) and determines initial stats
        doclength = len(textlist)
        uniquewords = len(set(textlist))
        averagelength = sum(len(word) for word in textlist) / len(textlist)
        sentiment = self.sentiment_score(textlist)
        pol, sub = self.polsubreader(textlist)
        common_words = self.get_top(textlist)
        self.stats[committee_year] = {'Word Count': doclength, 'Unique Words': uniquewords, 'Average Word Length':
                                      averagelength, 'Sentiment': sentiment, 'Polarity': pol, 'Subjectivity': sub,
                                      'Common Words': common_words}

    @staticmethod
    def word_sentiment_dict():
        # Gets the AFINN-111 word sentiment document into a dictionary
        word_dict = {}
        with open('AFINN-111.txt', 'r') as infile:
            while True:
                line = infile.readline().strip('\n').split('\t')
                if line == ['']:
                    break
                word_dict[line[0]] = int(line[1])
        return word_dict

    def sentiment_score(self, lst):
        # Calculates the average sentiment score per president using the AFINN-111 dict
        score = 0
        for i in range(len(lst)):
            lst[i] = lst[i].strip(',.;—–?!"\': ')
            if lst[i] in self.sentiment_dict.keys():
                score += self.sentiment_dict[lst[i]]
        return score / len(lst)

    @staticmethod
    def polsubreader(lst, minsub=0.0, maxsub=1.0, minpol=-1.0, maxpol=1.0):
        text = ' '.join([word for word in lst])
        pol, sub = TextBlob(text).sentiment
        return pol, sub

    @staticmethod
    def get_top(lst, k=10):
        # Takes in list of words, returns list of most common words
        most_common_words = collections.Counter([word for word in lst])
        inaugural_top = (most_common_words.most_common(k))
        return inaugural_top

    def comparison_bar(self, comparisonelement):
        # Generating a bar chart comparing a chosen statistic across the uploaded documents
        if len(self.stats) == 0:
            print('Please upload a file first')
            return
        height = [self.stats[subdict][comparisonelement] for subdict in self.stats]
        labels = [(str(year) + committee) for year, committee in self.file_details]

        plt.bar(x=range(len(height)), height=height)
        plt.xticks(range(len(height)), labels, rotation=45)
        plt.xlabel('Committee')
        plt.ylabel(str(comparisonelement))
        plt.title(str(comparisonelement) + ' by Committee and Year')
        plt.show()
        return

    def comparison_scatter(self, comparison1, comparison2):
        # Generating a scatter plot comparing two chosen statistics across the uploaded documents and
        # returns the correlation coefficient as an integer
        if len(self.stats) == 0:
            print('Please upload a file first')
            return
        xelement = [self.stats[subdict][comparison1] for subdict in self.stats]
        yelement = [self.stats[subdict][comparison2] for subdict in self.stats]
        plt.scatter(x=xelement, y=yelement)
        plt.xlabel(comparison1)
        plt.ylabel(comparison2)
        plt.title(str(comparison2) + ' by ' + str(comparison1))
        plt.show()
        corr, _ = pearsonr(xelement, yelement)
        return corr

    @staticmethod
    def vec(myinterests, all_interests):
        # Produces a vector that contains the number of times each word is used
        # per speech for all the most common words

        # Making a new list just to contain the words
        my_words = []
        for i in range(len(myinterests)):
            my_words.append(myinterests[i][0])

        all_interests = list(all_interests)
        vector = []

        # Appending the words if they are in the all words list
        for i in range(len(all_interests)):
            if all_interests[i] in my_words:
                idx = my_words.index(all_interests[i])
                vector.append(myinterests[idx][1])
            else:
                vector.append(0)

        return vector

    @staticmethod
    def mag(v):
        """ Magnitude of the vector, v = [vx, vy, vz, ...] """
        return sum([i ** 2 for i in v]) ** .5

    @staticmethod
    def dot(u, v):
        """ Dot product of two vectors """
        return sum([i * j for i, j in zip(u, v)])

    def cosine_similarity(self, u, v):
        cos_theta = self.dot(u, v) / (self.mag(u) * self.mag(v))
        return cos_theta

    def unique_words(self):
        # Gets the set of all unique words used across the speeches
        unique = set()
        for wordlist in self.all_text:
            for word in wordlist:
                unique.add(word)
        return unique

    def plot_cos_similarity(self, cos_similarity_arr):
        # Plotting data from the cos similarity array
        sns.heatmap(cos_similarity_arr)
        labels = [(str(year) + committee) for year, committee in self.file_details]
        plt.yticks(np.arange(len(cos_similarity_arr)) + .5, labels,
                   rotation='horizontal')
        plt.xticks(np.arange(len(cos_similarity_arr)) + 0.5, labels, rotation='vertical')
        plt.title('Cosine Similarity across Background Guides')
        plt.show()

    def similarity_heatmap(self):
        # Gets the cosine similarity between the documents and plots a heatmap
        if len(self.stats) == 0:
            print('Please upload a file first')
            return
        unique = self.unique_words()
        docs_lst = []
        for docwords in self.all_text:
            top100 = self.get_top(docwords, k=100)
            docs_lst.append(top100)
        lst_vector = [self.vec(lst, unique) for lst in docs_lst]
        number_files = len(self.file_details)
        cos_similarity_arr = np.zeros((number_files, number_files))
        for i in range(len(lst_vector)):
            for j in range(len(lst_vector)):
                cos = self.cosine_similarity(lst_vector[i], lst_vector[j])
                cos_similarity_arr[i][j] = cos
        self.plot_cos_similarity(cos_similarity_arr)

    @staticmethod
    def get_divisors(num):
        # Returns the two divisors (ints) of a number with the smallest sum
        smallestsum = num + 2
        for i in range(1, num + 1):
            if num % i == 0:
                num1 = i
                num2 = num / i
                twosum = num1 + num2
                if twosum < smallestsum:
                    smallest1, smallest2 = num1, num2
                    smallestsum = twosum
        return int(smallest1), int(smallest2)

    def wordclouds(self):
        # Generates wordclouds using subplots
        fig = plt.figure(figsize=(11, 4), dpi=200)
        fig.tight_layout()
        labels = [(str(year) + committee) for year, committee in self.file_details]

        # Gets divisors to be used to figure out the dimensions of the subplots
        num1, num2 = self.get_divisors(len(self.file_details))

        # Plots the wordclouds from each file
        for i in range(len(self.file_details)):
            string = ' '.join([word for word in self.all_text[i]])
            # Display the generated image:
            wordcloud = WordCloud().generate(string)
            fig.add_subplot(num1, num2, (i + 1))
            plt.imshow(wordcloud)
            plt.title(labels[i] + ' Words', fontdict={'fontsize': 10})
            plt.axis("off")
        plt.show()

    def sankey(self, k=5, word_list=None):
        # Creates a sankey diagram of the union of the k most common words from each text file
        topwordscount = []
        uniquewords = set()
        for docwords in self.all_text:
            top_doc = self.get_top(docwords, k)
            topwordscount.append(top_doc)
            for word, count in top_doc:
                uniquewords.add(word)
        uniquewords = list(uniquewords)

        # Allowing for the user to use their own words instead
        if word_list:
            uniquewords = word_list

        labels = [(str(year) + committee) for year, committee in self.file_details]
        labels.extend(uniquewords)

        # Getting the source, target, and value into their lists
        source, target, value = [], [], []
        for i in range(len(topwordscount)):
            for word, count in topwordscount[i]:
                source.append(i)
                target.append(labels.index(word))
                value.append(count)
        fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5),
                                                  label=labels, color="blue"),
                                        link=dict(source=source, target=target, value=value))])
        fig.show()


if __name__ == "__main__":
    test = annexedtext()
    test.load_file('2022SOC.txt', 2022, 'SOC', parser=atp.remove_arableague_headings)
    test.load_file('2022ECON.txt', 2022, 'ECON', parser=atp.remove_arableague_headings)
    test.load_file('2022HOS.txt', 2022, 'HOS', parser=atp.remove_arableague_headings)
    test.load_file('2022JDC.txt', 2022, 'JDC', parser=atp.remove_arableague_headings)
    test.load_file('2022PAL.txt', 2022, 'PAL', parser=atp.remove_arableague_headings)
    test.load_file('2022POL.txt', 2022, 'POL', parser=atp.remove_arableague_headings)
    test.load_file('2022ENVR.txt', 2022, 'ENVR', parser=atp.remove_arableague_headings)
    test.load_file('2022SPEC.txt', 2022, 'SPEC', parser=atp.remove_arableague_headings)

    # Visualization type 1- simple bar chart- the first 3 vizs all can count towards the third assigned viz
    test.comparison_bar('Unique Words')

    # Visualiztion type 2- simple scatter plot with two statistics
    print(test.comparison_scatter('Word Count', 'Unique Words'))

    # Visualization type 3- heatmap based on cosine similarity
    test.similarity_heatmap()

    # Visualization type 4- wordcloud- this is the visualization that uses subplots
    test.wordclouds()

    # Visualization type 5- sankey diagram
    test.sankey()
