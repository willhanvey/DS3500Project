import os
import sys
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

VOTING_RECORD = {'Yes': [], 'No': []}


class Country:

    def __init__(self, df, country, votes):
        self.words_df = df
        self.country = country
        self.votes_list = votes
        self.accuracy = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.rf = None
        self.mse = None
        self.create_model()

    def create_model(self):
        """
        Initializes a random forest classifier model
        """
        self.words_df['Votes'] = self.votes_list
        df = self.words_df[self.words_df['Votes'] != 'A']
        df = df.dropna()
        df['Votes'] = df['Votes'].replace('Y', 1)
        df['Votes'] = df['Votes'].replace(['N', 'X'], 0)
        x = df.loc[:, df.columns != 'Votes']
        y = df['Votes']
        # Training the model and reporting accuracy
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=.2, random_state=7)
        self.rf = RandomForestClassifier(n_estimators=100, random_state=7)
        self.rf.fit(self.X_train, self.y_train)
        y_pred = self.rf.predict(self.X_test)
        self.mse = mean_squared_error(self.y_test, y_pred)
        print(self.country, ': ', self.mse)

    def __get_words__(self, file):
        """
        :param file: A text file
        :return: A one row df with the columns as words and values as the number of occurrences of the words in the file
        """
        column_list = self.words_df.columns
        df_dict = {}
        # Only using words already in the model
        for word in column_list:
            df_dict[word] = 0
        with open(file, 'r', encoding='UTF-8') as f:
            word_list = []
            line = None
            while line != '':
                line = f.readline()
                sentence_list = line.split()
                for word in sentence_list:
                    word_list.append(word)
            for word in word_list:
                if word in column_list:
                    df_dict[word] += 1
        word_df = pd.DataFrame(df_dict, index=[0])
        return word_df

    def vote(self, file):
        """
        :param file: A text file containing the resolution the country will vote on
        :return: Appends the country to the VOTING_RECORD global dictionary based on the result
        """
        word_df = self.__get_words__(file)
        '''
        knn = KNeighborsClassifier(n_neighbors=21)
        knn.fit(self.X_train, self.y_train)
        y_pred = self.knn.predict(word_df)
        '''
        x = word_df.loc[:, word_df.columns != 'Votes']
        y_pred = self.rf.predict(x)
        if y_pred == [1]:
            VOTING_RECORD['Yes'].append(self.country)
        else:
            VOTING_RECORD['No'].append(self.country)

    def __str__(self):
        return self.country


def filter_columns(file):
    """
    :param file: A csv file to be turned into a dataframe
    :return: The dataframe
    """
    df = pd.read_csv(file)
    for column in df.columns:
        if sum(df[column].isnull()) > len(df) * .75:
            df = df.drop(columns=column, axis=1)
    # Dropping columns with errors
    df = df.drop(df.index[0])
    df = df.drop(df.index[5712])
    df = df.drop(df.index[6225])
    df = df.drop(df.index[7180]).reset_index()
    return df


def append_file(df, file):
    """
    :param df: A dataframe
    :param file: A file of equal length to the dataframe
    :return: The dataframe with the file as a new column
    """
    pdflist = []
    with open(file, 'r') as infile:
        for line in infile:
            pdflist.append(line.strip())
    df['PDF_List'] = pdflist
    return df


def cull_rows(df, folder):
    """
    :param df: Dataframe with a folder referencing files
    :param folder: A folder with files located inside
    :return: A df that contains only the rows that reference files in the folder
    """
    filelist = []
    for file in (os.listdir(sys.path[0] + folder)):
        filelist.append(file[:-7] + 'pdf')
    droplist = []
    for i in range(len(df)):
        if df['PDF_List'][i] not in filelist:
            droplist.append(i)
    df = df.drop(index=droplist).reset_index()
    return df


def create_words_df(df, folder):
    """
    :param df: Dataframe with references to files
    :param folder: Folder location where the files are stored
    :return: A dataframe with words as columns and resolutions as rows with the values as the # of occurrences
    """
    word_df = pd.DataFrame()
    print(len(df))
    for i in range(len(df)):
        file = df['PDF_List'][i][:-3] + 'txt.txt'
        with open(folder + '\\' + file) as f:
            word_list = f.readline().split(',')
            if len(word_df.columns) != 0:
                word_df.loc[len(word_df.index)] = [0] * len(word_df.columns)
            else:
                word_df[word_list[0]] = [0]
            for word in word_list:
                if word not in word_df.columns:
                    word_df[word] = [0] * (len(word_df))
                word_df.at[i, word] += 1
        print(i)
    print(word_df)
    word_df.to_csv('word_lists.csv')


def main():
    un_votes_df = filter_columns('UN DATA.csv')
    un_votes_df = append_file(un_votes_df, 'FinalPDFList.txt')
    un_votes_df = cull_rows(un_votes_df, '\\CleanedText')
    # words_df = create_words_df(un_votes_df, '\\CleanedText')
    words_df = pd.read_csv('Final_Words_DF.csv')
    words_df = words_df.drop(labels='Unnamed: 0', axis=1)

    # Dropping duplicate rows
    words_df = words_df.drop_duplicates(keep=False)
    idx = list(words_df.index.values)
    missing = []
    for num in range(1, len(un_votes_df)):
        if num not in idx:
            missing.append(num)
    un_votes_df = un_votes_df.drop(missing, axis=0)
    un_votes_df = un_votes_df.reset_index(drop=True)
    droplist = []

    # Dropping words that appear less than 100 times or in less than 5 documents- the vast majority of these are errors
    for column in words_df.columns:
        if sum(words_df[column]) < 100:
            droplist.append(column)
        elif (words_df[column] != 0).sum() < 5:
            droplist.append(column)
    words_df = words_df.drop(columns=droplist)
    words_df = words_df.reset_index(drop=True)

    # Opens the file containing the list of countries in the current UN and initializes classes
    with open('uncountries.txt') as infile:
        country_list = []
        for country in infile:
            country_list.append(Country(words_df, country.strip(), un_votes_df[country.strip()].tolist()))
    # Each country votes on the file provided in country.vote -> this can be changed to any text file and it will run
    for country in country_list:
        country.vote('testresolution4.txt')
    print(VOTING_RECORD)
    print(f'Yes Votes: {len(VOTING_RECORD["Yes"])} \nNo Votes: {len(VOTING_RECORD["No"])}')


if __name__ == "__main__":
    main()
