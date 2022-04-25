import pandas as pd


def trim_df(df, num):
    """
    :param df: A dataframe
    :param num: An integer
    :return: A dataframe with all words with less than num occurrences removed
    """
    droplist = []
    for column in df.columns:
        if df[column].sum() < num:
            droplist.append(column)
    df = df.drop(droplist, axis=1)
    return df


def merge_dfs(df, file):
    """
    :param df: A df to be merged
    :param file: A csv file containing a second df
    :return: A merged dataframe
    """
    df2 = pd.read_csv(file)
    df = pd.concat([df, df2], axis=0, ignore_index=True)
    return df


def main():
    # Merges the five word dataframes
    df = pd.read_csv('filteredwordlist1.csv')
    df = merge_dfs(df, 'filteredwordlist2.csv')
    df = merge_dfs(df, 'filteredwordlist3.csv')
    df = merge_dfs(df, 'filteredwordlist4.csv')
    df = merge_dfs(df, 'filteredwordlist5.csv')
    # Replaces all occurrences of NA with 0 and removes words that appear less than 50 times
    df = df.fillna(0)
    df = trim_df(df, 50)
    # Writes final df to a csv
    df.to_csv('Final_Words_DF.csv')


if __name__ == "__main__":
    main()
