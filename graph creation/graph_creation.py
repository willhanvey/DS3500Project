# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 02:13:54 2022

@author: alex
graph creation of country voting similarity
"""

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import datetime as dt
import re
from itertools import product
import sys


def main():
    data = pd.read_csv("../UN DATA.csv", engine='python')

    # They throw problems, manual inspection
    data.drop("COTE D’IVOIRE", axis=1, inplace=True)
    data.drop(" UNITED KINGDOM", axis=1, inplace=True)
    data.drop(" UNITED STATES", axis=1, inplace=True)
    data.drop(" CHILE", axis=1, inplace=True)
    data.drop(" SOMALIA", axis=1, inplace=True)
    data.drop(" ALGERIA", axis=1, inplace=True)
    data.drop(" BELGIUM", axis=1, inplace=True)
    data.drop("Aa UNITED STATES", axis=1, inplace=True)
    data.drop("AY UNION OF SOUTH AFRICA", axis=1, inplace=True)
    data.drop("AY DENMARK", axis=1, inplace=True)
    data.drop("AY SWEDEN", axis=1, inplace=True)

    # Keep only alphanumerical names of countries to then use as a mapping 
    colnames_country = {i: re.sub('[^A-Za-z0-9]+', '', i) for i in data.columns.tolist()}
    data = data.rename(columns=colnames_country)
    countries = data.columns[11:]

    data['Date'] = pd.to_datetime(data['Date'])
    if len(sys.argv)==2:
        data = data[data['Date'].dt.year >= int(sys.argv[1])]
    elif len(sys.argv)==3:
        data = data[(data['Date'].dt.year >= int(sys.argv[1])) & (data['Date'].dt.year <= int(sys.argv[2]))]

    # Create country-country DataFrame
    ctr_ctr = pd.DataFrame(list(product(countries, countries)))

    # Remove duplicated country pairs
    # https://stackoverflow.com/a/40475008
    ctr_ctr = pd.DataFrame(np.sort(ctr_ctr.values, axis=1), columns=ctr_ctr.columns).drop_duplicates()

    # Remove same country pairs
    # https://stackoverflow.com/a/43951580
    ctr_ctr = ctr_ctr[ctr_ctr[0] != ctr_ctr[1]].rename(columns={0: "countryA",1: "countryB"})

    # Get list of countries in use
    countryA = ctr_ctr.groupby(by="countryA").count().index.tolist()

    # Compute similarity
    ctr_ctr["similarity"] = np.nan
    for idx, cA in enumerate(tqdm(countryA)):
        for cB in countryA[idx+1:]:
            # Find data of 2 countries voting on the same resolution
            currentData = pd.concat([data[cA], data[cB]], axis=1).dropna()
            total = currentData[cA].count() # number of votes
            if total > 100:
                same = currentData[cA].eq(currentData[cB]).sum() # ocassions when they have the same vote
                ctr_ctr.loc[(ctr_ctr.countryA == cA) & (ctr_ctr.countryB == cB),'similarity'] = same/total

    ctr_ctr.dropna(inplace=True) # remove countries with non-relevant similarities

    # Map name of countries back to their original values
    colnames_country = {v: k for k, v in colnames_country.items()}
    colnames_country_func = lambda x: colnames_country[x] if x in colnames_country.keys() else x
    ctr_ctr['countryA'] = ctr_ctr['countryA'].map(colnames_country_func)
    ctr_ctr['countryB'] = ctr_ctr['countryB'].map(colnames_country_func)

    # save
    filename = "graph_similarity"
    if len(sys.argv)==2:
        filename += "_from_{}".format(int(sys.argv[1]))
    elif len(sys.argv)==3:
        filename += "_{}_{}".format(int(sys.argv[1]), int(sys.argv[2]))
    filename += ".csv"
    ctr_ctr.to_csv(filename, encoding='utf-8', index=False)


if __name__ == '__main__':
    main()

