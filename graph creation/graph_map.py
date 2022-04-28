# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 02:13:54 2022

@author: alex
text analysis of UN resolutions
"""
import geopandas
import matplotlib.pyplot as plt
import community
import pandas as pd
import networkx as nx
import numpy as np
import sys


def main():
	world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
	data = pd.read_csv(sys.argv[1])
	count = 0
	countries = sorted(data.countryA.unique())
	# print(world['name'].to_string())
	world_map = {
		"BOSNIA AND HERZEGOVINA": "Bosnia and Herz.",
		"BRUNEI DARUSSALAM": "Brunei",
		"COTE D'IVOIRE": "Côte d'Ivoire",
		"DEMOCRATIC REPUBLIC OF THE CONGO": "Dem. Rep. Congo",
		"DOMINICAN REPUBLIC": "Dominican Rep.",
		"EQUATORIAL GUINEA": "Eq. Guinea",
		"LAO PEOPLE'S DEMOCRATIC REPUBLIC": "Laos",
		"NORTH MACEDONIA": "Macedonia",
		"DEMOCRATIC PEOPLE'S REPUBLIC OF KOREA": "North Korea",
		"RUSSIAN FEDERATION": "Russia",
		"SOUTH SUDAN": "S. Sudan",
		"SOLOMON ISLANDS": "Solomon Is.",
		"REPUBLIC OF KOREA": "South Korea",
		"UNITED REPUBLIC OF TANZANIA": "Tanzania",
		"UNITED STATES": "United States of America",
		"VIET NAM": "Vietnam",
		"USSR": "Russia",
		"YUGOSLAVIA": "Bosnia and Herz."
	}
	for i in world.name.sort_values().tolist():
		if i.upper() in countries:
			world_map[i.upper()] = i

	world_map_func = lambda x: world_map[x] if x in world_map.keys() else x

	data['countryA'] = data['countryA'].map(world_map_func)
	data['countryB'] = data['countryB'].map(world_map_func)
	#world.plot()
	G = nx.from_pandas_edgelist(data,source="countryA", target="countryB",edge_attr="similarity")
	comms = community.best_partition(G, weight="similarity")

	comms_data = {"country": [], "cluster": []}
	for country, cluster in comms.items():
		comms_data["country"].append(country)
		comms_data["cluster"].append(cluster)

	comms_data = pd.DataFrame.from_dict(comms_data)
	world["cluster"] = np.nan
	print(countries)
	for i in world_map.values():
		cl = comms_data.loc[comms_data.country==i, "cluster"]
		if not cl.empty:
			world.loc[world.name==i, "cluster"] = cl.item()

	world.plot(column='cluster');
	plt.title(sys.argv[1])
	plt.show()

if __name__ == '__main__':
		main()

