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

NOT_COUNTRY = [
	"Human Development",
	"Regions",
	"Arab States",
	"Developing Countries",
	"East Asia and the Pacific",
	"Europe and Central Asia",
	"High human development",
	"Latin America and the Caribbean",
	"Least Developed Countries",
	"Low human development",
	"Medium human development",
	"Organization for Economic Co-operation and Development",
	"Small Island Developing States",
	"South Asia",
	"Sub-Saharan Africa",
	"Very high human development",
	"World"
]

def main():
	hdi = pd.read_csv("../HDI.csv", encoding='latin-1')[["Country", "2019"]].\
		rename(columns={"2019": "hdi"}).sort_values(by=["hdi"], ascending=False)
	hdi = hdi[hdi["Country"].isin(NOT_COUNTRY) == False]
	data = pd.read_csv(sys.argv[1])
	countries = sorted(data.countryA.unique())
	hdi_countries = sorted(hdi["Country"].unique())
	world_map = {
		"CONGO": "Congo (Democratic Republic of the)",
		"CONGO": "Côte d'Ivoire",
		"ESWATINI": "Eswatini (Kingdom of)",
		"REPUBLIC OF KOREA": "Korea (Republic of)",
		"REPUBLIC OF MOLDOVA": "Moldova (Republic of)",
		"UNITED REPUBLIC OF TANZANIA": "Tanzania (United Republic of)"
	}
	for i in hdi_countries:
		if i[1:].upper() in countries:
			world_map[i[1:].upper()] = i

	hdi = hdi[hdi["Country"].isin(list(world_map.values())) == True]
	if len(sys.argv)==3:
		if int(sys.argv[2])*2 > len(hdi):
			sys.stdout.write("number too big\n")
			return
		hdi_top = hdi.head(int(sys.argv[2]))
		hdi_bot = hdi.tail(int(sys.argv[2]))
		hdi = pd.concat([hdi_top, hdi_bot])


	world_map_func = lambda x: world_map[x] if x in world_map.keys() else x
	data['countryA'] = data['countryA'].map(world_map_func)
	data['countryB'] = data['countryB'].map(world_map_func)
	data['similarity'] = np.around(data['similarity'], 3)
	country_list = list(hdi["Country"])
	data = data[(data["countryA"].isin(country_list)) & (data["countryB"].isin(country_list))]
	G = nx.from_pandas_edgelist(data,source="countryA", target="countryB",edge_attr="similarity")

	#https://stackoverflow.com/a/62936131
	"""
	
	"""
	similarity = nx.get_edge_attributes(G, 'similarity')
	nodelist = G.nodes()
	plt.figure(figsize=(12,8))
	pos = nx.spring_layout(G, weight='similarity')
	nx.draw_networkx_nodes(G,pos,
						   nodelist=nodelist,
						   node_size=3000,
						   node_color='lightgray',
						   alpha=0.7)
	nx.draw_networkx_edges(G,pos,
						   edgelist = similarity.keys(),
						   width=list(map(lambda x: x*20, list(similarity.values()))),
						   edge_color='lightblue',
						   alpha=0.6)
	nx.draw_networkx_labels(G, pos=pos,
							labels=dict(zip(nodelist,nodelist)),
							font_color='black')
	# nx.draw_networkx_edge_labels(G,pos,edge_labels=similarity) # Edge labels
	plt.box(False)
	plt.show()


if __name__ == '__main__':
	if len(sys.argv) in [2,3]:
		main()
	else:
		sys.stdout.write("USE python graph_hdi_viz.py <graph file> <number of top/bottom countries to use>\n")

