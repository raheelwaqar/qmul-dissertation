import csv
import json

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd


def read_data(edges_file):
    nodes_df = pd.read_csv('dataset/london_transport_nodes.txt', delimiter=" ")
    edges_df = pd.read_csv(edges_file, delimiter=" ",
                           names=["layerID", "nodeID1", "nodeID2", 'weight'])
    return nodes_df, edges_df


def create_graph(nodes_df, layer_edges_df):
    G = nx.Graph()
    for _, row in nodes_df.iterrows():
        node_id = row['nodeID']
        node_label = row['nodeLabel']
        node_lat = row['nodeLat']
        node_long = row['nodeLong']
        if node_id in layer_edges_df['nodeID1'].unique() or node_id in layer_edges_df['nodeID2'].unique():
            G.add_node(node_id, label=node_label, pos=(node_lat, node_long))

    for _, row in layer_edges_df.iterrows():
        node1 = row['nodeID1']
        node2 = row['nodeID2']
        weight = row['weight']
        G.add_edge(node1, node2, weight=weight)

    return G


def get_top_n_values(centrality, node_id_to_name, n=20):
    degree_dict = dict()
    for node, value in centrality.items():
        node_name = node_id_to_name[node]
        degree_dict[node_name] = value
    sorted_degree = sorted(degree_dict.items(), key=lambda x: x[1])

    # Selecting the first 20 cities with the lowest temperatures
    return sorted_degree[:n]


def calculate_centrality(graph, nodes_df, layer_id, file_name):
    node_id_to_name = nodes_df.set_index('nodeID')['nodeLabel'].to_dict()
    degree_centrality = nx.degree_centrality(graph)
    ## Weighted Centrality
    weighted_closeness_centrality = nx.closeness_centrality(graph, distance='weight')
    weighted_betweenness_centrality = nx.betweenness_centrality(graph, weight='weight')
    weighted_pagerank_centrality = nx.pagerank(graph, weight='weight')
    ## UN-Weighted Centrality
    unweighted_closeness_centrality = nx.closeness_centrality(graph)
    unweighted_betweenness_centrality = nx.betweenness_centrality(graph)
    unweighted_pagerank_centrality = nx.pagerank(graph)

    return {
        "file_name": file_name.split("/")[-1],
        "layer_id": int(layer_id),
        "centrality":
            {
                "degree": {"weighted": get_top_n_values(degree_centrality, node_id_to_name),
                           "unweighted": get_top_n_values(degree_centrality, node_id_to_name)},
                "closeness": {"weighted": get_top_n_values(weighted_closeness_centrality, node_id_to_name),
                              "unweighted": get_top_n_values(unweighted_closeness_centrality, node_id_to_name)},
                "betweenness": {"weighted": get_top_n_values(weighted_betweenness_centrality, node_id_to_name),
                                "unweighted": get_top_n_values(unweighted_betweenness_centrality, node_id_to_name)},
                "pagerank": {"weighted": get_top_n_values(weighted_pagerank_centrality, node_id_to_name),
                             "unweighted": get_top_n_values(unweighted_pagerank_centrality, node_id_to_name)}
            }
    }


def visualize_graphs(nodes_df, edges_df, file_name, show_graph=True):
    layers = edges_df['layerID'].unique()
    centrality_list = list()
    for i, layer_id in enumerate(layers):
        layer_edges_df = edges_df[edges_df['layerID'] == layer_id]
        graph = create_graph(nodes_df, layer_edges_df)

        fig, ax = plt.subplots(figsize=(19, 10))
        ax.set_title(f'Layer {layer_id}')

        pos = nx.spring_layout(graph, seed=42)

        edge_labels = nx.get_edge_attributes(graph, 'weight')
        node_labels = nx.get_node_attributes(graph, 'label')

        nx.draw_networkx_edges(graph, pos, alpha=0.2, ax=ax)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', ax=ax)

        nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='lightblue', ax=ax)
        nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=6, font_color='black', ax=ax)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.format_coord = lambda x, y: ""

        ax.autoscale(enable=True)
        ax.margins(0.1)

        # print("\n\n************** Layer: " + str(layer_id) + " Centrality  **************")
        centrality_list.append(calculate_centrality(graph, nodes_df, layer_id, file_name))
        # print(data)
        if show_graph:
            plt.show()
    return centrality_list


def main():
    main_edge_file = 'dataset/london_transport_multiplex.edges'
    nodes_df, edges_df = read_data(main_edge_file)
    main_centrality_list = visualize_graphs(nodes_df, edges_df, main_edge_file, True)
    print(main_centrality_list)

    # Define the column names for the CSV files

    disruption_edge_files = ['dataset/Disruptions/london_transport_multiplex_DISR1.edges',
                             'dataset/Disruptions/london_transport_multiplex_DISR100.edges',
                             'dataset/Disruptions/london_transport_multiplex_DISR150.edges',
                             'dataset/Disruptions/london_transport_multiplex_DISR200.edges',
                             'dataset/Disruptions/london_transport_multiplex_DISR346.edges']
    disruption_centrality_list = list()
    for edge_file in disruption_edge_files:
        disruption_nodes_df, disruption_edges_df = read_data(edge_file)
        disruption_centrality_list.append(visualize_graphs(disruption_nodes_df, disruption_edges_df, edge_file, False))


    print(disruption_centrality_list)



main()
####################################
