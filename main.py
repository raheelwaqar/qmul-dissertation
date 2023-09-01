import json
import logging

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from scipy.cluster import hierarchy
from scipy.stats import kendalltau
from itertools import combinations
from config import main_edge_file, node_file, disruption_edge_files, kendalltau_matrix_output

# To show all rows and columns, adjust the display options:
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns


# Function to check whether a matrix is square or not
def is_square_matrix(df):
    # Check if it's a square matrix
    num_rows, num_cols = df.shape
    if num_rows == num_cols:
        print("The matrix is a square matrix.")
    else:
        print("The matrix is not a square matrix.")
    return num_rows == num_cols


# Function to read data from files
def read_data(edges_file):
    # Read nodes data from a file
    nodes_df = pd.read_csv(node_file, delimiter=" ")
    # Read edges data from the specified file and assign column names
    edges_df = pd.read_csv(edges_file, delimiter=" ", names=["layerID", "nodeID1", "nodeID2", 'weight'])
    return nodes_df, edges_df


# Function to create a graph from nodes and edges dataframes
def create_graph(nodes_df, layer_edges_df):
    G = nx.Graph()  # Create an empty graph
    # Iterate over each row in the nodes dataframe
    for _, row in nodes_df.iterrows():
        node_id = row['nodeID']
        node_label = row['nodeLabel']
        node_lat = row['nodeLat']
        node_long = row['nodeLong']
        # Add node to the graph if it is present in the edges dataframe
        if node_id in layer_edges_df['nodeID1'].unique() or node_id in layer_edges_df['nodeID2'].unique():
            G.add_node(node_id, label=node_label, pos=(node_lat, node_long))

    # Iterate over each row in the layer edges dataframe
    for _, row in layer_edges_df.iterrows():
        node1 = row['nodeID1']
        node2 = row['nodeID2']
        weight = row['weight']
        # Add an edge between node1 and node2 with the specified weight
        G.add_edge(node1, node2, weight=weight)

    return G


# Function to get the top n values from a centrality dictionary
def get_top_n_values(centrality, node_id_to_name, top20=True):
    degree_dict = dict()
    for node, value in centrality.items():
        node_name = node_id_to_name[node]
        degree_dict[node_name] = value
    sorted_degree = sorted(degree_dict.items(), key=lambda x: x[1])

    # Selecting the first 20 cities with the lowest temperatures if top20 is True
    if top20:
        return sorted_degree[-1 * 20:]
    return sorted_degree


# Function to calculate centrality measures for a given graph
def calculate_centrality(graph, nodes_df, layer_id, file_name, top20):
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

    result = {
        "file_name": file_name.split("/")[-1],
        "layer_id": int(layer_id),
        "centrality":
            {
                "degree": {"weighted": get_top_n_values(degree_centrality, node_id_to_name, top20),
                           "unweighted": get_top_n_values(degree_centrality, node_id_to_name, top20)},
                "closeness": {"weighted": get_top_n_values(weighted_closeness_centrality, node_id_to_name, top20),
                              "unweighted": get_top_n_values(unweighted_closeness_centrality, node_id_to_name, top20)},
                "betweenness": {"weighted": get_top_n_values(weighted_betweenness_centrality, node_id_to_name, top20),
                                "unweighted": get_top_n_values(unweighted_betweenness_centrality, node_id_to_name,
                                                               top20)},
                "pagerank": {"weighted": get_top_n_values(weighted_pagerank_centrality, node_id_to_name, top20),
                             "unweighted": get_top_n_values(unweighted_pagerank_centrality, node_id_to_name, top20)}
            }
    }

    return json.dumps(result)


def show_graph(nodes_df, edges_df):
    # Get unique layer IDs from the edges dataframe
    layers = edges_df['layerID'].unique()

    # Iterate over each layer in the edges dataframe
    for i, layer_id in enumerate(layers):
        # Filter edges dataframe to get edges for the current layer
        layer_edges_df = edges_df[edges_df['layerID'] == layer_id]

        # Create a graph using the nodes dataframe and layer-specific edges dataframe
        graph = create_graph(nodes_df, layer_edges_df)

        # Create a figure and axis for plotting the graph
        fig, ax = plt.subplots(figsize=(19, 10))
        ax.set_title(f'Layer {layer_id}')

        # Position nodes using the spring layout algorithm
        pos = nx.spring_layout(graph, seed=42)

        # Get edge labels and node labels for visualization
        edge_labels = nx.get_edge_attributes(graph, 'weight')
        node_labels = nx.get_node_attributes(graph, 'label')

        # Draw edges with transparency and edge labels
        nx.draw_networkx_edges(graph, pos, alpha=0.2, ax=ax)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', ax=ax)

        # Draw nodes with size and color
        nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='lightblue', ax=ax)
        nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=6, font_color='black', ax=ax)

        # Set the x and y limits of the plot
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.format_coord = lambda x, y: ""

        # Enable autoscaling and set margins
        ax.autoscale(enable=True)
        ax.margins(0.1)

        plt.show()


# Function to visualize graphs
def calculate_centrality_measure(nodes_df, edges_df, file_name, top20):
    # Get unique layer IDs from the edges dataframe
    layers = edges_df['layerID'].unique()
    centrality_list = list()  # List to store centrality data for each layer

    # Iterate over each layer in the edges dataframe
    for i, layer_id in enumerate(layers):
        # Filter edges dataframe to get edges for the current layer
        layer_edges_df = edges_df[edges_df['layerID'] == layer_id]

        # Create a graph using the nodes dataframe and layer-specific edges dataframe
        graph = create_graph(nodes_df, layer_edges_df)

        # Calculate centrality measures for the current layer and store the results in the centrality_list
        centrality_list.append(calculate_centrality(graph, nodes_df, layer_id, file_name, top20))

    return centrality_list  # Return the list of centrality data for each layer


def calculate_kendalltau(disruption_centrality_list, main_centrality_list):
    main_centrality_values = get_specific_centrality_values(main_centrality_list, centrality_type="betweenness",
                                                            is_weighted=True)
    stations = [item[0] for item in main_centrality_values]

    print("Main stations with centrality values", main_centrality_values)
    finalized_rank_dic = dict()
    main_ranks = list(range(20, 0, -1))
    finalized_rank_dic['main'] = main_ranks
    file_name = ""
    try:
        for individual_disruption in disruption_centrality_list:
            data = get_specific_centrality_values(individual_disruption, centrality_type="betweenness",
                                                  is_weighted=True)

            individual_disruption_data = json.loads(individual_disruption[0])
            file_name = individual_disruption_data.get("file_name")

            ranked_data = sorted(data, key=lambda x: x[1], reverse=True)
            ranks = [x + 1 for x in range(len(data))]
            ranked_data = [[station, rank] for (station, value), rank in zip(ranked_data, ranks)]

            ranked_dict = {station: rank for station, rank in ranked_data}
            disruption_ranks = [ranked_dict[station] for station in stations]

            if len(main_ranks) == len(disruption_ranks):
                file = file_name.split(".")[0].split("_")[-1]
                finalized_rank_dic[file] = disruption_ranks
    except KeyError as e:
        logging.exception("KeyError ", file_name, e)

    kendall_df = kendalltau_to_matrix(finalized_rank_dic)
    kendall_df.to_csv(kendalltau_matrix_output)
    return kendall_df


def kendalltau_to_matrix(finalized_rank_dic):
    pairs = combinations(finalized_rank_dic.keys(), 2)

    # Calculate Kendall's tau and p-value for each pair of keys and store the results in a list
    pair_kendall = [(1 - kendalltau(finalized_rank_dic[a], finalized_rank_dic[b]).statistic) / 2 for a, b in pairs]

    # Create a dictionary to store the values
    matrix_data = {}
    # Iterate over each pair of keys and corresponding Kendall's tau and p-value
    for pair, pair_kendall in zip(combinations(finalized_rank_dic.keys(), 2), pair_kendall):
        # Print the pair and its Kendall's tau
        file1, file2 = pair
        if file1 not in matrix_data:
            matrix_data[file1] = {}
        if file2 not in matrix_data:
            matrix_data[file2] = {}
        matrix_data[file1][file2] = pair_kendall
        matrix_data[file2][file1] = pair_kendall

    # Create a DataFrame from the matrix dictionary
    df = pd.DataFrame(matrix_data)
    return df.fillna(0)


def calculate_linkage(df):
    linkage_matrix = hierarchy.linkage(df.values, method='single', metric='euclidean')
    return linkage_matrix


def draw_dendrogram(df, linkage_matrix):
    # Plot the dendrogram using the linkage matrix
    plt.figure(figsize=(10, 6))
    hierarchy.dendrogram(linkage_matrix, labels=df.columns, leaf_font_size=10)
    plt.xlabel('Files')
    plt.ylabel('Distance')
    plt.title('Dendrogram')
    plt.show()


def get_specific_centrality_values(data, centrality_type, is_weighted):
    main_centrality_data = data[0]
    main_centrality_data = json.loads(main_centrality_data)
    centrality_values = main_centrality_data.get("centrality").get(centrality_type).get(
        "weighted" if is_weighted else "unweighted")
    return centrality_values


def calculate_disruption_centrality_measures(visualize_graph):
    # Initialize an empty list to store centrality data for each disruption edge file
    disruption_centrality_list = list()

    # Iterate over each disruption edge file
    for edge_file in disruption_edge_files:
        # Read data from the current disruption edge file
        disruption_nodes_df, disruption_edges_df = read_data(edge_file)

        show_graph(disruption_nodes_df, disruption_edges_df) if visualize_graph else None
        # Obtain centrality list for the current disruption edge file
        disruption_centrality_list.append(
            calculate_centrality_measure(disruption_nodes_df, disruption_edges_df, edge_file, top20=False))
    return disruption_centrality_list


# Main function
def main():
    # Read data from the main edge file
    nodes_df, edges_df = read_data(main_edge_file)

    show_graph(nodes_df, edges_df)
    # obtain centrality list for the main edge file
    main_centrality_list = calculate_centrality_measure(nodes_df, edges_df, main_edge_file, top20=True)

    disruption_centrality_list = calculate_disruption_centrality_measures(visualize_graph=False)

    kendall_df = calculate_kendalltau(disruption_centrality_list, main_centrality_list)

    # Linkage code works well with square matrices
    is_square_matrix(kendall_df)

    linkage_matrix = calculate_linkage(kendall_df)

    draw_dendrogram(kendall_df, linkage_matrix)


if __name__ == '__main__':
    main()
