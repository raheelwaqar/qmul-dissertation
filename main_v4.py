import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from scipy.stats import stats
from itertools import combinations


# Function to read data from files
def read_data(edges_file):
    # Read nodes data from a file
    nodes_df = pd.read_csv('dataset/london_transport_nodes.txt', delimiter=" ")
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

    return {
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


# Function to visualize graphs
def visualize_graphs(nodes_df, edges_df, file_name, top20, show_graph=True):
    # Get unique layer IDs from the edges dataframe
    layers = edges_df['layerID'].unique()
    centrality_list = list()  # List to store centrality data for each layer

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

        # Calculate centrality measures for the current layer and store the results in the centrality_list
        centrality_list.append(calculate_centrality(graph, nodes_df, layer_id, file_name, top20))

        # Show the graph if show_graph is True
        if show_graph:
            plt.show()

    return centrality_list  # Return the list of centrality data for each layer


# Function to get key-value centrality dictionary
def get_key_value_centrality(data, disruptions=False):
    # Check if disruptions flag is False
    if not disruptions:
        # Access the weighted closeness centrality data
        data = data['centrality']['closeness']['weighted']

    result_dict = dict()  # Dictionary to store key-value centrality pairs

    # Iterate over the data
    for i, v in enumerate(data):
        # Extract the node ID (key) and centrality value (value)
        result_dict[v[0]] = v[1]

    return result_dict  # Return the dictionary of key-value centrality pairs


# Function to calculate ranks based on centrality values
def get_ranks(data):
    # Extract the values from the dictionary
    values = list(data.values())

    # Find the index of the highest value
    max_index = values.index(max(values))

    # Calculate the ranks using rankdata from scipy.stats
    ranks = stats.rankdata([-val for val in values])

    # Create a new dictionary with ranks
    data_with_ranks = {key: rank for key, rank in zip(data.keys(), ranks)}

    return data_with_ranks


# Function to extract centrality data for specific stations from disruptions
def extract_stations(disruption_centrality_list, stations):
    # Initialize an empty list to store the extracted data
    extracted_data = []

    # Iterate over each inner list in the disruption centrality list
    for inner_list in disruption_centrality_list:
        # Extract the dictionary from the inner list
        dictionary = inner_list[0]

        # Extract the centrality dictionary from the main dictionary
        centrality_dict = dictionary['centrality']

        # Iterate over the keys in the centrality dictionary
        for key in centrality_dict.keys():
            # Check if the key is 'closeness'
            if key == 'closeness':
                # Extract the weighted values from the closeness centrality
                weighted_values = centrality_dict[key]['weighted']

                # Filter the weighted values based on the stations
                filtered_values = [item for item in weighted_values if item[0] in stations]

                # Create a new dictionary entry with the file name as the key and the filtered values as the value
                extracted_data.append({dictionary['file_name']: filtered_values})

    # Return the extracted data list
    return extracted_data


# Function to calculate Kendall's tau and p-value between two datasets
def calculate_kendalltau(dataset1, dataset2):
    # Convert the values of dataset1 to a list of ranks
    ranks1 = list(dataset1.values())

    # Convert the values of dataset2 to a list of ranks
    ranks2 = list(dataset2.values())

    # Calculate Kendall's tau and p-value using the ranks
    tau, p_value = stats.kendalltau(ranks1, ranks2)

    # Return Kendall's tau and p-value
    return tau, p_value


# Function to calculate ranks for extracted disruptions
def calculate_ranks(extracted_disruptions_list):
    # Create an empty dictionary to store the ranks
    rank_dic = dict()

    # Iterate over each extracted disruptions station in the list
    for extracted_disruptions_station in extracted_disruptions_list:
        # Get the key (file_name) from the extracted disruptions station dictionary
        key = list(extracted_disruptions_station.keys())[0]

        # Get the key-value centrality dictionary for the disruptions station with disruptions=True
        disruption_key_value_centrality = get_key_value_centrality(extracted_disruptions_station[key], True)

        # Calculate ranks for the key-value centrality dictionary
        disruption_ranks = get_ranks(disruption_key_value_centrality)

        # Store the calculated ranks in the rank_dic dictionary with the key as the file_name
        rank_dic[key] = disruption_ranks

    # Return the dictionary containing the calculated ranks
    return rank_dic


# Main function
def main():
    # Path to the main edge file
    main_edge_file = 'dataset/london_transport_multiplex.edges'

    # Read data from the main edge file
    nodes_df, edges_df = read_data(main_edge_file)

    # Visualize graphs and obtain centrality list for the main edge file
    main_centrality_list = visualize_graphs(nodes_df, edges_df, main_edge_file, True, False)

    # Obtain key-value centrality dictionary from the first element of the main centrality list
    key_value_centrality = get_key_value_centrality(main_centrality_list[0])

    # Obtain the top 20 stations from the key-value centrality dictionary
    top_20_stations = list(key_value_centrality.keys())

    # Calculate ranks for the key-value centrality dictionary
    main_ranks = get_ranks(key_value_centrality)

    print(main_ranks)

    # List of disruption edge files
    disruption_edge_files = ['dataset/Disruptions/london_transport_multiplex_DISR1.edges',
                             'dataset/Disruptions/london_transport_multiplex_DISR100.edges',
                             'dataset/Disruptions/london_transport_multiplex_DISR200.edges',
                             'dataset/Disruptions/london_transport_multiplex_DISR220.edges']

    # Initialize an empty list to store centrality data for each disruption edge file
    disruption_centrality_list = list()

    # Iterate over each disruption edge file
    for edge_file in disruption_edge_files:
        # Read data from the current disruption edge file
        disruption_nodes_df, disruption_edges_df = read_data(edge_file)

        # Visualize graphs and obtain centrality list for the current disruption edge file
        disruption_centrality_list.append(
            visualize_graphs(disruption_nodes_df, disruption_edges_df, edge_file, False, False))

    # Extract centrality data for the top 20 stations from the disruption centrality list
    extracted_disruptions_list = extract_stations(disruption_centrality_list, top_20_stations)

    # Calculate ranks for each extracted disruptions station and store the ranks in a dictionary
    rank_dic = calculate_ranks(extracted_disruptions_list)

    print(rank_dic)

    # Add the main ranks to the rank_dic dictionary with the key as the main edge file
    rank_dic['london_transport_multiplex.edges'] = main_ranks

    # Generate pairs of keys from rank_dic dictionary
    pairs = combinations(rank_dic.keys(), 2)

    # Calculate Kendall's tau and p-value for each pair of keys and store the results in a list
    pair_kendall = [calculate_kendalltau(rank_dic[a], rank_dic[b]) for a, b in pairs]

    # Iterate over each pair of keys and corresponding Kendall's tau and p-value
    for pair, pair_kendall in zip(combinations(rank_dic.keys(), 2), pair_kendall):
        # Print the pair and its Kendall's tau
        print(f"Kendall of {pair}: {pair_kendall}")


main()

# # kendall distance? Clustering, dendrogram?
# # create matrix
#
#                                         london_transport_multiplex_DISR100.edges
# london_transport_multiplex_DISR1.edges              1