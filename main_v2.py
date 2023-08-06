import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# sort top 20 and compare different centrality measure
# we need to play with weight. Approxi for the traffic
# calculate centrality of the distruptions on different days
# weighted degree, distance of a link will be the inverse of the weight.


# weighted measure
# run on typical day


def read_data():
    # Read node and edge data from CSV files
    nodes_df = pd.read_csv('dataset/london_transport_nodes.txt', delimiter=" ")
    edges_df = pd.read_csv('dataset/london_transport_multiplex.edges', delimiter=" ",
                           names=["layerID", "nodeID1", "nodeID2", 'weight'])
    return nodes_df, edges_df


def create_graph(nodes_df, layer_edges_df):
    # Create a graph object using NetworkX
    G = nx.Graph()

    # Add nodes to the graph based on the node data
    for _, row in nodes_df.iterrows():
        node_id = row['nodeID']
        node_lat = row['nodeLat']
        node_long = row['nodeLong']
        if node_id in layer_edges_df['nodeID1'].unique() or node_id in layer_edges_df['nodeID2'].unique():
            G.add_node(node_id, pos=(node_lat, node_long))

    # Add edges to the graph based on the edge data
    for _, row in layer_edges_df.iterrows():
        node1 = row['nodeID1']
        node2 = row['nodeID2']
        weight = row['weight']
        G.add_edge(node1, node2, weight=weight)

    return G


def visualize_graphs(nodes_df, edges_df):
    # Get unique layer IDs from the edge data
    layers = edges_df['layerID'].unique()
    num_layers = len(layers)

    # Create a mapping between node IDs and names
    node_id_to_name = nodes_df.set_index('nodeID')['nodeLabel'].to_dict()

    # Iterate over each layer
    for i, layer_id in enumerate(layers):
        # Filter edge data for the current layer
        layer_edges_df = edges_df[edges_df['layerID'] == layer_id]

        # Create a graph for the current layer
        G = create_graph(nodes_df, layer_edges_df)

        # Plot and visualize the graph
        fig, ax = plt.subplots(figsize=(15, 12))  # Increase the figure size for more space
        ax.set_title(f'Layer {layer_id}')

        # Adjust node positions using a spring layout algorithm
        pos = nx.spring_layout(G, seed=42)

        # Draw the graph with node and edge attributes
        nx.draw(G, pos, with_labels=True, node_size=400, node_color='lightblue', edge_color='gray', ax=ax)

        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        # closeness_centrality = nx.closeness_centrality(G)
        # betweenness_centrality = nx.betweenness_centrality(G)
        # eigenvector_centrality = nx.eigenvector_centrality(G)
        # pagerank_centrality = nx.pagerank(G)

        # # Print centrality measures for each node
        print("Degree Centrality:")
        for node, centrality in degree_centrality.items():
            node_name = node_id_to_name[node]
            print(f"Node {node_name}: {centrality}")
        #
        # print("\nCloseness Centrality:")
        # for node, centrality in closeness_centrality.items():
        #     node_name = node_id_to_name[node]
        #     print(f"Node {node_name}: {centrality}")
        #
        # print("\nBetweenness Centrality:")
        # for node, centrality in betweenness_centrality.items():
        #     node_name = node_id_to_name[node]
        #     print(f"Node {node_name}: {centrality}")
        #
        # print("\nEigenvector Centrality:")
        # for node, centrality in eigenvector_centrality.items():
        #     node_name = node_id_to_name[node]
        #     print(f"Node {node_name}: {centrality}")

        # print("\nPageRank Centrality:")
        # for node, centrality in pagerank_centrality.items():
        #     node_name = node_id_to_name[node]
        #     print(f"{node_name}: {centrality}")

        plt.show()


# Read data from CSV files
nodes_df, edges_df = read_data()

# Call the function to visualize the graphs and calculate centrality measures
visualize_graphs(nodes_df, edges_df)
