import networkx as nx
import matplotlib.pyplot as plt

# with the use networkx.
# Draw Spring layout.
# With Latitude AND LONGITUDE FIND THE POSITION
# POSITION for the stops should remain same.
# Multiplex network. These three layers are the part of multiplex network.
# centrality measure in notes
# Page rank is also important like Google use for ranking the pages.
# Distance notion. Station is important if it's distance is small with other stations.
#
# Define a function to read data from file


# Calculate centrality of different stations
# Highest centrality has rank 1 and then 2nd and so on
# Compare ranking in different situations. If there are some distruptions these ranking will be change let's say in case of a strike.


################# Next steps #########################
# With Latitude AND LONGITUDE FIND THE POSITION
# POSITION for the stops should remain same.
# chapter in centrality measure from notes.
# look all libraries in python
# test in 3 datasets
# provide ranking in list.

def read_data():
    with open("dataset/london_transport_multiplex.edges", "r") as file:
        # Read the contents of the file
        contents = file.read()

        # Split the contents into lines
        lines = contents.split("\n")

        # Split each line into values
        layer1 = []
        layer2 = []
        layer3 = []
        for line in lines:
            values = line.split(" ")
            # Categorize values into different layers based on the first value
            if values[0] == '1':
                layer1.append(values)
            elif values[0] == '2':
                layer2.append(values)
            else:
                layer3.append(values)
        # Return the values for each layer as a list of lists
        return [layer1, layer2, layer3]


# Define a function to create and print a graph based on data
def create_graph(data):
    G = nx.Graph()
    nodes = set()
    for path in data:
        n1 = path[1]
        n2 = path[2]
        nodes.add(n1)
        nodes.add(n2)
    for n in nodes:
        G.add_node(n)
    for path in data:
        n1 = path[1]
        n2 = path[2]
        w = path[3]
        G.add_edge(n1, n2, weight=w)

    # Return the graph object instead of printing it
    return G


if __name__ == '__main__':
    data_layers = read_data()
    # Create a 1x3 subplot figure
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 10))
    # For each layer of data, create and plot the corresponding graph in a subplot
    for i, layer in enumerate(data_layers):
        G = create_graph(layer)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, ax=axs[i])
        labels = nx.get_edge_attributes(G, "w")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=axs[i])
        axs[i].set_title(f"Layer {i + 1}")
    # Show the figure
    plt.show()
