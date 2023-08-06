import networkx as nx
import matplotlib.pyplot as plt

# Define a function to read data from file
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
def create_print_graph(data):
    # Create an empty graph object
    G = nx.Graph()

    # Add nodes
    nodes = set()
    for path in data:
        n1 = path[1]
        n2 = path[2]
        nodes.add(n1)
        nodes.add(n2)
    for n in nodes:
        G.add_node(n)

    # Add edges with weights
    for path in data:
        n1 = path[1]
        n2 = path[2]
        w = path[3]
        G.add_edge(n1, n2, weight=w)

    # Print some basic information about the graph
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    # Draw the graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, "w")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()


if __name__ == '__main__':
    # Read data from file and return it as a list of lists
    data_layers = read_data()
    print(data_layers)
    # Create and print a graph for each layer of data
    for data in data_layers:
        create_print_graph(data)
