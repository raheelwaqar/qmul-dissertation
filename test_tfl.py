import unittest
import pandas as pd


from main import is_square_matrix, read_data, create_graph
from config import main_edge_file


class TflTestCases(unittest.TestCase):

    def test_is_square_matrix(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        self.assertTrue(is_square_matrix(df))

        df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])
        self.assertFalse(is_square_matrix(df))

    def test_read_data(self):
        nodes_df, edges_df = read_data(main_edge_file)
        self.assertTrue(len(nodes_df) > 0)
        self.assertTrue(len(edges_df) > 0)

    def test_create_graph(self):
        nodes_df = pd.DataFrame({"nodeID": [1, 2], "nodeLabel": ["A", "B"],
                                 "nodeLat": [0.0, 1.0], "nodeLong": [0.0, 1.0]})
        edges_df = pd.DataFrame({"layerID": [1], "nodeID1": [1], "nodeID2": [2], "weight": [0.5]})

        graph = create_graph(nodes_df, edges_df)
        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(len(graph.edges), 1)


if __name__ == '__main__':
    unittest.main()
