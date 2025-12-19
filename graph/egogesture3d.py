import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 42

self_link = [(i, i) for i in range(num_node)]

single_hand_edges = [
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

left_hand_inward = single_hand_edges.copy()

right_hand_inward = [(i + 21, j + 21) for (i, j) in single_hand_edges]

inward_ori_index = left_hand_inward + right_hand_inward

inward = inward_ori_index
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward


class Graph:

    def __init__(self, labeling_mode='spatial', scale=1):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError(f"Unsupported labeling mode: {labeling_mode}")
        return A


if __name__ == '__main__':
    graph = Graph()
    print(f"Number of nodes: {graph.num_node}")
    print(f"Adjacency matrix shape: {graph.A.shape}")
    print(f"Number of inward edges: {len(graph.inward)}")
    print(f"Number of neighbor edges: {len(graph.neighbor)}")
