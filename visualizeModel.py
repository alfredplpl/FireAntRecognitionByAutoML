__author__ = "QQSong and Alfred Increment"
__version__ = "0.0.1"
__license__ = "Apache License 2.0"
#https://github.com/keras-team/autokeras/blob/0.4.0/examples/visualizations/visualize.py

import os
from graphviz import Digraph

from autokeras.utils import pickle_from_file
import Settings

def to_pdf(graph, path):
    dot = Digraph(comment='The Round Table')

    for index, node in enumerate(graph.node_list):
        dot.node(str(index), str(node.shape))

    for u in range(graph.n_nodes):
        for v, layer_id in graph.adj_list[u]:
            dot.edge(str(u), str(v), str(graph.layer_list[layer_id]))

    dot.render(path)


def visualize(path):
    cnn_module = pickle_from_file(os.path.join(path, 'module'))
    cnn_module.searcher.path = path
    for item in cnn_module.searcher.history:
        model_id = item['model_id']
        graph = cnn_module.searcher.load_model_by_id(model_id)
        to_pdf(graph, os.path.join(path, str(model_id)))


if __name__ == '__main__':
    visualize(Settings.TMP_PATH)