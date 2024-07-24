import pytest
from prepareimage import PrepareImage

def test():
    image = [[[1, 2, 1, 0, 0, 0], [0, 2, 0, 0, 0, 0], [1, 2, 1, 0, 0, 1]]]
    item = PrepareImage(image)
    item.prepare_objects()
    true_graphs = [{(0, 0): [],
                    (0, 1): [(1, 1)],
                    (0, 2): [],
                    (1, 1): [(0, 1), (2, 1)],
                    (2, 0): [],
                    (2, 1): [(1, 1)],
                    (2, 2): [],
                    (2, 5): []}]
    true_components = [[[(0, 0)],
                       [(0, 1), (1, 1), (2, 1)],
                       [(0, 2)],
                       [(2, 0)],
                       [(2, 2)],
                       [(2, 5)]]]
    assert item.graph_batch == true_graphs
    assert item.components_batch == true_components