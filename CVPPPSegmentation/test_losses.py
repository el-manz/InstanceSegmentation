import pytest
from losses import HaloLoss

def test_recoloring():

    predicted = [[[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 1, 0, 1, 1, 1]],
                 [[0.7, 1, 0.8, 0, 0, 0], [0, 0.005, 0, 0, 0, 0], [1, 0.45, 1, 0, 0, 1]]]
    labels = [[[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 1, 0, 1, 1, 2]],
              [[1, 2, 1, 0, 0, 0], [0, 2, 0, 0, 0, 0], [1, 2, 1, 0, 0, 1]]]
    neg_weight = 0 # not used in test
    size_threshold = 30 # not used in test

    item = HaloLoss(predicted, labels, neg_weight, size_threshold)

    item.make_graphs()

    true_graphs = [{(0, 2): [(1, 2)],
                    (1, 2): [(0, 2)],
                    (2, 0): [(2, 1)],
                    (2, 1): [(2, 0)],
                    (2, 3): [(2, 4)],
                    (2, 4): [(2, 3)],
                    (2, 5): []},
                   {(0, 0): [],
                    (0, 1): [(1, 1)],
                    (0, 2): [],
                    (1, 1): [(0, 1), (2, 1)],
                    (2, 0): [],
                    (2, 1): [(1, 1)],
                    (2, 2): [],
                    (2, 5): []}]
    
    item.find_objects()
    
    true_components = [[[(0, 2), (1, 2)], [(2, 0), (2, 1)], [(2, 3), (2, 4)], [(2, 5)]],
                       [[(0, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2)], [(2, 0)], [(2, 2)], [(2, 5)]]]
    
    # test making graph
    assert item.graphs == true_graphs

    # test finding objects
    assert item.components == true_components