import math
import pytest
from losses import HaloLoss

def test_loss():

    predicted = [[[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 1, 0, 1, 1, 1]],
                 [[0.7, 1, 0.8, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0.4, 1, 0, 0, 0.9]],
                 [[1, 0.5, 0.9, 0, 0, 0.8], [0.2, 1, 0.3, 0, 0, 0], [0.7, 0.1, 0.4, 0.5, 0, 0.05]]]
    labels = [[1, 2, 1, 0, 0, 0], [0, 2, 0, 0, 0, 0], [1, 2, 1, 0, 0, 1]]
    neg_weight = 0.3

    item = HaloLoss(predicted, labels, neg_weight)

    item.make_graphs()

    true_graph = {(0, 0): [],
                    (0, 1): [(1, 1)],
                    (0, 2): [],
                    (1, 1): [(0, 1), (2, 1)],
                    (2, 0): [],
                    (2, 1): [(1, 1)],
                    (2, 2): [],
                    (2, 5): []}
    
    # test making graph
    assert item.graph == true_graph
    
    item.find_objects()
    
    true_components = [[(0, 0)],
                       [(0, 1), (1, 1), (2, 1)],
                       [(0, 2)],
                       [(2, 0)],
                       [(2, 2)],
                       [(2, 5)]]

    # test finding objects
    assert item.components == true_components

    item.make_halo(max_distance = 1)

    true_halos = [[(0, 1), (1, 0)],
                  [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2)],
                  [(0, 1), (0, 3), (1, 2)],
                  [(1, 0), (2, 1)],
                  [(1, 2), (2, 1), (2, 3)],
                  [(1, 5), (2, 4)]]
    
    #test finding halos
    assert len(item.components) == len(item.halos)
    assert item.halos == true_halos

    item.recoloring_stage()

    true_recolored = [2, 2, 2, 1, 1, 1]

    # test recoloring components
    assert len(item.recolored_components) == len(item.components)
    assert item.recolored_components == true_recolored

    loss = item.compute_loss()

    true_loss = (math.log(0.95)) +\
                (math.log(0.5) + math.log(0.95) + math.log(0.1)) / 3 +\
                (math.log(0.9)) +\
                (math.log(0.95)) +\
                (math.log(0.95)) +\
                (math.log(0.9)) +\
                (7 * math.log(0.05) + 3 * math.log(0.95))
    true_loss *= -1
    
    assert loss == true_loss