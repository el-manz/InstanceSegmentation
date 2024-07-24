import math
import pytest
from losses import HaloLoss

def test_preliminary_steps():

    predicted = [[[[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 1, 0, 1, 1, 1]],
                 [[0.7, 1, 0.8, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0.4, 1, 0, 0, 0.9]],
                 [[1, 0.5, 0.9, 0, 0, 0.8], [0.2, 1, 0.3, 0, 0, 0], [0.7, 0.1, 0.4, 0.5, 0, 0.05]]]]
    labels = [[[1, 2, 1, 0, 0, 0], [0, 2, 0, 0, 0, 0], [1, 2, 1, 0, 0, 1]]]
    neg_weight = 0.3
    eps = 0.05
    max_distance = 1

    item = HaloLoss(predicted, labels, neg_weight, eps, max_distance)

    halos = item.make_halo(item.components_batch[0])

    true_halos = [[(0, 1), (1, 0)],
                  [(0, 0), (0, 2), (1, 0), (1, 2), (2, 0), (2, 2)],
                  [(0, 1), (0, 3), (1, 2)],
                  [(1, 0), (2, 1)],
                  [(1, 2), (2, 1), (2, 3)],
                  [(1, 5), (2, 4)]]
    
    #test finding halos
    assert len(item.components_batch[0]) == len(halos)
    assert halos == true_halos

    recolored_components = item.recoloring_stage(item.predicted[0], item.components_batch[0], halos)

    true_recolored = [2, 2, 2, 1, 1, 1]

    # test recoloring components
    assert len(recolored_components) == len(item.components_batch[0])
    assert recolored_components == true_recolored

def test_compute_loss():

    predicted = [[[[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 1, 0, 1, 1, 1]],
                 [[0.7, 1, 0.8, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0.4, 1, 0, 0, 0.9]],
                 [[1, 0.5, 0.9, 0, 0, 0.8], [0.2, 1, 0.3, 0, 0, 0], [0.7, 0.1, 0.4, 0.5, 0, 0.05]]]]
    labels = [[[1, 2, 1, 0, 0, 0], [0, 2, 0, 0, 0, 0], [1, 2, 1, 0, 0, 1]]]
    neg_weight = 0.3
    eps = 0.05
    max_distance = 1

    item = HaloLoss(predicted, labels, neg_weight, eps, max_distance)

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