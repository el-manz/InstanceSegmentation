import numpy as np
import pytest
from inference import Inference

def test_preliminary_steps():

    predicted = [[[[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 1, 0, 1, 1, 1]],
                 [[0.7, 1, 0.8, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0.4, 1, 0, 0, 0.9]],
                 [[1, 0.5, 0.9, 0, 0, 0.8], [0.2, 1, 0.3, 0, 0, 0], [0.7, 0.1, 0.4, 0.5, 0, 0.05]]]]
    size_threshold = 3
    proximity_threshold = 1

    item = Inference(predicted, size_threshold, proximity_threshold)

    # test reassigning colors
    true_reassigned = [[[2, 1, 0, 0, 0, 2], [2, 2, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]]
    assert item.reassigned_colors == true_reassigned

    # test filtering components by size_threshold
    components_filtered = item.filter_small_objets(item.components_batch[0])
    true_components_filtered = [[(0, 0), (1, 0), (1, 1)]]
    assert components_filtered == true_components_filtered

    # test merging objects
    final_components = item.merge_close_objects(item.reassigned_colors[0], components_filtered)
    true_final = [[(0, 0), (1, 0), (1, 1)]]
    assert final_components == true_final

    # replace components_filtered to test several components case
    replaced_components_filtered = [[(0, 0), (1, 0), (1, 1)],
                                    [(0, 2), (1, 2), (1, 3)],
                                    [(1, 5), (2, 5)]]
    replaced_reassigned = [[[2, 2, 2, 0, 0, 2], [2, 2, 2, 0, 0, 0], [0, 0, 1, 0, 0, 0]]]
    item.proximity_threshold = 2
    final_components = item.merge_close_objects(replaced_reassigned[0], replaced_components_filtered)
    true_replaced_final = [[(0, 0), (1, 0), (1, 1), (0, 2), (1, 2), (1, 3)],
                           [(1, 5), (2, 5)]]
    assert final_components == true_replaced_final


def test_compute_inference():
    predicted = [[[[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 1, 0, 1, 1, 1]],
                 [[0.7, 1, 0.8, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0.4, 1, 0, 0, 0.9]],
                 [[1, 0.5, 0.9, 0, 0, 0.8], [0.2, 1, 0.3, 0, 0, 0], [0.7, 0.1, 0.4, 0.5, 0, 0.05]]]]
    size_threshold = 3
    proximity_threshold = 1

    item = Inference(predicted, size_threshold, proximity_threshold)

    # test all pipeline
    result = item.compute_inference()
    true_final = [[(0, 0), (1, 0), (1, 1)]]
    assert result == [true_final]