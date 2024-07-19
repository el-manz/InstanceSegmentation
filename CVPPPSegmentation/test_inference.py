import pytest
from inference import Inference

def test_inference():

    predicted = [[[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 1, 0, 1, 1, 1]],
                 [[0.7, 1, 0.8, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0.4, 1, 0, 0, 0.9]],
                 [[1, 0.5, 0.9, 0, 0, 0.8], [0.2, 1, 0.3, 0, 0, 0], [0.7, 0.1, 0.4, 0.5, 0, 0.05]]]
    size_threshold = 3
    proximity_threshold = 1

    item = Inference(predicted, size_threshold, proximity_threshold)

    item.assign_most_probable()

    true_reassigned = [[2, 1, 0, 0, 0, 2], [2, 2, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]

    # test reassigning colors
    assert item.reassigned_colors == true_reassigned

    item.make_graphs()
    item.find_objects()

    true_components = [[(0, 0), (1, 0), (1, 1)],
                  [(0, 1)],
                  [(0, 5)],
                  [(2, 2)]]
    
    # test finding object
    assert item.components == true_components

    item.filter_small_objets()

    true_components_filtered = [[(0, 0), (1, 0), (1, 1)]]

    # test filtering components by size_threshold
    assert item.components_filtered == true_components_filtered

    item.merge_close_objects()

    true_final = [[(0, 0), (1, 0), (1, 1)]]

    # test merging objects
    assert item.final_components == true_final

    # replace components_filtered to test several components case
    replaced_components_filtered = [[(0, 0), (1, 0), (1, 1)],
                                    [(0, 2), (1, 2), (1, 3)],
                                    [(1, 5), (2, 5)]]
    replaced_reassigned = [[2, 2, 2, 0, 0, 2], [2, 2, 2, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
    item.components_filtered = replaced_components_filtered
    item.reassigned_colors = replaced_reassigned
    item.proximity_threshold = 2
    item.merge_close_objects()
    
    true_replaced_final = [[(0, 0), (1, 0), (1, 1), (0, 2), (1, 2), (1, 3)],
                           [(1, 5), (2, 5)]]
    
    assert item.final_components == true_replaced_final