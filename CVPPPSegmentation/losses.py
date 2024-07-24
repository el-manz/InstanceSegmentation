import math
import numpy as np
import torch
import torch.nn.functional as F

from prepareimage import PrepareImage

class HaloLoss(PrepareImage):
    
    def __init__(self, predicted, labels, neg_weight=0.3, eps=0.05, max_distance=1):
        self.predicted = predicted
        self.labels = labels
        self.neg_weight = neg_weight
        self.colors_number = len(self.predicted[0])
        self.eps = eps
        self.max_distance = max_distance

        super().__init__(labels)
        self.prepare_objects()

    def make_halo(self, components):

        def squared_distance(point1, point2):
            x1, y1 = point1
            x2, y2 = point2
            return (x1 - x2) ** 2 + (y1 - y2) ** 2
        
        halos = []
        for object in components:
            halo = []
            # find framing rectangle
            min_x = self.width
            min_y = self.height
            max_x = 0
            max_y = 0
            for point in object:
                min_x = min(min_x, point[0])
                max_x = max(max_x, point[0])
                min_y = min(min_y, point[1])
                max_y = max(max_y, point[1])
            # get halo
            for i in range(max(min_x - self.max_distance, 0), min(max_x + self.max_distance + 1, self.height)):
                for j in range(max(min_y - self.max_distance, 0), min(max_y + self.max_distance + 1, self.width)):
                    if (i, j) in object:
                        continue
                    for point in object:
                        if squared_distance((i, j), point) <= self.max_distance ** 2:
                            halo.append((i, j))
            halos.append(halo)
        return halos
            
    
    def recoloring_stage(self, predicted_elem, components, halos):

        # add/subtract eps to avoid log(0)
        for c in range(self.colors_number):
            for i in range(self.height):
                for j in range(self.width):
                    if predicted_elem[c][i][j] == 0:
                        predicted_elem[c][i][j] += self.eps
                    elif predicted_elem[c][i][j] == 1:
                        predicted_elem[c][i][j] -= self.eps

        # functional to maximize
        def recoloring_functional(color, object, halo):
            object_sum = 0
            for point in object:
                object_sum += math.log(predicted_elem[color][point[0]][point[1]])
            object_sum /= len(object)

            halo_sum = 0
            for point in halo:
                halo_sum += math.log(1 - predicted_elem[color][point[0]][point[1]])
            halo_sum /= len(halo)

            return object_sum + self.neg_weight * halo_sum
        
        # find best color for each component
        recolored_components = []
        for i in range(len(components)):
            object = components[i]
            halo = halos[i]
            max_color = 1
            max_value = recoloring_functional(1, object, halo)
            for color in range(2, self.colors_number):
                value = recoloring_functional(color, object, halo)
                if value > max_value:
                    max_color = color
                    max_value = value
            recolored_components.append(max_color)

        return recolored_components

    
    def compute_loss(self):

        def compute_gain(object, recolored_color, elem):
            object_sum = 0
            for point in object:
                object_sum += math.log(self.predicted[elem][recolored_color][point[0]][point[1]])
            object_sum /= len(object)
            return object_sum
        
        total_loss = 0
        
        for elem in range(self.batch_size):

            # preliminary steps
            halos = self.make_halo(self.components_batch[elem])
            recolored_components = self.recoloring_stage(self.predicted[elem],
                                                         self.components_batch[elem], halos)

            background_sum = 0
            for i in range(self.height):
                for j in range(self.width):
                    if self.labels[elem][i][j] == 0:
                        background_sum += math.log(self.predicted[elem][0][i][j])

            for i in range(len(self.components_batch[elem])):
                object = self.components_batch[elem][i]
                recolored_color = recolored_components[i]
                background_sum += compute_gain(object, recolored_color, elem)

            background_sum *= -1

            total_loss += background_sum
        
        return total_loss / self.batch_size
    
# predicted = [[[[0, 0, 1, 0, 0, 0], [0, 0, 1, 0, 0, 0], [1, 1, 0, 1, 1, 1]],
#                  [[0.7, 1, 0.8, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0.4, 1, 0, 0, 0.9]],
#                  [[1, 0.5, 0.9, 0, 0, 0.8], [0.2, 1, 0.3, 0, 0, 0], [0.7, 0.1, 0.4, 0.5, 0, 0.05]]]]
# labels = [[[1, 2, 1, 0, 0, 0], [0, 2, 0, 0, 0, 0], [1, 2, 1, 0, 0, 1]]]
# neg_weight = 0.3
# eps = 0.05
# max_distance = 1
# item = HaloLoss(predicted, labels)

# print(item.batch_size, item.height, item.width)
# print(item.graph_batch)