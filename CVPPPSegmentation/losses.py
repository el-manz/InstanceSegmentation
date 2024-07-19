import math
import numpy
import torch
import torch.nn.functional as F

class HaloLoss:
    
    def __init__(self, predicted, labels, neg_weight, eps=0.05):
        self.predicted = predicted
        self.labels = labels
        self.height = len(self.labels)
        self.width = len(self.labels[0])
        self.neg_weight = neg_weight
        self.colors_number = len(self.predicted)
        self.eps = eps

        self.graph = {}
        self.components = []
        self.halos = []
        self.recolored_components = []

    def make_graphs(self):

        def get_neighbors(i, j):
            result = []
            if i != 0:
                result.append((i - 1, j))
            if j != 0:
                result.append((i, j - 1))
            if i != self.height - 1:
                result.append((i + 1, j))
            if j != self.width - 1:
                result.append((i, j + 1))
            return result

        for i in range(self.height):
            for j in range(self.width):
                if self.labels[i][j] <= 0:
                    continue
                self.graph[(i, j)] = []
                neighbors = get_neighbors(i, j)
                for i_neigh, j_neigh in neighbors:
                    if self.labels[i][j] == self.labels[i_neigh][j_neigh]:
                        self.graph[(i, j)].append((i_neigh, j_neigh))

    
    def find_objects(self):

        def dfs(u, used, component):
            used[u] = True
            component.append(u)
            for v in self.graph[u]:
                if not used[v]:
                    dfs(v, used, component)

        used = {}
        for i in range(self.height):
            for j in range(self.width):
                used[(i, j)] = False

        for i in range(self.height):
            for j in range(self.width):
                if self.labels[i][j] > 0 and not used[(i, j)]:
                    new_component = []
                    dfs((i, j), used, new_component)
                    self.components.append(new_component)


    def make_halo(self, max_distance):

        def squared_distance(point1, point2):
            x1, y1 = point1
            x2, y2 = point2
            return (x1 - x2) ** 2 + (y1 - y2) ** 2
        
        for object in self.components:

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
            for i in range(max(min_x - max_distance, 0), min(max_x + max_distance + 1, self.height)):
                for j in range(max(min_y - max_distance, 0), min(max_y + max_distance + 1, self.width)):
                    if (i, j) in object:
                        continue
                    for point in object:
                        if squared_distance((i, j), point) <= max_distance ** 2:
                            halo.append((i, j))

            self.halos.append(halo)
            
    
    def recoloring_stage(self):

        # add/subtract eps to avoid log(0)
        for c in range(self.colors_number):
            for i in range(self.height):
                for j in range(self.width):
                    if self.predicted[c][i][j] == 0:
                        self.predicted[c][i][j] += self.eps
                    elif self.predicted[c][i][j] == 1:
                        self.predicted[c][i][j] -= self.eps

        # functional to maximize
        def recoloring_functional(color, object, halo):
            object_sum = 0
            for point in object:
                object_sum += math.log(self.predicted[color][point[0]][point[1]])
            object_sum /= len(object)

            halo_sum = 0
            for point in halo:
                halo_sum += math.log(1 - self.predicted[color][point[0]][point[1]])
            halo_sum /= len(halo)

            return object_sum + self.neg_weight * halo_sum
        
        # find best color for each component
        for i in range(len(self.components)):
            object = self.components[i]
            halo = self.halos[i]
            max_color = 1
            max_value = recoloring_functional(1, object, halo)
            for color in range(2, self.colors_number):
                print(i, color)
                value = recoloring_functional(color, object, halo)
                print("RECOLORING object: ", object, ", color: ", color, ", value: ", value)
                if value > max_value:
                    max_color = color
                    max_value = value
            self.recolored_components.append(max_color)

    
    def compute_loss(self):

        def compute_gain(object, recolored_color):
            object_sum = 0
            for point in object:
                object_sum += math.log(self.predicted[recolored_color][point[0]][point[1]])
            object_sum /= len(object)
            return object_sum
        
        background_sum = 0
        for i in range(self.height):
            for j in range(self.width):
                if self.labels[i][j] == 0:
                    background_sum += math.log(self.predicted[0][i][j])

        for i in range(len(self.components)):
            object = self.components[i]
            recolored_color = self.recolored_components[i]
            background_sum += compute_gain(object, recolored_color)

        background_sum *= -1
        
        return background_sum