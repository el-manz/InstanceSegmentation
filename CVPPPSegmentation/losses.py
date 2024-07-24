import math
import numpy
import torch
import torch.nn.functional as F

class HaloLoss:
    
    def __init__(self, predicted, labels, neg_weight=0.3, eps=0.05, max_distance=1):
        self.predicted = predicted
        self.labels = labels
        self.batch_size = len(self.labels)
        self.height = len(self.labels[0])
        self.width = len(self.labels[0][0])
        self.neg_weight = neg_weight
        self.colors_number = len(self.predicted[0])
        self.eps = eps
        self.max_distance = max_distance

    def make_graphs(self, elem):

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
                if self.labels[elem][i][j] <= 0:
                    continue
                self.graph[(i, j)] = []
                neighbors = get_neighbors(i, j)
                for i_neigh, j_neigh in neighbors:
                    if self.labels[elem][i][j] == self.labels[elem][i_neigh][j_neigh]:
                        self.graph[(i, j)].append((i_neigh, j_neigh))

    
    def find_objects(self, elem):

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
                if self.labels[elem][i][j] > 0 and not used[(i, j)]:
                    new_component = []
                    dfs((i, j), used, new_component)
                    self.components.append(new_component)


    def make_halo(self):

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
            for i in range(max(min_x - self.max_distance, 0), min(max_x + self.max_distance + 1, self.height)):
                for j in range(max(min_y - self.max_distance, 0), min(max_y + self.max_distance + 1, self.width)):
                    if (i, j) in object:
                        continue
                    for point in object:
                        if squared_distance((i, j), point) <= self.max_distance ** 2:
                            halo.append((i, j))

            self.halos.append(halo)
            
    
    def recoloring_stage(self, elem):

        print(self.predicted)
        print(self.colors_number)

        # add/subtract eps to avoid log(0)
        for c in range(self.colors_number):
            for i in range(self.height):
                for j in range(self.width):
                    if self.predicted[elem][c][i][j] == 0:
                        self.predicted[elem][c][i][j] += self.eps
                    elif self.predicted[elem][c][i][j] == 1:
                        self.predicted[elem][c][i][j] -= self.eps

        print(self.predicted)

        # functional to maximize
        def recoloring_functional(color, object, halo):
            object_sum = 0
            for point in object:
                object_sum += math.log(self.predicted[elem][color][point[0]][point[1]])
            object_sum /= len(object)

            halo_sum = 0
            for point in halo:
                print(elem, color, point[0], point[1])
                print(self.predicted[elem][color][point[0]][point[1]])
                halo_sum += math.log(1 - self.predicted[elem][color][point[0]][point[1]])
                print("ok")
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

        def compute_gain(object, recolored_color, elem):
            object_sum = 0
            for point in object:
                object_sum += math.log(self.predicted[elem][recolored_color][point[0]][point[1]])
            object_sum /= len(object)
            return object_sum
        
        total_loss = 0
        
        for elem in range(self.batch_size):

            self.graph = {}
            self.components = []
            self.halos = []
            self.recolored_components = []

            # preliminary steps
            self.make_graphs(elem)
            self.find_objects(elem)
            self.make_halo()
            self.recoloring_stage(elem)

            background_sum = 0
            for i in range(self.height):
                for j in range(self.width):
                    if self.labels[elem][i][j] == 0:
                        background_sum += math.log(self.predicted[elem][0][i][j])

            for i in range(len(self.components)):
                object = self.components[i]
                recolored_color = self.recolored_components[i]
                background_sum += compute_gain(object, recolored_color, elem)

            background_sum *= -1

            total_loss += background_sum
        
        return total_loss / self.batch_size