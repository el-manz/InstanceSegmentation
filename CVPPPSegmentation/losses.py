import numpy
import torch
import torch.nn.functional as F

class HaloLoss:
    
    def __init__(self, predicted, labels, neg_weight, size_threshold):
        self.predicted = predicted
        self.labels = labels
        self.height = len(self.labels)
        self.width = len(self.labels[0])
        self.neg_weight = neg_weight
        self.size_threshold = size_threshold
        self.graph = {}
        self.components = []
        self.halos = []

        # for i in range(predicted.size(0)):

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
                max_x = min(max_x, point[0])
                min_y = min(min_y, point[0])
                max_y = min(max_y, point[0])

            # get halo
            for i in range(max(min_x - max_distance, 0), min(max_x + max_distance, self.width)):
                for j in range(max(min_y - max_distance, 0), min(max_y + max_distance, self.height)):
                    for point in object:
                        if squared_distance((i, j), point) <= max_distance:
                            halo.append((i, j))

            self.halos.append(halo)
        
        # assert len(self.components) == len(self.halos) - вынести в тест


    def recoloring_stage(predicted, halo):
        pass
