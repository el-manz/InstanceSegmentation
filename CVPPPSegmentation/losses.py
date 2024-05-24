import numpy
import torch
import torch.nn.functional as F

class HaloLoss:
    
    def __init__(self, predicted, labels, neg_weight, size_threshold):
        self.predicted = predicted
        self.labels = labels
        self.neg_weight = neg_weight
        self.size_threshold = size_threshold
        self.graph = {}
        self.components = []

        # for i in range(predicted.size(0)):

    def make_graphs(self):

        def get_neighbors(i, j):
            result = []
            if i != 0:
                result.append((i - 1, j))
            if j != 0:
                result.append((i, j - 1))
            if i != height - 1:
                result.append((i + 1, j))
            if j != width - 1:
                result.append((i, j + 1))
            return result

        height = len(self.labels)
        width = len(self.labels[0])

        for i in range(height):
            for j in range(width):
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

        height = len(self.labels)
        width = len(self.labels[0])

        used = {}
        for i in range(height):
            for j in range(width):
                used[(i, j)] = False

        for i in range(height):
            for j in range(width):
                if self.labels[i][j] > 0 and not used[(i, j)]:
                    new_component = []
                    dfs((i, j), used, new_component)
                    self.components.append(new_component)


    def recoloring_stage(predicted, halo):
        pass
