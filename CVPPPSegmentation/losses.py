import numpy
import torch
import torch.nn.functional as F

class HaloLoss:
    
    def __init__(self, predicted, labels, neg_weight, size_threshold):
        self.predicted = predicted
        self.labels = labels
        self.neg_weight = neg_weight
        self.size_threshold = size_threshold
        self.graphs = []
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

        for channel in range(len(self.labels)):
            labels_channel = self.labels[channel]

            height = len(labels_channel)
            width = len(labels_channel[0])

            graph = {}

            for i in range(height):
                for j in range(width):
                    if labels_channel[i][j] <= 0:
                        continue
                    graph[(i, j)] = []
                    neighbors = get_neighbors(i, j)
                    for i_neigh, j_neigh in neighbors:
                        if labels_channel[i][j] == labels_channel[i_neigh][j_neigh]:
                            graph[(i, j)].append((i_neigh, j_neigh))
            
            self.graphs.append(graph)

    
    def find_objects(self):

        def dfs(u, used, component, channel):
            used[u] = True
            component.append(u)
            for v in self.graphs[channel][u]:
                if not used[v]:
                    dfs(v, used, component, channel)
        
        for channel in range(len(self.labels)):
            labels_channel = self.labels[channel]

            height = len(labels_channel)
            width = len(labels_channel[0])

            channel_components = []

            used = {}
            for i in range(height):
                for j in range(width):
                    used[(i, j)] = False

            for i in range(height):
                for j in range(width):
                    if labels_channel[i][j] > 0 and not used[(i, j)]:
                        new_component = []
                        dfs((i, j), used, new_component, channel)
                        channel_components.append(new_component)
            
            self.components.append(channel_components)
                    

    def recoloring_stage(predicted, halo):
        pass
