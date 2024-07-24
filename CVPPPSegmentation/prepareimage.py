import numpy as np

class PrepareImage:

    def __init__(self, images_batch):
        self.images_batch = images_batch
        self.batch_size = len(self.images_batch)
        self.height = len(self.images_batch[0])
        self.width = len(self.images_batch[0][0])

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
        
        graph = {}

        for i in range(self.height):
            for j in range(self.width):
                if self.images_batch[elem][i][j] <= 0:
                    continue
                graph[(i, j)] = []
                neighbors = get_neighbors(i, j)
                for i_neigh, j_neigh in neighbors:
                    if self.images_batch[elem][i][j] == self.images_batch[elem][i_neigh][j_neigh]:
                        graph[(i, j)].append((i_neigh, j_neigh))
        return graph
                        

    def find_objects(self, elem, graph):
        def dfs(u, used, component):
            used[u] = True
            component.append(u)
            for v in graph[u]:
                if not used[v]:
                    dfs(v, used, component)

        used = {}
        for i in range(self.height):
            for j in range(self.width):
                used[(i, j)] = False
        components = []
        for i in range(self.height):
            for j in range(self.width):
                if self.images_batch[elem][i][j] > 0 and not used[(i, j)]:
                    new_component = []
                    dfs((i, j), used, new_component)
                    components.append(new_component)

        return components


    def prepare_objects(self):
        self.graph_batch = []
        self.components_batch = []
        for elem in range(self.batch_size):
            graph = self.make_graphs(elem)
            components = self.find_objects(elem, graph)
            self.graph_batch.append(graph)
            self.components_batch.append(components)