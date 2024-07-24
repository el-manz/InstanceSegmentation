import numpy as np
from scipy.spatial.distance import directed_hausdorff

class Inference:
    
    def __init__(self, predicted, size_threshold, proximity_threshold):
        self.predicted = predicted
        self.batch_size = len(self.predicted)
        self.colors_number = len(self.predicted[0])
        self.height = len(self.predicted[0][0])
        self.width = len(self.predicted[0][0][0])
        self.size_threshold = size_threshold
        self.proximity_threshold = proximity_threshold

    def assign_most_probable(self, elem):
        for i in range(self.height):
            for j in range(self.width):
                most_probable_color = 0
                max_probability = self.predicted[elem][0][i][j]
                for c in range(self.colors_number):
                    probability = self.predicted[elem][c][i][j]
                    if probability > max_probability:
                        most_probable_color = c
                        max_probability = probability
                self.reassigned_colors[i][j] = most_probable_color
    
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
                if self.reassigned_colors[i][j] <= 0:
                    continue
                self.graph[(i, j)] = []
                neighbors = get_neighbors(i, j)
                for i_neigh, j_neigh in neighbors:
                    if self.reassigned_colors[i][j] == self.reassigned_colors[i_neigh][j_neigh]:
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
                if self.reassigned_colors[i][j] > 0 and not used[(i, j)]:
                    new_component = []
                    dfs((i, j), used, new_component)
                    self.components.append(new_component)

    def filter_small_objets(self):
        for object in self.components:
            if len(object) >= self.size_threshold:
                self.components_filtered.append(object)

    def merge_close_objects(self):

        def symmetric_hausdorff(u, v):
            return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
        
        def merge_dfs(u, used, merged_component):
            used[u] = True
            merged_component += self.components_filtered[u]
            for v in self.merge_graph[u]:
                if not used[v]:
                    merge_dfs(v, used, merged_component)

        for i in range(len(self.components_filtered)):
            self.merge_graph[i] = []

        for i in range(len(self.components_filtered)):
            for j in range(i + 1, len(self.components_filtered)):
                object_i = self.components_filtered[i]
                object_j = self.components_filtered[j]
                color_i = self.reassigned_colors[object_i[0][0]][object_i[0][1]]
                color_j = self.reassigned_colors[object_j[0][0]][object_j[0][1]]
                if color_i != color_j:
                    continue
                if symmetric_hausdorff(object_i, object_j) <= self.proximity_threshold:
                    # remember that these should be merged
                    self.merge_graph[i].append(j)
                    self.merge_graph[j].append(i)
        
        # merge final components through dfs
        self.final_components = []
        used = [False] * len(self.components_filtered)

        for i in range(len(self.components_filtered)):
            if not used[i]:
                new_merged_component = []
                merge_dfs(i, used, new_merged_component)
                self.final_components.append(new_merged_component)
    
    def compute_inference(self):
        self.result = []
        for elem in range(self.batch_size):

            self.reassigned_colors = np.zeros([self.height, self.width]).tolist()
            self.graph = {}
            self.components = []
            self.components_filtered = []
            self.merge_graph = {}
            self.final_components = []

            # steps
            self.assign_most_probable(elem)
            self.make_graphs()
            self.find_objects()
            self.filter_small_objets()
            self.merge_close_objects()
            self.result.append(self.final_components)
        return self.result