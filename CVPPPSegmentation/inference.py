import numpy as np
from scipy.spatial.distance import directed_hausdorff

from prepareimage import PrepareImage

class Inference(PrepareImage):
    
    def __init__(self, predicted, size_threshold, proximity_threshold):
        self.predicted = predicted
        self.batch_size = len(self.predicted)
        self.colors_number = len(self.predicted[0])
        self.height = len(self.predicted[0][0])
        self.width = len(self.predicted[0][0][0])
        self.size_threshold = size_threshold
        self.proximity_threshold = proximity_threshold

        self.assign_most_probable()
        super().__init__(self.reassigned_colors)
        self.prepare_objects()

    def assign_most_probable(self):
        self.reassigned_colors = []
        for elem in range(self.batch_size):
            new_colormap = np.zeros([self.height, self.width]).tolist()
            for i in range(self.height):
                for j in range(self.width):
                    most_probable_color = 0
                    max_probability = self.predicted[elem][0][i][j]
                    for c in range(self.colors_number):
                        probability = self.predicted[elem][c][i][j]
                        if probability > max_probability:
                            most_probable_color = c
                            max_probability = probability
                    new_colormap[i][j] = most_probable_color
            self.reassigned_colors.append(new_colormap)

    def filter_small_objets(self, components):
        components_filtered = []
        for object in components:
            if len(object) >= self.size_threshold:
                components_filtered.append(object)
        return components_filtered

    def merge_close_objects(self, reassigned_colors, components_filtered):

        def symmetric_hausdorff(u, v):
            return max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
        
        def merge_dfs(u, used, merged_component):
            used[u] = True
            merged_component += components_filtered[u]
            for v in merge_graph[u]:
                if not used[v]:
                    merge_dfs(v, used, merged_component)

        final_components = []
        merge_graph = {}

        for i in range(len(components_filtered)):
            merge_graph[i] = []

        for i in range(len(components_filtered)):
            for j in range(i + 1, len(components_filtered)):
                object_i = components_filtered[i]
                object_j = components_filtered[j]
                color_i = reassigned_colors[object_i[0][0]][object_i[0][1]]
                color_j = reassigned_colors[object_j[0][0]][object_j[0][1]]
                if color_i != color_j:
                    continue
                if symmetric_hausdorff(object_i, object_j) <= self.proximity_threshold:
                    # remember that these should be merged
                    merge_graph[i].append(j)
                    merge_graph[j].append(i)
        
        # merge final components through dfs
        used = [False] * len(components_filtered)

        for i in range(len(components_filtered)):
            if not used[i]:
                new_merged_component = []
                merge_dfs(i, used, new_merged_component)
                final_components.append(new_merged_component)
        return final_components
    
    def compute_inference(self):
        self.result = []
        for elem in range(self.batch_size):
            # steps
            components_filtered = self.filter_small_objets(self.components_batch[elem])
            final_components = self.merge_close_objects(self.reassigned_colors[elem], components_filtered)
            self.result.append(final_components)
        return self.result