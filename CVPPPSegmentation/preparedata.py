import numpy as np
from skimage.io import imread
from skimage.transform import resize
from torch.utils.data import DataLoader
from tqdm import tqdm

class PrepareData():
    
    def __init__(self, data_folder, splits, size, batch_size):
        self.data_folder = data_folder
        self.size = size
        self.height = size[0]
        self.width = size[1]
        self.batch_size = batch_size

        self.dataloaders = {}

        for name in splits:
            split = splits[name]
            self.dataloaders[name] = self.create_dataloader(self.make_images_list(split))
        

    def prepare_inst_array(self, inst_array):
        moved_array = np.moveaxis(inst_array, 0, 2)
        colors = []
        for i in range(self.height):
            for j in range(self.width):
                colors.append(moved_array[i][j])
        unique_colors = np.unique(colors, axis=0)
        print(unique_colors)
        colors_mapping = {}
        for i in range(len(unique_colors)):
            colors_mapping[tuple(unique_colors[i])] = i
        # print(colors_mapping)
        result_array = np.zeros([self.height, self.width])
        for i in range(self.height):
            for j in range(self.width):
                result_array[i][j] = colors_mapping[tuple(moved_array[i][j])]
        return np.array(result_array)


    def make_images_list(self, split):
        images = {"img": [], "sem": [], "inst": []}
        for i in tqdm(range(len(split))):
            img_path = self.data_folder + split.iloc[i]['img_path']
            inst_path = self.data_folder + split.iloc[i]['inst_path']

            img_array = np.rollaxis(resize(imread(img_path)[:, :, :3], self.size, mode='constant'), 2, 0)
            inst_array = np.rollaxis(resize(imread(inst_path), self.size, mode='constant'), 2, 0)

            inst_array = self.prepare_inst_array(inst_array)

            images["img"].append(np.asarray(img_array))
            images["inst"].append(np.asarray(inst_array))
        return images

    def create_dataloader(self, images):
        return DataLoader(list(zip(images["img"], images["inst"])), batch_size=self.batch_size, shuffle=True)