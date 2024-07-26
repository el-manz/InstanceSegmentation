import pandas as pd
import pytest
from preparedata import PrepareData

def test_shapes():

    data_folder = '/Users/elizavetamanzula/Desktop/у(ч)еба/InstanceSegmentation/CVPPPSegmentation/CVPPPSegmData/'

    split_data = pd.read_csv('/Users/elizavetamanzula/Desktop/у(ч)еба/InstanceSegmentation/CVPPPSegmentation/CVPPPSegmData/split.csv')
    train_split = split_data[split_data['split'] == 'train']
    val_split = split_data[split_data['split'] == 'dev']
    test_split = split_data[split_data['split'] == 'test']

    splits = {"train": train_split, "val": val_split, "test": test_split}
    size = (480, 480)
    batch_size = 10

    item = PrepareData(data_folder, splits, size, batch_size)

    # test images_lists shapes
    val_list = item.make_images_list(val_split)
    assert val_list["img"][0].shape == (3, size[0], size[1])
    assert val_list["inst"][0].shape == (size[0], size[1])
    assert val_list["img"].shape == (len(val_split), 3, size[0], size[1])
    assert val_list["inst"].shape == (len(val_split), size[0], size[1])

    # test dataloaders shapes
    val_loader = item.dataloaders["val"]
    batch, label = next(iter(val_loader))
    assert batch.shape == (batch_size, 3, size[0], size[1])
    assert label.shape == (batch_size, size[0], size[1])