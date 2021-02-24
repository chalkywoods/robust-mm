import os

import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import RandomSampler, ConcatDataset, DataLoader, random_split

from mm_fit.utils.dataset import MMFit, SequentialStridedSampler
from mm_fit.utils.data_transforms import Unit, Resample

def load_data(modalities, data_path, ids, splits=[1], loader=False, samplers=[], 
              batch_size=128, window_stride = 0.2, window_length = 5, 
              skeleton_sampling_rate = 30, target_sensor_sampling_rate = 50, workers=0, device=None):

    # dataset __len__ uses pose_3d dims (workaround)
    if not 'pose_3d' in modalities:
        modalities.append('pose_3d')

    window_stride = int(window_stride * skeleton_sampling_rate)
    skeleton_window_length = int(window_length * skeleton_sampling_rate)
    sensor_window_length = int(window_length * target_sensor_sampling_rate)

    data_transforms = {
            'skeleton': transforms.Compose([
                Unit()
            ]),
            'sensor': transforms.Compose([
                Resample(target_length=sensor_window_length)
            ])
        }

    datasets = []
    for w_id in ids:
        modality_filepaths = {}
        workout_path = os.path.join(data_path, 'w' + w_id)
        files = os.listdir(workout_path)
        label_path = None
        for file in files:
            if 'labels' in file:
                label_path = os.path.join(workout_path, file)
                continue
            for modality_type in modalities:
                if modality_type in file:
                    modality_filepaths[modality_type] = os.path.join(workout_path, file)
        if label_path is None:
            raise Exception('Error: Label file not found for workout {}.'.format(w_id))

        datasets.append(MMFit(modality_filepaths, label_path, window_length, skeleton_window_length,
                        sensor_window_length, skeleton_transform=data_transforms['skeleton'],
                        sensor_transform=data_transforms['sensor']))

    dataset = ConcatDataset(datasets)
    lengths = [int(split*len(dataset)) for split in splits]
    lengths[np.argmax(splits)] += len(dataset) - sum(lengths) # ensure splits add up to dataset size
    print('Dataset splits: {}'.format(str(lengths)))

    datasets = random_split(dataset, lengths)

    if loader:
        loaders = []
        for idx, dataset in enumerate(datasets):
            loaders.append(DataLoader(dataset=dataset, batch_size=batch_size,
                sampler=samplers[idx](dataset), pin_memory=True, num_workers=workers))
        return loaders
    else:
        return datasets