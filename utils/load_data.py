import os

import torchvision.transforms as transforms
from torch.utils.data import RandomSampler, ConcatDataset, DataLoader

from mm_fit.utils.dataset import MMFit, SequentialStridedSampler
from mm_fit.utils.data_transforms import Unit, Resample

def load_data(modalities, data_path, train_ids=[], val_ids=[], test_ids=[], loader=False, 
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

    train_datasets, val_datasets, test_datasets = [], [], []
    for w_id in train_ids+ val_ids + test_ids:
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

        if w_id in train_ids:
            train_datasets.append(MMFit(modality_filepaths, label_path, window_length, skeleton_window_length,
                                        sensor_window_length, skeleton_transform=data_transforms['skeleton'],
                                        sensor_transform=data_transforms['sensor'], device=device))
        elif w_id in val_ids:
            val_datasets.append(MMFit(modality_filepaths, label_path, window_length, skeleton_window_length,
                                    sensor_window_length, skeleton_transform=data_transforms['skeleton'],
                                    sensor_transform=data_transforms['sensor'], device=device))
        elif w_id in test_ids:
            test_datasets.append(MMFit(modality_filepaths, label_path, window_length, skeleton_window_length,
                                    sensor_window_length, skeleton_transform=data_transforms['skeleton'],
                                    sensor_transform=data_transforms['sensor'], device=device))
        else:
            raise Exception('Error: Workout {} not assigned to train, test, or val datasets'.format(w_id))

    train_dataset = ConcatDataset(train_datasets) if train_datasets else []
    val_dataset = ConcatDataset(val_datasets) if val_datasets else []
    test_dataset = ConcatDataset(test_datasets) if test_datasets else []

    if loader:
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                                  sampler=RandomSampler(train_dataset), pin_memory=True, num_workers=workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                                sampler=SequentialStridedSampler(val_dataset, window_stride), pin_memory=True, num_workers=workers)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 sampler=SequentialStridedSampler(test_dataset, window_stride), pin_memory=True, num_workers=workers)
        return (train_loader, val_loader, test_loader)
    else:
        return (train_dataset, val_dataset, test_dataset)