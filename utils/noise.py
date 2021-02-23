import os

import numpy as np
import tqdm
import torch
from sklearn.mixture import GaussianMixture

from mm_fit.utils.utils import load_modality

class GMMNoise:

    def __init__(self, data_path, modalities, workouts, n_components, params=None, step='auto'):
        self.modalities = modalities
        if params is not None:
            self.load_models(params)
        else:
            self.models = {}
            with tqdm.tqdm(total=len(modalities)) as pbar:
                for modality in modalities:
                    mod_data = []
                    for workout in workouts:
                        pbar.set_description('Loading {} data'.format(modality))
                        workout_path = os.path.join(data_path, 'w' + workout)
                        files = os.listdir(workout_path)
                        for file in files:
                            if modality in file:
                                mod_data.append(load_modality(os.path.join(workout_path, file))[:,2:])
                    mod_data = np.concatenate(mod_data)
                    self.models[modality] = []
                    for i in range(mod_data.shape[1]):
                        pbar.set_description('Fitting GMM {} for {}'.format(i, modality))
                        step = max(int(mod_data.shape[0]/100000), 1) if step == 'auto' else step
                        self.models[modality].append(GaussianMixture(n_components).fit(mod_data[0:-1:step,i].reshape((-1,1))))
                        pbar.update(1/mod_data.shape[1])

    def get_noise(self, shape, modality=None, modality_idx=None):
        modality = self.modalities[modality_idx] if modality is None else modality
        samples = []
        if isinstance(shape, int):
            samples = [model.sample(shape)[0] for model in self.models[modality]]
            return np.concatenate(samples, axis=1)
        else:
            all_samples = []
            samples = [model.sample(shape[0]*shape[2])[0] for model in self.models[modality]]
            samples = np.stack([sample.reshape((shape[0], shape[2])) for sample in samples], axis=1)
            return samples
            # for i in range(shape[2]):
            #     samples = [model.sample(shape[0])[0] for model in self.models[modality]]
            #     all_samples.append(np.concatenate(samples, axis=1))
            # return np.stack(all_samples, axis=2)

    def add_noise(self, data, snr=1, modality=None, modality_idx=None):
        noise = self.get_noise(data.shape, modality=modality, modality_idx=modality_idx)
        if isinstance(data, torch.Tensor):
            noise = torch.Tensor(noise)
        return (noise + snr*data)/(1 + snr)
