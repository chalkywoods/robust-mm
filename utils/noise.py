import os

import numpy as np
import tqdm
import torch
from sklearn.mixture import GaussianMixture

from mm_fit.utils.utils import load_modality

class GMMNoise_mmfit:

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

class GMMNoise:
    def __init__(self, data, n_components):
        modalities = []
        for data, _ in data:
            with tqdm.tqdm(total=len(data)) as pbar:
                pbar.set_description('Generating GMM for dataset')
                for mod in data:
                    modality = []
                    for i in range(mod.shape[2]):
                        row = []
                        for j in range(mod.shape[3]):
                            components = min(n_components, len(np.unique(mod[:,0,i,j])))
                            row.append(GaussianMixture(components).fit(mod[:,0,i,j].reshape((-1,1))))
                        modality.append(row)
                    modalities.append(modality)
                    pbar.update(1)
                self.models = np.array(modalities)
    
    def get_noise(self, shape, modality):
        samples = []
        if isinstance(shape, int):
            noise = []
            for i in range(self.models.shape[1]):
                noise.append([model.sample(shape)[0].reshape((-1)) for model in self.models[modality,i,:]])
            return np.array(noise)
        else:
            all_samples = []
            samples = [model.sample(shape[0]*shape[2])[0] for model in self.models[modality]]
            samples = np.stack([sample.reshape((shape[0], shape[2])) for sample in samples], axis=1)
            return samples

    def add_noise(self, data, snr=1, modality=None):
        modality = range(len(data)) if modality is None else modality # corrupt all by default
        modalities = []
        for i in range(len(data)):
            if i in modality:
                noise = torch.as_tensor(np.moveaxis(self.get_noise(data[0].shape[0], i), -1, 0).reshape(data[i].shape), dtype=torch.float)
                modalities.append((noise + torch.as_tensor(snr*data[i], dtype=torch.float))/(1 + snr))
            else:
                modalities.append(torch.as_tensor(data[i], dtype=torch.float))
        return modalities

def get_gaussian_noise(shape):
    return np.random.default_rng().normal(0,1,shape)        
        
def add_gaussian_noise(data, snr=1, modality=None):
    modality = range(len(data)) if modality is None else modality
    modalities = []
    for i in range(len(data)):
        if i in modality:
            noise = torch.as_tensor(np.moveaxis(get_gaussian_noise(data[0].shape), -1, 0).reshape(data[i].shape), dtype=torch.float)
            modalities.append((noise + torch.as_tensor(snr*data[i], dtype=torch.float))/(1 + snr))
        else:
            modalities.append(torch.as_tensor(data[i], dtype=torch.float))
    return modalities

