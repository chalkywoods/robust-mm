import numpy as np
import tqdm
import torch
import dgcca.dgcca as dgcca
from scipy.stats import norm
import matplotlib.pyplot as plt


class CcaAnomalyDetector:

    def __init__(self, dgcca, device='cpu'):
        self.dgcca = dgcca
        self.device = device

    def train(self, data, embedding_dim=None, window=50, stride=False, method='threshold', plot=False):
        if method == 'threshold':
            print('Generating noise...')
            noise = [torch.tensor(noise_like(modality)) for modality in data]
            print('Getting data embeddings...')
            data_embedding = [self.dgcca.get_embedding(modality, i) for (i, modality) in enumerate(data)]
            print('Getting noise embeddings...')
            noise_embedding = [self.dgcca.get_embedding(modality, i) for (i, modality) in enumerate(noise)]

            thresholds = np.ones((self.dgcca.modalities, self.dgcca.modalities))
            type_1 = np.zeros((self.dgcca.modalities, self.dgcca.modalities))
            type_2 = np.zeros((self.dgcca.modalities, self.dgcca.modalities))
            if plot:
                fig, ax = plt.subplots(nrows=self.dgcca.modalities, ncols=self.dgcca.modalities, sharex=True, figsize=(15,15))
                x = np.linspace(-1, 1, 100)
            with tqdm.tqdm(total=(self.dgcca.modalities*(self.dgcca.modalities-1))/2) as pbar_embed:
                for i in range(self.dgcca.modalities):
                    for j in range(i+1, self.dgcca.modalities):
                        pbar_embed.set_description('Computing ({},{}) threshold'.format(i,j))
                        true_mean, true_std, _ = dgcca.window_corr(data_embedding[i], data_embedding[j], window, stride=stride)
                        noise_mean, noise_std, _ = dgcca.window_corr(np.append(data_embedding[i], noise_embedding[i], 0), 
                                                                    np.append(noise_embedding[j], data_embedding[j], 0), window, stride=stride)
                        intersections = solve(true_mean, noise_mean, true_std, noise_std)
                        thresholds[i,j] = np.max(intersections)
                        thresholds[j,i] = thresholds[i,j]
                        true_dist = norm()
                        type_1[i,j] = norm.cdf(thresholds[i,j], loc=true_mean, scale=true_std)
                        type_2[i,j] = 1-norm.cdf(thresholds[i,j], loc=noise_mean, scale=noise_std)
                        if plot:
                            ax[i,j].plot(x, norm.pdf(x, true_mean, true_std), c='green')
                            ax[i,j].plot(x, norm.pdf(x, noise_mean, noise_std), c='red')
                        pbar_embed.update(1)
            self.thresholds = thresholds
            self.type_1 = type_1
            self.type_2 = type_2
            self.classifier = self.threshold_classifier
            if plot:
                return fig
            

    def detect_anomalies(self, data, grace=0):
        return self.classifier(data, grace=grace)

    def threshold_classifier(self, data, grace=0):
        corrs = self.dgcca.get_corrs(data)
        clean = corrs>self.thresholds
        cleanness = clean.sum()/(clean.shape[0]*clean.shape[1]-self.dgcca.modalities)
        return (clean.sum(axis=0)/(self.dgcca.modalities-1-grace)) >= cleanness

def noise_like(data):
    mean = data.mean().item()
    std = data.std().item()
    return np.random.default_rng().normal(mean, std, data.shape)

def solve(m1,m2,std1,std2):
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    return np.roots([a,b,c])
