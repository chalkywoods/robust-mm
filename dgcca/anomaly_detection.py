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

    def train(self, clean, corrupt, embedding_dim=None, window=50, stride=False, method='threshold', plot=False, snr=1):
        if method == 'threshold':
            print('Getting data embeddings...')
            clean_embedding = [self.dgcca.get_embedding(modality, i) for (i, modality) in enumerate(clean)]
            print('Getting noise embeddings...')
            corrupt_embedding = [self.dgcca.get_embedding(modality, i) for (i, modality) in enumerate(corrupt)]

            thresholds = np.ones((self.dgcca.modalities, self.dgcca.modalities))
            type_1 = np.zeros((self.dgcca.modalities, self.dgcca.modalities))
            type_2 = np.zeros((self.dgcca.modalities, self.dgcca.modalities))
            if stride == 'auto':
                stride = int(clean[0].shape[0]/5000)
            if plot:
                fig, ax = plt.subplots(nrows=self.dgcca.modalities, ncols=self.dgcca.modalities, sharex=True, sharey=True, figsize=(15,15))
                x = np.linspace(-1, 1, 100)
            with tqdm.tqdm(total=(self.dgcca.modalities*(self.dgcca.modalities-1))/2) as pbar_embed:
                for i in range(self.dgcca.modalities):
                    for j in range(i+1, self.dgcca.modalities):
                        pbar_embed.set_description('Computing ({},{}) threshold'.format(i,j))
                        true_mean, true_std, true_corrs = dgcca.window_corr(clean_embedding[i], clean_embedding[j], window, stride=stride)
                        noise_mean, noise_std, noise_corrs = dgcca.window_corr(np.append(clean_embedding[i], corrupt_embedding[i], 0), 
                                                                    np.append(corrupt_embedding[j], clean_embedding[j], 0), window, stride=stride)
                        threshold = get_thresh(true_mean, noise_mean, true_std, noise_std)
                        thresholds[i,j] = threshold
                        thresholds[j,i] = threshold
                        type_1[i,j] = norm.cdf(thresholds[i,j], loc=true_mean, scale=true_std)
                        type_2[i,j] = 1-norm.cdf(thresholds[i,j], loc=noise_mean, scale=noise_std)
                        if plot:
                            ax[i,j].plot(x, norm.pdf(x, true_mean, true_std), c='green')
                            ax[i,j].hist(true_corrs, color='green', alpha=0.5, density=True)
                            ax[i,j].plot(x, norm.pdf(x, noise_mean, noise_std), c='red')
                            ax[i,j].hist(noise_corrs, color='red', alpha=0.5, density=True)
                            ax[j,i].plot(x, norm.pdf(x, true_mean, true_std), c='green')
                            ax[j,i].hist(true_corrs, color='green', alpha=0.5, density=True)
                            ax[j,i].plot(x, norm.pdf(x, noise_mean, noise_std), c='red')
                            ax[j,i].hist(noise_corrs, color='red', alpha=0.5, density=True)

                        pbar_embed.update(1)
            self.thresholds = thresholds
            self.type_1 = type_1
            self.type_2 = type_2
            self.classifier = self.threshold_classifier
            if plot:
                return fig
            

    def detect_anomalies(self, data, grace=0, evaluating=False):
        return self.classifier(data, grace=grace, evaluating=evaluating)

    def threshold_classifier(self, data, grace=0, evaluating=False):
        corrs = self.dgcca.get_corrs(data)
        clean = corrs>self.thresholds
        cleanness = clean.sum()/(clean.shape[0]*clean.shape[1]-self.dgcca.modalities)
        pred = (clean.sum(axis=0)/(self.dgcca.modalities-1-grace)) >= cleanness
        if evaluating:
            return (pred, clean)
        else:
            return pred

def noise_like(data):
    mean = data.mean().item()
    std = data.std().item()
    return np.random.default_rng().normal(mean, std, data.shape)

def get_thresh(mtrue, mfalse, stdtrue, stdfalse):
    roots = solve(mtrue, mfalse, stdtrue, stdfalse)
    if stdtrue > stdfalse:
        return np.max(roots)
    else:
        return np.min(roots)

def solve(m1,m2,std1,std2):
    a = 1/(2*std1**2) - 1/(2*std2**2)
    b = m2/(std2**2) - m1/(std1**2)
    c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
    return np.roots([a,b,c])
