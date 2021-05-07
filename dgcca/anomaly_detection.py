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

    def train(self, clean, corrupt, embedding_dim=None, window=50, stride=False, method='intersection', method_param=None, classifier='proportional', plot=False, snr=1, grace=0, round_func=np.round, correlation_weighting=np.mean):
        thresholds = np.ones((self.dgcca.modalities, self.dgcca.modalities))
        if method == 'hard' and not plot:
            for i in range(self.dgcca.modalities):
                for j in range(i+1, self.dgcca.modalities):
                    thresholds[i,j] = method_param
                    thresholds[j,i] = method_param
            self.thresholds = thresholds
            self.set_classifier(classifier, grace, round_func)
            return

        if window == 1:
            self.weighting = 'flat'
        else:
            self.weighting = correlation_weighting
        print('Getting data embeddings...')
        clean_embedding = [self.dgcca.get_embedding(modality, i) for (i, modality) in enumerate(clean)]
        print('Getting noise embeddings...')
        corrupt_embedding = [self.dgcca.get_embedding(modality, i) for (i, modality) in enumerate(corrupt)]
        type_1 = np.zeros((self.dgcca.modalities, self.dgcca.modalities))
        type_2 = np.zeros((self.dgcca.modalities, self.dgcca.modalities))
        if stride == 'auto':
            stride = int(clean[0].shape[0]/5000)
        if plot:
            fig, ax = plt.subplots(nrows=self.dgcca.modalities, ncols=self.dgcca.modalities, sharey=True, figsize=(15,15))
            method_string = '{}, Threshold parameter: {}'.format(method, method_param) if method in ['hard', 'ppf'] else method
            #fig.tight_layout()
            #fig.subplots_adjust(top=0.92, hspace = 0.22)
            #fig.suptitle('Pairwise correlation distributions and thresholds over {} samples. Threshold method: {}'.format(window, method), fontsize=20)
            x = np.linspace(-1, 1, 100)
        distributions = np.empty((self.dgcca.modalities, self.dgcca.modalities, 2), dtype='object') if classifier == 'prob' else None
        with tqdm.tqdm(total=(self.dgcca.modalities*(self.dgcca.modalities-1))/2) as pbar_embed:
            for i in range(self.dgcca.modalities):
                for j in range(i+1, self.dgcca.modalities):
                    pbar_embed.set_description('Computing ({},{}) threshold'.format(i,j))
                    true_mean, true_std, true_corrs = dgcca.window_corr(clean_embedding[i], clean_embedding[j], window, stride=stride, weighting=self.weighting)
                    noise_mean, noise_std, noise_corrs = dgcca.window_corr(np.append(clean_embedding[i], corrupt_embedding[i], 0), 
                                                                np.append(corrupt_embedding[j], clean_embedding[j], 0), window, stride=stride, weighting=self.weighting)
                    if classifier == 'prob':
                        distributions[i,j,0] = norm(loc=true_mean, scale=true_std)
                        distributions[j,i,0] = norm(loc=true_mean, scale=true_std)
                        distributions[i,j,1] = norm(loc=noise_mean, scale=noise_std)
                        distributions[j,i,1] = norm(loc=noise_mean, scale=noise_std)
                    if method == 'hard':
                        thresholds[i,j] = method_param
                        thresholds[j,i] = method_param
                    elif method == 'ppf':
                        threshold = norm.ppf(q=method_param, loc=true_mean, scale=true_std)
                        thresholds[i,j] = threshold
                        thresholds[j,i] = threshold
                    elif method == 'noise_mean':
                        thresholds[i,j] = noise_mean
                        thresholds[j,i] = noise_mean
                    elif method == 'intersection':
                        threshold = get_thresh(true_mean, noise_mean, true_std, noise_std)
                        thresholds[i,j] = threshold
                        thresholds[j,i] = threshold
                    else:
                        raise ValueError('threshold method {} not recognised'.format(method))
                    type_1[i,j] = norm.cdf(thresholds[i,j], loc=true_mean, scale=true_std)
                    type_2[i,j] = 1-norm.cdf(thresholds[i,j], loc=noise_mean, scale=noise_std)
                    if plot:
                        for row, col in zip((i,j), (j,i)):
                            ax[row,col].plot(x, norm.pdf(x, true_mean, true_std), c='green')
                            ax[row,col].hist(true_corrs, color='green', alpha=0.5, density=True)
                            ax[row,col].plot(x, norm.pdf(x, noise_mean, noise_std), c='red')
                            ax[row,col].hist(noise_corrs, color='red', alpha=0.5, density=True)
                            ax[row,col].axvline(thresholds[i,j], c='black')
                            #ax[row,col].text(0.5,-0.15, 'Type 1: {:.2f}, Type 2: {:.2f}'.format(type_1[i,j], type_2[i,j]), ha="center", transform=ax[row,col].transAxes)
                            #ax[row,col].set_title('Type 1: {:.2f}, Type 2: {:.2f}'.format(type_1[i,j], type_2[i,j]))
                            ax[row,col].set_xlim(-1,1)
                    pbar_embed.update(1)
        self.thresholds = thresholds
        self.type_1 = type_1
        self.type_2 = type_2
        self.set_classifier(classifier, grace, round_func, distributions)
        if plot:
            return fig

    def set_classifier(self, classifier='proportional', grace=0, round_func=np.round, distributions=None):
        if classifier == 'proportional':
            self.grace = grace
            self.classifier = self.proportional_classifier
        elif classifier == 'est_corrupt':
            self.round_func = round_func
            self.classifier = self.est_corrupt_classifier
        elif classifier == 'delta':
            self.classifier = self.delta_classifier
        elif classifier == 'prob':
            self.distributions = distributions
            self.classifier = self.prob_classifier
        else:
            raise ValueError('classifier type not recognised')
            
    def detect_anomalies(self, data, evaluating=False):
        return self.classifier(data, evaluating=evaluating)

    def prob_classifier(self, data, evaluating=False):
        corrs = self.dgcca.get_corrs(data, weighting=self.weighting)
        probs = np.zeros_like(self.distributions)
        for row in range(corrs.shape[0]):
            for col in range(corrs.shape[1]):
                if row != col:
                    probs[row,col,0] = self.distributions[row,col,0].cdf(corrs[row,col])
                    probs[row,col,1] = 1 - self.distributions[row,col,1].cdf(corrs[row,col])
        corrupt = []
        net = net_prob(probs)
        while (net<0).any():
            next_corrupt = np.argmin(net)
            corrupt.append(next_corrupt)
            probs[next_corrupt,:,:] = 0                           # Zero out this row so its corruption is not attributed to a clean modality ...
            probs[:,next_corrupt,:] = 0                           # .. and column
            net = net_prob(probs)
        pred = np.array([False if mod in corrupt else True for mod in range(self.dgcca.modalities)])
        if evaluating:
            return (pred, corrs>self.thresholds)
        else:
            return pred

    def delta_classifier(self, data, evaluating=False):
        corrs = self.dgcca.get_corrs(data, weighting=self.weighting)
        deltas = corrs - self.thresholds                         # Calculate distance from thresholds
        corrupt = []
        while (np.sum(deltas, axis=1)<0).any():                  # Continue until all rows are net positive
            next_corrupt = np.argmin(np.sum(deltas, axis=1))     # Consider row furthest below thresholds corrupt
            corrupt.append(next_corrupt)
            deltas[next_corrupt,:] = 0                           # Zero out this row so its corruption is not attributed to a clean modality ...
            deltas[:,next_corrupt] = 0                           # .. and column
        pred = np.array([False if mod in corrupt else True for mod in range(self.dgcca.modalities)])
        if evaluating:
            return (pred, corrs>self.thresholds)
        else:
            return pred

    def proportional_classifier(self, data, evaluating=False):
        corrs = self.dgcca.get_corrs(data, weighting=self.weighting) 
        clean = corrs>self.thresholds                                                   # Identify corrupt pairs
        cleanness = clean.sum()/(clean.shape[0]*clean.shape[1]-self.dgcca.modalities)   # Calculate proportion of pairs corrupted
        pred = (clean.sum(axis=0)/(self.dgcca.modalities-1-self.grace)) >= cleanness    # Consider modalities with a higher proportion of corruption corrupted
        if evaluating:
            return (pred, clean)
        else:
            return pred

    def est_corrupt_classifier(self, data, evaluating=False):
        corrs = self.dgcca.get_corrs(data, weighting=self.weighting)
        clean = corrs>self.thresholds                                                        # Identify corrupt pairs
        clean_amount = corrs-self.thresholds
        corruptness = 1 - clean.sum()/(clean.shape[0]*clean.shape[1]-self.dgcca.modalities)  # Calculate proportion of pairs corrupted
        corrupt_est = min(self.est_num_corrupted(corruptness)[1], self.dgcca.modalities - 2) # Estimate number of corrupted modalities using this proportion
        num_corrupt = int(self.round_func(corrupt_est))                                      # Round to int using specified function
        confidence = np.abs(num_corrupt - corrupt_est)                                       # If estimate was int, confidence is high (not used)
        corrupt = np.argsort(clean.sum(axis=0))[-num_corrupt:]                               # Mark the top num_corrupt modalities as corrupt
        pred = np.array([False if mod in corrupt else True for mod in range(self.dgcca.modalities)])
        if evaluating:
            return (pred, clean)
        else:
            return pred

    def est_num_corrupted(self, prop):
        n = self.dgcca.modalities
        a = 1
        b = 1-2*n
        c = n**2 - n*(1+(n-1)*(1-prop))
        r1 = (-b + np.sqrt(b**2-4*a*c))/(2*a)
        r2 = (-b - np.sqrt(b**2-4*a*c))/(2*a)
        return (r1, r2)

def net_prob(probs):
    sums = np.sum(probs, axis=1)
    return sums[:,0] - sums[:,1]

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
    return np.roots(np.nan_to_num([a,b,c]))
