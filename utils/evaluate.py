import numpy as np
import tqdm
import torch
import torch.nn as nn

import utils.mm_mnist
from utils.noise import GMMNoise
import utils.noise
import dgcca.dgcca
import dgcca.anomaly_detection

def embed(classifier, loader, noise=None):
    classifier.eval()
    with tqdm.tqdm(total=len(loader)) as pbar:
        pbar.set_description('Embedding')
        out = []
        for batch, (data, _) in enumerate(loader):
            data = data if noise is None else noise(data)
            out.append([mod.detach() for mod in classifier(data, heads=True)])
            pbar.update(1)
    output = []
    for i in range(4):
        output.append(np.concatenate([batch[i] for batch in out]))
    return output

def eval_classifier(classifier, loader, criterion=nn.CrossEntropyLoss()):
    classifier.eval()

    correct = 0
    with tqdm.tqdm(total=len(loader)) as pbar:
        total_loss = 0
        for batch, (data, label) in enumerate(loader):
            out = classifier(data).detach()
            loss = criterion(out, label)
            total_loss += loss
            total_loss_avg = total_loss / ((batch + 1))
            pred = np.argmax(out, axis=1)
            correct += np.sum((pred==label).numpy().astype(int))
            pbar.update(1)
            pbar.set_description('Accuracy: {:.2%} | Error: {:.4f}'.format(correct/((batch+1)*batch_size), total_loss_avg))
    return correct/len(ds_test)
    
def eval_ad(classifier, data, detector, noise, num_corrupt=0, window=20, grace=0):
    classifier.eval()
    tp_pred, tn_pred, fp_pred, fn_pred, tp_ind, tn_ind, fp_ind, fn_ind = (0,0,0,0,0,0,0,0)
    with tqdm.tqdm(total=len(range(0, len(data[0])-window, window))) as pbar:
        for i in range(0, len(data[0])-window, window):
            clean = np.ones(len(data))
            corrupt = np.random.default_rng().choice(len(data), size=num_corrupt, replace=False)
            clean[corrupt] = 0
            corrupt_data = noise([mod[i:i+window,:].numpy() for mod in data], corrupt)
            corrupt_data = [torch.FloatTensor(mod) for mod in corrupt_data]

            embedding = classifier(corrupt_data, heads=True)

            pred, individual = detector.detect_anomalies([mod.double() for mod in embedding], grace=grace, evaluating=True)

            for j in range(0,4):
                for k in range(j+1,4):
                    if j in corrupt or k in corrupt:
                        if not individual[j,k]:
                            tn_ind += 1
                        else:
                            fp_ind += 1
                    else:
                        if individual[j,k]:
                            tp_ind += 1
                        else:
                            fn_ind += 1
                        
            tp_pred += ((clean == pred) & (pred == True)).sum()
            tn_pred += ((clean == pred) & (pred == False)).sum()
            fp_pred += ((clean != pred) & (pred == True)).sum()
            fn_pred += ((clean != pred) & (pred == False)).sum()

            pbar.update(1)
            pbar.set_description('Prediction accuracy: {:.2%} | Individual accuracy: {:.2%} '.format((tp_pred+tn_pred)/(4*(1+(i/window))), (tp_ind+tn_ind)/(6*(1+(i/window)))))
    return np.array([[tp_pred, tn_pred, fp_pred, fn_pred],
                     [tp_ind, tn_ind, fp_ind, fn_ind]])


def eval_both(classifier, data, labels, detector, noise, num_corrupt=0, window=20, grace=0, round_func=np.round):
    classifier.eval()
    detector.grace=grace
    detector.round_func=round_func

    tp_pred, tn_pred, fp_pred, fn_pred, tp_ind, tn_ind, fp_ind, fn_ind = (0,0,0,0,0,0,0,0)
    correct_raw = 0
    correct_cleaned = 0
    correct_ad_cleaned = 0
    with tqdm.tqdm(total=len(range(0, len(data[0])-window, window))) as pbar:
        for i in range(0, len(data[0])-window, window):
            clean = np.ones(len(data))
            corrupt = np.random.default_rng().choice(len(data), size=num_corrupt, replace=False)
            clean[corrupt] = 0
            corrupt_data = noise([mod[i:i+window,:].numpy() for mod in data], corrupt)
            corrupt_data = [torch.FloatTensor(mod) for mod in corrupt_data]

            embedding = classifier(corrupt_data, heads=True)
            pred, individual = detector.detect_anomalies([mod.double() for mod in embedding], evaluating=True)

            for j in range(0,4):
                for k in range(j+1,4):
                    if j in corrupt or k in corrupt:
                        if not individual[j,k]:
                            tn_ind += 1
                        else:
                            fp_ind += 1
                    else:
                        if individual[j,k]:
                            tp_ind += 1
                        else:
                            fn_ind += 1
                        
            tp_pred += ((clean == pred) & (pred == True)).sum()
            tn_pred += ((clean == pred) & (pred == False)).sum()
            fp_pred += ((clean != pred) & (pred == True)).sum()
            fn_pred += ((clean != pred) & (pred == False)).sum()

            # Accuracy for raw data
            raw_pred = classifier(embedding, tails=True).detach()
            for sample in range(embedding[0].shape[0]):
                if np.argmax(raw_pred[sample]) == labels[i+sample]:
                    correct_raw += 1

            # Accuracy for perfectly cleaned data (labels)
            cleaned_data = embedding
            for mod in corrupt:
                cleaned_data[mod] = torch.zeros_like(cleaned_data[mod])

            cleaned_pred = classifier(cleaned_data, tails=True).detach()
            for sample in range(embedding[0].shape[0]):
                if np.argmax(cleaned_pred[sample]) == labels[i+sample]:
                    correct_cleaned += 1

            # Accuracy for ad cleaned data (predictions)
            ad_cleaned_data = embedding
            for mod in range(len(pred)):
                if pred[mod] == False:
                    ad_cleaned_data[mod] = torch.zeros_like(ad_cleaned_data[mod])

            ad_cleaned_pred = classifier(ad_cleaned_data, tails=True).detach()
            for sample in range(embedding[0].shape[0]):
                if np.argmax(ad_cleaned_pred[sample]) == labels[i+sample]:
                    correct_ad_cleaned += 1


            pbar.update(1)
            pbar.set_description('Prediction accuracy: {:.2%} | Individual accuracy: {:.2%} | Raw accuracy: {:.2%} | Cleaned accuracy: {:.2%} | AD cleaned accuracy: {:.2%}'.format((tp_pred+tn_pred)/(4*(1+(i/window))), (tp_ind+tn_ind)/(6*(1+(i/window))), correct_raw/(i+window), correct_cleaned/(i+window), correct_ad_cleaned/(i+window)))
    return (np.array([[tp_pred, tn_pred, fp_pred, fn_pred],
                     [tp_ind, tn_ind, fp_ind, fn_ind]]), correct_raw/(i+window), correct_cleaned/(i+window), correct_ad_cleaned/(i+window))

def pipeline(classifier, train_cca, train_ad, test_ad, cca_dim, snr, gmm_components, window_size, grace, cca_path, noise_gen=None, noise_type='gmm', thresh_method='intersection', thresh_method_param=None, corruption_classifier='proportional', train_snr=None, round_func=np.round, cca_weighting=np.mean):
    classifier.eval()
    train_snr = snr if train_snr is None else train_snr
    cca = dgcca.dgcca.DGCCA([[512, 125, 64]]*4, 32, device='cpu', use_all_singular_values=False)
    cca.load_checkpoint(cca_path)
    train_cca_embeddings = [torch.DoubleTensor(mod) for mod in embed(classifier, train_cca)]
    cca.train(train_cca_embeddings, epochs=50, batch_size=128, cca_dim=cca_dim, cca_hidden_dim=1000)

    clean = [torch.DoubleTensor(mod) for mod in embed(classifier, train_ad)]
    
    if noise_gen is None and noise_type=='gmm':
        noise_gen = utils.noise.GMMNoise(train_ad, gmm_components) # Train GMM for each pixel to generate noise

    if noise_type=='gmm':
        noise = lambda data: noise_gen.add_noise([mod.numpy() for mod in data], snr=train_snr) # noise and image weighted equally
    else:
        noise = lambda data: utils.noise.add_gaussian_noise(data, snr=train_snr)
    
    corrupt = [torch.DoubleTensor(mod) for mod in embed(classifier, train_ad, noise)]
    
    detector = dgcca.anomaly_detection.CcaAnomalyDetector(cca)
    detector.train(clean, corrupt, stride='auto', window=window_size, 
                   method=thresh_method, method_param=thresh_method_param, classifier=corruption_classifier, 
                   grace=grace, round_func=round_func, correlation_weighting=cca_weighting)

    if noise_type=='gmm':
        test_noise = lambda data, modality: noise_gen.add_noise(data, snr=snr, modality=modality)
    else:
        test_noise = lambda data, modality: utils.noise.add_gaussian_noise(data, snr=snr, modality=modality)

    results = {}
    for data, label in test_ad:
        data = [mod.double() for mod in data]
        for num_corrupt in [0,1,2]:
            results[num_corrupt] = {}
            results[num_corrupt]['ad'], results[num_corrupt]['raw'], results[num_corrupt]['clean'], results[num_corrupt]['ad_clean'] = eval_both(classifier, data, label, detector, test_noise, num_corrupt, window_size)
    
    return results
