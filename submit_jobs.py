#!/usr/bin/env python

from __future__ import print_function
import drmaa
import os


def main():
    """
    Create a DRMAA session then submit a job.
    Note, need file called myjob.sh in the current directory.
    """
    noise_type = 'gmm'
    #thresh_method = 'intersection'
    seed=0
    train_snr=-1
    with drmaa.Session() as s:
        jt = s.createJobTemplate()
        jt.remoteCommand = os.path.join(os.getcwd(), 'pipeline.sh')
        jt.nativeSpecification = '-l h_rt=00:10:00 -l rmem=4G'
        jt.blockEmail = False
        jt.email = ['pipeline@hggwoods.com']
        for ad_classifier in ['ceiling', 'round', 'floor']:#, 'delta', 'prob', 'none']:
            for thresh_method in ['intersection', 'noise_mean', 0.1, 0.2, 0.3, 0.4]:
                for batch_size in [128]:
                    for snr in [0.0,0.2,0.4,0.6,0.8,1.0,1.2,1.4,1.6,1.8,2.0]:
                        for gmm_components in [5]:
                            for cca_dim in [15]:
                                for window_size in [30]:
                                    for grace in [0]:
                                        for repeat in range(1):
                                            param_string = 'batch_size: {}, snr: {}, gmm_components: {}, cca_dim: {}, window_size: {}, grace: {}, noise_type: {}, repeat: {}, threshold_method: {}, ad_classifier: {}'.format(batch_size, snr, gmm_components, cca_dim, window_size, grace, noise_type, repeat, thresh_method, ad_classifier)
                                            jt.args = [batch_size, snr, gmm_components, cca_dim, window_size, grace, thresh_method, noise_type, repeat, seed, train_snr, ad_classifier]
                                            job_id = s.runJob(jt)
                                            print('Job {} submitted with params {}'.format(job_id, param_string))
        s.deleteJobTemplate(jt)


if __name__ == '__main__':
    main()