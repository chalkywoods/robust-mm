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
    with drmaa.Session() as s:
        for batch_size in [128]:
            for snr in [0,0.25,0.5,0.75,1,2]:
                for gmm_components in [5]:
                    for cca_dim in [15]:
                        for window_size in [100]:
                            for grace in [0,1]:
                                param_string = 'batch_size: {}, snr: {}, gmm_components: {}, cca_dim: {}, window_size: {}, grace: {}, noise_type: {}'.format(batch_size, snr, gmm_components, cca_dim, window_size, grace, noise_type)
                                jt = s.createJobTemplate()

                                jt.remoteCommand = os.path.join(os.getcwd(), 'pipeline.sh')
                                jt.args = [batch_size, snr, gmm_components, cca_dim, window_size, grace, noise_type]
                                jt.nativeSpecification = '-l h_rt=00:30:00 -l rmem=4G'
                                jt.blockEmail = False
                                jt.email = ['pipeline@hggwoods.com']
                                job_id = s.runJob(jt)
                                print('Job {} submitted with params {}'.format(job_id, param_string))
                                s.deleteJobTemplate(jt)


if __name__ == '__main__':
    main()