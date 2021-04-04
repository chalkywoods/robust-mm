#!/usr/bin/env python

from __future__ import print_function
import drmaa
import os


def main():
    """
    Create a DRMAA session then submit a job.
    Note, need file called myjob.sh in the current directory.
    """
    with drmaa.Session() as s:
        for batch_size in [128]:
            for snr in [0,0.25,0.5,0.75,1,2]:
                for gmm_components in [1,3,5,7,9]:
                    for cca_dim in [15]:
                        for window_size in [30]:
                            for grace in [0]:
                                param_string = 'batch_size: {}, snr: {}, gmm_components: {}, cca_dim: {}, window_size: {}, grace: {}'.format(batch_size, snr, gmm_components, cca_dim, window_size, grace)
                                #print('Running anomaly detection with {}'.format(param_string))
                                jt = s.createJobTemplate()

                                jt.remoteCommand = os.path.join(os.getcwd(), 'pipeline.sh')
                                jt.args = [batch_size, snr, gmm_components, cca_dim, window_size, grace]
                                
                                job_id = s.runJob(jt)
                                print('Job {} submitted with params {}'.format(job_id, param_string))
                                s.deleteJobTemplate(jt)


if __name__ == '__main__':
    main()