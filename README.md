# Real-Time Robustness to Modality Corruption in Multimodal Machine Learning

Machine learning has been an active area of research in recent years and has been deployed in a number of tasks. Machine learning models often use data from a single view, or modality, such as an array of sensors on a single device or the image stream of a video. However, research has shown that combining data from multiple modalities can increase performance. Multimodal learning comes with a number of challenges, including deciding when to fuse data from each modality and dealing with failure in individual modalities.

Existing research and implementations often assume that the multimodal data supplied to the model is complete and clean. However, in real world conditions data from a particular
modality or sensor may be corrupted or completely missing, resulting in inaccurate results from such a model. Multimodal models should either understand when they have been fed
corrupted samples, suggesting the output could be inaccurate, or be trained in such a way that they are robust to them.

This project aims to maximise model performance in the presence of missing and corrupt modalities. A novel anomaly detection pipeline based on Canonical Correlation Analysis is presented, able to detect corruption in all modalities simultaneously, provided at least 2 remain clean. The pipeline is applied to a multimodal MNIST classifier, achieving accuracy increases of up to 28.1% on highly corrupted data, whilst giving no detriment to accuracy when no corruption is present.

The [Final Report](./docs/Dissertation/Dissertation.pdf) got a grade of 80, and I hope to continue working on and potentially publishing it in future.

Demonstration provided using the mm-fit multimodal human activity recognition dataset. Data available from the [MM-Fit website](https://mmfit.github.io/).
