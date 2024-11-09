# [TGRS2024] PIN-DGA: Enhanced Generalization Ability of Infrared Small Target Detection via Perturbed Training and Adaptive Test

# Abstract: 

While deep learning (DL) methods based on empirical risk minimization (ERM) have achieved superior detection performance on public infrared small target detection (IRSTD) datasets, the learned models suffer severe performance degradation when dealing with out-of-distribution (OOD) data. The primary challenge in practical infrared systems lies in enhancing the generalization ability to ensure reliable detection performance in diverse scenarios. In this article, we propose a single-source domain generalization (SDG) and test-time adaptation (TTA) approach for IRSTD, aiming to train a domain-agnostic DL model in the presence of only one source domain and enabling it to perform well on any unseen target domain. During the training process, perturbed images and networks (PINs) are utilized to enhance the diversity of features extracted from a single source domain with limited samples. A Y-shaped multibranch network employs a combination of the self-supervised recovery task and the supervised detection task. Their losses jointly guide the network to converge explicitly to the flat minima in the loss landscape. During the test process, we apply domain-guided adaptation (DGA) to the learned model. The feature statistics of the dynamically inputted image batches are considered as cues to the target domain. Batch normalization (BN) layers are adaptively fine-tuned online based on the differences in feature statistics between the source model and target images. We conducted extensive ablation studies to demonstrate the rationality and effectiveness of each component of the framework. Compared to other SDG, TTA, and model-driven IRSTD methods, PINDGA
can more effectively improve the detection performance of DL methods on unseen target domains. As a model-agnostic framework, we verified its compatibility with current state-of-theart networks. On three OOD datasets, our method can improve
the average probability of detection (PD) by 16.08 and reduce false alarm targets (FATs) by 33%.

![image](https://github.com/user-attachments/assets/ea9be1f3-cfb4-4f09-9fb9-6e84fcbf3ea9)

# Usage

1. Download datasets by https://drive.google.com/file/d/1BRZvfkM3_8IbSp95dmjzHifWLcyErB35/view?usp=drive_link and unzip it. (Sequence f2 in the IRSTScenes dataset is private.)
2. Train a network: run 'train.py'
3. Infer by a learned model: run 'inference.py'
4. Get metrics: run 'get_metrics.py'

# Full-text

https://ieeexplore.ieee.org/abstract/document/10736658

Chen G, Wang W, Wang Z, et al. Enhanced Generalization Ability of Infrared Small Target Detection via Perturbed Training and Adaptive Test[J]. IEEE Transactions on Geoscience and Remote Sensing, 2024.


If you find the work helpful, please give me a star and cite the paper.

Thank you!

