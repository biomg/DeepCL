# Prediction of Drug-Target Interactions Based on Deep Contrastive Learning
During the drug discovery process, determining the binding of molecules to protein targets is one of the most expensive steps. Therefore, precise, high-throughput computational prediction of drug-target interactions (DTI) can help prioritize potential experimental screening candidates. By using projectors to preprocess drug molecules and target protein data, we can improve the accuracy of protein sequence data, enhance the performance of prediction models, and quickly identify potential drug-target interactions. In our study, we propose a deep contrastive learning (DeepCL) method to predict kinase-drug interactions. First, we integrates drug and target data into a unified dataset, simplifying the process of establishing relationships between drugs and targets. Second, we utilizes a pre-trained model to process both drug molecules and target protein data, ensuring efficient and accurate feature extraction.
Third, the processed data is fed into a projector composed of fully connected neural networks for training, enhancing the model's ability to learn and predict interactions effectively. This method achieved competitive experimental results compared to other state-of-the-art methods on three low-coverage benchmark public datasets. Compared to other models, it demonstrated faster computational speed and improved prediction accuracy.
# Dependency:
Python 3.9 <br>
Pytorch 1.10.1 <br>
numpy 1.18.5 <br>
scikit-learn 0.23.2
# Supported GPUs
It now supports GPUs.The code supports both GPUs and CPUs.It automatically checks if the server has a GPU installed and prioritizes the GPU if it is present.In addition, DeepCL works with protein binding sites and identifies the binding affinity of a protein target to a drug. It supports model training and testing.

# Usage:
python train.py
python predict.py

# Contact
Jinlong Li: lijinlong07@foxmail.com
# Updates:
24/8/2024
