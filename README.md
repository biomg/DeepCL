# Convolutional networks with parallel structure for metastatic prostate cancer prediction
Accurately predicting the future cancer status of cancer patients is crucial for their treatment. Numerous scientific studies indicate a strong correlation between cancer and the genetic mutations of patients. With the continuous development of genetic mutation detection technologies, discovering one's mutated genes has become increasingly accessible. Additionally, a substantial amount of somatic gene mutation data has been generated in medical and research processes. Using artificial intelligence to process this data is of paramount importance. In this study, firstly, we propose a mutation data preprocessing method for easier extraction of mutation data features. Second, we use convolutional networks with parallel structure (CNPS) to extract features of gene mutation data in different dimensions to make more accurate judgment. Finally, CNPS is highly interpretable, and by analyzing CNPS, we can extract genes that play crucial roles in metastatic prostate cancer. After training, CNPS achieves higher accuracy results on both the MPC dataset and the MSK-MET dataset.  
# Dependency:
Python 3.7 <br>
Pytorch 1.12.1 <br>
numpy 1.18.5 <br>
scikit-learn 0.23.2
# Supported GPUs
Now it supports GPUs. The code support GPUs and CPUs, it automatically check whether you server install GPU or not, it will proritize using the GPUs if there exist GPUs.
In addition, WVDL can also be adapted to protein binding sites on DNAs and identify DNA binding speciticity of proteins.
It supports model training, testing.
# Usage:
python DCSN.py
python data_analizyy.py
python get_crucial_gene.py
# Contact
Junjiang Liu:junjiang.liu@foxmail.com
# Updates:
21/12/2022
