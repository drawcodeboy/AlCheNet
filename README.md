# Introduction
* Keyword: Alzheimer's Disease
# Install
```
docker pull ubuntu:22.04
docker run -itd --gpus=all --shm-size --name <container_name> ubuntu:22.04

apt-get update
apt-get install sudo
sudo apt-get install git
git clone https://github.com/drawcodeboy/EEG-Sleep.git .

sudo apt-get install curl
sudo apt-get install unzip
curl -L -o ./data/open-nuro-dataset.zip https://www.kaggle.com/api/v1/datasets/download/yosftag/open-nuro-dataset
unzip data/open-nuro-dataset.zip -d data/open-nuro-dataset
```
# Result
```
=====================[Test Channel-wise encoder]=====================
device: cuda:1
Load Dataset Nuro
Evaluate: 100.00%
Test Time: 00m 01s
Accuracy: 0.4882
F1-Score(Macro): 0.4613
Precision(Macro): 0.4609
Recall(Macro): 0.4677
(.venv) root@9199914e73eb:/home# python test.py --config=transformer
=====================[Test Transformer]=====================
device: cuda:1
Load Dataset Nuro
Evaluate: 100.00%
Test Time: 00m 01s
Accuracy: 0.4645
F1-Score(Macro): 0.4371
Precision(Macro): 0.4383
Recall(Macro): 0.4457
(.venv) root@9199914e73eb:/home# python test.py --config=gcnet
=====================[Test GCNet]=====================
device: cuda:1
Load Dataset Nuro
Evaluate: 100.00%
Test Time: 00m 02s
Accuracy: 0.5570
F1-Score(Macro): 0.5263
Precision(Macro): 0.5229
Recall(Macro): 0.5341
(.venv) root@9199914e73eb:/home# python test.py --config=chebnet
=====================[Test ChebNet]=====================
device: cuda:1
Load Dataset Nuro
Evaluate: 100.00%
Test Time: 00m 02s
Accuracy: 0.5871
F1-Score(Macro): 0.5724
Precision(Macro): 0.5799
Recall(Macro): 0.5735
(.venv) root@9199914e73eb:/home# python test.py --config=gsnet
=====================[Test GSNet]=====================
device: cuda:1
Load Dataset Nuro
Evaluate: 100.00%
Test Time: 00m 02s
Accuracy: 0.4613
F1-Score(Macro): 0.4496
Precision(Macro): 0.4544
Recall(Macro): 0.4461
(.venv) root@9199914e73eb:/home# python test.py --config=gatnet 
=====================[Test GATNet]=====================
device: cuda:1
Load Dataset Nuro
Evaluate: 100.00%
Test Time: 00m 02s
Accuracy: 0.4753
F1-Score(Macro): 0.4413
Precision(Macro): 0.4438
Recall(Macro): 0.4555
```