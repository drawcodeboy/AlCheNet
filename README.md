# Introduction
* Keyword: Mamba-VaDE, Clustering, EEG, Sleep-Stage
# Install
```
docker pull ubuntu:22.04
docker run -itd --gpus=all --shm-size --name <container_name> ubuntu:22.04

apt-get update
apt-get install sudo
sudo apt-get install git
git clone https://github.com/drawcodeboy/EEG-Sleep.git .

sudo apt-get install wget
sudo apt-get install curl
sudo apt-get install unzip
wget -r -N -c -np https://physionet.org/files/siena-scalp-eeg/1.0.0/ -P data/
curl -L -o ./data/open-nuro-dataset.zip https://www.kaggle.com/api/v1/datasets/download/yosftag/open-nuro-dataset
unzip data/open-nuro-dataset.zip -d data/open-nuro-dataset
```

# References
VaDE Repository: https://github.com/mori97/VaDE/blob/master/vade.py