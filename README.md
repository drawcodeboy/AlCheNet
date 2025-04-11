
```
docker pull ubuntu:22.04
docker run -itd --gpus=all --shm-size --name <container_name> ubuntu:22.04

apt-get update
apt-get install sudo
sudo apt-get install git
git clone https://github.com/drawcodeboy/EEG-Sleep.git .

sudo apt-get install wget
wget -r -N -c -np https://physionet.org/files/siena-scalp-eeg/1.0.0/ -P data/

```