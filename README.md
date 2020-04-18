# Image Inpainting


### Installation commands

```
sudo conda install ignite -c pytorch-nightly
sudo conda install pip
sudo pip install neptune-client 

git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..

```

### Small Dataset Test Script
```
cd dataset
mkdir train
cd train
aws s3 --no-sign-request cp s3://open-images-dataset/tar/validation.tar.gz .
tar -xzf validation.tar.gz
rm validation.tar.gz
mkdir test
cd validation
mv `ls | head -100` ../test
```