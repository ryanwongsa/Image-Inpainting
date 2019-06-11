## Image Inpainting

Start Instance

- added tag jupyter to instance

jupyter notebook --generate-config

```
c = get_config()
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
```

git clone https://github.com/ryanwongsa/image-inpainting.git

===============================================================
tmux
virtualenv venv
source venv/bin/activate
pip install awscli
cd image-inpainting
mkdir dataset
aws s3 --no-sign-request sync s3://open-images-dataset/train dataset/train

===============================================================
tmux
cd image-inpainting/dataset
wget https://www.dropbox.com/s/qp8cxqttta4zi70/irregular_mask.zip?dl=0
unzip irregular_mask.zip\?dl\=0 

===============================================================
tmux
cd image-inpainting
jupyter notebook