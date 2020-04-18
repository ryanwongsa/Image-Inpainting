# Image Inpainting

A PyTorch Implmentation of the paper, [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/pdf/1804.07723.pdf). Architecture may not be an exact match of due to the limited description of hyperparameters and architecture details.

*NOTE: See lightning branch for original pytorch-lightning version*

*TODO: Update README with latest pytorch ignite training and evaluating commands*


## Instructions

```
sudo conda install ignite -c pytorch-nightly
sudo conda install pip
sudo pip install neptune-client 
```

## Results

Currently results shows training done on the Goolge Landmark v2 Dataset on a single P100 GPU.

> Note: Training for these results were only for 4.5 hours, while original paper trained the model on a V100 GPU for 10 days.


### Example Results (so far)

- 4.5 Hours of Training
- tv loss is very low at the moment, might need to increase the scaling factor.
- highest loss is style out.


![Example Images](res/sample_training.JPG)

### Validation Set

![Example Images](res/individualImage.png)

> Currently optimizing training and hyperparameters before training for full duration longer.

## Training the model

Use command below for more parameter details.
```cmd
python main.py -h
```

### Training from scratch

```cmd
python main.py \
    --train_dir "/content/data/train" \
    --valid_dir "/content/data/test" \
    --mask_dir "/content/data/irregular_mask/disocclusion_img_mask" \
    --save_path "/content/gdrive/My Drive/image-inpainting" \
    --train_percent_check 1.0 \
    --val_check_interval 0.5

```

### Continue from pretrained model

```cmd
python main.py \
    --version_number 1 \
    --checkpoint_dir "/content/gdrive/My Drive/image-inpainting/default/version_0/checkpoints/_ckpt_epoch_2.ckpt" \
    --train_dir "/content/data/train" \
    --valid_dir "/content/data/test" \
    --mask_dir "/content/data/irregular_mask/disocclusion_img_mask" \
    --save_path "/content/gdrive/My Drive/image-inpainting" \
    --train_percent_check 1.0 \
    --val_check_interval 0.5
```

## References

- [Partial Conv](https://github.com/NVIDIA/partialconv)
- [Keras Implementation](https://github.com/MathiasGruber/PConv-Keras)
- [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/pdf/1804.07723.pdf)

## TODO

- [x] Create architecture
- [x] Implement training architecture
- [x] Save the model to an external location
- [x] Document code
- [ ] Train for full duration
- [ ] Add mixed precision support
- [x] Update dataloader to use nvidia-dali
- [x] Refactor code to use pytorch-lightning
- [ ] Create Colab prediction implementation


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