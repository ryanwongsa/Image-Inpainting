# Image Inpainting

A PyTorch Implmentation of the paper, [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/pdf/1804.07723.pdf). Architecture may not be an exact match of due to the limited description of hyperparameters and architecture details.

## Instructions

1. conda env create -f imageinpaintingenv.yml
2. source activate imageinpaintingenv

<!-- TODO: Use nvidia dali dataloader for faster data loading -->
<!-- 6. pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/9.0 nvidia-dali -->
<!-- pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/10.0 nvidia-dali -->


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
    --height 512 \
    --width 512 \
    --num_workers 2 \
    --learning_rate 0.0002 \
    --train_percent_check 1.0 \
    --val_check_interval 1.0 \
    --batch_size 6 \
    --save_path "/content/data/logs" \

```

### Continue from pretrained model

```cmd
python main.py \
    --logs_dir "/content/data/logs/lightning_logs/version_1" \
    --checkpoint_name "_ckpt_epoch_1.ckpt" \
    --train_dir "/content/data/train" \
    --valid_dir "/content/data/test" \
    --mask_dir "/content/data/irregular_mask/disocclusion_img_mask" \
    --height 512 \
    --width 512 \
    --num_workers 2 \
    --learning_rate 0.0002 \
    --train_percent_check 1.0 \
    --val_check_interval 1.0 \
    --batch_size 6 \
    --save_path "/content/data/logs"
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
- [ ] Update dataloader to use nvidia-dali
- [ ] Refactor code to use pytorch-lightning
- [ ] Create Colab prediction implementation
