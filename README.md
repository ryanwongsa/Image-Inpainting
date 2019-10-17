# Image Inpainting

A PyTorch Implmentation of the paper, [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/pdf/1804.07723.pdf). Architecture may not be an exact match of due to the limited description of hyperparameters and architecture details.

## Instructions

1. conda create -n imageinpaintingenv

2. source activate imageinpaintingenv

3. conda install pip
4. conda install pytorch torchvision cudatoolkit=9.2 -c pytorch -c defaults -c numba/label/dev
5. conda install jupyter
6. pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/9.0 nvidia-dali
7. pip install pytorch-lightning
8. conda install -c menpo opencv


## Limitations

Trained on Open Images dataset for 5 days. Requires more 4 more days of training and 1 day of fine tuning (around $300 of GCP credit).

Latest model available [here](https://drive.google.com/drive/folders/1FgRREp38REeVGc1FVAoFThS1IUWGvbeB?usp=sharing).

## References

- Partial Conv: https://github.com/NVIDIA/partialconv
- Keras Implementation: https://github.com/MathiasGruber/PConv-Keras

## TODO

- [x] Create architecture
- [x] Implement training architecture
- [x] Save the model to an external location
- [x] Document code
- [ ] Train for full duration
- [ ] Update dataloader to use nvidia-dali
- [ ] Refactor code to use pytorch-lightning
- [ ] Create Colab prediction implementation
