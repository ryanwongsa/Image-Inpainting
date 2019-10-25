# Image Inpainting

A PyTorch Implmentation of the paper, [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/pdf/1804.07723.pdf). Architecture may not be an exact match of due to the limited description of hyperparameters and architecture details.

See `lightning` branch for latest developments in generalising the training and monitoring process.

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
- [ ] Refactor code to use pytorch-lightning
- [ ] Create Colab prediction implementation
