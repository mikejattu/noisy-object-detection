# noisy-object-detection

## Models

We compare vision models along axes that should interact with image
corruption. One **primary axis** (shape vs. texture bias) is spanned with an
architecture-matched pair; the remaining models are controls borrowed from
adjacent axes (spatial integration, training objective, explicit robustness).
Each model is included to carry a specific contrast.

| Model | Axis / role | Source |
|---|---|---|
| ResNet-50 (ImageNet) | Reference + texture-biased pole | `timm` |
| ResNet-50 (Stylized-ImageNet) | Shape-biased pole, matched to reference | Geirhos `texture-vs-shape` |
| ViT-B/16 (supervised) | Spatial integration (attention vs. conv) | `timm` |
| ConvNeXt-B | Conv operation vs. modern recipe | `timm` |
| CLIP ViT-B/16 | Training objective (language-aligned) | OpenCLIP |
| ResNet-50 (AugMix / DeepAugment) | Explicit-robustness control | RobustBench |

Weights download by name on first use; they are not committed to the repo.