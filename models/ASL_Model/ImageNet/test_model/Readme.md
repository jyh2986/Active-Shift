### Test model for ImageNet

This directory contains trained model for ImageNet using AS-ResNet.

| Model | Top-1 (%) | Top-5 (%) | Params(M) |
|:-----:|:---------:|:---------:|:---------:|
| w32   | 64.1      |    85.4   |    0.9    |
| w50   | 69.9      |    89.3   |    1.96   |
| w68   | 72.2      |    90.7   |    3.42   |


#### Caution

Deploy files are only valid for a inference. Batch normalization (BN) layer is merged with scale layer for the fast inference. 

Do not forget to add BN layer for the training.
