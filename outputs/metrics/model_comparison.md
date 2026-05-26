## Model Comparison

| Model | Parameters (M) | Model Size (MB) | Pixel Accuracy (%) | mIoU (%) | Inference (ms/image) | Training Time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| fcn_resnet18 | 11.181 | 42.69 | 92.56 | 73.63 | 2.12 | 245.37 |
| fcn_resnet34 | 21.289 | 81.28 | 94.35 | 79.85 | 2.57 | 498.36 |

## Per-Class IoU

| Model | background IoU (%) | person IoU (%) | car IoU (%) | cat IoU (%) | dog IoU (%) |
| --- | --- | --- | --- | --- | --- |
| fcn_resnet18 | 92.39 | 73.38 | 73.67 | 66.97 | 61.74 |
| fcn_resnet34 | 94.00 | 78.52 | 78.94 | 76.35 | 71.42 |

