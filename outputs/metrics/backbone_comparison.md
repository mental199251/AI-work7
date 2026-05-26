## Backbone Comparison

| Model | Backbone | Upsampling | Task | Parameters (M) | Model Size (MB) | Pixel Accuracy (%) | mIoU (%) | Inference (ms/image) | Training Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fcn_resnet18 | resnet18 | bilinear | five_class | 11.181 | 42.69 | 92.56 | 73.63 | 1.55 | 245.37 |
| fcn_resnet34 | resnet34 | bilinear | five_class | 21.289 | 81.28 | 94.35 | 79.85 | 2.59 | 498.36 |
| fcn_resnet50 | resnet50 | bilinear | five_class | 23.526 | 89.95 | 94.31 | 79.60 | 3.25 | 406.23 |

## Per-Class IoU

| Model | background IoU (%) | person IoU (%) | car IoU (%) | cat IoU (%) | dog IoU (%) |
| --- | --- | --- | --- | --- | --- |
| fcn_resnet18 | 92.39 | 73.38 | 73.67 | 66.97 | 61.74 |
| fcn_resnet34 | 94.00 | 78.52 | 78.94 | 76.35 | 71.42 |
| fcn_resnet50 | 93.99 | 77.59 | 78.42 | 76.45 | 71.55 |

