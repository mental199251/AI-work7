## Upsampling Comparison

| Model | Backbone | Upsampling | Task | Parameters (M) | Model Size (MB) | Pixel Accuracy (%) | mIoU (%) | Inference (ms/image) | Training Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fcn_resnet34 | resnet34 | bilinear | five_class | 21.289 | 81.28 | 94.35 | 79.85 | 2.59 | 498.36 |
| fcn_resnet34_deconv | resnet34 | deconv | five_class | 21.291 | 81.28 | 94.45 | 79.68 | 3.20 | 433.74 |

## Per-Class IoU

| Model | background IoU (%) | person IoU (%) | car IoU (%) | cat IoU (%) | dog IoU (%) |
| --- | --- | --- | --- | --- | --- |
| fcn_resnet34 | 94.00 | 78.52 | 78.94 | 76.35 | 71.42 |
| fcn_resnet34_deconv | 94.07 | 79.03 | 76.99 | 77.00 | 71.30 |

