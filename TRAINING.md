# Training

Below, we provide the training commands for training Segmenter + ALGM on the following datasets:
* [ADE20K](#ade20k)
* [Cityscapes](#cityscapes)
* [Pascal-Context](#pascal-context)

Different backbone possibilities:
* ViT-Ti: `vit_tiny_patch16_384`
* ViT-S: `vit_small_patch16_384`
* ViT-B: `vit_base_patch16_384`
* ViT-L: `vit_large_patch16_384`

For more configuration options, see [segm/config.yml](segm/config.yml) and [segm/train.py](segm/train.py).

## ADE20K

Segmenter + ALGM, and ViT-Ti backbone:

```bash
python -m segm.train    --log-dir runs/vit_tiny_layers_1_5_T_0.9 \
                        --dataset ade20k \
                        --backbone vit_tiny_patch16_384 \
                        --decoder mask_transformer \
                        --patch-type algm \
                        --selected-layers 1 5 \
                        --threshold 0.9

```

Segmenter + ALGM, and ViT-S backbone:

```bash
python -m segm.train  --log-dir runs/vit_small_layers_1_5_T_0.88/ \
                      --dataset ade20k \
                      --backbone vit_small_patch16_384 \
                      --decoder mask_transformer \
                      --patch-type algm \
                      --selected-layers 1 5 \
                      --merging-window-size 2 2 \
                      --threshold 0.88 
```

Segmenter + ALGM, and ViT-B backbone:
```bash
python -m segm.train    --log-dir runs/vit_base_layers_1_5_T_0.94 \
                        --dataset ade20k \
                        --backbone vit_base_patch16_384 \
                        --decoder mask_transformer \
                        --patch-type algm \
                        --selected-layers 1 5 \
                        --threshold 0.94
```

Segmenter + ALGM, and ViT-L backbone:

```bash
python -m segm.train    --log-dir runs/vit_large_layers_1_7_T_0.95 \
                        --dataset ade20k_large \
                        --backbone vit_large_patch16_384 \
                        --decoder mask_transformer \
                        --patch-type algm \
                        --selected-layers 1 7 \
                        --threshold 0.95
```

## Cityscapes 

Segmenter + ALGM, and ViT-S backbone:

```bash
python -m segm.train  --log-dir runs/vit_small_layers_1_5_T_0./ \
                      --dataset Cityscapes \
                      --backbone vit_small_patch16_384 \
                      --decoder mask_transformer \
                      --patch-type algm \
                      --selected-layers 1 5 \
                      --merging-window-size 2 2 \
                      --threshold  0.955
```


## Pascal-Context

Segmenter + ALGM, and ViT-S backbone:

```bash
python -m segm.train  --log-dir runs/vit_small_layers_1_5_T_0./ \
                      --dataset  pascal_context \
                      --backbone vit_small_patch16_384 \
                      --decoder mask_transformer \
                      --patch-type algm \
                      --selected-layers 1 5 \
                      --merging-window-size 2 2 \
                      --threshold 0.88
```