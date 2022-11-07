# QueryOTR

## Outpainting by Queries, ECCV 2022. [ArXiv](https://arxiv.org/abs/2207.05312)

we propose a novel hybrid vision-transformer-based encoder-decoder framework, named Query Outpainting TRansformer (QueryOTR), for extrapolating visual context all-side around a given image. Patch-wise mode's global modeling capacity allows us to extrapolate images from the attention mechanism's query standpoint. A novel Query Expansion Module (QEM) is designed to integrate information from the predicted queries based on the encoder's output, hence accelerating the convergence of the pure transformer even with a relatively small dataset. To further enhance connectivity between each patch, the proposed Patch Smoothing Module (PSM) re-allocates and averages the overlapped regions, thus providing seamless predicted images. We experimentally show that QueryOTR could generate visually appealing results smoothly and realistically against the state-of-the-art image outpainting approaches.

<div style="align: center">
<img src="./assets/demo.jpg" width="700px">
</div>

## 1. Requirements
PyTorch >= 1.10.1;
python >= 3.7;
CUDA >= 11.3;
torchvision;

NOTE: The code was tested to work well on Linux with torch 1.7, 1.9 and Win10 with torch 1.10.1. However, there is potential "Inplace Operation Error" bug if you use PyTorch < 1.10, which is quiet subtle and not fixed. If you found why the bug occur, pls let me know.

## News:
\[2022/11/7\] We update the code. We found the [official MAE](https://github.com/facebookresearch/mae) code may degrade the performance by somehow (about 0.5-1 in terms of FID) and we go back to [unofficial MAE](https://github.com/pengzhiliang/MAE-pytorch). Meanwhile, we upload a trained checkpoints on Scenery [google drive](https://drive.google.com/drive/folders/1s_Qs6m314a5vwLzdQ58uKOveK6fZjgaB?usp=share_link) which can reach FID 20.38, IS 3.959.


## 2. Data preparation

### Scenery
Scenery consists of about 6,000 images, and we randomly select 1,000 images for evaluation. The training and test dataset can be down [here](https://github.com/z-x-yang/NS-Outpainting)

Meanwhile, we also provide the Scenery dataset that we have split here [baidu_pan](https://pan.baidu.com/s/1Zn5X3jfqr6x3ho705VMHZA?pwd=qotr).

### Building
Building contains about 16,000 images in the training set and 1,500 images in the testing set, which can be found in [here](https://github.com/PengleiGao/UTransformer)

### WikiArt
The WikiArt datasets can be downloaded [here](https://github.com/cs-chan/ArtGAN/tree/master/WikiArt%20Dataset). We perform a split manner of genres datasets, which contains 45,503 training images and 19,492 testing images

## 3. Training and evaluation
Before you reimplement our results, you need to download the ViT pretrain checkpoint [here](https://drive.google.com/drive/folders/1ZVzOD-ZGPBNtJ4HtsR-8IIH7Cm40LiMW?usp=share_link), and then initialize the encoder weight.


Training on your datasets, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --name=EXPERIMENT_NAME --data_root=YOUR_TRAIN_PATH --patch_mean=YOUR_PATCH_MEAN --patch_std=YOUR_PATCH_STD
```

Evaluate on your datasets, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python evaluate.py --r=EXPERIMENT_NAME --data_root=YOUR_TEST_PATH --patch_mean=YOUR_PATCH_MEAN --patch_std=YOUR_PATCH_STD
```




## Acknowledgements

Our codes are built upon MAE, [pytroch-fid](https://github.com/mseitzer/pytorch-fid) and [inception score](https://github.com/sbarratt/inception-score-pytorch)

## Citation

```
@inproceedings{yao2022qotr,
  title={Outpainting by Queries},
  author={Yao, Kai and Gao, Penglei and Yang, Xi and Sun, Jie and Zhang, Rui and Huang, Kaizhu},
  booktitle={European Conference on Computer Vision},
  pages={153--169},
  year={2022},
  organization={Springer}
}
```
