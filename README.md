# Disentangling Object Motion and Occlusion for Unsupervised Multi-frame Monocular Depth

This paper has been accepted by [ECCV 2022](https://eccv2022.ecva.net/)

By [Ziyue Feng](https://ziyue.cool), [Liang Yang](https://ericlyang.github.io/), [Longlong Jing](https://longlong-jing.github.io/), [Haiyan Wang](https://haiyan-chris-wang.github.io/), [Yingli Tian](https://www.ccny.cuny.edu/profiles/yingli-tian), and [Bing Li](https://www.clemson.edu/cecas/departments/automotive-engineering/people/li.html).

Arxiv: [Link](https://arxiv.org/abs/2203.15174)

![teaser.png](assets/teaser.png)

## Architecture:

![Architecture.png](assets/Architecture.png)

## ⚙️ Setup

You can install the dependencies with:

```shell
conda create -n dynamicdepth python=3.6.6
conda activate dynamicdepth
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
pip install tensorboardX==1.4
conda install opencv=3.3.1   # just needed for evaluation
pip install open3d
pip install wandb
pip install scikit-image
```

We ran our experiments with PyTorch 1.8.0, CUDA 11.1, Python 3.6.6 and Ubuntu 18.04.

## 💾 KITTI Data Prepare

**Download Data**

First download the KITTI RAW dataset, put in the `kitti_data` folder.

Our default settings expect that you have converted the png images to jpeg with this command, which also deletes the raw KITTI `.png` files:

```shell
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```

or you can skip this conversion step and train from raw png files by adding the flag `--png` when training, at the expense of slower load times.

## 💾 Cityscapes Data Prepare

First preprocess the Cityscapes dataset using SfMLearner's prepare_train_data.py script.
We used the following command:

```bash
python prepare_train_data.py \
    --img_height 512 \
    --img_width 1024 \
    --dataset_dir <path_to_downloaded_cityscapes_data> \
    --dataset_name cityscapes \
    --dump_root <your_preprocessed_cityscapes_path> \
    --seq_length 3 \
    --num_threads 8
```

## ⏳ Training

By default models and log event files are saved to `log/mdp/`.

```shell
python -m dynamicdepth.train  # the configs are defined in options.py
```

## ⏳ Evaluating

```shell
python -m dynamicdepth.evaluate_depth  # the configs are defined in options.py
```

### Citation

```
@article{dynamicdepth,
  title={Disentangling Object Motion and Occlusion for Unsupervised Multi-frame Monocular Depth},
  author={Feng, Ziyue and Yang, Liang and Jing, Longlong and Wang, Haiyan and Tian, YingLi and Li, Bing},
  journal={arXiv preprint arXiv:2203.15174},
  year={2022}
}
```
