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
python -m pip install cityscapesscripts
```

We ran our experiments with PyTorch 1.8.0, CUDA 11.1, Python 3.6.6 and Ubuntu 18.04.

## 💾 Cityscapes Data Prepare

Pull the repository and make a folder named CS_RAW for cityscapes raw data:

```bash
git clone https://github.com/AutoAILab/DynamicDepth.git
cd DynamicDepth
cd data
mkdir CS_RAW
```

From [Cityscapes official website](https://www.cityscapes-dataset.com/) download the following packages: 1) `leftImg8bit_sequence_trainvaltest.zip`, 2) `camera_trainvaltest.zip` into the `CS_RAW` folder.

Preprocess the Cityscapes dataset using the `prepare_train_data.py`(from SfMLearner) script with following command:

```bash
cd CS_RAW
unzip leftImg8bit_sequence_trainvaltest.zip
unzip camera_trainvaltest.zip
cd ..

python prepare_train_data.py \
    --img_height 512 \
    --img_width 1024 \
    --dataset_dir CS_RAW \
    --dataset_name cityscapes \
    --dump_root CS \
    --seq_length 3 \
    --num_threads 8
```

Prepare dynamic object mask `doj_mask` (To be updated soon)

## ⏳ Training

By default models and log event files are saved to `log/mdp/`.

```shell
python -m dynamicdepth.train  # the configs are defined in options.py
```

## ⏳ Evaluating

```shell
python -m dynamicdepth.evaluate_depth  # the configs are defined in options.py
```

## Citation

```
@article{feng2022disentangling,
  title={Disentangling Object Motion and Occlusion for Unsupervised Multi-frame Monocular Depth},
  author={Feng, Ziyue and Yang, Liang and Jing, Longlong and Wang, Haiyan and Tian, YingLi and Li, Bing},
  journal={arXiv preprint arXiv:2203.15174},
  year={2022}
}
```

## Reference

InstaDM: https://github.com/SeokjuLee/Insta-DM
ManyDepth: https://github.com/nianticlabs/manydepth

## Contact

If you have any concern with this paper or implementation, welcome to open an issue or email me at 'zfeng@clemson.edu'