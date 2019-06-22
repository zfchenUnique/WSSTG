# Weakly-Supervised Spatio-Temporally Grounding Natural Sentence in Video  

This repo contains the main baselines of VID-sentence dataset introduced in WSSTG.
Please refer to [our paper](https://arxiv.org/abs/1906.02549) and the [repo](https://github.com/JeffCHEN2017/VID-sentence-private) for the information of VID-sentence dataset.


### Task

<p align="center">
<figcaption>Description: "A brown and white dog is lying on the grass and then it stands up."</figcaption>
</p>
<p align="center">
<img src="images/task.png" alt="task" width="500px">
</p>
<p align="center">
<figcaption>The proposed WSSTG task aims to localize a spatio-temporal tube (i.e., the sequence of green bounding boxes) in the video which semantically corresponds to the given sentence, with no reliance on any spatio-temporal annotations during training.</figcaption>
</p>

### Architecture

<p align="center">
<img src="images/frm.png" alt="architecture" width="500px">
</p>
<p align="center">
<figcaption>The architecture of the proposed method.</figcaption>
</p>

### Contents
1. [Requirements: software](#requirements-software)
2. [Installation](#installation)
3. [Training](#Training)
4. [Testing](#Testing)

### Requirements: software

- Pytorch (version=0.4.0)
- python 3.6 

### Installation

1. Clone the WSSTG repository and VID-sentence reposity

```Shell
  git clone https://github.com/JeffCHEN2017/WSSTL-private.git
  git clone https://github.com/JeffCHEN2017/VID-Sentence-private.git
```
2. Download [tube proposals](), [RGB feature]() and [I3D feature]() from Google Drive.

3. Extract  *.tar files and make symlinks between the download data and the desired data folder

```Shell
tar xvf tubePrp.tar
ln -s tubePrp $WSSTG_ROOT/data/tubePrp

tar xvf vid_i3d.tar vid_i3d
ln -s vid_i3d $WSSTG_ROOT/data/vid_i3d

tar xvf vid_rgb.tar vid_rgb
ln -s vid_rgb $WSSTG_ROOT/data/vid_rgb
```

  Note: We extract the tube proposals using the method proposed by [Gkioxari and Malik](https://arxiv.org/abs/1411.6031) .A python implementation [here](ttps://www.mi.t.u-tokyo.ac.jp/projects/person_search/) is provided by Yamaguchi etal..
  We extract singel-frame propsoals and RGB feature for each frame using a [faster-RCNN](https://arxiv.org/abs/1506.01497) model pretrained on COCO dataset, which is provided by [Jianwei Yang](https://github.com/jwyang/faster-rcnn.pytorch).
  We extract [I3D-RGB and I3D-flow features](https://arxiv.org/abs/1705.07750) using the model provided by [Carreira and Zisserman](https://github.com/deepmind/kinetics-i3d.git).


### Training
```Shell
$'WSSTG_ROOT/data'
```
Notice: Because the changes of batch sizes and the random seed, the performance may be slightly different from our submission. We provide a checkpoint here which achieves similar performance (VS) to the model we report in the paper.

### Testing
Download the checkpoint here and run
```Shell
$'WSSTG_ROOT/data'
```

### License

WSSTG is released under the CC-BY-NC 4.0 LICENSE (refer to the LICENSE file for details).

### Citing WSSTG

If you find this repo useful in your research, please consider citing:

    @inproceedings{chen2019weakly,
        Title={Weakly-Supervised Spatio-Temporally Grounding Natural Sentence in Video},
        Author={Chen, Zhenfang and Ma, Lin and Luo, Wenhan and Wong, Kwan-Yee K},
        Booktitle={ACL},
        year={2019}
    }

### Acknowledgement
Some codes are borrowed/ inspired from [Yamaguchi etal.](https://www.mi.t.u-tokyo.ac.jp/projects/person_search/)

### Contact

You can contact Zhenfang Chen by sending email to chenzhenfang2013@gmail.com
