# Weakly-Supervised Spatio-Temporally Grounding Natural Sentence in Video  

This repo contains the main baselines of VID-sentence dataset introduced in WSSTG.
Please refer our paper and the repo for the information of VID-sentence dataset.


### Task

<p align="center">
<figcaption>Description: "A brown and white dog is lying on the grass and then it stands up."</figcaption>
</p>
<p align="center">
<img src="images/task.png" alt="task" width="500px">
</p>
<p align="center">
<figcaption>The proposed WSSTG task aims to localize a spatio-temporal tube (ie, the sequence of green bounding boxes) in the video which semantically corresponds to the given sentence, with no reliance on any spatio-temporal annotations during training.</figcaption>
</p>

### Architecture

<p align="center">
<img src="images/frm.png" alt="method compare" width="500px">
</p>
<p align="center">
<figcaption>The architecture of the proposed method.</figcaption>
</p>

### Contents
1. [Requirements: software](#requirements-software)
3. [Installation](#installation)
4. [Training](#Training)
5. [Testing](#Testing)

### Requirements: software
Pytorch (version=0.4.0)
python 3.6 

### Installation

1. Clone the WSSTG repository and VID-sentence reposity
  ```Shell
  git clone https://github.com/JeffCHEN2017/WSSTL-private.git
  git clone https://github.com/JeffCHEN2017/VID-Sentence-private.git
  ```

2. Download the extracted tubes and the corresponding features ( i.e. RGB feature and I3D feature)
  ```Shell
  
  ```
3. Making softlinks between the download data and the desired data folder
```Shell
$'WSSTG_ROOT/data'
```

### Training
```Shell
$'WSSTG_ROOT/data'
```
Because the changes of batch sizes and the random seed, the performance may be slightly different from our submission. We provide a checkpoint here which achieves similar performance (VS) to the model we report in the paper.

### Testing
Download the checkpoint here and run
```Shell
$'WSSTG_ROOT/data'
```

### License

WSSTG is released under the MIT License (refer to the LICENSE file for details).

### Citing WSSTG

If you find this repo useful in your research, please consider citing:

    @inproceedings{chen2019weakly,
        Title={Weakly-Supervised Spatio-Temporally Grounding Natural Sentence in Video},
        Author={Chen, Zhenfang and Ma, Lin and Luo, Wenhan and Wong, Kwan-Yee K},
        Booktitle={ACL},
        year={2019}
    }

### Acknowledgement


### Contact

You can contact Zhenfang Chen by sending email to chenzhenfang2013@gmail.com
