# Within the Dynamic Context: Inertia-aware 3D Human Modeling with Pose Sequence 
### [Project Page](https://ai4sports.opengvlab.com/Dyco/) | [Paper](https://arxiv.org/abs/2403.19160) | [Data](https://ai4sports.opengvlab.com/Dyco/)
## A. Prerequisite
### `Configure environment`
Create and activate a virtual environment.

    conda create --name Dyco python=3.7
    conda activate Dyco

Install the required packages.

    pip install -r requirements.txt

    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

    pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

### `Download SMPL model`

Copy the smpl model.

    SMPL_DIR=/path/to/smpl

    MODEL_DIR=$SMPL_DIR/smplify_public/code/models

    cp $MODEL_DIR/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl third_parties/smpl/models

Follow [this page](https://github.com/vchoutas/smplx/tree/master/tools) to remove Chumpy objects from the SMPL model.

### `Download vgg.pth`

Download the vgg.pth from [here](https://github.com/richzhang/PerceptualSimilarity/tree/master/lpips/weights/v0.1).

    VGG_DIR=/path/to/vgg.pth

    cp $VGG_DIR third_parties/lpips/weights/v0.1/

## B. I3D-Human Dataset
The I3D-Human Dataset focuses on capturing variations in clothing appearance under approximately identical poses. Compared with existing benchmarks, we outfit the subjects in loose clothing such as dresses and light jackets and
encourage movements involving acceleration or deceleration, such as sudden stops
after spinning, swaying, and flapping sleeves. Our capturing equipment consists of
10 DJI Osmo Action cameras, shooting at a frame rate of 100fps while synchronized
with an audio signal. The final processed dataset records 10k frames of sequence
from 6 subjects in total. Click [here](https://github.com/Yifever20002/Dyco) to download our I3D-Human Dataset and copy it to /path/to/Dyco's parent/dataset/.

![image](https://github.com/Yifever20002/Dyco/blob/main/images/dataset.png)

## C. Train and Test

### Baseline

    sh scripts/pjlab_mocap/ID1_1/ID1_1_humannerf.sh

    sh scripts/pjlab_mocap/ID1_1/ID1_1_humannerf_test.sh

### + Conditions

    sh scripts/pjlab_mocap/ID1_1/ID1_1_posedelta.sh

    sh scripts/pjlab_mocap/ID1_1/ID1_1_posedelta_test.sh




