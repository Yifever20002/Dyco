# Within the Dynamic Context: Inertia-aware 3D Human Modeling with Pose Sequence 
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
from 6 subjects in total. Click [here](https://github.com/Yifever20002/Dyco) to download our I3D-Human Dataset.



### 3. [Optional] Extract CNN features  

```
export CUDA_VISIBLE_DEVICES=0
DIRNAME=dataset/pjlab_mocap/xianbei_v1.0-
for split in trainview_all
do
python tools/extract_features.py/compute_features.py \
    --net resnet34 \
    --output_dir ${DIRNAME}${split}/rgb_features/ \
    --folder ${DIRNAME}${split}/
done
```
The precomputed RGB features will be saved in dataset/pjlab_mocap/xianbei_v1.0-${split}/rgb_features/ and be loaded during training when needed.

## C. Train and Test

### 1. Baseline (See scripts/examples/baseline.sh)
```
export CUDA_VISIBLE_DEVICES=0
python train.py \
    --cfg configs/human_nerf/xianbei_v1.0.yaml \
    experiment baseline/humannerf_baseline
    
for type in movement novelview novelpose_autoregressive
do
python run.py \
    --type ${type} \
    --cfg configs/human_nerf/xianbei_v1.0.yaml \
    experiment baseline/humannerf_baseline 
done

```

### 2. + Conditions
```
# baseline, w/o extra condition input to canonical-mlp
sh scripts/examples/baseline.sh 

# Input pose condition to canonical-mlp and offset-cmlp
sh scripts/examples/pose_condition_len1.sh

# Input pose-sequence condition to canonical-mlp and offset-cmlp
sh scripts/examples/pose-seq_condition_len4.sh

# Input pose-delta condition to canonical-mlp and offset-mlp
sh scripts/examples/pose-delta_condition_len4.sh

```


