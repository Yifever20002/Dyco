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

### `Down SMPL model`

Copy the smpl model.
``````
    SMPL_DIR=/path/to/smpl
    MODEL_DIR=$SMPL_DIR/smplify_public/code/models
    cp $MODEL_DIR/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl third_parties/smpl/models
``````
Follow [this page](https://github.com/vchoutas/smplx/tree/master/tools) to remove Chumpy objects from the SMPL model.

## B. Prepare dataset
### 1. Download and extract
    ```
    DATASET_NAME=xianbei_v1.0
    cp -r /mnt/data/Gvlab/chenyutong/data/pjlab/xianbei_v1.0 data/
    ```
### 2. Prepare images and poses
Generate the folder in dataset/pjlab_mocap/\${DATASET_NAME}-\${SPLIT} which contains
* cameras.pkl - the camera intrinsics and extrinsics.
* canonical_joints.pkl -  the joints' positions in the canonical pose.
* frameid_pose.pkl - the Rh,Th,poses of each frame over the entire video
* mesh_infos.pkl - the image/mask path and poses
* images - the resized and cropped images
* masks - the resized and cropped masks
```
cd tools/prepare_pjlab_mocap
for split in train novelview novelpose
do
python prepare_dataset.py --cfg=xianbei_v1.0_crop-resize/xianbei_v1.0_${split}.yaml 
done
```


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


