experiment=humannerf_baseline

for type in  novelview novelpose

do
torchrun --nproc_per_node=3 --nnodes=1 --master_port 2950${CUDA_VISIBLE_DEVICES} run.py \
    --type ${type} \
    --cfg configs/human_nerf/ID3_1.yaml \
    use_amp True \
    frame_interval 3 \
    train.lossweights.lpips 1.0 train.lossweights.mse 0.2 \
    experiment ${experiment} \
    non_rigid_motion_mlp.mlp_depth 6 \
    canonical_mlp.mlp_depth 8 \
    canonical_mlp.triplane False
    done
