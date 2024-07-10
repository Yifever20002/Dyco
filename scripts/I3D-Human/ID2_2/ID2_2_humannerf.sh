# export CUDA_VISIBLE_DEVICES=0
experiment=humannerf_baseline

torchrun --nproc_per_node=1 --nnodes=1 --master_port 2950${CUDA_VISIBLE_DEVICES} train.py \
    --cfg configs/human_nerf/ID2_2.yaml \
    use_amp True \
    frame_interval 3 \
    train.lossweights.lpips 1.0 train.lossweights.mse 0.2 \
    experiment ${experiment} \
    non_rigid_motion_mlp.mlp_depth 6 \
    canonical_mlp.mlp_depth 8 \
    canonical_mlp.triplane False

