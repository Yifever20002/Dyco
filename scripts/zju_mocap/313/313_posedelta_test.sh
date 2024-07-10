pose_length=1
pose_step=0 #does not matter
pose_D1=32
pose_D2=32

posedelta_length=6
posedelta_step=25
posedelta_deltastep=25
posedelta_D1=16    #↑
posedelta_D2=32   #↑
posedelta_representation=axis-angle   #quaternion  matrix  axis-angle
experiment=posedelta_condition_addskip/pose-len-${pose_length}_D1-${pose_D1}_D2-${pose_D2}/posedelta-${posedelta_representation}-len-${posedelta_length}-step-${posedelta_step}_deltastep-${posedelta_deltastep}_D1-${posedelta_D1}_D2-${posedelta_D2}_single-gpu

subject=313

for type in novelview novelpose
do
torchrun --nproc_per_node=4 --nnodes=1 --master_port 2950${CUDA_VISIBLE_DEVICES} run.py \
    --type ${type} \
    --cfg configs/human_nerf/standard_zju/${subject}.yaml \
    use_amp True \
    train.lossweights.lpips 1.0 train.lossweights.mse 0.2 \
    random_seed 7 \
    patch.N_patches 6 \
    resume False \
    \
    non_rigid_motion_mlp.pose_condition.length ${pose_length} \
    non_rigid_motion_mlp.pose_condition.step ${pose_step} \
    non_rigid_motion_mlp.pose_condition.localize.enable True \
    non_rigid_motion_mlp.pose_condition.localize.fg_threshold 0.2 \
    non_rigid_motion_mlp.pose_condition.bg_condition zero_input \
    non_rigid_motion_mlp.pose_condition.network PoseSeq_Encoder \
    non_rigid_motion_mlp.pose_condition.PoseSeq_Encoder.D1 ${pose_D1} \
    non_rigid_motion_mlp.pose_condition.PoseSeq_Encoder.D2 ${pose_D2} \
    \
    non_rigid_motion_mlp.posedelta_condition.representation ${posedelta_representation} \
    non_rigid_motion_mlp.posedelta_condition.length ${posedelta_length} \
    non_rigid_motion_mlp.posedelta_condition.step ${posedelta_step} \
    non_rigid_motion_mlp.posedelta_condition.deltastep ${posedelta_deltastep} \
    non_rigid_motion_mlp.posedelta_condition.localize.enable True \
    non_rigid_motion_mlp.posedelta_condition.localize.fg_threshold 0.2 \
    non_rigid_motion_mlp.posedelta_condition.bg_condition zero_input \
    non_rigid_motion_mlp.posedelta_condition.network PoseSeq_Encoder \
    non_rigid_motion_mlp.posedelta_condition.PoseSeq_Encoder.D1 ${posedelta_D1} \
    non_rigid_motion_mlp.posedelta_condition.PoseSeq_Encoder.D2 ${posedelta_D2} \
    \
    canonical_mlp.pose_condition.length ${pose_length} \
    canonical_mlp.pose_condition.step ${pose_step} \
    canonical_mlp.pose_condition.localize.enable True \
    canonical_mlp.pose_condition.localize.fg_threshold 0.2 \
    canonical_mlp.pose_condition.bg_condition zero_input \
    canonical_mlp.pose_condition.network PoseSeq_Encoder \
    canonical_mlp.pose_condition.PoseSeq_Encoder.D1 ${pose_D1} \
    canonical_mlp.pose_condition.PoseSeq_Encoder.D2 ${pose_D2} \
    \
    canonical_mlp.posedelta_condition.representation ${posedelta_representation} \
    canonical_mlp.posedelta_condition.length ${posedelta_length} \
    canonical_mlp.posedelta_condition.step ${posedelta_step} \
    canonical_mlp.posedelta_condition.deltastep ${posedelta_deltastep} \
    canonical_mlp.posedelta_condition.localize.enable True \
    canonical_mlp.posedelta_condition.localize.fg_threshold 0.2 \
    canonical_mlp.posedelta_condition.bg_condition zero_input \
    canonical_mlp.posedelta_condition.network PoseSeq_Encoder \
    canonical_mlp.posedelta_condition.PoseSeq_Encoder.D1 ${posedelta_D1} \
    canonical_mlp.posedelta_condition.PoseSeq_Encoder.D2 ${posedelta_D2} \
    experiment ${experiment}

done
