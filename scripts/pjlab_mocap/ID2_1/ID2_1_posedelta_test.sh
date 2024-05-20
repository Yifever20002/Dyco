pose_length=1
pose_step=24 #does not matter
pose_D1=32
pose_D2=32

posedelta_length=12
posedelta_step=24    #pose_frame interval /3
posedelta_deltastep=24
posedelta_D1=16    #↑
posedelta_D2=32   #↑

localize=True
posedelta_representation=axis-angle   #quaternion  matrix  axis-angle
experiment=posedelta_condition_addskip/pose-len-${pose_length}_D1-${pose_D1}_D2-${pose_D2}/posedelta-${posedelta_representation}-len-${posedelta_length}-step-${posedelta_step}_deltastep-${posedelta_deltastep}_D1-${posedelta_D1}_D2-${posedelta_D2}

for type in novelview novelpose

do
torchrun --nproc_per_node=4 --nnodes=1 run.py \
    --type ${type} \
    --cfg configs/human_nerf/ID2_1.yaml \
    train.lossweights.lpips 1.0 train.lossweights.mse 0.2 \
    netchunk_per_gpu 150000\
    random_seed 7 \
    \
    non_rigid_motion_mlp.pose_condition.length ${pose_length} \
    non_rigid_motion_mlp.pose_condition.step ${pose_step} \
    non_rigid_motion_mlp.pose_condition.localize.enable ${localize} \
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
    non_rigid_motion_mlp.posedelta_condition.localize.enable ${localize} \
    non_rigid_motion_mlp.posedelta_condition.localize.fg_threshold 0.2 \
    non_rigid_motion_mlp.posedelta_condition.bg_condition zero_input \
    non_rigid_motion_mlp.posedelta_condition.network PoseSeq_Encoder \
    non_rigid_motion_mlp.posedelta_condition.PoseSeq_Encoder.D1 ${posedelta_D1} \
    non_rigid_motion_mlp.posedelta_condition.PoseSeq_Encoder.D2 ${posedelta_D2} \
    \
    canonical_mlp.pose_condition.length ${pose_length} \
    canonical_mlp.pose_condition.step ${pose_step} \
    canonical_mlp.pose_condition.localize.enable ${localize} \
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
    canonical_mlp.posedelta_condition.localize.enable ${localize} \
    canonical_mlp.posedelta_condition.localize.fg_threshold 0.2 \
    canonical_mlp.posedelta_condition.bg_condition zero_input \
    canonical_mlp.posedelta_condition.network PoseSeq_Encoder \
    canonical_mlp.posedelta_condition.PoseSeq_Encoder.D1 ${posedelta_D1} \
    canonical_mlp.posedelta_condition.PoseSeq_Encoder.D2 ${posedelta_D2} \
    experiment ${experiment}
done
