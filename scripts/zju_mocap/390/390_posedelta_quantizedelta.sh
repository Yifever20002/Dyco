pose_length=1
pose_step=0 #does not matter
pose_D1=32
pose_D2=32
ca_quantize=Notuse     # rotate-only(keep vector while quantize rotation) axis-angle(quantize all)
nr_quantize=Notuse
quantized_pose_step=1024

posedelta_length=2
posedelta_step=25
posedelta_deltastep=25
posedelta_D1=16    #↑
posedelta_D2=32   #↑
pd_ca_quantize=axis-angle    
pd_nr_quantize=axis-angle
quantized_deltapose_step=360
posedelta_representation=axis-angle   #quaternion  matrix  axis-angle

experiment=posedelta_condition_qdelta/pose-len-${pose_length}_D1-${pose_D1}_D2-${pose_D2}_quanstep-${quantized_pose_step}_caq-${ca_quantize}_nrq-${nr_quantize}/posedelta-${posedelta_representation}-len-${posedelta_length}-step-${posedelta_step}_deltastep-${posedelta_deltastep}_D1-${posedelta_D1}_D2-${posedelta_D2}_quanstep-${quantized_deltapose_step}_caq-${pd_ca_quantize}_nrq-${pd_nr_quantize}

export CUDA_VISIBLE_DEVICES=0
subject=390

torchrun --nproc_per_node=1 --nnodes=1 --master_port 2950${CUDA_VISIBLE_DEVICES} train.py \
    --cfg configs/human_nerf/standard_zju/${subject}.yaml \
    use_amp True \
    train.lossweights.lpips 1.0 train.lossweights.mse 0.2 \
    random_seed 7 \
    \
    quantized_pose_step ${quantized_pose_step}\
    quantized_deltapose_step ${quantized_deltapose_step}\
    \
    patch.N_patches 6 \
    resume True \
    \
    non_rigid_motion_mlp.pose_condition.length ${pose_length} \
    non_rigid_motion_mlp.pose_condition.step ${pose_step} \
    non_rigid_motion_mlp.pose_condition.quantize_type ${nr_quantize} \
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
    non_rigid_motion_mlp.posedelta_condition.quantize_type ${pd_nr_quantize} \
    non_rigid_motion_mlp.posedelta_condition.localize.enable True \
    non_rigid_motion_mlp.posedelta_condition.localize.fg_threshold 0.2 \
    non_rigid_motion_mlp.posedelta_condition.bg_condition zero_input \
    non_rigid_motion_mlp.posedelta_condition.network PoseSeq_Encoder \
    non_rigid_motion_mlp.posedelta_condition.PoseSeq_Encoder.D1 ${posedelta_D1} \
    non_rigid_motion_mlp.posedelta_condition.PoseSeq_Encoder.D2 ${posedelta_D2} \
    \
    canonical_mlp.pose_condition.length ${pose_length} \
    canonical_mlp.pose_condition.step ${pose_step} \
    canonical_mlp.pose_condition.quantize_type ${ca_quantize} \
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
    canonical_mlp.posedelta_condition.quantize_type ${pd_ca_quantize} \
    canonical_mlp.posedelta_condition.localize.enable True \
    canonical_mlp.posedelta_condition.localize.fg_threshold 0.2 \
    canonical_mlp.posedelta_condition.bg_condition zero_input \
    canonical_mlp.posedelta_condition.network PoseSeq_Encoder \
    canonical_mlp.posedelta_condition.PoseSeq_Encoder.D1 ${posedelta_D1} \
    canonical_mlp.posedelta_condition.PoseSeq_Encoder.D2 ${posedelta_D2} \
    experiment ${experiment}
