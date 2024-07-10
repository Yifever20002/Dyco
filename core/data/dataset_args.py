from configs import cfg

class DatasetArgs(object):
    dataset_attrs = {}

    if cfg.category == 'human_nerf' and cfg.task == 'I3D-Human':
        dataset_attrs.update({
            ##ID1_1
            f"ID1_1-train": {
            "dataset_path": f"../dataset/I3D-Human/ID1_1-train",
            "train_dataset_path": f"../dataset/I3D-Human/ID1_1-train",
            "keyfilter": cfg.train_keyfilter,
            "ray_shoot_mode": cfg.train.ray_shoot_mode}, # for train

            f"ID1_1-train_render": {
            "dataset_path": f"../dataset/I3D-Human/ID1_1-train",
            "train_dataset_path": f"../dataset/I3D-Human/ID1_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "skip":1, "maxframes": -1}, # for train

            f"ID1_1-movement": {
            "dataset_path": f"../dataset/I3D-Human/ID1_1-train",
            "train_dataset_path": f"../dataset/I3D-Human/ID1_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "select_views": [0], "skip":4}, # for novelview

            f"ID1_1-novelview": {
            "dataset_path": f"../dataset/I3D-Human/ID1_1-novelview",
            "train_dataset_path": f"../dataset/I3D-Human/ID1_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',}, # for novelview

            f"ID1_1-novelview_res": {
            "dataset_path": f"../dataset/I3D-Human/ID1_1-novelview_hres",
            "train_dataset_path": f"../dataset/I3D-Human/ID1_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',}, # for novelview

            f"ID1_1-novelview-stop": {
            "dataset_path": f"../dataset/I3D-Human/ID1_1-stop",
            "train_dataset_path": f"../dataset/I3D-Human/ID1_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "skip":1, "maxframes": -1}, # for train

            f"ID1_1-novelpose": {
            "dataset_path": f"../dataset/I3D-Human/ID1_1-novelpose",
            "train_dataset_path": f"../dataset/I3D-Human/ID1_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image'}, # for novelpose

            f"ID1_1-novelpose-autoregressive": {
            "dataset_path": f"../dataset/I3D-Human/ID1_1-novelpose-autoregressive",
            "train_dataset_path": f"../dataset/I3D-Human/ID1_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image'}, # for novelpose
            
            
            ##ID1_2
            f"ID1_2-train": {
            "dataset_path": f"../dataset/I3D-Human/ID1_2-train",
            "train_dataset_path": f"../dataset/I3D-Human/ID1_2-train",
            "keyfilter": cfg.train_keyfilter,
            "ray_shoot_mode": cfg.train.ray_shoot_mode}, # for train

            f"ID1_2-train_render": {
            "dataset_path": f"../dataset/I3D-Human/ID1_2-train",
            "train_dataset_path": f"../dataset/I3D-Human/ID1_2-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "skip":1, "maxframes": -1}, # for train

            f"ID1_2-movement": {
            "dataset_path": f"../dataset/I3D-Human/ID1_2-train",
            "train_dataset_path": f"../dataset/I3D-Human/ID1_2-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "select_views": [0], "skip":4}, # for novelview

            f"ID1_2-novelview": {
            "dataset_path": f"../dataset/I3D-Human/ID1_2-novelview",
            "train_dataset_path": f"../dataset/I3D-Human/ID1_2-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',}, # for novelview

            f"ID1_2-novelview_res": {
            "dataset_path": f"../dataset/I3D-Human/ID1_2-novelview_hres",
            "train_dataset_path": f"../dataset/I3D-Human/ID1_2-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',}, # for novelview

            f"ID1_2-novelpose": {
            "dataset_path": f"../dataset/I3D-Human/ID1_2-novelpose",
            "train_dataset_path": f"../dataset/I3D-Human/ID1_2-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image'}, # for novelpose

            f"ID1_2-novelpose-autoregressive": {
            "dataset_path": f"../dataset/I3D-Human/ID1_2-novelpose-autoregressive",
            "train_dataset_path": f"../dataset/I3D-Human/ID1_2-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image'}, # for novelpose


            ##ID2_1
            f"ID2_1-train": {
            "dataset_path": f"../dataset/I3D-Human/ID2_1-train",
            "train_dataset_path": f"../dataset/I3D-Human/ID2_1-train",
            "keyfilter": cfg.train_keyfilter,
            "ray_shoot_mode": cfg.train.ray_shoot_mode}, # for train

            f"ID2_1-train_render": {
            "dataset_path": f"../dataset/I3D-Human/ID2_1-train",
            "train_dataset_path": f"../dataset/I3D-Human/ID2_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "skip":1, "maxframes": -1}, # for train

            f"ID2_1-movement": {
            "dataset_path": f"../dataset/I3D-Human/ID2_1-train",
            "train_dataset_path": f"../dataset/I3D-Human/ID2_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "select_views": [0], "skip":4}, # for novelview

            f"ID2_1-novelview": {
            "dataset_path": f"../dataset/I3D-Human/ID2_1-novelview",
            "train_dataset_path": f"../dataset/I3D-Human/ID2_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',}, # for novelview

            f"ID2_1-novelview_res": {
            "dataset_path": f"../dataset/I3D-Human/ID2_1-novelview_hres",
            "train_dataset_path": f"../dataset/I3D-Human/ID2_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',}, # for novelview

            f"ID2_1-novelview-stop": {
            "dataset_path": f"../dataset/I3D-Human/ID2_1-stop",
            "train_dataset_path": f"../dataset/I3D-Human/ID2_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "skip":1, "maxframes": -1}, # for train

            f"ID2_1-novelpose": {
            "dataset_path": f"../dataset/I3D-Human/ID2_1-novelpose",
            "train_dataset_path": f"../dataset/I3D-Human/ID2_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image'}, # for novelpose

            f"ID2_1-novelpose-autoregressive": {
            "dataset_path": f"../dataset/I3D-Human/ID2_1-novelpose-autoregressive",
            "train_dataset_path": f"../dataset/I3D-Human/ID2_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image'}, # for novelpose

            ##ID2_2
            f"ID2_2-train": {
            "dataset_path": f"../dataset/I3D-Human/ID2_2-train",
            "train_dataset_path": f"../dataset/I3D-Human/ID2_2-train",
            "keyfilter": cfg.train_keyfilter,
            "ray_shoot_mode": cfg.train.ray_shoot_mode}, # for train

            f"ID2_2-train_render": {
            "dataset_path": f"../dataset/I3D-Human/ID2_2-train",
            "train_dataset_path": f"../dataset/I3D-Human/ID2_2-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "skip":1, "maxframes": -1}, # for train

            f"ID2_2-movement": {
            "dataset_path": f"../dataset/I3D-Human/ID2_2-train",
            "train_dataset_path": f"../dataset/I3D-Human/ID2_2-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "select_views": [0], "skip":4}, # for novelview

            f"ID2_2-novelview": {
            "dataset_path": f"../dataset/I3D-Human/ID2_2-novelview",
            "train_dataset_path": f"../dataset/I3D-Human/ID2_2-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',}, # for novelview

            f"ID2_2-novelview-stop": {
            "dataset_path": f"../dataset/I3D-Human/ID2_2-stop",
            "train_dataset_path": f"../dataset/I3D-Human/ID2_2-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "skip":1, "maxframes": -1}, # for train

            f"ID2_2-novelpose": {
            "dataset_path": f"../dataset/I3D-Human/ID2_2-novelpose",
            "train_dataset_path": f"../dataset/I3D-Human/ID2_2-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image'}, # for novelpose

            f"ID2_2-novelpose-autoregressive": {
            "dataset_path": f"../dataset/I3D-Human/ID2_2-novelpose-autoregressive",
            "train_dataset_path": f"../dataset/I3D-Human/ID2_2-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image'}, # for novelpose

            ##ID3_1
            f"ID3_1-train": {
            "dataset_path": f"../dataset/I3D-Human/ID3_1-train",
            "train_dataset_path": f"../dataset/I3D-Human/ID3_1-train",
            "keyfilter": cfg.train_keyfilter,
            "ray_shoot_mode": cfg.train.ray_shoot_mode}, # for train

            f"ID3_1-train_render": {
            "dataset_path": f"../dataset/I3D-Human/ID3_1-train",
            "train_dataset_path": f"../dataset/I3D-Human/ID3_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "skip":1, "maxframes": -1}, # for train

            f"ID3_1-movement": {
            "dataset_path": f"../dataset/I3D-Human/ID3_1-train",
            "train_dataset_path": f"../dataset/I3D-Human/ID3_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "select_views": [0], "skip":4}, # for novelview

            f"ID3_1-novelview": {
            "dataset_path": f"../dataset/I3D-Human/ID3_1-novelview",
            "train_dataset_path": f"../dataset/I3D-Human/ID3_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',}, # for novelview

            f"ID3_1-novelview_res": {
            "dataset_path": f"../dataset/I3D-Human/ID3_1-novelview_hres",
            "train_dataset_path": f"../dataset/I3D-Human/ID3_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',}, # for novelview

            f"ID3_1-novelview-stop": {
            "dataset_path": f"../dataset/I3D-Human/ID3_1-stop",
            "train_dataset_path": f"../dataset/I3D-Human/ID3_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "skip":1, "maxframes": -1}, # for train

            f"ID3_1-novelpose": {
            "dataset_path": f"../dataset/I3D-Human/ID3_1-novelpose",
            "train_dataset_path": f"../dataset/I3D-Human/ID3_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image'}, # for novelpose

            f"ID3_1-novelpose-autoregressive": {
            "dataset_path": f"../dataset/I3D-Human/ID3_1-novelpose-autoregressive",
            "train_dataset_path": f"../dataset/I3D-Human/ID3_1-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image'}, # for novelpose

            ##ID3_2
            f"ID3_2-train": {
            "dataset_path": f"../dataset/I3D-Human/ID3_2-train",
            "train_dataset_path": f"../dataset/I3D-Human/ID3_2-train",
            "keyfilter": cfg.train_keyfilter,
            "ray_shoot_mode": cfg.train.ray_shoot_mode}, # for train

            f"ID3_2-train_render": {
            "dataset_path": f"../dataset/I3D-Human/ID3_2-train",
            "train_dataset_path": f"../dataset/I3D-Human/ID3_2-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "skip":1, "maxframes": -1}, # for train

            f"ID3_2-movement": {
            "dataset_path": f"../dataset/I3D-Human/ID3_2-train",
            "train_dataset_path": f"../dataset/I3D-Human/ID3_2-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "select_views": [0], "skip":4}, # for novelview

            f"ID3_2-novelview": {
            "dataset_path": f"../dataset/I3D-Human/ID3_2-novelview",
            "train_dataset_path": f"../dataset/I3D-Human/ID3_2-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',}, # for novelview

            f"ID3_2-novelview-stop": {
            "dataset_path": f"../dataset/I3D-Human/ID3_2-stop",
            "train_dataset_path": f"../dataset/I3D-Human/ID3_2-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "skip":1, "maxframes": -1}, # for train

            f"ID3_2-novelpose": {
            "dataset_path": f"../dataset/I3D-Human/ID3_2-novelpose",
            "train_dataset_path": f"../dataset/I3D-Human/ID3_2-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image'}, # for novelpose

            f"ID3_2-novelpose-autoregressive": {
            "dataset_path": f"../dataset/I3D-Human/ID3_2-novelpose-autoregressive",
            "train_dataset_path": f"../dataset/I3D-Human/ID3_2-train",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image'}, # for novelpose



            'flexible':{
                "dataset_path":  cfg.flexible_testpath,
                "keyfilter": cfg.test_keyfilter,
                "ray_shoot_mode": 'image',
                "skip":1, "maxframes": -1} 
        }
        )       
    if cfg.category == 'human_nerf' and cfg.task == 'zju_mocap':    
        subjects = ['313', '315', '377', '386', '387', '390', '392', '393', '394','xiao']
        for sub in subjects:
            dataset_attrs.update({
                f"zju_{sub}_standard_train":{
                    "dataset_path": f"dataset/zju_mocap/standard/{sub}/{sub}_train",
                    "source_path": f"data/zju/CoreView_{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "subject": sub,
                },
                f"zju_{sub}_standard_train_render":{
                    "dataset_path": f"dataset/zju_mocap/standard/{sub}/{sub}_train",
                    "source_path": f"data/zju/CoreView_{sub}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "skip": 1, "maxframes": -1,
                    "subject": sub,
                },
                f"zju_{sub}_standard_novelview":{
                    "dataset_path": f"dataset/zju_mocap/standard/{sub}/{sub}_novelview",
                    "source_path": f"data/zju/CoreView_{sub}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "subject": sub,
                    "train_dataset_path": f"dataset/zju_mocap/standard/{sub}/{sub}_train",
                },
                f"zju_{sub}_standard_novelpose":{
                    "dataset_path": f"dataset/zju_mocap/standard/{sub}/{sub}_novelpose",
                    "source_path": f"data/zju/CoreView_{sub}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "subject": sub,
                    "train_dataset_path": f"dataset/zju_mocap/standard/{sub}/{sub}_train",
                },
                f"zju_{sub}_standard_novelpose_autoregressive":{
                    "dataset_path": f"dataset/zju_mocap/standard/{sub}/{sub}_novelpose_autoregressive",
                    "source_path": f"data/zju/CoreView_{sub}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "subject": sub,
                    "train_dataset_path": f"dataset/zju_mocap/standard/{sub}/{sub}_train",
                },
                f"zju_{sub}_standard_resized_train":{
                    "dataset_path": f"dataset/zju_mocap/standard_resized/{sub}/{sub}_train",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "subject": sub,
                },
                f"zju_{sub}_standard_resized_novelview":{
                    "dataset_path": f"dataset/zju_mocap/standard_resized/{sub}/{sub}_novelview",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "subject": sub,
                    "train_dataset_path": f"dataset/zju_mocap/standard_resized/{sub}/{sub}_train",
                },
                f"zju_{sub}_standard_resized_novelpose":{
                    "dataset_path": f"dataset/zju_mocap/standard_resized/{sub}/{sub}_novelpose",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "subject": sub,
                    "train_dataset_path": f"dataset/zju_mocap/standard_resized/{sub}/{sub}_train",
                },
                f"zju_{sub}_nb_4view_train": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_nb_4view_train",
                    "source_path": f"data/zju/CoreView_{sub}",
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    "subject": sub,
                },
                f"zju_{sub}_nb_1view_test": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_nb_4view_train",
                    "source_path": f"data/zju/CoreView_{sub}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,
                    "select_views":[1],
                    "skip":4,
                },
                f"zju_{sub}_nb_1view_test_progress": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_nb_4view_train",
                    "source_path": f"data/zju/CoreView_{sub}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,
                    "select_views":[1],
                    "skip": 18
                },
                f"zju_{sub}_nb_4view_novelpose": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_nb_4view_novelpose",
                    "source_path": f"data/zju/CoreView_{sub}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,
                },
                f"zju_{sub}_nb_1view_novelpose": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_nb_1view_novelpose_all",
                    "source_path": f"data/zju/CoreView_{sub}",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,
                },
                f"zju_{sub}_nb_4view_novelview": {
                    "dataset_path": f"dataset/zju_mocap/{sub}_nb_4view_novelview",
                    "source_path": f"data/zju/CoreView_{sub}", 
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'zju_mocap',
                    "subject": sub,
                    "train_dataset_path": f"dataset/zju_mocap/{sub}_nb_4view_train", 
                    #the same source path
                },
                f"zju_387_nb_rightlimb_32":{
                    "dataset_path": f"dataset/zju_mocap/387_nb_pose_rightlimb_32",
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "subject": sub,                        
                },
            })


    if cfg.category == 'human_nerf' and cfg.task == 'wild':
        dataset_attrs.update({
            "monocular_train": {
                "dataset_path": 'dataset/wild/monocular',
                "keyfilter": cfg.train_keyfilter,
                "ray_shoot_mode": cfg.train.ray_shoot_mode,
            },
            "monocular_test": {
                "dataset_path": 'dataset/wild/monocular',  
                "keyfilter": cfg.test_keyfilter,
                "ray_shoot_mode": 'image',
                "src_type": 'wild'
            },
        })


    @staticmethod
    def get(name):
        attrs = DatasetArgs.dataset_attrs[name]
        return attrs.copy()
