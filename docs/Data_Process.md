本文档用于介绍如何生成训练数据。

1. 关于Argoverse V2数据集的简单说明
```
Argoverse V2 Dataset Root Path
|---train
|-------00a6ffc1-6ce9-3bc3-a060-6006e9893a1a
|-----------calibration
|---------------egovehicle_SE3_sensor.feather
|---------------intrinsics.feather
|-----------map
|-----------sensors
|---------------cameras
|-------------------ring_front_center(about 300)
|-----------------------315967376899927209.jpg
|-----------------------315967376949927221.jpg
|-----------------------...
|-------------------ring_front_left
|-------------------ring_front_right
|-------------------ring_side_left
|-------------------ring_side_right
|-------------------ring_rear_left
|-------------------ring_rear_right
|-------------------stereo_front_left
|-------------------stereo_front_right
|---------------lidar(about 150)
|-------------------315967376859506000.feather
|-------------------315967376959702000.feather
|-------------------...
|-----------annotations.feather
|-----------city_SE3_egovehicle.feather
|-------01bb304d-7bd8-35f8-bbef-7086b688e35e
|-------022af476-9937-3e70-be52-f65420d52703
|-------...
|---val
|-------...
|---trainval
|-------av2_infos_train.pkl
|-------av2_infos_val.pkl
|-------av2_infos_val_mono3d.coco.json
```
下载后的文件目录如上，包含train/val/trainval三个文件，其中train和val文件下包含原始的数据，以train目录下的00a6ffc1-6ce9-3bc3-a060-6006e9893a1a样本为例：
```
1. 00a6ffc1-6ce9-3bc3-a060-6006e9893a1a：一个约15s的数据片段样本，采样频率为10Hz
2. calibration：该目录下的egovehicle_SE3_sensor.feather和intrinsics.feather文件分别包含sensor到egovehicle的空间位姿变换关系以及相机的内参
3. sensors/cameras/xx_camera: 7个单目摄像头+2个双目摄像头，每个摄像头采样频率20Hz，15s对应约300张照片，每张照片名称对应触发的时间戳(ns)
4. sensors/lidar/xx.feather: 点云数据，雷达采样频率为10Hz，15s对应约150个文件
5. annotations.feather: 存储object的标注信息，读取标注数据的变量类型为<class 'pandas.core.frame.DataFrame'>，包含字段如下
                        annotations.columns: Index(['timestamp_ns', 'track_uuid', 'category', 'length_m', 'width_m', 'height_m', 'qw', 'qx', 'qy', 'qz', 'tx_m', 'ty_m', 'tz_m', 'num_interior_pts'], dtype='object')
                        其中，timestamp_ns是和lidar时间戳一致的，length_m/width_m/height_m分别表示3D目标框的长宽高，qw/qx/qy/qz表示相对于egovehicle frame的四元数，tx_m/ty_m/tz_m表示相对于egovehicle frame的位移，num_interior_pts表示3D目标框内的雷达点数，即标注框是否有效
6. city_SE3_egovehicle.feather：egovehicle到global frame的位姿变换关系
7. trainval/av2_XX_infos.pkl: 数据集目录结构如下，其中trainval文件下包含两个.pkl文件，分别对应训练集和验证集，每个文件中包含若干
                            条"dict()"类型的数据，在之后的训练过程中直接从.pkl文件中加载字典数据。
```

2. 关于dict()数据说明
以pkl文件中的一条字典类型数据为例，数据字典中包含以下字段：
```
info = {
        'log_id': log_id, # '00a6ffc1-6ce9-3bc3-a060-6006e9893a1a' 每个15s的样本的唯一标识
        'timestamp': timestamp_ns, # 315967376859506000(ns)
        'gt_bboxes' : gt_bboxes_3d, # 当前timestamp时，所有object的3D box真值[[x, y, z, l, w, h, r], ...]
        'gt_labels' : gt_labels, # 当前timestamp时，所有object的类别标签 [13, 12, 12, 12, 6, 6, 6, 6, ...]
        'gt_names' : gt_names, # 当前timestamp时，所有object的类别标签 ['BOX_TRUCK', 'BUS', 'BUS', 'BUS', 'CONSTRUCTION_CONE', 'CONSTRUCTION_CONE', 'CONSTRUCTION_CONE', 'CONSTRUCTION_CONE', ...]
        'gt_num_pts' : gt_num_pts, # 当前timestamp时，所有3D框中的lidar点数，用于后续判断是否需要提取姿态[137, 90, 1654, 638, 5, 11, 11, ...]
        'gt_velocity' : gt_velocity, #  当前timestamp时，所有object的速度 [[v_x, v_y, v_z], ...]
        'gt_uuid' : gt_uuid, # 每一个object的唯一标识 ['136ad4d1-1a85-455a-8e33-b376ff442dd5', 'b5839925-de5a-46ac-a39d-69e58bf9589e', ...]
        'gt_city_SE3_ego' : gt_city_SE3_ego, # 当前timestamp下，自车相对全局的坐标变换关系 SE3(rotation=array([[-0.89443817,  0.44706894,  0.01047527],
       [-0.44692674, -0.8944692 ,  0.01346646],
       [ 0.01539024,  0.00736324,  0.99985445]]), translation=array([1618.79915463,  263.39290419,   14.0770558 ]))
    }
```
最后，每一条15s的样本数据生成150条info，所有样本的info都存储到.pkl文件中。
同样可以按照这样的字段处理其他的数据集，生成.pkl文件后，就可以直接输入到av2_dataset.py中使用。
