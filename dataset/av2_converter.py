# Copyright (c) OpenMMLab. All rights reserved.
import os
from collections import OrderedDict
from os import path as osp
from typing import List, Tuple, Union

import mmcv
import numpy as np
import pickle 
from tqdm import tqdm 
from pathlib import Path
import math
import copy
import cv2
from mmdet3d.core.visualizer.image_vis import draw_camera_bbox3d_on_img
from mmdet3d.core.bbox import Box3DMode, LiDARInstance3DBoxes, CameraInstance3DBoxes, get_box_type

from av2.datasets.sensor.sensor_dataloader import read_city_SE3_ego
from av2.datasets.sensor.splits import TRAIN, VAL, TEST
from av2.utils.io import read_feather
from av2.geometry.geometry import quat_to_mat
from av2.geometry.se3 import SE3
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.structures.cuboid import Cuboid, CuboidList
from shapely.geometry import MultiPoint, box
from pyquaternion import Quaternion
from mmdet3d.core.bbox import points_cam2img
from scipy.spatial.transform import Rotation

class_names = ('REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER',
    'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS',
    'MESSAGE_BOARD_TRAILER', 'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG')

TRAIN_SAMPLE_RATE = 0
VAL_SAMPLE_RATE = 0
VELOCITY_SAMPLING_RATE = 5

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points

def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.
    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.
    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])
        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str, cat_name: str) -> OrderedDict:
    """Generate one 2D annotation record given various information on top of
    the 2D bounding box coordinates.
    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.
    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): file name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    coco_rec = dict()

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = class_names.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec
    
def yaw_to_quaternion3d(yaw: float) -> np.ndarray:
    """Convert a rotation angle in the xy plane (i.e. about the z axis) to a quaternion.
    Args:
        yaw: angle to rotate about the z-axis, representing an Euler angle, in radians
    Returns:
        array w/ quaternion coefficients (qw,qx,qy,qz) in scalar-first order, per Argoverse convention.
    """
    qx, qy, qz, qw = Rotation.from_euler(seq="z", angles=yaw, degrees=False).as_quat()
    return np.array([qw, qx, qy, qz])
    

def box_velocity(current_annotation, current_timestamp_ns, all_timestamps, annotations, log_dir):
    timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir) # {timestamp_ns: SE3(rotation, translation)}
    city_SE3_ego_reference = timestamp_city_SE3_ego_dict[current_timestamp_ns] # 获取当前时间戳自车坐标系相对全局坐标系的姿态变换

    curr_index = all_timestamps.index(current_timestamp_ns)
    prev_index = curr_index - VELOCITY_SAMPLING_RATE # 5 估算速度: delta_x(m) / 0.5(s) 
    next_index = curr_index + VELOCITY_SAMPLING_RATE

    track_uuid = current_annotation[1]["track_uuid"]

    if prev_index > 0:
        prev_timestamp_ns = all_timestamps[prev_index]

        #get annotation in prev timestamp
        prev_annotations = annotations[annotations["timestamp_ns"] == int(prev_timestamp_ns)]
        prev_annotation = prev_annotations[prev_annotations["track_uuid"] == track_uuid]

        if len(prev_annotation) == 0:
            prev_annotation = None
    else:
        prev_annotation = None 

    if next_index < len(all_timestamps):
        next_timestamp_ns = all_timestamps[next_index]

        #get annotation in next timestamp
        next_annotations = annotations[annotations["timestamp_ns"] == int(next_timestamp_ns)]
        next_annotation = next_annotations[next_annotations["track_uuid"] == track_uuid]

        if len(next_annotation) == 0:
            next_annotation = None
    else:
        next_annotation = None 

    if prev_annotation is None and next_annotation is None:
        return np.array([0, 0, 0])

    # take centered average of displacement for velocity
    if prev_annotation is not None and next_annotation is not None:
        city_SE3_ego_prev = timestamp_city_SE3_ego_dict[prev_timestamp_ns]
        reference_SE3_ego_prev = city_SE3_ego_reference.inverse().compose(city_SE3_ego_prev)

        city_SE3_ego_next = timestamp_city_SE3_ego_dict[next_timestamp_ns]
        reference_SE3_ego_next = city_SE3_ego_reference.inverse().compose(city_SE3_ego_next)

        prev_translation = np.array([prev_annotation["tx_m"].item(), prev_annotation["ty_m"].item(), prev_annotation["tz_m"].item()])   
        next_translation = np.array([next_annotation["tx_m"].item(), next_annotation["ty_m"].item(), next_annotation["tz_m"].item()])   

        #convert prev and next annotations into the current annotation reference frame
        prev_translation = reference_SE3_ego_prev.transform_from(prev_translation)
        next_translation = reference_SE3_ego_next.transform_from(next_translation)

        delta_t = (next_timestamp_ns - prev_timestamp_ns) * 1e-9
        return (next_translation - prev_translation) / delta_t, city_SE3_ego_reference

    # take one-sided average of displacement for velocity
    else:
        if prev_annotation is not None:
            city_SE3_ego_prev = timestamp_city_SE3_ego_dict[prev_timestamp_ns]
            reference_SE3_ego_prev = city_SE3_ego_reference.inverse().compose(city_SE3_ego_prev)

            prev_translation = np.array([prev_annotation["tx_m"].item(), prev_annotation["ty_m"].item(), prev_annotation["tz_m"].item()])   
            current_translation = np.array([current_annotation[1]["tx_m"], current_annotation[1]["ty_m"], current_annotation[1]["tz_m"]])   

            #convert prev annotation into the current annotation reference frame
            prev_translation = reference_SE3_ego_prev.transform_from(prev_translation)

            delta_t = (current_timestamp_ns - prev_timestamp_ns) * 1e-9
            return (current_translation - prev_translation) / delta_t, city_SE3_ego_reference

        if next_annotation is not None:
            city_SE3_ego_next = timestamp_city_SE3_ego_dict[next_timestamp_ns]
            reference_SE3_ego_next = city_SE3_ego_reference.inverse().compose(city_SE3_ego_next)

            current_translation = np.array([current_annotation[1]["tx_m"], current_annotation[1]["ty_m"], current_annotation[1]["tz_m"]])   
            next_translation = np.array([next_annotation["tx_m"].item(), next_annotation["ty_m"].item(), next_annotation["tz_m"].item()])   

            #convert next annotations into the current annotation reference frame
            next_translation = reference_SE3_ego_next.transform_from(next_translation)
            
            delta_t = (next_timestamp_ns - current_timestamp_ns) * 1e-9
            return (next_translation - current_translation) / delta_t, city_SE3_ego_reference


def aggregate_sweeps(log_dir, timestamp_ns, num_sweeps = 5):
    lidar_dir = log_dir / "sensors" / "lidar"
    sweep_paths = sorted(lidar_dir.glob("*.feather"))
    timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir)
    city_SE3_ego_reference = timestamp_city_SE3_ego_dict[timestamp_ns]

    reference_index = sweep_paths.index(lidar_dir / f"{timestamp_ns}.feather")
    start = max(0, reference_index - num_sweeps + 1)
    end = reference_index + 1
    sweeps_list = []
    transform_list = []
    delta_list = []

    for i in range(start, end):
        timestamp_ns_i = int(sweep_paths[i].stem)

        sweeps_list.append(sweep_paths[i])
        timestamp_delta = abs(timestamp_ns_i - timestamp_ns)

        delta_list.append(timestamp_delta)
        assert timestamp_ns >= timestamp_ns_i
        if timestamp_delta != 0:
            city_SE3_ego_ti = timestamp_city_SE3_ego_dict[timestamp_ns_i]
            reference_SE3_ego_ti = city_SE3_ego_reference.inverse().compose(city_SE3_ego_ti)
            transform_list.append(reference_SE3_ego_ti)
        else:
            city_SE3_ego_t = timestamp_city_SE3_ego_dict[timestamp_ns]
            reference_SE3_ego_t = city_SE3_ego_reference.inverse().compose(city_SE3_ego_t)
            transform_list.append(reference_SE3_ego_t)
    
    while len(sweeps_list) < num_sweeps:
        sweeps_list.append(sweeps_list[-1])
        transform_list.append(transform_list[-1])
        delta_list.append(delta_list[-1])

    sweeps_list = sweeps_list[::-1]
    transform_list = transform_list[::-1]
    delta_list = delta_list[::-1]

    return sweeps_list, transform_list, delta_list

def generate_info(filename, log_id, log_dir, annotations, name2cid, all_timestamps, n_sweep):
    timestamp_ns = int(filename.split(".")[0])
    lidar_path = "{log_dir}/sensors/lidar/{timestamp_ns}.feather".format(log_dir=log_dir,timestamp_ns=timestamp_ns)

    mmcv.check_file_exist(lidar_path)
    
    if annotations is None:
        gt_bboxes_3d = []
        gt_labels = []
        gt_names = [] 
        gt_num_pts = []
        gt_velocity = []
        gt_uuid = []
        
    else:
        curr_annotations = annotations[annotations["timestamp_ns"] == timestamp_ns] # maybe exists N agents at the timestamp
        # curr_annotations = curr_annotations[curr_annotations["num_interior_pts"] > 0] # only keep the agents with lidar points
        valid_flag =  list(curr_annotations["num_interior_pts"] > 0)
        gt_bboxes_3d = []
        gt_labels = []
        gt_names = [] 
        gt_num_pts = []
        gt_velocity = []
        gt_uuid = []
        gt_city_SE3_ego = []
        '''
            (0, 
            timestamp_ns                          315969904359876000
            track_uuid          4d8b2da8-f934-4738-8257-b9a24a34966e
            category                                       BICYCLIST
            length_m                                          1.0324
            width_m                                          1.04356
            height_m                                        1.915459
            qw                                             -0.004864
            qx                                                   0.0
            qy                                                   0.0
            qz                                              0.999988
            tx_m                                          127.652988
            ty_m                                            5.051275
            tz_m                                           -1.473346
            num_interior_pts                                       4
            Name: 0, dtype: object)
        '''
        for i, annotation in enumerate(curr_annotations.iterrows()): # annotation: [row_id, info]
            class_name = annotation[1]["category"]
#             if class_name == 'PEDESTRIAN' and annotation[1]["track_uuid"] == '18156413-9706-435b-a2e3-b645a9d5649c':
#                 print(f"{annotation[1]['timestamp_ns'] * 1e-9} : x={annotation[1]['tx_m']}, y={annotation[1]['ty_m']}")
# #                 breakpoint()
            if class_name not in class_names:
                del valid_flag[i]
                continue 

            track_uuid = annotation[1]["track_uuid"]
            num_interior_pts = annotation[1]["num_interior_pts"]

            gt_labels.append(name2cid[class_name])
            gt_names.append(class_name)
            gt_num_pts.append(num_interior_pts)
            gt_uuid.append(track_uuid)

            translation = np.array([annotation[1]["tx_m"], annotation[1]["ty_m"], annotation[1]["tz_m"]])
            lwh = np.array([annotation[1]["length_m"], annotation[1]["width_m"], annotation[1]["height_m"]])
            rotation = quat_to_mat(np.array([annotation[1]["qw"], annotation[1]["qx"], annotation[1]["qy"], annotation[1]["qz"]])) # 四元数 to 旋转矩阵
            ego_SE3_object = SE3(rotation=rotation, translation=translation) # 3D object到ego vehicle的变换矩阵

            rot = ego_SE3_object.rotation
            lwh = lwh.tolist()
            center = translation.tolist()
            center[2] = center[2] - lwh[2] / 2 # 底面中心
            yaw = math.atan2(rot[1, 0], rot[0, 0])

            gt_bboxes_3d.append([*center, *lwh, yaw])

            velocity, city_SE3_ego = box_velocity(annotation, timestamp_ns, all_timestamps, annotations, Path(log_dir))[:2]
            
            gt_velocity.append(velocity) # m/s
            gt_city_SE3_ego.append(city_SE3_ego) #ego到global的变换矩阵
        assert len(valid_flag) == len(gt_names) == len(gt_bboxes_3d), breakpoint()

    sweeps, transforms, deltas = aggregate_sweeps(Path(log_dir), timestamp_ns, n_sweep)
    
    # info of many objects at the same timestamp
    info = {
        'lidar_path': lidar_path,
        'log_id': log_id,
        'sweeps' : sweeps, 
        'transforms' : transforms,
        'timestamp': timestamp_ns,
        'timestamp_deltas' :deltas,
        'gt_bboxes' : gt_bboxes_3d,
        'gt_labels' : gt_labels,
        'gt_names' : gt_names, 
        'gt_num_pts' : gt_num_pts,
        'gt_velocity' : gt_velocity,
        'gt_uuid' : gt_uuid,
        'gt_city_SE3_ego' : gt_city_SE3_ego,
        'valid_flag': valid_flag,
    }

    return info 

class AV2_Converter:
    def __init__(self, data_root, out_dir, split, workers):
        self.data_root = data_root
        self.out_dir = out_dir
        self.split = split
        self.workers = workers
        self.log_ids = os.listdir(os.path.join(self.data_root, self.split))
        self.classes = (
            'REGULAR_VEHICLE', 'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 
            'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS',
            'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER')
        self.name2cid = {c: i for i, c in enumerate(self.classes)}
        self.global_coordinate = True
        self.debug = False
        self.camera_types = [
            'ring_front_center',
            'ring_front_left',
            'ring_front_right',
            'ring_side_left',
            'ring_side_right',
            'ring_rear_left',
            'ring_rear_right',
                        ]
        self.image_shape = {
            'ring_front_center': (2048, 1550, 3),
            'ring_front_left': (1550, 2048, 3),
            'ring_front_right': (1550, 2048, 3),
            'ring_side_left': (1550, 2048, 3),
            'ring_side_right': (1550, 2048, 3),
            'ring_rear_left': (1550, 2048, 3),
            'ring_rear_right': (1550, 2048, 3),
        }
        os.makedirs(out_dir, exist_ok=True)
    
    def __len__(self):
        return len(self.log_ids)
    
    def convert(self):
        print('Start converting ...')
        """Convert action."""
        mmcv.track_parallel_progress(self.create_single_info, range(len(self)),
                                        self.workers)
        print('\nFinished ...')

    def create_single_info(self, idx):
        infos = []
        log_id = self.log_ids[idx]
        assert log_id in TRAIN or log_id in VAL or log_id in TEST
        
        log_dir = "{root_path}/{split}/{log_id}".format(root_path=self.data_root, split=self.split, log_id=log_id)
        lidar_paths = "{log_dir}/sensors/lidar".format(log_dir=log_dir)
        annotations_path = "{log_dir}/annotations.feather".format(log_dir=log_dir)
        annotations = read_feather(Path(annotations_path))

        mmcv.check_file_exist(annotations_path)

        all_timestamps = sorted([int(filename.split(".")[0]) for filename in os.listdir(lidar_paths)])
        timestamp_city_SE3_ego_dict = read_city_SE3_ego(log_dir) # EGO -> city SE3 at different timestamps
        for i, filename in enumerate(sorted(os.listdir(lidar_paths))):
            info = self.generate_info(filename, log_id, log_dir, annotations, self.name2cid, all_timestamps, timestamp_city_SE3_ego_dict)
            info = self.create_boxes_info(info, root_path=os.path.join(self.data_root, self.split))
            infos.append(info)
            
        out_file = "{out_dir}/{split}/{seq_id}.pkl".format(out_dir=self.out_dir, split=self.split, seq_id=idx)
        pickle.dump(infos, open(out_file, "wb"))
    
    def generate_info(self, filename, log_id, log_dir, annotations, name2cid, all_timestamps, timestamp_city_SE3_ego_dict):
        timestamp_ns = int(filename.split(".")[0])
        lidar_path = "{log_dir}/sensors/lidar/{timestamp_ns}.feather".format(log_dir=log_dir,timestamp_ns=timestamp_ns)

        mmcv.check_file_exist(lidar_path)
        
        if annotations is None:
            gt_bboxes_3d = []
            gt_labels = []
            gt_names = [] 
            gt_num_pts = []
            gt_velocity = []
            gt_uuid = []
            
        else:
            curr_annotations = annotations[(annotations["timestamp_ns"] == timestamp_ns)] 
            curr_annotations = curr_annotations[curr_annotations["category"].isin(self.classes)] # filter out the agents that are not in the classes
            # curr_annotations = curr_annotations[curr_annotations["num_interior_pts"] > 0] # only keep the agents with lidar points
            valid_flag =  [] # only extract the agents pose with lidar points
            gt_bboxes_3d = []
            gt_labels = []
            gt_names = [] 
            gt_num_pts = []
            gt_velocity = []
            gt_uuid = []
            gt_city_SE3_ego = []
            '''
                (0, 
                timestamp_ns                          315969904359876000
                track_uuid          4d8b2da8-f934-4738-8257-b9a24a34966e
                category                                       BICYCLIST
                length_m                                          1.0324
                width_m                                          1.04356
                height_m                                        1.915459
                qw                                             -0.004864
                qx                                                   0.0
                qy                                                   0.0
                qz                                              0.999988
                tx_m                                          127.652988
                ty_m                                            5.051275
                tz_m                                           -1.473346
                num_interior_pts                                       4
                Name: 0, dtype: object)
            '''
            for i, annotation in enumerate(curr_annotations.iterrows()): 
                annotation = annotation[1] # annotation: [row_id, info]
                class_name = annotation["category"]
                # if class_name == 'PEDESTRIAN' and annotation[1]["track_uuid"] == '18156413-9706-435b-a2e3-b645a9d5649c':
                #     print(f"{annotation[1]['timestamp_ns'] * 1e-9} : x={annotation[1]['tx_m']}, y={annotation[1]['ty_m']}")

                track_uuid = annotation["track_uuid"]
                num_interior_pts = annotation["num_interior_pts"]

                gt_labels.append(name2cid[class_name])
                gt_names.append(class_name)
                gt_num_pts.append(num_interior_pts)
                gt_uuid.append(track_uuid)

                translation = np.array([annotation["tx_m"], annotation["ty_m"], annotation["tz_m"]])
                lwh = np.array([annotation["length_m"], annotation["width_m"], annotation["height_m"]])
                rotation = quat_to_mat(np.array([annotation["qw"], annotation["qx"], annotation["qy"], annotation["qz"]])) # queration to rotation matrix
                ego_SE3_object = SE3(rotation=rotation, translation=translation) # 3D object到ego vehicle的变换矩阵

                rot = ego_SE3_object.rotation
                lwh = lwh.tolist()
                center = translation.tolist()
                center[2] = center[2] - lwh[2] / 2 # 底面中心
                yaw = math.atan2(rot[1, 0], rot[0, 0])

                gt_bboxes_3d.append([*center, *lwh, yaw])

                velocity = self.box_velocity(annotation, timestamp_city_SE3_ego_dict, timestamp_ns, all_timestamps, annotations)
                gt_velocity.append(velocity[:2]) # [vx, vy] m/s
                
                city_SE3_ego = timestamp_city_SE3_ego_dict[timestamp_ns]
                gt_city_SE3_ego.append(city_SE3_ego) # EGO -> CITY SE3
                valid_flag.append(annotation["num_interior_pts"] > 0)
            assert len(valid_flag) == len(gt_names) == len(gt_bboxes_3d)
        
        # all 3D objects' infos at current timestamp 
        info = {
            'lidar_path': lidar_path,
            'log_id': log_id,
            'timestamp': timestamp_ns,
            'gt_bboxes' : gt_bboxes_3d,
            'gt_labels' : gt_labels,
            'gt_names' : gt_names, 
            'gt_num_pts' : gt_num_pts,
            'gt_velocity' : gt_velocity,
            'gt_uuid' : gt_uuid,
            'gt_city_SE3_ego' : gt_city_SE3_ego,
            'valid_flag': valid_flag,
        }

        return info 

    def box_velocity(self, current_annotation, timestamp_city_SE3_ego_dict, current_timestamp_ns, all_timestamps, annotations):
        '''
        Function:
            compute the velocity of each object at the current timestamp
        Args:   
            current_annotation: pd.DataFrame
            timestamp_city_SE3_ego_dict: dict[SE3]
            current_timestamp_ns: int
            all_timestamps: list[int]
            annotations: pd.DataFrame
        '''
        city_SE3_ego_reference = timestamp_city_SE3_ego_dict[current_timestamp_ns]
        curr_index = all_timestamps.index(current_timestamp_ns)
        prev_index = curr_index - VELOCITY_SAMPLING_RATE # 5 估算速度: delta_x(m) / 0.5(s) 
        next_index = curr_index + VELOCITY_SAMPLING_RATE

        track_uuid = current_annotation["track_uuid"]

        if prev_index > 0:
            prev_timestamp_ns = all_timestamps[prev_index]

            #get annotation in prev timestamp
            prev_annotations = annotations[annotations["timestamp_ns"] == int(prev_timestamp_ns)]
            prev_annotation = prev_annotations[prev_annotations["track_uuid"] == track_uuid]

            if len(prev_annotation) == 0:
                prev_annotation = None
        else:
            prev_annotation = None 

        if next_index < len(all_timestamps):
            next_timestamp_ns = all_timestamps[next_index]

            #get annotation in next timestamp
            next_annotations = annotations[annotations["timestamp_ns"] == int(next_timestamp_ns)]
            next_annotation = next_annotations[next_annotations["track_uuid"] == track_uuid]

            if len(next_annotation) == 0:
                next_annotation = None
        else:
            next_annotation = None 

        if prev_annotation is None and next_annotation is None:
            return np.array([0, 0, 0])

        # take centered average of displacement for velocity
        if prev_annotation is not None and next_annotation is not None:
            city_SE3_ego_prev = timestamp_city_SE3_ego_dict[prev_timestamp_ns]
            reference_SE3_ego_prev = city_SE3_ego_reference.inverse().compose(city_SE3_ego_prev)

            city_SE3_ego_next = timestamp_city_SE3_ego_dict[next_timestamp_ns]
            reference_SE3_ego_next = city_SE3_ego_reference.inverse().compose(city_SE3_ego_next)

            prev_translation = np.array([prev_annotation["tx_m"].item(), prev_annotation["ty_m"].item(), prev_annotation["tz_m"].item()])   
            next_translation = np.array([next_annotation["tx_m"].item(), next_annotation["ty_m"].item(), next_annotation["tz_m"].item()])   

            #convert prev and next annotations into the current annotation reference frame
            prev_translation = reference_SE3_ego_prev.transform_from(prev_translation)
            next_translation = reference_SE3_ego_next.transform_from(next_translation)

            delta_t = (next_timestamp_ns - prev_timestamp_ns) * 1e-9
            return (next_translation - prev_translation) / delta_t

        # take one-sided average of displacement for velocity
        else:
            if prev_annotation is not None:
                city_SE3_ego_prev = timestamp_city_SE3_ego_dict[prev_timestamp_ns]
                reference_SE3_ego_prev = city_SE3_ego_reference.inverse().compose(city_SE3_ego_prev)

                prev_translation = np.array([prev_annotation["tx_m"].item(), prev_annotation["ty_m"].item(), prev_annotation["tz_m"].item()])   
                current_translation = np.array([current_annotation["tx_m"], current_annotation["ty_m"], current_annotation["tz_m"]])   

                #convert prev annotation into the current annotation reference frame
                prev_translation = reference_SE3_ego_prev.transform_from(prev_translation)

                delta_t = (current_timestamp_ns - prev_timestamp_ns) * 1e-9
                return (current_translation - prev_translation) / delta_t

            if next_annotation is not None:
                city_SE3_ego_next = timestamp_city_SE3_ego_dict[next_timestamp_ns]
                reference_SE3_ego_next = city_SE3_ego_reference.inverse().compose(city_SE3_ego_next)

                current_translation = np.array([current_annotation["tx_m"], current_annotation["ty_m"], current_annotation["tz_m"]])   
                next_translation = np.array([next_annotation["tx_m"].item(), next_annotation["ty_m"].item(), next_annotation["tz_m"].item()])   

                #convert next annotations into the current annotation reference frame
                next_translation = reference_SE3_ego_next.transform_from(next_translation)
                
                delta_t = (next_timestamp_ns - current_timestamp_ns) * 1e-9
                return (next_translation - current_translation) / delta_t
    
    def create_boxes_info(self, av2_info, root_path):
        """Export 2d annotation from the info file and raw data.
        Args:
            root_path (str): Root path of the raw data.
            info_path (str): Path of the info file.
            mono3d (bool, optional): Whether to export mono3d annotation.
                Default: True.
        """
        coco_2d_dict = dict(annotations=[], images=[], categories=self.name2cid)
        
        log_id = av2_info["log_id"]
        timestamp = av2_info["timestamp"]

        av2_info['gt_2d_boxes'] = {}
        av2_info['image_files'] = {}
        av2_info['gt_2d_corners'] = {}
        av2_info['cam2ego_rotation'] = {}
        av2_info['cam2ego_translation'] = {}
        av2_info['cam_intrinsic'] = {}
        
        log_dir = Path("{}/{}".format(root_path, log_id))
        
        cam_imgs, cam_models = {}, {}
        for cam_name in self.camera_types:
            cam_models[cam_name] = PinholeCamera.from_feather(log_dir, cam_name)

            cam_path = root_path + "/{}/sensors/cameras/{}/".format(log_id, cam_name)
            closest_dst = np.inf
            closest_img = None
            for filename in os.listdir(cam_path):
                if not filename.endswith(".jpg"):
                    continue

                img_timestamp = int(filename.split(".")[0])
                delta = abs(timestamp - img_timestamp)

                if delta < closest_dst:
                    closest_img = cam_path + filename
                    closest_dst = delta
                    
            cam_imgs[cam_name] = closest_img

        for cam_name in self.camera_types:
            (height, width, _) = self.image_shape[cam_name]
            self.get_2d_boxes(av2_info, copy.deepcopy(cam_models[cam_name]), width, height, cam_imgs[cam_name])
        
        # convert dict to list
        av2_info['valid_flag'] = [av2_info['gt_2d_boxes'][uuid] != None for uuid in av2_info['gt_uuid']]
        av2_info['gt_2d_boxes'] = [av2_info['gt_2d_boxes'][uuid] for uuid in av2_info['gt_uuid']]
        av2_info['image_files'] = [av2_info['image_files'][uuid] for uuid in av2_info['gt_uuid']]
        av2_info['gt_2d_corners'] = [av2_info['gt_2d_corners'][uuid] for uuid in av2_info['gt_uuid']]
        av2_info['cam2ego_rotation'] = [av2_info['cam2ego_rotation'][uuid] for uuid in av2_info['gt_uuid']]
        av2_info['cam2ego_translation'] = [av2_info['cam2ego_translation'][uuid] for uuid in av2_info['gt_uuid']]
        av2_info['cam_intrinsic'] = [av2_info['cam_intrinsic'][uuid] for uuid in av2_info['gt_uuid']]
        return av2_info

    def get_2d_boxes(self, info, cam_model, width, height, cam_img):
        timestamp = info["timestamp"]
        ego_SE3_cam = cam_model.ego_SE3_cam
        camera_intrinsic = cam_model.intrinsics.K
        cam2ego_rotation = ego_SE3_cam.rotation
        cam2ego_translation = ego_SE3_cam.translation
        
        for name, bbox, velocity, uuid in zip(info["gt_names"], info["gt_bboxes"], info["gt_velocity"], info["gt_uuid"]):
            quat = yaw_to_quaternion3d(bbox[-1]).tolist()
            cuboid_ego = Cuboid.from_numpy(np.array(bbox[:-1] + quat), name, timestamp)
            
            cuboid_ego_av2 = copy.deepcopy(cuboid_ego)
            cuboid_ego_av2.dst_SE3_object.translation[2] = cuboid_ego_av2.dst_SE3_object.translation[2] + cuboid_ego_av2.height_m / 2
            cuboid_cam = cuboid_ego_av2.transform(ego_SE3_cam.inverse())
            
            cam_box = CuboidList([cuboid_cam])
            cuboids_vertices_cam = cam_box.vertices_m
            N, V, D = cuboids_vertices_cam.shape

            # Collapse first dimension to allow for vectorization.
            cuboids_vertices_cam = cuboids_vertices_cam.reshape(-1, D)
            _, _, is_valid = cam_model.project_cam_to_img(cuboids_vertices_cam)

            num_valid = np.sum(is_valid)
            if num_valid > 0:
                corner_coords = view_points(cuboid_cam.vertices_m.T, camera_intrinsic, True).T[:, :2].tolist()

                # Keep only corners that fall within the image.
                final_coords = post_process_coords(corner_coords, (width, height))
                # Skip if the convex hull of the re-projected corners
                # does not intersect the image canvas.
                if final_coords is None:
                    continue
                # a object may be projected to different view images and choose the bigger 2D box 
                if info['gt_2d_boxes'].get(uuid, None) is None:
                    info['gt_2d_boxes'][uuid] = final_coords
                    info['image_files'][uuid] = cam_img
                    info['gt_2d_corners'][uuid] = np.array(corner_coords)
                    info['cam2ego_rotation'][uuid] = cam2ego_rotation
                    info['cam2ego_translation'][uuid] = cam2ego_translation
                    info['cam_intrinsic'][uuid] = camera_intrinsic
                else:
                    origin_delta_x = abs(info['gt_2d_boxes'][uuid][2] - info['gt_2d_boxes'][uuid][0])
                    origin_delta_y = abs(info['gt_2d_boxes'][uuid][3] - info['gt_2d_boxes'][uuid][1])
                    curr_delta_x = abs(final_coords[2] - final_coords[0])
                    curr_delta_y = abs(final_coords[3] - final_coords[1])
                    if curr_delta_x > origin_delta_x or curr_delta_y > origin_delta_y:
                        info['gt_2d_boxes'][uuid] = final_coords
                        info['image_files'][uuid] = cam_img
                        info['gt_2d_corners'][uuid] = np.array(corner_coords)
                        info['cam2ego_rotation'][uuid] = cam2ego_rotation
                        info['cam2ego_translation'][uuid] = cam2ego_translation
                        info['cam_intrinsic'][uuid] = camera_intrinsic
                    if self.debug: 
                        corner_1 = (int(info['gt_2d_boxes'][uuid][0]), int(info['gt_2d_boxes'][uuid][1]))
                        corner_2 = (int(info['gt_2d_boxes'][uuid][2]), int(info['gt_2d_boxes'][uuid][3]))
                        img_tmp = cv2.imread(cam_img)
                        cv2.rectangle(img_tmp, corner_1, corner_2, (0, 255, 0), 2)
                        img_name = cam_img.split("/")[-1].split(".")[0]
                        cv2.imwrite(f"/root/autodl-tmp/debug_imgs/{img_name}.png", img_tmp)
            else:
                info['gt_2d_boxes'][uuid] = None
                info['image_files'][uuid] = None
                info['gt_2d_corners'][uuid] = None        
                info['cam2ego_rotation'][uuid] = None
                info['cam2ego_translation'][uuid] = None
                info['cam_intrinsic'][uuid] = None
                
def create_av2_infos(root_path, out_dir, max_sweeps=5):
    os.makedirs(out_dir, exist_ok=True)

    name2cid = {c: i for i, c in enumerate(class_names)}
    n_sweep = max_sweeps

    train_infos = []
    val_infos = []
    test_infos = []
    seq_id = 0
    for log_id in tqdm(TRAIN):
        print("log_id : ", log_id)
        split = "train"
        log_dir = "{root_path}/{split}/{log_id}".format(root_path=root_path, split=split, log_id=log_id)
        lidar_paths = "{log_dir}/sensors/lidar".format(log_dir=log_dir)
        annotations_path = "{log_dir}/annotations.feather".format(log_dir=log_dir)
        annotations = read_feather(Path(annotations_path))

        mmcv.check_file_exist(annotations_path)

        all_timestamps = sorted([int(filename.split(".")[0]) for filename in os.listdir(lidar_paths)])
        for i, filename in tqdm(enumerate(sorted(os.listdir(lidar_paths)))):
            # if i % TRAIN_SAMPLE_RATE != 0:
            #     continue 
#             import time
#             time_start = time.time()
            info = generate_info(filename, log_id, log_dir, annotations, name2cid, all_timestamps, n_sweep)
#             time_tmp = time.time()
            info_bbox = create_boxes_info(info, root_path=os.path.join(root_path, split), mono3d=True)
#             time_end = time.time()
            
            train_infos.append(info_bbox)
            
        out_file = "{out_dir}/{split}/{seq_id}.pkl".format(out_dir=out_dir, split=split, seq_id=seq_id)
        pickle.dump(train_infos, open(out_file, "wb"))
        train_infos = []
    
    seq_id = 0
    for log_id in tqdm(VAL):
        split = "val"
        log_dir = "{root_path}/{split}/{log_id}".format(root_path=root_path, split=split, log_id=log_id)
        lidar_paths = "{log_dir}/sensors/lidar".format(log_dir=log_dir)
        annotations_path = "{log_dir}/annotations.feather".format(log_dir=log_dir)
        annotations = read_feather(Path(annotations_path))
        mmcv.check_file_exist(annotations_path)

        all_timestamps = sorted([int(filename.split(".")[0]) for filename in os.listdir(lidar_paths)])
        for i, filename in enumerate(sorted(os.listdir(lidar_paths))):
            # if i % VAL_SAMPLE_RATE != 0:
            #     continue 

            info = generate_info(filename, log_id, log_dir, annotations, name2cid, all_timestamps, n_sweep) # t时刻n个智能体对应的n个3D标注框
            info_bbox = create_boxes_info(info, root_path=os.path.join(root_path, split), mono3d=True)
            val_infos.append(info)

        out_file = "{out_dir}/{split}/{seq_id}.pkl".format(out_dir=out_dir, split=split, seq_id=seq_id)
        pickle.dump(val_infos, open(out_file, "wb"))
        val_infos = []
    seq_id = 0
    # for log_id in tqdm(TRAIN[40:50]):
    #     split = "test"
    #     log_dir = "{root_path}/{split}/{log_id}".format(root_path=root_path, split=split, log_id=log_id)
    #     lidar_paths = "{log_dir}/sensors/lidar".format(log_dir=log_dir)
    #     annotations = None

    #     all_timestamps = sorted([int(filename.split(".")[0]) for filename in os.listdir(lidar_paths)])
    #     for i, filename in enumerate(sorted(os.listdir(lidar_paths))):
    #         if i % VAL_SAMPLE_RATE != 0:
    #             continue 

    #         info = generate_info(filename, log_id, log_dir, annotations, name2cid, all_timestamps, n_sweep)

    #         test_infos.append(info)

    # out_file = "{out_dir}/{info_prefix}_infos_{split}.pkl".format(out_dir=out_dir,info_prefix=info_prefix, split=split)
    # pickle.dump(test_infos, open(out_file, "wb"))

def get_2d_boxes(info, cam_model, width, height, cam_img, mono3d=True):
    timestamp = info["timestamp"]
    log_id = info["log_id"]
    ego_SE3_cam = cam_model.ego_SE3_cam
    camera_intrinsic = cam_model.intrinsics.K

    coco_infos = []
    for name, bbox, velocity, uuid in zip(info["gt_names"], info["gt_bboxes"], info["gt_velocity"], info["gt_uuid"]):
        quat = yaw_to_quaternion3d(bbox[-1]).tolist()
        cuboid_ego = Cuboid.from_numpy(np.array(bbox[:-1] + quat), name, timestamp)
        
        cuboid_ego_av2 = copy.deepcopy(cuboid_ego)
        cuboid_ego_av2.dst_SE3_object.translation[2] = cuboid_ego_av2.dst_SE3_object.translation[2] + cuboid_ego_av2.height_m / 2
        cuboid_cam = cuboid_ego_av2.transform(ego_SE3_cam.inverse())
        
        cam_box = CuboidList([cuboid_cam])
        cuboids_vertices_cam = cam_box.vertices_m
        N, V, D = cuboids_vertices_cam.shape

        # Collapse first dimension to allow for vectorization.
        cuboids_vertices_cam = cuboids_vertices_cam.reshape(-1, D)
        _, _, is_valid = cam_model.project_cam_to_img(cuboids_vertices_cam)

        num_valid = np.sum(is_valid)
        if num_valid > 0:
            corner_coords = view_points(cuboid_cam.vertices_m.T, camera_intrinsic, True).T[:, :2].tolist()

            # Keep only corners that fall within the image.
            final_coords = post_process_coords(corner_coords, (width, height))
            # Skip if the convex hull of the re-projected corners
            # does not intersect the image canvas.
            if final_coords is None:
                continue
            else:
                min_x, min_y, max_x, max_y = final_coords

            # repro_rec = generate_record(min_x, min_y, max_x, max_y, log_id, cam_img, name)

#             if mono3d and (repro_rec is not None):
#                 rot = cuboid_ego.dst_SE3_object.rotation
#                 size = [cuboid_ego.length_m, cuboid_ego.width_m, cuboid_ego.height_m]
#                 center = cuboid_ego.dst_SE3_object.translation.tolist()
#                 yaw = math.atan2(rot[1, 0], rot[0, 0]) - cam_model.egovehicle_yaw_cam_rad

#                 repro_rec['bbox_cam3d'] = [*center, *size, yaw]
# #                 breakpoint()
#                 repro_rec['velo_cam3d'] = ego_SE3_cam.transform_from([*velocity, 0])[:2]

#                 center2d = points_cam2img(cuboid_cam.dst_SE3_object.translation.tolist(), camera_intrinsic, with_depth=True)

#                 repro_rec['center2d'] = center2d.squeeze().tolist()
#                 # normalized center2D + depth
#                 # if samples with depth < 0 will be removed
#                 if repro_rec['center2d'][2] <= 0:
#                     continue

            # repro_rec['attribute_name'] = "None"
            # repro_rec['attribute_id'] = 0
            # repro_rec['gt_num_pts'] = num_pts
            # repro_rec['gt_uuid'] = uuid

            # coco_infos.append(repro_rec)
            
            # choose the bigger 2D box 
            if info['gt_2d_boxes'].get(uuid, None) is None:
                info['gt_2d_boxes'][uuid] = final_coords
                info['image_files'][uuid] = cam_img
            else:
                origin_delta_x = abs(info['gt_2d_boxes'][uuid][2] - info['gt_2d_boxes'][uuid][0])
                origin_delta_y = abs(info['gt_2d_boxes'][uuid][3] - info['gt_2d_boxes'][uuid][1])
                curr_delta_x = abs(final_coords[2] - final_coords[0])
                curr_delta_y = abs(final_coords[3] - final_coords[1])
                if curr_delta_x > origin_delta_x or curr_delta_y > origin_delta_y:
                    info['gt_2d_boxes'][uuid] = final_coords
                    info['image_files'][uuid] = cam_img
                if name == 'PEDESTRIAN':
                    import cv2
                    corner_1 = (int(info['gt_2d_boxes'][uuid][0]), int(info['gt_2d_boxes'][uuid][1]))
                    corner_2 = (int(info['gt_2d_boxes'][uuid][2]), int(info['gt_2d_boxes'][uuid][3]))
                    img_tmp = cv2.imread(cam_img)
                    cv2.rectangle(img_tmp, corner_1, corner_2, (0, 255, 0), 2)
                    img_name = cam_img.split("/")[-1].split(".")[0]
                    cv2.imwrite(f"/root/autodl-tmp/debug_imgs/{img_name}.png", img_tmp)

    return coco_infos

def export_2d_annotation(root_path, info_path, mono3d=True):
    """Export 2d annotation from the info file and raw data.
    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        mono3d (bool, optional): Whether to export mono3d annotation.
            Default: True.
    """
    # get bbox annotations for camera
    camera_types = [
        'ring_front_center',
        'ring_front_left',
        'ring_front_right',
        'ring_side_left',
        'ring_side_right',
        'ring_rear_left',
        'ring_rear_right',
    ]
    av2_infos = mmcv.load(info_path)
    cat2Ids = [
        dict(id=class_names.index(cat_name), name=cat_name)
        for cat_name in class_names
    ]

    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)

    for info in mmcv.track_iter_progress(av2_infos):
        log_id = info["log_id"]
        timestamp = info["timestamp"]

        log_dir = Path("{}/{}".format(root_path, log_id))
        
        cam_imgs, cam_models = {}, {}
        for cam_name in camera_types:
            cam_models[cam_name] = PinholeCamera.from_feather(log_dir, cam_name)

            cam_path = root_path + "/{}/sensors/cameras/{}/".format(log_id, cam_name)
            closest_dst = np.inf
            closest_img = None
            for filename in os.listdir(cam_path):
                img_timestamp = int(filename.split(".")[0])
                delta = abs(timestamp - img_timestamp)

                if delta < closest_dst:
                    closest_img = cam_path + filename
                    closest_dst = delta
                    
            cam_imgs[cam_name] = closest_img

        for cam_name in camera_types:
            cam_img = mmcv.imread(cam_imgs[cam_name])
            (height, width, _) = cam_img.shape

            coco_infos = get_2d_boxes(info, copy.deepcopy(cam_models[cam_name]), width, height, cam_imgs[cam_name], mono3d=True)
            
            coco_2d_dict['images'].append(
                dict(
                file_name=cam_imgs[cam_name],
                timestamp=timestamp,
                id=cam_name,
                token=log_id,
                ego_SE3_cam_rotation=cam_models[cam_name].ego_SE3_cam.rotation,
                ego_SE3_cam_translation=cam_models[cam_name].ego_SE3_cam.translation,
                ego_SE3_cam_intrinsics=cam_models[cam_name].intrinsics.K,
                width=width,
                height=height))

            coco_2d_dict['annotations'].append(coco_infos)
                    
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'

    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')

def create_boxes_info(av2_info, root_path, mono3d=True):
    """Export 2d annotation from the info file and raw data.
    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        mono3d (bool, optional): Whether to export mono3d annotation.
            Default: True.
    """
    # get bbox annotations for camera
    camera_types = [
        'ring_front_center',
        'ring_front_left',
        'ring_front_right',
        'ring_side_left',
        'ring_side_right',
        'ring_rear_left',
        'ring_rear_right',
    ]

    cat2Ids = [
        dict(id=class_names.index(cat_name), name=cat_name)
        for cat_name in class_names
    ]

    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    
    log_id = av2_info["log_id"]
    timestamp = av2_info["timestamp"]

    av2_info['gt_2d_boxes'] = {}
    av2_info['image_files'] = {}
    
    log_dir = Path("{}/{}".format(root_path, log_id))
    
    cam_imgs, cam_models = {}, {}
    for cam_name in camera_types:
        cam_models[cam_name] = PinholeCamera.from_feather(log_dir, cam_name)

        cam_path = root_path + "/{}/sensors/cameras/{}/".format(log_id, cam_name)
        closest_dst = np.inf
        closest_img = None
        for filename in os.listdir(cam_path):
            if not filename.endswith(".jpg"):
                continue

            img_timestamp = int(filename.split(".")[0])
            delta = abs(timestamp - img_timestamp)

            if delta < closest_dst:
                closest_img = cam_path + filename
                closest_dst = delta
                
        cam_imgs[cam_name] = closest_img

    for cam_name in camera_types:
        cam_img = mmcv.imread(cam_imgs[cam_name])
        (height, width, _) = cam_img.shape

        coco_infos = get_2d_boxes(av2_info, copy.deepcopy(cam_models[cam_name]), width, height, cam_imgs[cam_name], mono3d=False)
        
        coco_2d_dict['images'].append(
            dict(
            file_name=cam_imgs[cam_name],
            timestamp=timestamp,
            id=cam_name,
            token=log_id,
            ego_SE3_cam_rotation=cam_models[cam_name].ego_SE3_cam.rotation,
            ego_SE3_cam_translation=cam_models[cam_name].ego_SE3_cam.translation,
            ego_SE3_cam_intrinsics=cam_models[cam_name].intrinsics.K,
            width=width,
            height=height))

        coco_2d_dict['annotations'].append(coco_infos)
    av2_info.update(coco_2d_dict)
    return av2_info
            
        
def create_av2_full_infos(root_path,
                          info_prefix,
                          out_dir,
                          max_sweeps=10):
    """Create info file of argoverse V2 dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """
    os.makedirs(out_dir, exist_ok=True)

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    from nuscenes.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}/{}_infos_test.pkl'.format(out_dir, info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}/{}_infos_train.pkl'.format(out_dir, info_prefix))
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(root_path,
                                 '{}/{}_infos_val.pkl'.format(out_dir, info_prefix))
        mmcv.dump(data, info_val_path)