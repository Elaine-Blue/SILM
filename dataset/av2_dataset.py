import sys
sys.path.append('/root/Demo/')
import os
from itertools import permutations
from itertools import product
from typing import Callable, Dict, List, Optional, Tuple, Union
from uuid import UUID
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
from tqdm import tqdm
import logging
# from models import SinglePoseDetector, opts
import cv2
from av2.geometry.se3 import SE3
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

from utils import *

AV2_CLASSES = ('REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER',
    'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS',
    'MESSAGE_BOARD_TRAILER', 'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG')

CLASSES_VEHICLE = ['REGULAR_VEHICLE', 'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS']
CLASSES_PEDESTRIAN = ['PEDESTRIAN']
CLASSES_CYCLIST = ['BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER']
CLASSES_FOR_FORECAST = CLASSES_VEHICLE + CLASSES_PEDESTRIAN + CLASSES_CYCLIST

HISTORICAL_FRAMES = 20
MIN_HISTORICAL_FRAMES = 5
MIN_PREDICT_FRAMES = 5
PREDICT_FRAMES =30
STEP_FRAMES = 1
EGO_DIST = 30
EXTEND_SIZE = 0
THRESHOLD = 0.8

# init pose extractor
# opt = opts().init()
# pose_extractor = SinglePoseDetector(opt)

velocity = {}
samples = {}
for cls in CLASSES_FOR_FORECAST:
    velocity[cls] = 0.0
    samples[cls] = 0

class ArgoverseV2Dataset(Dataset):
    def __init__(self,
                 root: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 local_radius: float = 50,
                 agent_type: str = 'PEDESTRIAN',
                 preprocess: bool = True) -> None:
        self._split = split
        self._local_radius = local_radius
        self.root = root
        self.split = split
        self.pkl_file_path = os.path.join(root, 'trainval', f"av2_infos_{split}.pkl")
        self.pkl_root_path = os.path.join(root, 'trainval_test', self.split)
        self.data_df, self.log_ids = load_forecast_data_seq(self.pkl_root_path)
        # self.log_ids = os.listdir(os.path.join(self.root, '', split))
        self.patches_info = {}
        self.agent_type = agent_type
        self.show = True # for debug
        self.miss_num = 0
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        self.seq_len = 0
        self._processed_paths = [] 
        for file in self._processed_paths:
            if not os.path.exists(file):
                self._processed_paths.remove(file)
        self.preprocess = preprocess
        # if self.preprocess:
        #     self.pose_extractor = pose_extractor
        super(ArgoverseV2Dataset, self).__init__(root, transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.split, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed', self.split)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return os.listdir(os.path.join(self.root, self.split))

    @property
    def processed_paths(self) -> List[str]:
        return [os.path.join(self.processed_dir, f) for f in os.listdir(self.processed_dir)]

    def process(self) -> None:
        for i in tqdm(range(len(self.log_ids))):
            log_id = self.log_ids[i]
            data_df_part = self.data_df[log_id]
            data_processed = self.process_argoverse_v2(data_df_part, self.split, log_id)
            for key, value in data_processed.items():
                # breakpoint()
                data_processed_single = TemporalData(**value)
                pt_path = os.path.join(self.processed_dir, str(self.seq_len) + '.pt')
                self._processed_paths.append(pt_path)
                torch.save(data_processed_single, pt_path)  
                self.seq_len += 1
                # logging.debug(f"Process Seq_ID: {i}, Miss Trajectory: {self.miss_num}, Current Seq_ID : {self.seq_len}")
                
    def len(self) -> int:
        return len(os.listdir(self.processed_dir)) - 2

    def get(self, idx) -> Data:
        try:
            data = torch.load(self.processed_paths[idx])
        except:
            logging.debug(f"Load data failed: {self.processed_paths[idx]}")
            return None
        return data
    
    def process_argoverse_v2(self,
                            raw_df: pd.DataFrame,
                            split: str,
                            log_id: str) -> Dict:
        # filter out actors that are unseen during the historical time steps
        raw_df = raw_df.sort_values(by='timestamp')
        timestep_range = [(x, x + HISTORICAL_FRAMES + PREDICT_FRAMES) for x in range(101) if x % STEP_FRAMES == 0]
        timestamps = list(np.sort(raw_df['timestamp'].unique()))
        timestamps_lst = [timestamps[x[0]:x[1]] for x in timestep_range]
        fragment_df_lst = [raw_df[raw_df['timestamp'].isin(ts)] for ts in timestamps_lst] # split into fragments of 5s

        infos = {}

        # logging.info(f'Processing log {log_id} with {len(fragment_df_lst)} trajectory fragments')
        
        for idx, timestamps, fragment_df in zip(range(len(fragment_df_lst)), timestamps_lst, fragment_df_lst):
            historical_timestamps = timestamps[:HISTORICAL_FRAMES]
            historical_df = fragment_df[fragment_df['timestamp'].isin(historical_timestamps)]
            # 1. filter the agents who didn't appear in the last 20 framess
            actor_ids = list(historical_df['gt_uuid'].unique())
            fragment_df = fragment_df[fragment_df['gt_uuid'].isin(actor_ids)]
            
            for actor_id, actor_df in fragment_df.groupby('gt_uuid'):
                # ignore the object if the object's historical frames are less than 5
                past_timestamps = list(actor_df['timestamp'])
                if (timestamps[HISTORICAL_FRAMES -1] not in past_timestamps) | (len(past_timestamps) < 5):
                    fragment_df = fragment_df[fragment_df['gt_uuid'] != actor_id]
                    continue
                # ignore the object if the object's distance from ego car over EGO_DIST limitation
                ego_xy = actor_df[actor_df['timestamp'] == timestamps[HISTORICAL_FRAMES-1]]['gt_bbox'].values[0][:2]
                ego_dist = np.linalg.norm(np.array(ego_xy)) # last observed timestamp distance from ego car
                if ego_dist > EGO_DIST:
                    fragment_df = fragment_df[fragment_df['gt_uuid'] != actor_id]
            
            # update actors id for prediction 
            actor_ids = list(fragment_df['gt_uuid'].unique())
            num_nodes = len(actor_ids)
            
            if num_nodes == 0:
                # logging.warning(f'No nodes in {log_id} fragment {idx}')
                # self.miss_num += 1
                continue
            elif num_nodes == 1:
                # logging.warning(f'Only one node in {log_id} fragment {idx}')
                # self.miss_num += 1
                continue
            
            # initialization 
            x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
            edge_indexes = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous() # 生成fully-connected边
            padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)
            bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)
            rotate_angles = torch.zeros(num_nodes, dtype=torch.float)
            gt_labels = torch.zeros(num_nodes, dtype=torch.int32)
            
            # take ego-vehicle's pose at the last observed time step as origin
            city_SE3_origin = fragment_df[fragment_df['timestamp'] == timestamps[HISTORICAL_FRAMES -1]]['gt_city_SE3_ego'].values[0]
            origin_SE3_ego_all = []
            for timestamp in timestamps:
                if (timestamp == timestamps[HISTORICAL_FRAMES -1]) or (timestamp not in fragment_df['timestamp'].values):
                    origin_SE3_ego = SE3(rotation=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), translation=np.array([0, 0, 0]))
                else:
                    city_SE3_ego = fragment_df[fragment_df['timestamp'] == timestamp]['gt_city_SE3_ego'].values[0]
                    origin_SE3_ego = city_SE3_origin.inverse().compose(city_SE3_ego)
                origin_SE3_ego_all.append(origin_SE3_ego)
                    
            # boxes_2d = torch.zeros(num_nodes, 20, 4, dtype=torch.int32)
            # use_pose = torch.zeros(num_nodes, dtype=torch.bool)
            # valid_flag = torch.zeros(num_nodes, 20, dtype=torch.bool)
            # images_names = [[''] * HISTORICAL_FRAMES for _ in range(num_nodes)]
            # keypoints = torch.zeros(num_nodes, 20, 51, dtype=torch.float)

            for actor_id, actor_df in fragment_df.groupby('gt_uuid'):
                node_idx = actor_ids.index(actor_id)
                node_steps = [timestamps.index(timestamp) for timestamp in actor_df['timestamp']]
                xyz = np.stack([np.array(list(actor_df['gt_bbox']))[:, 0], np.array(list(actor_df['gt_bbox']))[:, 1], np.array(list(actor_df['gt_bbox']))[:, 2]], axis=-1)
                # cobvert to ego-vehicle coordinate
                for i, j in enumerate(node_steps):
                    x[node_idx, j, :] = torch.from_numpy(origin_SE3_ego_all[j].transform_from(xyz[i]))[:2]
                heading_vector = x[node_idx, HISTORICAL_FRAMES-1] - x[node_idx, HISTORICAL_FRAMES-2]
                rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
                padding_mask[node_idx, node_steps] = False
                # bos_mask is True if time step t is valid and time step t-1 is invalid
                bos_mask[:, 0] = ~padding_mask[:, 0]
                bos_mask[:, 1: 20] = padding_mask[:, : 19] & ~padding_mask[:, 1: 20]
                
                if actor_df['gt_name'].iloc[0] in CLASSES_VEHICLE:
                    gt_labels[node_idx] = 0
                elif actor_df['gt_name'].iloc[0] in CLASSES_PEDESTRIAN:
                    gt_labels[node_idx] = 1
                else:
                    gt_labels[node_idx] = 2

                # if self.preprocess:
                #     if actor_id in target_agent_ids:
                #         try:
                #             boxes_2d[node_idx, node_historical_steps] = torch.tensor(list(actor_df['gt_2d_box']), dtype=torch.int32)[:len(node_historical_steps)]
                #         except:
                #             breakpoint()
                #         for k, node_step in enumerate(node_historical_steps):
                #             images_names[node_idx][node_step] = list(actor_df['image_file'])[k]
                        
                #         use_pose[node_idx] = True
                        
                #         valid_flag[node_idx, node_historical_steps] = torch.tensor(list(actor_df['valid_flag']))[:len(node_historical_steps)]
                #         images = {idx: cv2.imread(image_path) for idx, image_path in enumerate(images_names[node_idx]) if image_path != '' }
                #         for step in node_historical_steps:
                #             if valid_flag[node_idx, step]:
                #                 corner = boxes_2d[node_idx, step].tolist()
                #                 w, h = corner[2] - corner[0], corner[3] - corner[1]
                #                 if w > 50 and h > 80:
                #                     img = images[step]
                #                     img_cropped = img[corner[1]-EXTEND_SIZE:corner[3]+EXTEND_SIZE, corner[0]-EXTEND_SIZE:corner[2]+EXTEND_SIZE, :]
                #                     ret = torch.from_numpy(self.pose_extractor.run(img_cropped)['results'])
                #                     keypoints[node_idx, step, :] = ret.reshape(1, -1) # torch.where(ret[:, [2]] > THRESHOLD, ret[:, :2], torch.tensor([0., 0.])).reshape(1, -1)
                #                 else:
                #                     valid_flag[node_idx, step] = False
                #         use_pose[node_idx] = valid_flag[node_idx, :].any()
                #         # if use_pose[node_idx]:
                #         #     cv2.imwrite(f'./debug/{self.seq_len}_{node_idx}_{p}.jpg', img_cropped)

            positions = x.clone()
            x[:, 20:] = torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
                                        torch.zeros(num_nodes, 30, 2),
                                        x[:, 20:] - x[:, 19].unsqueeze(-2))
            x[:, 1: 20] = torch.where((padding_mask[:, : 19] | padding_mask[:, 1: 20]).unsqueeze(-1),
                                    torch.zeros(num_nodes, 19, 2),
                                    x[:, 1: 20] - x[:, : 19])
            x[:, 0] = torch.zeros(num_nodes, 2)
            y = None if split == 'test' else x[:, 20:]
                
            infos[idx] = {
                'x': x[:, : 20],  # [N, 20, 2]
                'positions': positions,  # [N, 50, 2]
                'edge_index': edge_indexes,  # [2, N x N - 1]
                'y': y,  # [N, 30, 2]
                'num_nodes': num_nodes,
                'padding_mask': padding_mask,  # [N, 50]
                'bos_mask': bos_mask,  # [N, 20]
                'rotate_angles': rotate_angles,  # [N]
                'gt_labels': gt_labels, # [N]
                'actors_id': np.array([UUID(id).int for id in actor_ids]),
                'seq_id': self.seq_len + idx
            }

            # if self.agent_type == 'PEDESTRIAN':
            #     tmp = {
            #         'boxes_2d': boxes_2d,
            #         'use_pose': use_pose,
            #         'image_files': images_names,
            #         'valid_flags': valid_flag,
            #         'keypoints': keypoints,
            #     }
            #     infos[i].update(tmp)
        return infos

def get_center_agent(df, agent_type):
    '''
    function: get agents' id of target type 
    :param df: dataframe
    :param agent_type: str
    :return: [agent_id]
    '''
    agent_ids = list(df['gt_uuid'].unique())
    agent_ids_selected = []
    for agent_id in agent_ids:
        agent_df = df[df['gt_uuid'] == agent_id]
        if agent_df['gt_name'].iloc[0] == agent_type:
            agent_ids_selected.append(agent_id)
    return agent_ids_selected

def get_agent_node_steps(df, timestamps, type = 'all'):
    node_steps = [timestamps.index(timestamp) for timestamp in df['timestamp']]
    if type == 'all':
        return node_steps
    elif type == 'observed':
        observed_node_steps = [step for step in node_steps if step < HISTORICAL_FRAMES]
        return observed_node_steps
    elif type == 'predict':
        predict_node_steps = [step for step in node_steps if step >= HISTORICAL_FRAMES]
        return predict_node_steps