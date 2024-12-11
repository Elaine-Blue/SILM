import sys
sys.path.append('/root/LT3D')
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
from models import SinglePoseDetector, opts
import cv2

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
STEP_FRAMES = 10
EGO_DIST = 20
EXTEND_SIZE = 0
THRESHOLD = 0.8

# init pose extractor
opt = opts().init()
pose_extractor = SinglePoseDetector(opt)

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
        assert os.path.exists(self.pkl_file_path), f"{self.pkl_file_path} does not exist"
        self.data_df = load_forecast_data(self.pkl_file_path)
        self.log_ids = os.listdir(os.path.join(self.root, '', split))
        self.patches_info = {}
        self.agent_type = agent_type
        self.show = True # for debug
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        self.seq_len = 0
        self._processed_paths = [] 
        for file in self._processed_paths:
            if not os.path.exists(file):
                self._processed_paths.remove(file)
        self.preprocess = preprocess
        if self.preprocess:
            self.pose_extractor = pose_extractor
        super(ArgoverseV2Dataset, self).__init__(root, transform)

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.split, 'data')

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, 'processed_test', self.split)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return os.listdir(os.path.join(self.root, self.split))

    @property
    def processed_paths(self) -> List[str]:
        return [os.path.join(self.processed_dir, f) for f in os.listdir(self.processed_dir)]

    def process(self) -> None:
        # if os.path.exists(self.processed_dir):
        #     return 
        for i, log_id in tqdm(enumerate(self.log_ids)):
            data_df_part = self.data_df[self.data_df['log_id'] == log_id]
            data_processed = self.process_argoverse_v2(data_df_part, self.split, log_id)
            for key, value in data_processed.items():
                data_processed_single = TemporalData(**value)
                pt_path = os.path.join(self.processed_dir, str(self.seq_len) + '.pt')
                self._processed_paths.append(pt_path)
                torch.save(data_processed_single, pt_path)  
                self.seq_len += 1

    def len(self) -> int:
        return len(os.listdir(self.processed_dir)) - 2

    def get(self, idx) -> Data:
        return torch.load(self.processed_paths[idx])
    
    def process_argoverse_v2(self,
                            raw_df: pd.DataFrame,
                            split: str,
                            log_id: str) -> Dict:
        # filter out actors that are unseen during the historical time steps
        raw_df = raw_df.sort_values(by='timestamp')
        raw_df = raw_df[raw_df['gt_name'].isin(CLASSES_FOR_FORECAST)] # filter out traffic signs, etc.
        timestep_range = [(x, x + HISTORICAL_FRAMES + PREDICT_FRAMES) for x in range(101) if x % STEP_FRAMES == 0]
        timestamps = list(np.sort(raw_df['timestamp'].unique()))
        timestamps_lst = [timestamps[x[0]:x[1]] for x in timestep_range]
        fragment_df_lst = [raw_df[raw_df['timestamp'].isin(ts)] for ts in timestamps_lst]

        infos = {}
        
        i = 0

        logging.info(f'Processing log {log_id} with {len(fragment_df_lst)} trajectory fragments')
        
        for idx in tqdm(range(len(fragment_df_lst))):
            timestamps = timestamps_lst[idx]
            fragment_df = fragment_df_lst[idx]
            historical_timestamps = timestamps[:HISTORICAL_FRAMES]
            historical_df = fragment_df[fragment_df['timestamp'].isin(historical_timestamps)]
            actor_ids = list(historical_df['gt_uuid'].unique())
            # filter the agents who didn't appear in the last 20 frames
            fragment_df = fragment_df[fragment_df['gt_uuid'].isin(actor_ids)]
            
            for actor_id, actor_df in fragment_df.groupby('gt_uuid'):
                # ignore the object if the object's historical frames are less than 5
                node_steps_observed = get_agent_node_steps(actor_df, timestamps, 'observed')
                if HISTORICAL_FRAMES -1 not in node_steps_observed:
                    fragment_df = fragment_df[fragment_df['gt_uuid'] != actor_id]
                    actor_ids.remove(actor_id)
                    continue
                # ignore the object if the object's distance from ego car over EGO_DIST limitation
                current_xy = actor_df[actor_df['timestamp'] == timestamps[HISTORICAL_FRAMES-1]]['gt_bbox'].values[0][:2]
                ego_dist = np.linalg.norm(np.array(current_xy)) # last observed timestamp distance from ego car
                if ego_dist == 0 or ego_dist > EGO_DIST:
                    fragment_df = fragment_df[fragment_df['gt_uuid'] != actor_id]
                    actor_ids.remove(actor_id)
            
            num_nodes = len(actor_ids)
            
            if num_nodes == 0:
                logging.warning(f'No nodes in {log_id} fragment {idx}')
                continue
            elif num_nodes == 1:
                logging.warning(f'Only one node in {log_id} fragment {idx}')
                continue
            
            # initialization 
            x = torch.zeros(num_nodes, 50, 2, dtype=torch.float)
            edge_indexes = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous() # 生成fully-connected边
            
            edge_attrs = torch.zeros(edge_indexes.size(1), 50, 2, dtype=torch.float)

            padding_mask = torch.ones(num_nodes, 50, dtype=torch.bool)
            bos_mask = torch.zeros(num_nodes, 20, dtype=torch.bool)
            rotate_angles = torch.zeros(num_nodes, dtype=torch.float)
            gt_labels = torch.zeros(num_nodes, dtype=torch.int32)
            boxes_2d = torch.zeros(num_nodes, 20, 4, dtype=torch.int32)
            use_pose = torch.zeros(num_nodes, dtype=torch.bool)
            valid_flag = torch.zeros(num_nodes, 20, dtype=torch.bool)
            images_names = [[''] * HISTORICAL_FRAMES for _ in range(num_nodes)]
            keypoints = torch.zeros(num_nodes, 20, 51, dtype=torch.float)

            target_agent_ids = get_center_agent(fragment_df, self.agent_type)
            for actor_id, actor_df in fragment_df.groupby('gt_uuid'):
                node_idx = actor_ids.index(actor_id)
                node_steps = [timestamps.index(timestamp) for timestamp in actor_df['timestamp']]
                padding_mask[node_idx, node_steps] = False
                
                if actor_df['gt_name'].iloc[0] in CLASSES_VEHICLE:
                    gt_labels[node_idx] = 0
                elif actor_df['gt_name'].iloc[0] in CLASSES_PEDESTRIAN:
                    gt_labels[node_idx] = 1
                else:
                    gt_labels[node_idx] = 2

                gt_labels[node_idx] = actor_df['gt_label'].iloc[0]
                xy = torch.from_numpy(np.stack([np.array(list(actor_df['gt_bbox']))[:, 0], np.array(list(actor_df['gt_bbox']))[:, 1]], axis=-1)).float()
                x[node_idx, node_steps, :] = xy
                
                node_historical_steps = list(filter(lambda node_step: node_step < 20, node_steps))
                
                if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
                    heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
                    rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
                else:  # make no predictions for the actor if the number of valid time steps is less than 2
                    padding_mask[node_idx, 20:] = True

                if self.preprocess:
                    if actor_id in target_agent_ids:
                        try:
                            boxes_2d[node_idx, node_historical_steps] = torch.tensor(list(actor_df['gt_2d_box']), dtype=torch.int32)[:len(node_historical_steps)]
                        except:
                            breakpoint()
                        for k, node_step in enumerate(node_historical_steps):
                            images_names[node_idx][node_step] = list(actor_df['image_file'])[k]
                        
                        use_pose[node_idx] = True
                        
                        valid_flag[node_idx, node_historical_steps] = torch.tensor(list(actor_df['valid_flag']))[:len(node_historical_steps)]
                        images = {idx: cv2.imread(image_path) for idx, image_path in enumerate(images_names[node_idx]) if image_path != '' }
                        for step in node_historical_steps:
                            if valid_flag[node_idx, step]:
                                corner = boxes_2d[node_idx, step].tolist()
                                w, h = corner[2] - corner[0], corner[3] - corner[1]
                                if w > 50 and h > 80:
                                    img = images[step]
                                    img_cropped = img[corner[1]-EXTEND_SIZE:corner[3]+EXTEND_SIZE, corner[0]-EXTEND_SIZE:corner[2]+EXTEND_SIZE, :]
                                    ret = torch.from_numpy(self.pose_extractor.run(img_cropped)['results'])
                                    keypoints[node_idx, step, :] = ret.reshape(1, -1) # torch.where(ret[:, [2]] > THRESHOLD, ret[:, :2], torch.tensor([0., 0.])).reshape(1, -1)
                                else:
                                    valid_flag[node_idx, step] = False
                        use_pose[node_idx] = valid_flag[node_idx, :].any()
                        # if use_pose[node_idx]:
                        #     cv2.imwrite(f'./debug/{self.seq_len}_{node_idx}_{p}.jpg', img_cropped)
                            
            # bos_mask is True if time step t is valid and time step t-1 is invalid
            bos_mask[:, 0] = ~padding_mask[:, 0]
            bos_mask[:, 1: 20] = padding_mask[:, : 19] & ~padding_mask[:, 1: 20]

            positions = x.clone()
            # relative vectors from t=0 to t in ego vehicle frame
            x[:, 20:] = torch.where((padding_mask[:, 19].unsqueeze(-1) | padding_mask[:, 20:]).unsqueeze(-1),
                                    torch.zeros(num_nodes, 30, 2),
                                    x[:, 20:] - x[:, 19].unsqueeze(-2))
            x[:, 1: 20] = torch.where((padding_mask[:, : 19] | padding_mask[:, 1: 20]).unsqueeze(-1),
                                    torch.zeros(num_nodes, 19, 2),
                                    x[:, 1: 20] - x[:, : 19])
            x[:, 0] = torch.zeros(num_nodes, 2)

            y = None if split == 'test' else x[:, 20:]
            
            infos[i] = {
                'x': x[:, : 20],  # [N, 20, 2]
                'positions': positions,  # [N, 50, 2]
                'edge_index': edge_indexes,  # [2, N x N - 1]
                'edge_attrs': edge_attrs,  # [N x N - 1, 20, 2]
                'y': y,  # [N, 30, 2]
                'num_nodes': num_nodes,
                'padding_mask': padding_mask,  # [N, 50]
                'bos_mask': bos_mask,  # [N, 20]
                'rotate_angles': rotate_angles,  # [N]
                'gt_labels': gt_labels, # [N]
                'target_agent_indexes': target_agent_ids, # [N]
                'actors_id': np.array([UUID(id).int for id in actor_ids]),
                'seq_id': self.seq_len + i
            }

            if self.agent_type == 'PEDESTRIAN':
                tmp = {
                    'boxes_2d': boxes_2d,
                    'use_pose': use_pose,
                    'image_files': images_names,
                    'valid_flags': valid_flag,
                    'keypoints': keypoints,
                }
                infos[i].update(tmp)
            i += 1
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

