import pandas as pd
import pickle
import os
import glob
import numpy as np
from typing import Any, Dict, Iterable, List, Optional, Union, Tuple
from torch_geometric.data import Data
import torch

class TemporalData(Data):
    def __init__(self,
                 x: Optional[torch.Tensor] = None,
                 positions: Optional[torch.Tensor] = None,
                 edge_index: Optional[torch.Tensor] = None,
                 edge_attrs: Optional[List[torch.Tensor]] = None,
                 y: Optional[torch.Tensor] = None,
                 num_nodes: Optional[int] = None,
                 padding_mask: Optional[torch.Tensor] = None,
                 bos_mask: Optional[torch.Tensor] = None,
                 rotate_angles: Optional[torch.Tensor] = None,
                 seq_id: Optional[int] = None,
                 target_agent_idx:  Optional[int] = None,
                 actors_id: Optional[List] = None,
                 **kwargs) -> None:
        if x is None:
            super(TemporalData, self).__init__()
            return
        super(TemporalData, self).__init__(x=x, positions=positions, edge_index=edge_index, egde_attrs=edge_attrs, y=y, num_nodes=num_nodes,
                                           padding_mask=padding_mask, bos_mask=bos_mask, rotate_angles=rotate_angles,
                                           seq_id=seq_id, target_agent_idx=target_agent_idx ,actors_id=actors_id, **kwargs)
        if edge_attrs is not None:
            for t in range(self.x.size(1)):
                self[f'edge_attr_{t}'] = edge_attrs[:, t]

    def __inc__(self, key, value):
        if key == 'lane_actor_index':
            return torch.tensor([[self['lane_vectors'].size(0)], [self.num_nodes]])
        else:
            return super().__inc__(key, value)


def load(path: str) -> Any:
    """
    Returns
    -------
        object or None: returns None if the file does not exist
    """
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def load_forecast_data(data_path):
    raw_data = load(data_path)
    forecast_data = []
    keys_to_delete = ['lidar_path', 'sweeps', 'timestamp_deltas']
    for info in raw_data:
        for key in keys_to_delete:
            del info[key]
        for i in range(len(info['gt_names'])):
            uuid = info['gt_uuid'][i]
            row = {
                'log_id': info['log_id'],
                'timestamp': info['timestamp'],
                'transform': info['transforms'][0],  
                'gt_bbox': info['gt_bboxes'][i],
                'gt_label': info['gt_labels'][i],
                'gt_name': info['gt_names'][i],
                'gt_num_pt': info['gt_num_pts'][i],
                'gt_velocity': info['gt_velocity'][i],
                'gt_uuid': info['gt_uuid'][i],
                'gt_2d_box': info['gt_2d_boxes'].get(uuid, (0, 0, 0, 0)),
                'image_file': info['image_files'].get(uuid, ''),
                'valid_flag': info['valid_flag'][i]
            }

            forecast_data.append(row)
            
    df = pd.DataFrame(forecast_data)

    return df

def load_forecast_data_seq(root_path):
    forecast_data = {}
    log_ids = []
    for data_path in glob.glob(os.path.join(root_path, '*.pkl')):
        raw_data = load(data_path) # length is about 150 
        log_id = raw_data[0]['log_id']
        log_ids.append(log_id)
        info_pd = []
        for info in raw_data:
            for i in range(len(info['gt_names'])):
                row = {
                    'log_id': info['log_id'],
                    'timestamp': info['timestamp'], 
                    'gt_bbox': info['gt_bboxes'][i],
                    'gt_label': info['gt_labels'][i],
                    'gt_name': info['gt_names'][i],
                    'gt_num_pt': info['gt_num_pts'][i],
                    'gt_velocity': info['gt_velocity'][i],
                    'gt_uuid': info['gt_uuid'][i],
                    'gt_2d_box': info['gt_2d_boxes'][i],
                    'image_file': info['image_files'][i],
                    'valid_flag': info['valid_flag'][i],
                    'gt_city_SE3_ego': info['gt_city_SE3_ego'][i],
                    'gt_2d_corner': info['gt_2d_corners'][i],
                    'cam2ego_rot': info['cam2ego_rotation'][i],
                    'cam2ego_tran': info['cam2ego_translation'][i],
                    'cam_intrinsic': info['cam_intrinsic'][i],
                }
                info_pd.append(row)
        forecast_data[log_id] = pd.DataFrame(info_pd)
    return forecast_data, log_ids