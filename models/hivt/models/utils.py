import os
import pickle
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Union, Tuple

import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch_geometric.data import Data
import pandas as pd

NDArrayFloat = npt.NDArray[np.float64]
Frame = Dict[str, Any]
Frames = List[Frame]
Sequences = Dict[str, Frames]
CAMERAS = ['ring_front_center', 'ring_front_left', 'ring_front_right', 'ring_rear_left', 'ring_rear_right', 'ring_side_left', 'ring_side_right']


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


class DistanceDropEdge(object):

    def __init__(self, max_distance: Optional[float] = None) -> None:
        self.max_distance = max_distance

    def __call__(self,
                 edge_index: torch.Tensor,
                 edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.max_distance is None:
            return edge_index, edge_attr
        row, col = edge_index
        mask = torch.norm(edge_attr, p=2, dim=-1) < self.max_distance
        edge_index = torch.stack([row[mask], col[mask]], dim=0)
        edge_attr = edge_attr[mask]
        return edge_index, edge_attr


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

def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        fan_in = m.in_channels / m.groups
        fan_out = m.out_channels / m.groups
        bound = (6.0 / (fan_in + fan_out)) ** 0.5
        nn.init.uniform_(m.weight, -bound, bound)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.MultiheadAttention):
        if m.in_proj_weight is not None:
            fan_in = m.embed_dim
            fan_out = m.embed_dim
            bound = (6.0 / (fan_in + fan_out)) ** 0.5
            nn.init.uniform_(m.in_proj_weight, -bound, bound)
        else:
            nn.init.xavier_uniform_(m.q_proj_weight)
            nn.init.xavier_uniform_(m.k_proj_weight)
            nn.init.xavier_uniform_(m.v_proj_weight)
        if m.in_proj_bias is not None:
            nn.init.zeros_(m.in_proj_bias)
        nn.init.xavier_uniform_(m.out_proj.weight)
        if m.out_proj.bias is not None:
            nn.init.zeros_(m.out_proj.bias)
        if m.bias_k is not None:
            nn.init.normal_(m.bias_k, mean=0.0, std=0.02)
        if m.bias_v is not None:
            nn.init.normal_(m.bias_v, mean=0.0, std=0.02)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(4, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(4, 0):
                    nn.init.orthogonal_(hh)
            elif 'weight_hr' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
                nn.init.ones_(param.chunk(4, 0)[1])
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                for ih in param.chunk(3, 0):
                    nn.init.xavier_uniform_(ih)
            elif 'weight_hh' in name:
                for hh in param.chunk(3, 0):
                    nn.init.orthogonal_(hh)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)

def progressbar(itr: Iterable, desc: Optional[str] = None, **kwargs) -> Iterable:
    pbar = tqdm(itr, **kwargs)
    if desc:
        pbar.set_description(desc)
    return pbar


def save(obj, path: str) -> None:
    dir = os.path.dirname(path)
    if dir != "":
        os.makedirs(dir, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


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


def unpack_predictions(frames: Frames, classes: List[str]) -> Frames:
    """Convert data from mmdetection3D format to numpy format.

    Args:
        frames: list of frames
        classes: list of class names

    Returns:
        List of prediction item where each is a dictionary with keys:
            translation_m: ndarray[instance, [x, y, z]]
            size: ndarray[instance, [l, w, h]]
            yaw: ndarray[instance, float]
            velocity_m_per_s: ndarray[instance, [x, y]]
            label: ndarray[instance, int]
            score: ndarray[instance, float]
            frame_index: ndarray[instance, int]
    """
    unpacked_frames = []
    for frame_dict in frames:
        prediction = frame_dict["pts_bbox"]
        bboxes = prediction["boxes_3d"].tensor.numpy()
        label = prediction["labels_3d"].numpy()
        unpacked_frames.append(
            {
                "translation_m": bboxes[:, :3],
                "size": bboxes[:, 3:6],
                "yaw": wrap_pi(bboxes[:, 6]),
                "velocity_m_per_s": bboxes[:, -2:],
                "label": label,
                "name": np.array(
                    [classes[id] if id < len(classes) else "OTHER" for id in label]
                ),
                "score": prediction["scores_3d"].numpy(),
            }
        )
    return unpacked_frames


def annotate_frame_metadata(
    prediction_frames: Frames, label_frames: Frames, metadata_keys: List[str]
) -> None:
    """Copy annotations with provided keys from label to prediction frames.

    Args:
        prediction_frames: list of prediction frames
        label_frames: list of label frames
        metadata_keys: keys of the annotations to be copied
    """
    assert len(prediction_frames) == len(label_frames)
    for prediction, label in zip(prediction_frames, label_frames):
        for key in metadata_keys:
            prediction[key] = label[key]


def group_frames(frames_list: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Parameters
    ----------
    frames_list: list
        list of frames, each containing a detections snapshot for a timestamp
    """
    frames_by_seq_id = defaultdict(list)
    frames_list = sorted(frames_list, key=lambda f: f["timestamp_ns"])
    for frame in frames_list:
        frames_by_seq_id[frame["seq_id"]].append(frame)
    return dict(frames_by_seq_id)


def ungroup_frames(frames_by_seq_id: Dict[str, List[Dict]]):
    ungrouped_frames = []
    for frames in frames_by_seq_id.values():
        ungrouped_frames.extend(frames)
    return ungrouped_frames


def index_array_values(array_dict: Dict, index: Union[int, np.ndarray]) -> Dict:
    return {
        k: v[index] if isinstance(v, np.ndarray) else v for k, v in array_dict.items()
    }


def array_dict_iterator(array_dict: Dict, length: int):
    return (index_array_values(array_dict, i) for i in range(length))


def concatenate_array_values(array_dicts: List[Dict]) -> Dict[str, np.ndarray]:
    """
    Concatenates numpy arrays in list of dictionaries
    Handles inconsistent keys (will skip missing keys)
    Does not concatenate non-numpy values (int, str), sets to value if all values are equal
    """
    combined = defaultdict(list)
    for array_dict in array_dicts:
        for k, v in array_dict.items():
            combined[k].append(v)
    concatenated = {}
    for k, vs in combined.items():
        if all(isinstance(v, np.ndarray) for v in vs):
            if any(v.size > 0 for v in vs):
                concatenated[k] = np.concatenate([v for v in vs if v.size > 0])
            else:
                concatenated[k] = vs[0]
        elif all(vs[0] == v for v in vs):
            concatenated[k] = vs[0]
    return concatenated


def filter_by_class_names(frames_by_seq_id: Dict, class_names) -> Dict:
    frames = ungroup_frames(frames_by_seq_id)
    return group_frames(
        [
            index_array_values(frame, np.isin(frame["name"], class_names))
            for frame in frames
        ]
    )


def filter_by_class_thresholds(
    frames_by_seq_id: Dict, thresholds_by_class: Dict[str, float]
) -> Dict:
    frames = ungroup_frames(frames_by_seq_id)
    return group_frames(
        [
            concatenate_array_values(
                [
                    index_array_values(
                        frame,
                        (frame["name"] == class_name) & (frame["score"] >= threshold),
                    )
                    for class_name, threshold in thresholds_by_class.items()
                ]
            )
            for frame in frames
        ]
    )


def filter_by_ego_xy_distance(frames_by_seq_id: Sequences, distance_threshold: float):
    frames = ungroup_frames(frames_by_seq_id)
    return group_frames(
        [
            index_array_values(
                frame,
                np.linalg.norm(
                    frame["translation_m"][:, :2]
                    - np.array(frame["ego_translation_m"])[:2],
                    axis=1,
                )
                <= distance_threshold,
            )
            for frame in frames
        ]
    )


def group_by_track_id(frames: Frames) -> Sequences:
    tracks_by_track_id = defaultdict(list)
    for frame_idx, frame in enumerate(frames):
        for instance in array_dict_iterator(frame, len(frame["translation_m"])):
            instance["frame_idx"] = frame_idx
            tracks_by_track_id[instance["track_id"]].append(instance)
    return dict(tracks_by_track_id)


def wrap_pi(theta: NDArrayFloat) -> NDArrayFloat:
    theta = np.remainder(theta, 2 * np.pi)
    theta[theta > np.pi] -= 2 * np.pi
    return theta


# def unpack_labels(labels: List[Dict]) -> List[Dict]:
#     """
#     Returns
#     -------
#     list:
#     """
#     unpacked_labels = []
#     for label in labels:
#         bboxes = (
#             np.array(label["gt_bboxes"]) # [array(x_ego, y_ego, z_ego, l, w, h, theta, prob, cls), ...]
#             if len(label["gt_bboxes"]) > 0
#             else np.zeros((0, 7))
#         )
#         velocity = (
#             np.array(label["gt_velocity"])
#             if len(label["gt_velocity"]) > 0
#             else np.zeros((0, 2))
#         )
#         base_root = '/'.join(label['lidar_path'].split('/')[:-2])
#         img_paths = [os.path.join(base_root, 'cameras', camera, f'{label['timestamp']}.jpg') for camera in CAMERAS]
#         assert os.path.exists(img_paths), print(img_paths)
        
#         unpacked_labels.append(
#             {
#                 "translation_m": bboxes[:, :3],
#                 "size": bboxes[:, 3:6],
#                 "yaw": wrap_pi(bboxes[:, 6]),
#                 "velocity_m_per_s": velocity,
#                 "label": np.array(label["gt_labels"], dtype=int),
#                 "name": np.array(label["gt_names"]),
#                 "track_id": np.array([UUID(id).int for id in label["gt_uuid"]]),
#                 "timestamp_ns": label["timestamp"],
#                 "seq_id": label["log_id"],
#                 "img_paths": img_paths,
#                 "lidar2img": np.array(), 
#             }
#         )
#     return unpacked_labels
