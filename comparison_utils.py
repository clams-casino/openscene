import os
import re
import numpy as np
import torch


# def combine_scan_regions(scan_name, data_3d_dir, fused_feat_dir):
#     """
#     Combine the data from the different regions of a scan.

#     Args:
#     scan_name: str
#         The name of the scan.
#     data_3d_dir: str
#         The directory where the pre-processed 3D data for the scan's regions are stored.
#     fused_feat_dir: str
#         The directory where the fused features for the scan's regions are stored.

#     Returns:
#     scan_points: np.ndarray
#         The combined points of the scan, (N, 3)
#     scan_colors: np.ndarray
#         The combined colors of the scan, (N, 3)
#     scan_fused_feats: np.ndarray
#         The combined fused features of the scan, (N, E), where E is the feature dimension
#     """
#     scan_points = []
#     scan_colors = []
#     scan_fused_feats = []
    
#     data_3d_filenames = [fname for fname in os.listdir(data_3d_dir) if scan_name in fname]
    
#     for fname in data_3d_filenames:

#         match = re.search(
#             r'_region(\d+)\.pth', 
#             fname
#         )
#         if match:
#             region_number = match.group(1)
#         else:
#             raise RuntimeError(f"No region number for file {fname}")
            
#         fused_feat_fname = f"{scan_name}_region{region_number}_0.pt"
        
#         data_fused_feat = torch.load(
#             os.path.join(fused_feat_dir, fused_feat_fname)
#         )
        
#         data_3d = torch.load(
#             os.path.join(data_3d_dir, fname)
#         )
        
#         points = data_3d[0]
#         colors = data_3d[1]
        
#         # Check dimensions are the same with mask full and the number of points!!!
#         assert points.shape[0] == data_fused_feat["mask_full"].shape[0]
        
#         scan_points.append(
#             points[data_fused_feat["mask_full"]]
#         )
#         scan_colors.append(
#             colors[data_fused_feat["mask_full"]]
#         )
#         scan_fused_feats.append(
#             data_fused_feat["feat"]
#         )
        
#     scan_points = np.concatenate(scan_points, axis=0)
#     scan_colors = np.concatenate(scan_colors, axis=0)
#     scan_fused_feats = np.concatenate(scan_fused_feats, axis=0)
    
#     return scan_points, scan_colors, scan_fused_feats



def get_region_features(
        data_3d_path, 
        fused_feat_path, 
        disnet_runner,
):
    data_fused_feat = torch.load(fused_feat_path)
    data_3d = torch.load(data_3d_path)
    points, colors, _ = data_3d

    # Check dimensions are the same with mask full and the number of points!!!
    assert points.shape[0] == data_fused_feat["mask_full"].shape[0]

    # Get only the points which have a corresponding fused feature
    points = points[data_fused_feat["mask_full"]]
    colors = colors[data_fused_feat["mask_full"]]

    # make sure fused features are normalized
    fused_feats = data_fused_feat["feat"]
    fused_feats = (np.linalg.norm(fused_feats, axis=1, keepdims=True) + 1e-5)

    # get 3D features from distill model
    distill_feats = disnet_runner.run(points)

    fused_feats = torch.Tensor(fused_feats).to(torch.float16).to('cuda')
    distill_feats.to(torch.float16).to('cuda')

    return points, colors, fused_feats, distill_feats




import MinkowskiEngine as ME
from dataset.voxelization_utils import sparse_quantize

class DisNetRunner:
    def __init__(self, model, model_voxel_size=0.02):
        self._model_voxel_size = model_voxel_size
        self._model = model
        self._model.eval()
        self._model.cuda()

    def run(self, points):
        """
        Expects a point cloud of N points as an numpy array of shape (N,3)
        Creates a voxel representation of the point cloud with M voxels

        Returns:
            voxel_embeddings as a torch tensor of shape (N,E), where E is the embedding dimension
            NOTE, the embeddings are normalized
        """
        
        # voxelize the point cloud
        unique_coords, inverse_map = self._voxelize_points(points)
        
        # add batch dimension to the coords
        unique_coords_batched = ME.utils.batched_coordinates([unique_coords])
        
        # 3D distill model trained with no color input, uses all ones as the feature
        feats = torch.ones(unique_coords.shape[0], 3)
        
        # move inputs to gpu
        unique_coords_batched = unique_coords_batched.to('cuda')
        feats = feats.to('cuda')
        
        input_st = ME.SparseTensor(features=feats, coordinates=unique_coords_batched)
        
        with torch.no_grad():
            voxel_embeddings = self._model(input_st)

            # normalize embeddings
            voxel_embeddings /= (voxel_embeddings.norm(dim=-1, keepdim=True) + 1e-5)

        # use inverse_map to map embeddings to all points
        return voxel_embeddings[inverse_map]

    def _voxelize_points(self, points):
        coords_np = np.floor(points / self._model_voxel_size)
        unique_map, inverse_map = sparse_quantize(coords_np, return_index=True)
        unique_coords = torch.Tensor(coords_np[unique_map])
        return unique_coords, inverse_map



from util.util import extract_clip_feature

# Modified from original implementation in Openscene
def extract_text_feature(
    labelset, 
    prompt_eng=True,
    feature_2d_extractor='openseg',
    dataset='matterport_3d'
):
    if prompt_eng:
        print('Use prompt engineering: a XX in a scene')
        labelset = [ "a " + label + " in a scene" for label in labelset]
        
        if dataset == 'scannet_3d':
            labelset[-1] = 'other'
        if dataset == 'matterport_3d':
            labelset[-2] = 'other'
            
    if 'lseg' in feature_2d_extractor:
        text_features = extract_clip_feature(labelset)
    elif 'openseg' in feature_2d_extractor:
        text_features = extract_clip_feature(labelset, model_name="ViT-L/14@336px")
    else:
        raise NotImplementedError

    return text_features.to(torch.float16).to('cuda')


def compute_predictions(
    label_feats,
    fused_feats,
    distill_feats,
    method='ensemble',
):
    assert fused_feats.shape == distill_feats.shape
    assert fused_feats.shape[1] == label_feats.shape[1]
    
    if method == 'fusion':
        sim = fused_feats @ label_feats.T
        pred = torch.argmax(sim, dim=1)
        
    elif method == 'distill':
        sim = distill_feats @ label_feats.T
        pred = torch.argmax(sim, dim=1)
        
    elif method == 'ensemble':
        sim_fusion = fused_feats @ label_feats.T
        sim_distill = distill_feats @ label_feats.T
        
        max_sim_fusion, argmax_sim_fusion = torch.max(sim_fusion, dim=1)
        max_sim_distill, argmax_sim_distill = torch.max(sim_distill, dim=1)
        
        pred = argmax_sim_distill
        use_fusion = max_sim_fusion > max_sim_distill
        pred[use_fusion] = argmax_sim_fusion[use_fusion]
        
    return pred.cpu().numpy()
