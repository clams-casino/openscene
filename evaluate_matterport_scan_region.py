import os
import pickle
import logging
from tqdm import tqdm
from typing import Dict, Optional, Union

import numpy as np
import open3d as o3d

from copy import deepcopy

from tag_mapping.evaluation import (
    LatticeNavigationGraph,
    assign_label_box_lattice_graph_nodes,
    assign_proposal_box_lattice_graph_nodes,
)
from tag_mapping.utils import get_box_corners
from tag_mapping.datasets.matterport import read_matterport_region_bounding_boxes


def evaluate_matterport_scan_region_localizations(
    params: Dict,
    scan_dir: Union[str, os.PathLike],
    openscene_region_segs_dir: Union[str, os.PathLike],
    lattice_graph_path: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Evaluates the tag map region/room localizations on a matterport scan by comparing
    the localized regions against the labeled ground-truth bounding boxes.

    Evaluation outputs are saved to a pickle file in output_dir.

    Args:
        params: Dictionary of parameters for the evaluation.
        scan_dir: Path to the matterport scan directory.
            The basename of the scan_dir is used as the scan name.
        openscene_region_segmentations_dir: Path to the direction
            containing the OpenScene region segmentations.
        lattice_graph_path: Path to the lattice graph corresponding to the scan.
            The filename of the lattice graph must contain the scan name.
        output_dir: Directory to save the evaluation outputs to.
        logger: Logger to use, if None a logger will be created at debug level.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.DEBUG)

    scan_name = os.path.basename(scan_dir)
    logger.info(f"running evaluation on matterport scan {scan_name}")
    assert scan_name in os.path.basename(
        lattice_graph_path
    ), f"Lattice graph does not match scan {scan_name}. Lattice graph at {lattice_graph_path}."

    house_file_path = os.path.join(
        scan_dir, "house_segmentations", f"{scan_name}.house"
    )
    logger.info(f"loaded house file from {house_file_path}")
    ply_file_path = os.path.join(scan_dir, "house_segmentations", f"{scan_name}.ply")
    logger.info(f"loaded ply file from {ply_file_path}")

    # Go through and find scan name files in the segmentations directory
    region_seg_filenames = []
    for filename in os.listdir(openscene_region_segs_dir):
        if scan_name in filename:
            region_seg_filenames.append(filename)
    if len(region_seg_filenames) == 0:
        raise FileNotFoundError(
            f"No region segmentation files found for scan {scan_name} in {openscene_region_segs_dir}"
        )
    else:
        logger.info(
            f"found the following region segmentation files for scan {scan_name}:\n" + \
            "\n\t".join(region_seg_filenames)
        )

    # Load the labeled ground-truth bounding boxes
    label_gt_boxes = read_matterport_region_bounding_boxes(house_file_path)

    # Load the lattice graph
    lattice_graph = LatticeNavigationGraph.load(lattice_graph_path)
    rc_scene = o3d.t.geometry.RaycastingScene()
    rc_scene.add_triangles(
        o3d.t.geometry.TriangleMesh.from_legacy(
            o3d.io.read_triangle_mesh(ply_file_path)
        )
    )
    logger.info(f"loaded lattice graph from {lattice_graph_path}")

    # Aggregate region segmentations for the scan
    points, colors, preds = ([], [], [])
    label_to_pred_ind = None

    for fname in region_seg_filenames:
            
        with open(os.path.join(openscene_region_segs_dir, fname), 'rb') as f:
            data = pickle.load(f)
            
        points.append(data["points"])
        colors.append(data["colors"])
        preds.append(data["preds"])
        
        # This assumes that label_to_ind is the same for all regions preds generated
        if label_to_pred_ind == None:
            label_to_pred_ind = data["label_to_ind"]
            
    points = np.concatenate(points, axis=0)
    colors = np.concatenate(colors, axis=0)
    preds = np.concatenate(preds, axis=0)

    # print(points.shape)
    # print(preds.shape)
    # print(label_to_pred_ind)


    logger.info(f"aggregated region segmentations")

    # Run localization for each label
    logger.info(f"started localization pipeline")
    label_proposals = {}
    for label in tqdm(label_gt_boxes.keys()):
        if label in params["label_params"]["blacklisted_labels"]:
            continue

        """
        Some labeled categories in the scene may not be in the object categories we are evaluating on,
        so we skip them.

        NOTE: For the tag map evaluation we didn't do this, we tried to process all labels categories
        in the scene and post-process the results to only consider the object categories.
        """
        if label not in label_to_pred_ind.keys():
            continue

        # print(label)


        pred_ind = label_to_pred_ind[label]
        label_points = points[preds == pred_ind]

        label_pcd = o3d.geometry.PointCloud()
        label_pcd.points = o3d.utility.Vector3dVector(label_points)

        cluster_inds = label_pcd.cluster_dbscan(**params["dbscan_params"])

        p_boxes, p_box_confidences, p_box_tags = ([], [], [])
        for i in np.unique(cluster_inds):
            if i == -1:
                continue
            
            cluster_points = label_pcd.select_by_index(
                np.where(cluster_inds == i)[0]
            )
            
            cluster_box = cluster_points.get_axis_aligned_bounding_box()
            cluster_box = expand_box_to_min_side_length(
                cluster_box, params["min_proposal_box_length"])

            p_boxes.append(cluster_box)
            p_box_confidences.append(1.0)
            p_box_tags.append("n/a")

        label_proposals[label] = {
            "boxes": p_boxes,
            "confidences": p_box_confidences,
            "tags": p_box_tags,
        }

    # print(label_proposals)


    logger.info(f"finished running localization pipeline")

    # Assign nodes of the lattice navigation graph to the labeled and proposed boxes
    label_lattice_inds = {}
    for label, gt_boxes in label_gt_boxes.items():

        """
        Some labeled categories in the scene may not be in the object categories we are evaluating on,
        so we skip them.

        NOTE: For the tag map evaluation we didn't do this, we tried to process all labels categories
        in the scene and post-process the results to only consider the object categories.
        """
        if label not in label_to_pred_ind.keys():
            continue

        gt_boxes_lattice_inds = []
        for gt_box in gt_boxes:
            inds = assign_label_box_lattice_graph_nodes(
                lattice_graph,
                rc_scene,
                get_box_corners(gt_box),
                # NOTE: DISABLE inflation for region box label assignment since regions are usually large
                enable_inflation=False,
            )
            gt_boxes_lattice_inds.append(inds)
        label_lattice_inds[label] = gt_boxes_lattice_inds

    for label, proposals in label_proposals.items():
        p_boxes = proposals["boxes"]
        p_boxes_lattice_inds = []
        for p_box in p_boxes:
            inds = assign_proposal_box_lattice_graph_nodes(
                lattice_graph,
                rc_scene,
                get_box_corners(p_box),
            )
            p_boxes_lattice_inds.append(inds)
        proposals["lattice_inds"] = p_boxes_lattice_inds

    # print(label_gt_boxes)
    # print(label_lattice_inds)
    # print(label_proposals)

    # Evaluate precision metric over all proposals
    logger.info(f"started computing precision metrics")
    for label, proposals in tqdm(label_proposals.items()):
        # get all lattice inds for all ground-truth boxes of this label
        all_gt_lattice_inds = set()
        for inds in label_lattice_inds[label]:
            all_gt_lattice_inds.update(inds)
        all_gt_lattice_inds = np.array(list(all_gt_lattice_inds))

        if len(all_gt_lattice_inds) == 0:
            logger.debug(f"no ground-truth lattice inds for label {label}, skipping it")
            continue

        proposals["metrics"] = []
        for p_lattice_inds in proposals["lattice_inds"]:
            p_lattice_inds = np.array(p_lattice_inds)

            if len(p_lattice_inds) == 0:
                logger.debug(
                    f"proposal with no assigned lattice nodes in {label}, setting mean shortest path length to np.nan"
                )
                proposals["metrics"].append(
                    {
                        "mean_spl": np.nan,
                        "portion_at_gt": 0.0,
                    }
                )
                continue

            # query shortest path for all pairs of label and proposal lattice inds
            batch_p_inds, batch_l_inds = np.meshgrid(
                p_lattice_inds, all_gt_lattice_inds, indexing="ij"
            )
            batch_p_inds, batch_l_inds = (
                batch_p_inds.flatten(),
                batch_l_inds.flatten(),
            )

            all_pairs_spl = lattice_graph.batch_shortest_path_length(
                batch_p_inds, batch_l_inds
            )
            all_pairs_spl = all_pairs_spl.reshape(
                len(p_lattice_inds), len(all_gt_lattice_inds)
            )

            p_spl = all_pairs_spl.min(axis=1)  # min over all the ground-truth dim

            # compute mean shortest path length
            # NOTE: some paths lengths could be np.inf indicating it's not possible to reach any
            # ground-truth lattice node from this proposal node
            if np.isinf(p_spl).all():
                mean_spl = np.inf
            else:
                mean_spl = np.mean(p_spl[np.isfinite(p_spl)])

            # compute portion of the proposal lattice nodes that are directly at a ground-truth lattice node
            portion_at_gt = np.sum(p_spl == 0) / len(p_spl)

            proposals["metrics"].append(
                {
                    "mean_spl": mean_spl,
                    "portion_at_gt": portion_at_gt,
                }
            )

        # print(proposals["metrics"])
    logger.info(f"finished computing precision metrics")

    # Evaluate recall metric over all labels
    logger.info(f"started computing recall metrics")
    label_gt_boxes_metrics = {}
    for label, gt_boxes_lattice_inds in label_lattice_inds.items():
        if label in params["label_params"]["blacklisted_labels"]:
            continue

        # get all lattice inds for all proposal boxes of this label
        all_p_lattice_inds = set()
        for inds in label_proposals[label]["lattice_inds"]:
            all_p_lattice_inds.update(inds)
        all_p_lattice_inds = np.array(list(all_p_lattice_inds))

        if len(all_p_lattice_inds) == 0:
            logger.debug(f"no proposal lattice inds for label {label}")
            label_gt_boxes_metrics[label] = len(gt_boxes_lattice_inds) * [
                {
                    "mean_spl": np.nan,
                }
            ]
            continue

        label_gt_boxes_metrics[label] = []
        for gt_lattice_inds in gt_boxes_lattice_inds:
            gt_lattice_inds = np.array(gt_lattice_inds)

            if len(gt_lattice_inds) == 0:
                logger.debug(
                    f"ground-truth box with no assigned lattice nodes in {label}, setting mean shortest path length to np.nan"
                )
                label_gt_boxes_metrics[label].append(
                    {
                        "mean_spl": np.nan,
                    }
                )
                continue

            # query shortest path for all pairs of label and proposal lattice inds
            batch_p_inds, batch_l_inds = np.meshgrid(
                all_p_lattice_inds, gt_lattice_inds, indexing="ij"
            )
            batch_p_inds, batch_l_inds = (
                batch_p_inds.flatten(),
                batch_l_inds.flatten(),
            )

            all_pairs_spl = lattice_graph.batch_shortest_path_length(
                batch_p_inds, batch_l_inds
            )
            all_pairs_spl = all_pairs_spl.reshape(
                len(all_p_lattice_inds), len(gt_lattice_inds)
            )

            p_spl = all_pairs_spl.min(axis=0)  # min over all the proposal dim

            # compute min shortest path length
            # NOTE: some paths lengths could be np.inf indicating it's not possible to reach any
            # ground-truth lattice node from this proposal node
            if np.isinf(p_spl).all():
                mean_spl = np.inf
            else:
                mean_spl = np.mean(p_spl[np.isfinite(p_spl)])

            label_gt_boxes_metrics[label].append(
                {
                    "mean_spl": mean_spl,
                }
            )

    # print(label_gt_boxes_metrics)

    logger.info(f"finished computing recall metrics")

    # this is a workaround since the box is an open3d type which isn't pickleable
    for label, proposals in label_proposals.items():
        boxes_corners = []
        for p_box in proposals["boxes"]:
            boxes_corners.append(get_box_corners(p_box))
        proposals["boxes_corners"] = boxes_corners
        del proposals["boxes"]

    # Save computed metrics
    save_filename = f"{scan_name}.pkl"
    save_path = os.path.join(output_dir, save_filename)
    out = {
        "scan_dir": scan_dir,
        "house_file_path": house_file_path,
        "ply_file_path": ply_file_path,
        "openscene_region_segs_dir": openscene_region_segs_dir,
        "lattice_graph_path": lattice_graph_path,
        "label_gt_boxes": label_gt_boxes,
        "label_lattice_inds": label_lattice_inds,
        "label_proposals": label_proposals,
        "label_gt_boxes_metrics": label_gt_boxes_metrics,
    }
    with open(save_path, "wb") as f:
        pickle.dump(out, f)
    logger.info(f"saved evaluation outputs to {save_path}")


def expand_box_to_min_side_length(box, min_side_length):
    """
    Expands the box if needed such that each side is at least min_side_length long.
    """
    max_bound, min_bound = box.get_max_bound(), box.get_min_bound()
    lengths = max_bound - min_bound
    expand_lengths = np.maximum(min_side_length - lengths, 0)
    expand_lengths /= 2
    max_bound += expand_lengths
    min_bound -= expand_lengths
    return o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    

if __name__ == '__main__':

    SCAN_DIR = "/media/rsl_admin/T7/matterport/data/v1/scans/2t7WUuJeko7/2t7WUuJeko7"
    OPENSCENE_REGION_SEGS_DIR = "/home/rsl_admin/openscene/comparison_outputs/segmentations/region-openseg-matterport-test"
    LATTICE_GRAPH_PATH = "/media/rsl_admin/T7/matterport/lattice_graphs/2t7WUuJeko7_lattice_graph.pkl"
    OUTPUT_DIR = "/home/rsl_admin/openscene/comparison_outputs/test_eval_outputs/region"

    params = {
        "label_params": {
            "blacklisted_labels": (
                "other room",

                # no appropriate tag
                "dining booth",
                "entryway/foyer/lobby",
                "outdoor",
            )
        },
        "min_proposal_box_length": 0.1,
        "dbscan_params": {
            "eps": 0.5,
            "min_points": 50,
            "print_progress": False,
        },
    }

    print(params)

    evaluate_matterport_scan_region_localizations(
        params=params,
        scan_dir=SCAN_DIR,
        openscene_region_segs_dir=OPENSCENE_REGION_SEGS_DIR,
        lattice_graph_path=LATTICE_GRAPH_PATH,
        output_dir=OUTPUT_DIR,
    )