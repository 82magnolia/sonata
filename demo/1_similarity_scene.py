# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import sonata
import torch
import torch.nn.functional as F
import argparse
import os
import cv2
from scipy.spatial.transform import Rotation as R
import pandas as pd
import trimesh
from trimesh.sample import sample_surface
try:
    import flash_attn
except ImportError:
    flash_attn = None
import math
from typing import Union, Tuple, List
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from tqdm import trange
from matplotlib import colormaps


def get_color_wheel() -> torch.Tensor:
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    color_wheel = torch.zeros((RY + YG + GC + CB + BM + MR, 3), dtype=torch.float32)
    counter = 0
    color_wheel[0:RY, 0] = 255
    color_wheel[0:RY, 1] = torch.floor(255 * torch.arange(0, RY) / RY)
    counter += RY
    color_wheel[counter:counter + YG, 0] = 255 - torch.floor(255 * torch.arange(0, YG) / YG)
    color_wheel[counter:counter + YG, 1] = 255
    counter += YG
    color_wheel[counter:counter + GC, 1] = 255
    color_wheel[counter:counter + GC, 2] = torch.floor(255 * torch.arange(0, GC) / GC)
    counter += GC
    color_wheel[counter:counter + CB, 1] = 255 - torch.floor(255 * torch.arange(0, CB) / CB)
    color_wheel[counter:counter + CB, 2] = 255
    counter += CB
    color_wheel[counter:counter + BM, 2] = 255
    color_wheel[counter:counter + BM, 0] = torch.floor(255 * torch.arange(0, BM) / BM)
    counter += BM
    color_wheel[counter:counter + MR, 2] = 255 - torch.floor(255 * torch.arange(0, MR) / MR)
    color_wheel[counter:counter + MR, 0] = 255
    return color_wheel / 255


def map_coordinates_to_color(point_cloud, color_wheel):

    xyz_max = point_cloud.max(axis=0)
    xyz_min = point_cloud.min(axis=0)
    xyz_center = (xyz_max + xyz_min) / 2

    x = point_cloud[:, 0] - xyz_min[0]
    y = point_cloud[:, 1] - xyz_min[1]
    z = point_cloud[:, 2] - xyz_min[2]

    # Calculate distances to x+z=0
    dist = np.abs(x + z) / np.sqrt(2)
    # Normalization
    dist = (dist - dist.min()) / (dist.max() - dist.min()) * 0.9 * (color_wheel.shape[0] - 1)

    # Assign colors in the color wheel
    k0 = np.floor(dist).astype(int)
    k1 = (k0 + 1) % color_wheel.shape[0]
    f = dist - k0

    # Interpolate between colors in the color wheel
    colors = (1 - f[:, None]) * color_wheel[k0] + f[:, None] * color_wheel[k1]

    return colors


def choice_without_replacement(l: Union[List, np.array], n, return_idx=False):
    if isinstance(l, list):
        idx_list = np.random.permutation(len(l))[:n].tolist()
        choice_list = [l[idx] for idx in idx_list]

        if return_idx:
            return choice_list, idx_list
        else:
            return choice_list
    elif isinstance(l, np.ndarray):
        idx_arr = np.random.permutation(len(l))[:n]
        choice_arr = l[idx_arr]

        if return_idx:
            return choice_arr, idx_arr
        else:
            return choice_arr
    else:
        raise ValueError("Invalid input type")


def farthest_point_down_sample(points_np: np.array, n, return_idx=False, max_input_size=10000):
    if max_input_size != -1:  # First random sample to designated size
        in_points_np, in_idx_np = choice_without_replacement(points_np, max_input_size, return_idx=True)
    else:
        in_points_np, in_idx_np = points_np, np.arange(points_np.shape[0])

    pcd = o3d.geometry.PointCloud()
    in_idx_np = np.stack([in_idx_np.astype(float)] * 3, axis=-1)  # (N, 3)
    pcd.points = o3d.utility.Vector3dVector(in_points_np)
    pcd.colors = o3d.utility.Vector3dVector(in_idx_np)
    pcd = pcd.farthest_point_down_sample(n)

    if return_idx:
        idx_np = np.asarray(pcd.colors)[:, 0].astype(int).tolist()
        return np.asarray(pcd.points), idx_np
    else:
        return np.asarray(pcd.points)


def generate_yaw_points(num_rot: int, device='cpu'):
    yaw_arr = torch.arange(num_rot, dtype=torch.float, device=device)
    yaw_arr = yaw_arr * 2 * np.pi / num_rot

    return yaw_arr


def yaw2rot_mtx(yaw_arr: torch.Tensor, apply_xz_flip=False):
    # Initialize rotation matrices from yaw values
    def _yaw2mtx(yaw):
        # yaw is assumed to be a scalar
        yaw = yaw.reshape(1, )

        tensor_0 = torch.zeros(1, device=yaw.device)
        tensor_1 = torch.ones(1, device=yaw.device)

        R = torch.stack([
            torch.stack([torch.cos(yaw), tensor_0, -torch.sin(yaw)]),
            torch.stack([tensor_0, tensor_1, tensor_0]),
            torch.stack([torch.sin(yaw), tensor_0, torch.cos(yaw)])
        ]).reshape(3, 3)

        return R

    tot_mtx = []
    for yaw in yaw_arr:
        if apply_xz_flip:
            tot_mtx.append(_yaw2mtx(yaw))
            tot_mtx.append(_yaw2mtx(yaw) @ np.array([[-1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))  # X flip (Z flip is subsumed by X flip + rotation)
        else:
            tot_mtx.append(_yaw2mtx(yaw))
    return torch.stack(tot_mtx)


def check_in_polygon(poly_points: np.ndarray, query_points: np.ndarray):
    # poly_points are floorplan points arranged clockwise
    valid_poly_points = poly_points[poly_points.sum(-1) != np.inf]
    poly_points_sply = Polygon(valid_poly_points)

    in_polygon = []
    for point in query_points:
        point_sply = Point(point)
        in_polygon.append(poly_points_sply.contains(point_sply))
    in_polygon = np.array(in_polygon, dtype=bool)
    return in_polygon


def generate_uniform_query_points(num_query: Union[int, Tuple], qmin_x, qmax_x, qmin_y, qmax_y):
    if isinstance(num_query, tuple):
        qpoints_x = np.linspace(qmin_x, qmax_x, num_query[0])
        qpoints_y = np.linspace(qmin_y, qmax_y, num_query[1])
    else:
        num_query_per_axis = math.ceil(math.sqrt(num_query))
        qpoints_x = np.linspace(qmin_x, qmax_x, num_query_per_axis)
        qpoints_y = np.linspace(qmin_y, qmax_y, num_query_per_axis)
    qpoints_x, qpoints_y = np.meshgrid(qpoints_x, qpoints_y)
    qpoints = np.stack([qpoints_x, qpoints_y], axis=-1).reshape(-1, 2)
    return qpoints


def get_uniform_points_from_contour(scene_contour: np.array, num_points: int, point_height: float):
    # Set initial particles with fixed height uniformly covering scene contour
    scene_min_x = scene_contour[:, 0].min()
    scene_max_x = scene_contour[:, 0].max()
    scene_min_y = scene_contour[:, 2].min()
    scene_max_y = scene_contour[:, 2].max()
    init_num_query = 1000  # Used for determining area ratio between floorplan & bounding box
    init_query_points = generate_uniform_query_points(init_num_query, scene_min_x, scene_max_x, scene_min_y, scene_max_y)
    init_in_polygon = check_in_polygon(scene_contour[:scene_contour.shape[0] // 2, [0, 2]], init_query_points)
    area_ratio = init_in_polygon.sum() / init_in_polygon.shape[0]
    num_surplus = num_points / area_ratio  # Discount for area_ratio when generating query points
    surplus_query_points = generate_uniform_query_points(num_surplus, scene_min_x, scene_max_x, scene_min_y, scene_max_y)
    surplus_in_polygon = check_in_polygon(scene_contour[:scene_contour.shape[0] // 2, [0, 2]], surplus_query_points)
    query_points = surplus_query_points[surplus_in_polygon]
    vert_points = np.ones_like(query_points[:, 0:1]) * point_height
    query_points = np.concatenate([query_points[:, 0:1], vert_points, query_points[:, 1:2]], axis=-1)
    return query_points


def trimesh_load_with_postprocess(mesh_path, postprocess_type=None):
    tr_mesh = trimesh.load(mesh_path, force="mesh")
    if postprocess_type == 'bottom_crop':
        plane_origin = np.zeros([3, ])
        plane_origin[0] = tr_mesh.vertices.mean(0)[0]
        plane_origin[1] = tr_mesh.vertices.min(axis=0)[1]
        plane_origin[2] = tr_mesh.vertices.mean(0)[2]
        processed_tr_mesh = tr_mesh.slice_plane(plane_origin=plane_origin.tolist(), plane_normal=[0., 1., 0.])
    else:
        processed_tr_mesh = tr_mesh

    return processed_tr_mesh


def stream_geometry(log_dir, render_method_list, add_geometry_list, rm_geometry_list, vis_sample_idx=0, repeat_idx=0, num_repeats=1, close_window_at_end=False, save_prefix="default", video=None, visualizer=None):  # Helper function for visualizing each scene in batch
    # vis_sample_idx is the visualization index to use for saving rendered image / video
    # NOTE 1: Indices only matter for render_img and render_video modes
    # NOTE 2: add_geometry_list specifies geometry to be added for rendering, and rm_geometry_list states geometry to be removed after rendering

    if not os.path.exists(os.path.join(log_dir, f'{save_prefix}_rendering')):
        os.makedirs(os.path.join(log_dir, f'{save_prefix}_rendering'), exist_ok=True)

    # Initialize open3d visualizer for new scene at the first frame
    if repeat_idx == 0:
        visualizer = o3d.visualization.Visualizer()
        visualizer.create_window()
        for geometry in add_geometry_list:
            visualizer.add_geometry(geometry)
            visualizer.update_geometry(geometry)
    else:
        assert visualizer is not None
        for geometry in add_geometry_list:
            visualizer.add_geometry(geometry)
            visualizer.update_geometry(geometry)

    # Change to top-down view
    ctr = visualizer.get_view_control()
    rot = np.eye(4)
    rot[:3, :3] = R.from_euler('x', 80, degrees=True).as_matrix()
    cam = ctr.convert_to_pinhole_camera_parameters()
    cam.extrinsic = cam.extrinsic @ rot

    # Fix camera parameters to near-orthographic
    new_cam_extrinsic = np.copy(cam.extrinsic)
    new_cam_extrinsic[0, -1] = 0.  # Make camera centered around scene
    new_cam_extrinsic[1, -1] = 0.  # Make camera centered around scene
    new_cam_extrinsic[2, -1] += 300.
    cam.extrinsic = new_cam_extrinsic
    new_cam_intrinsic = np.copy(cam.intrinsic.intrinsic_matrix)
    new_cam_intrinsic[0, 0] *= 20.
    new_cam_intrinsic[1, 1] *= 20.
    cam.intrinsic.intrinsic_matrix = new_cam_intrinsic
    ctr.convert_from_pinhole_camera_parameters(cam, True)

    visualizer.poll_events()
    visualizer.update_renderer()

    if "render_img" in render_method_list:
        visualizer.capture_screen_image(os.path.join(log_dir, f'{save_prefix}_rendering', f'render_{vis_sample_idx}_repeat_{repeat_idx}.png'))
    if "render_img_top_down" in render_method_list:
        visualizer.capture_screen_image(os.path.join(log_dir, f'{save_prefix}_rendering', f'render_top_{vis_sample_idx}_repeat_{repeat_idx}.png'))
        ctr = visualizer.get_view_control()
        rot = np.eye(4)
        rot[:3, :3] = R.from_euler('x', -180, degrees=True).as_matrix()
        cam = ctr.convert_to_pinhole_camera_parameters()
        cam.extrinsic = cam.extrinsic @ rot
        ctr.convert_from_pinhole_camera_parameters(cam, True)
        ctr.set_zoom(0.4)

        visualizer.poll_events()
        visualizer.update_renderer()
        visualizer.capture_screen_image(os.path.join(log_dir, f'{save_prefix}_rendering', f'render_down_{vis_sample_idx}_repeat_{repeat_idx}.png'))
    if "render_video" in render_method_list:  # Render to video
        frame = visualizer.capture_screen_float_buffer()
        frame = (255 * np.asarray(frame)).astype(np.uint8)

        if repeat_idx == 0:
            video = cv2.VideoWriter(
                os.path.join(log_dir, f'{save_prefix}_rendering', f'render_{vis_sample_idx}' + '.mp4'),
                cv2.VideoWriter_fourcc(*'mp4v'),
                5.,
                (frame.shape[1], frame.shape[0]))

        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if repeat_idx == num_repeats - 1:
            video.release()

    for geometry in rm_geometry_list:
        visualizer.remove_geometry(geometry)

    repeat_idx += 1
    if close_window_at_end:
        visualizer.destroy_window()

    if "render_video" in render_method_list:
        return video, visualizer


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def get_point_cloud(coord, color=None, verbose=True):
    if not isinstance(coord, list):
        coord = [coord]
        if color is not None:
            color = [color]

    pcd_list = []
    for i in range(len(coord)):
        coord_ = to_numpy(coord[i])
        if color is not None:
            color_ = to_numpy(color[i])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord_)
        pcd.colors = o3d.utility.Vector3dVector(
            np.ones_like(coord_) if color is None else color_
        )
        pcd_list.append(pcd)
    if verbose:
        o3d.visualization.draw_geometries(pcd_list)
    return pcd_list


def get_line_set(coord, line, color=(1.0, 0.0, 0.0), verbose=True):
    coord = to_numpy(coord)
    line = to_numpy(line)
    colors = np.array([color for _ in range(len(line))])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(coord)
    line_set.lines = o3d.utility.Vector2iVector(line)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    if verbose:
        o3d.visualization.draw_geometries([line_set])
    return line_set


def keypoints_to_spheres(keypoints, paint_color=None, radius=0.015, alpha=1.0):
    spheres = o3d.geometry.TriangleMesh()
    for point_idx, alphapoint in enumerate(keypoints.points):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(alphapoint)
        if len(keypoints.colors) != 0:
            sphere_colors_np = keypoints.colors[point_idx][None, :].repeat(len(sphere.vertices), axis=0)
            sphere.vertex_colors = o3d.utility.Vector3dVector(sphere_colors_np)
        spheres += sphere
    if paint_color is not None:
        spheres.paint_uniform_color(paint_color)
    return spheres


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # General configs
    parser.add_argument("--log_dir", help="Log directory for saving experiment results", default="./log/")
    parser.add_argument("--seed", help="Seed value to use for reproducing experiments", default=0, type=int)
    parser.add_argument("--scene_root", help="Root directory containing scene data", required=True, type=str)
    parser.add_argument("--scene_pair_file", type=str, help="Path to .tsv file containing list of scenes with ground-truth matching objects", default=None)
    parser.add_argument("--num_floor_points", help="Number of floor points for extracting sonata features", default=0, type=int)
    parser.add_argument("--feat_sample_points", type=int, help="Number of points to sample per object model for feature extraction", default=2048)
    parser.add_argument("--normalize_feats", help="If True, normalize network outputs to unit norm", action="store_true")
    parser.add_argument("--rot_aug", help="Number of uniformly sampled y-axis (height-axis) rotations for feature averaging", default=1, type=int)
    parser.add_argument("--flip_aug", help="If True, apply flip augmentations and average them for feature extraction", action="store_true")
    parser.add_argument("--mutual_nn_sample_rate", help="Downsampling rate for mutual NN computation", default=0.01, type=float)

    # Visualization configs
    parser.add_argument("--vis_margin", help="Amount of margins to apply for reference scene during visualization", default=10., type=float)
    parser.add_argument("--vis_every", help="Point cloud sampling rate for visualization", default=500, type=int)
    parser.add_argument("--vis_indices", help="Point indices for visualization", nargs="+", type=int)
    parser.add_argument("--save_vis", help="Save point cloud used for visualization", action="store_true")
    parser.add_argument("--vis_gamma", help="Gamma value to use for visualization", default=1., type=float)
    parser.add_argument("--colorize_method", help="Method for visualizing feature distances", default="dist", type=str)
    parser.add_argument("--vis_mode", help="Visualization mode of features", default="feat_dist")

    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir, exist_ok=True)

    # set random seed
    sonata.utils.set_seed(args.seed)
    # Load model
    if flash_attn is not None:
        model = sonata.load("sonata", repo_id="facebook/sonata").cuda()
    else:
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],  # reduce patch size if necessary
            enable_flash=False,
        )
        model = sonata.load(
            "sonata", repo_id="facebook/sonata", custom_config=custom_config
        ).cuda()
    # Load default data transform pipeline
    transform = sonata.transform.default()

    # Iterate over scene point clouds
    scene_pair_table = pd.read_table(args.scene_pair_file)
    ref_scene_path_list = [os.path.join(args.scene_root, path, "scene.npz") for path in scene_pair_table.ref]
    tgt_scene_path_list = [os.path.join(args.scene_root, path, "scene.npz") for path in scene_pair_table.tgt]

    for scene_idx, (ref_scene_path, tgt_scene_path) in enumerate(zip(ref_scene_path_list, tgt_scene_path_list)):
        for scene_path in [ref_scene_path, tgt_scene_path]:
            scene_triplet = np.load(scene_path)
            if "arkit" in scene_path:
                scene_name = "pos"  # For manual pairs in ARKiT, use "pos" field as this contains real scans
            else:
                scene_name = "pair_pos"
            obj_trans = scene_triplet[scene_name + "_trans"]
            obj_rot = scene_triplet[scene_name + "_rot"]
            obj_scene_scales = scene_triplet[scene_name + "_obj_scene_scales"]

            # Load object mesh files and extract features (TODO: We need to write a separate floor point sampler for ARKiT dataset)
            curr_points_list = []
            curr_colors_list = []
            for inst_idx in trange(scene_triplet[scene_name + "_obj_id"].shape[0], desc=f"Build Scene ({scene_name} / {scene_idx})"):
                obj_id = scene_triplet[scene_name + '_obj_id'][inst_idx]
                obj_model_path = scene_triplet[scene_name + '_obj_path'][inst_idx]
                if "3d_front" in scene_path:
                    tr_mesh = trimesh_load_with_postprocess(
                        obj_model_path, postprocess_type='bottom_crop'
                    )
                elif "arkit" in scene_path:
                    tr_mesh = trimesh.load(obj_model_path)
                else:
                    raise NotImplementedError("Other scene datasets not supported")
                feat_sample_pcd_np, _, feat_sample_color_np = sample_surface(tr_mesh, args.feat_sample_points, sample_color=True)
                feat_sample_color_np = feat_sample_color_np[:, :-1]

                # Apply transforms
                feat_sample_pcd_np *= obj_scene_scales[inst_idx]
                feat_sample_pcd_np = feat_sample_pcd_np @ obj_rot[inst_idx].T + obj_trans[inst_idx]

                curr_points_list.append(feat_sample_pcd_np)
                curr_colors_list.append(feat_sample_color_np)

            # Optionally sample floorplan points
            if getattr(args, "num_floor_points", 0) > 0:
                fp_points = scene_triplet[scene_name + "_fp_points"]
                floor_points = get_uniform_points_from_contour(fp_points, args.num_floor_points, fp_points.min(0)[1])
                curr_points_list.append(floor_points)
                curr_colors_list.append(np.zeros_like(floor_points))

            curr_points = np.concatenate(curr_points_list, axis=0)
            curr_colors = np.concatenate(curr_colors_list, axis=0) / 255.

            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(curr_points)
            o3d_pcd.estimate_normals()

            if scene_path == ref_scene_path:  # Global point cloud
                global_data = {
                    "coord": curr_points,
                    "color": curr_colors,
                    "normal": np.asarray(o3d_pcd.normals)
                }
            else:  # Local point cloud
                local_data = {
                    "coord": curr_points,
                    "color": curr_colors,
                    "normal": np.asarray(o3d_pcd.normals)
                }

        # Obtain list of rotation matrices
        yaw_points = generate_yaw_points(args.rot_aug)
        rot_matrices = yaw2rot_mtx(yaw_points, apply_xz_flip=args.flip_aug).float().cpu().numpy()

        rot_local_data_list = [
            {
                "coord": (local_data["coord"] - local_data["coord"].mean(0, keepdims=True)) @ rot_mtx.T + local_data["coord"].mean(0, keepdims=True),
                "color": np.copy(local_data["color"]),
                "normal": local_data["normal"] @ rot_mtx.T
            } for rot_mtx in rot_matrices
        ]
        rot_global_data_list = [
            {
                "coord": (global_data["coord"] - global_data["coord"].mean(0, keepdims=True)) @ rot_mtx.T + global_data["coord"].mean(0, keepdims=True),
                "color": np.copy(global_data["color"]),
                "normal": global_data["normal"] @ rot_mtx.T
            } for rot_mtx in rot_matrices
        ]

        # model forward:
        rot_local_point_list = []
        rot_global_point_list = []
        with torch.inference_mode():
            for rot_local_data, rot_global_data in zip(rot_local_data_list, rot_global_data_list):
                rot_global_data = transform(rot_global_data)
                rot_local_data = transform(rot_local_data)

                for key in rot_global_data.keys():
                    if isinstance(rot_global_data[key], torch.Tensor):
                        rot_global_data[key] = rot_global_data[key].cuda(non_blocking=True)
                for key in rot_local_data.keys():
                    if isinstance(rot_local_data[key], torch.Tensor):
                        rot_local_data[key] = rot_local_data[key].cuda(non_blocking=True)
                rot_global_point = model(rot_global_data)
                rot_local_point = model(rot_local_data)
                # upcast point feature
                # Point is a structure contains all the information during forward
                for _ in range(2):
                    assert "pooling_parent" in rot_global_point.keys()
                    assert "pooling_inverse" in rot_global_point.keys()
                    parent = rot_global_point.pop("pooling_parent")
                    inverse = rot_global_point.pop("pooling_inverse")
                    parent.feat = torch.cat([parent.feat, rot_global_point.feat[inverse]], dim=-1)
                    rot_global_point = parent
                while "pooling_parent" in rot_global_point.keys():
                    assert "pooling_inverse" in rot_global_point.keys()
                    parent = rot_global_point.pop("pooling_parent")
                    inverse = rot_global_point.pop("pooling_inverse")
                    parent.feat = rot_global_point.feat[inverse]
                    rot_global_point = parent
                for _ in range(2):
                    assert "pooling_parent" in rot_local_point.keys()
                    assert "pooling_inverse" in rot_local_point.keys()
                    parent = rot_local_point.pop("pooling_parent")
                    inverse = rot_local_point.pop("pooling_inverse")
                    parent.feat = torch.cat([parent.feat, rot_local_point.feat[inverse]], dim=-1)
                    rot_local_point = parent
                while "pooling_parent" in rot_local_point.keys():
                    assert "pooling_inverse" in rot_local_point.keys()
                    parent = rot_local_point.pop("pooling_parent")
                    inverse = rot_local_point.pop("pooling_inverse")
                    parent.feat = rot_local_point.feat[inverse]
                    rot_local_point = parent
                rot_local_point_list.append(rot_local_point)
                rot_global_point_list.append(rot_global_point)

            # Aggregate augmented results
            local_point_coord = rot_local_point_list[0].coord[rot_local_point_list[0].inverse]
            global_point_coord = rot_global_point_list[0].coord[rot_global_point_list[0].inverse]
            local_point_color = local_data["color"]
            global_point_color = global_data["color"]
            local_point_offset = local_point_coord.shape[0]
            global_point_offset = global_point_coord.shape[0]

            full_local_point_feat = torch.stack([rot_local_point.feat[rot_local_point.inverse] for rot_local_point in rot_local_point_list], dim=0)
            full_global_point_feat = torch.stack([rot_global_point.feat[rot_global_point.inverse] for rot_global_point in rot_global_point_list], dim=0)

            local_point_feat = full_local_point_feat.mean(0)
            global_point_feat = full_global_point_feat.mean(0)

            if args.vis_mode == "feat_dist":
                # Get point orderings
                local_points = local_point_coord.cpu().numpy()
                sort_dt = np.dtype([('x', local_points.dtype), ('y', local_points.dtype), ('z', local_points.dtype)])
                local_points_for_sort = np.zeros(local_points.shape[0], dtype=sort_dt)
                local_points_for_sort['x'] = local_points[:, 0]
                local_points_for_sort['y'] = local_points[:, 1]
                local_points_for_sort['z'] = local_points[:, 2]
                view_indices = np.argsort(local_points_for_sort, order=['x', 'z', 'y'])

                if args.vis_indices is not None:
                    view_indices = view_indices[args.vis_indices]
                else:
                    view_indices = view_indices[::args.vis_every]

                prev_pcds = []

                for repeat_index, view_index in enumerate(view_indices):
                    select_index = [[view_index]]

                    if args.normalize_feats:
                        target = F.normalize(local_point_feat, p=2, dim=-1)
                        refer = F.normalize(global_point_feat, p=2, dim=-1)
                    else:
                        target = local_point_feat.clone().detach()
                        refer = global_point_feat.clone().detach()

                    if args.colorize_method == "dist":
                        dist_self = (target[select_index] - target).norm(dim=-1).reshape(-1)  # (N_unif, )
                        dist_cross = (target[select_index] - refer).norm(dim=-1).reshape(-1)
                        max_norm_dist_self = (dist_self - dist_self.min()) / (dist_self.max() - dist_self.min() + 1e-5)
                        max_norm_dist_self = max_norm_dist_self ** args.vis_gamma
                        max_norm_dist_self = max_norm_dist_self.cpu().numpy()
                        local_heat_color = colormaps['jet'](max_norm_dist_self, alpha=False, bytes=False)[:, :3]

                        max_norm_dist_cross = (dist_cross - dist_cross.min()) / (dist_cross.max() - dist_cross.min() + 1e-5)
                        max_norm_dist_cross = max_norm_dist_cross ** args.vis_gamma
                        max_norm_dist_cross = max_norm_dist_cross.cpu().numpy()
                        global_heat_color = colormaps['jet'](max_norm_dist_cross, alpha=False, bytes=False)[:, :3]

                        matched_index = torch.argmin(dist_cross)
                    else:  # Sigmoid-based visualization as in Sonata
                        inner_self = target[select_index] @ target.t()
                        inner_cross = target[select_index] @ refer.t()

                        oral = 0.02
                        highlight = 0.1
                        reject = 0.5
                        cmap = plt.get_cmap("Spectral_r")
                        sorted_inner = torch.sort(inner_cross, descending=True)[0]
                        oral = sorted_inner[0, int(global_point_offset * oral)]
                        highlight = sorted_inner[0, int(global_point_offset * highlight)]
                        reject = sorted_inner[0, -int(global_point_offset * reject)]

                        inner_self = inner_self - highlight
                        inner_self[inner_self > 0] = F.sigmoid(
                            inner_self[inner_self > 0] / (oral - highlight)
                        )
                        inner_self[inner_self < 0] = (
                            F.sigmoid(inner_self[inner_self < 0] / (highlight - reject)) * 0.9
                        )

                        inner_cross = inner_cross - highlight
                        inner_cross[inner_cross > 0] = F.sigmoid(
                            inner_cross[inner_cross > 0] / (oral - highlight)
                        )
                        inner_cross[inner_cross < 0] = (
                            F.sigmoid(inner_cross[inner_cross < 0] / (highlight - reject)) * 0.9
                        )

                        matched_index = torch.argmax(inner_cross)

                        local_heat_color = cmap(inner_self.squeeze(0).cpu().numpy())[:, :3]
                        global_heat_color = cmap(inner_cross.squeeze(0).cpu().numpy())[:, :3]

                    # shift local view from global view
                    bias = torch.tensor([[-args.vis_margin, 0., 0.]]).cuda()  # original bias in our paper
                    pcds = get_point_cloud(
                        coord=[global_point_coord, local_point_coord + bias],
                        color=[global_heat_color, local_heat_color],
                        verbose=False,
                    )
                    pcds.append(
                        get_line_set(
                            coord=torch.cat(
                                [
                                    local_point_coord[select_index] + bias,
                                    global_point_coord[matched_index.unsqueeze(0)],
                                ]
                            ),
                            line=np.array([[0, 1]]),
                            color=np.array([0, 0, 0]) / 255,
                            verbose=False,
                        )
                    )

                    global_match_pcd = o3d.geometry.PointCloud()
                    global_match_pcd.points = o3d.utility.Vector3dVector((global_point_coord[matched_index.unsqueeze(0)]).cpu().numpy())
                    global_match_pcd.paint_uniform_color((1., 0., 1.))
                    global_match_mesh = keypoints_to_spheres(global_match_pcd, radius=0.2)
                    pcds.append(global_match_mesh)

                    local_match_pcd = o3d.geometry.PointCloud()
                    local_match_pcd.points = o3d.utility.Vector3dVector((local_point_coord[select_index] + bias).cpu().numpy())
                    local_match_pcd.paint_uniform_color((1., 0., 1.))
                    local_match_mesh = keypoints_to_spheres(local_match_pcd, radius=0.2)
                    pcds.append(local_match_mesh)

                    if repeat_index == 0:
                        video = None  # Initialize video for logging
                        visualizer = None  # Initialize visualizer for logging

                    video, visualizer = stream_geometry(args.log_dir, ["render_img", "render_video"], pcds, pcds,
                                                        save_prefix="feat_dist", vis_sample_idx=scene_idx, repeat_idx=repeat_index, num_repeats=len(view_indices), video=video, visualizer=visualizer)

                    if args.save_vis:
                        o3d.io.write_point_cloud(os.path.join(args.log_dir, f"similarity_{scene_idx}_{view_index}_global.ply"), pcds[0])
                        o3d.io.write_point_cloud(os.path.join(args.log_dir, f"similarity_{scene_idx}_{view_index}_local.ply"), pcds[1])
                        o3d.io.write_line_set(os.path.join(args.log_dir, f"similarity_{scene_idx}_{view_index}_line.ply"), pcds[2])
                        o3d.io.write_triangle_mesh(os.path.join(args.log_dir, f"similarity_{scene_idx}_{view_index}_match_global.ply"), pcds[3])
                        o3d.io.write_triangle_mesh(os.path.join(args.log_dir, f"similarity_{scene_idx}_{view_index}_match_local.ply"), pcds[4])
            elif args.vis_mode == "mutual_nn":
                local_points = local_point_coord.cpu().numpy()
                global_points = global_point_coord.cpu().numpy()
                local_feats = local_point_feat.cpu().numpy()
                global_feats = global_point_feat.cpu().numpy()

                local_sample_points, local_sample_idx = farthest_point_down_sample(local_points, int(local_points.shape[0] * args.mutual_nn_sample_rate), return_idx=True)
                global_sample_points, global_sample_idx = farthest_point_down_sample(global_points, int(global_points.shape[0] * args.mutual_nn_sample_rate), return_idx=True)
                local_sample_feats = local_feats[local_sample_idx]
                global_sample_feats = global_feats[global_sample_idx]
                local_sample_feats = torch.from_numpy(local_sample_feats).cuda()
                global_sample_feats = torch.from_numpy(global_sample_feats).cuda()

                full_dist_mtx = (local_sample_feats[:, None] - global_sample_feats[None, :]).norm(dim=-1).cpu().numpy()  # (N_local, N_global)

                # Mutual NN assignment (https://gist.github.com/mihaidusmanu/20fd0904b2102acc1330bad9b4badab8)
                match_local_to_global = full_dist_mtx.argmin(-1)  # (N_local)
                match_global_to_local = full_dist_mtx.argmin(0)  # (N_global)

                range_local = np.arange(local_sample_points.shape[0])
                valid_matches = match_global_to_local[match_local_to_global] == range_local

                match_local_idx = range_local[valid_matches]
                match_global_idx = match_local_to_global[valid_matches]

                match_local_points = local_sample_points[match_local_idx]
                match_global_points = global_sample_points[match_global_idx]

                bias = np.array([[-args.vis_margin, 0., 0.]])  # original bias in our paper
                pcds = get_point_cloud(
                    coord=[global_point_coord, local_point_coord.cpu().numpy() + bias],
                    color=[global_point_color, local_point_color],
                    verbose=False,
                )

                color_wheel = get_color_wheel().numpy()
                idx_color = map_coordinates_to_color(np.array(match_local_points), color_wheel)

                global_match_pcd = o3d.geometry.PointCloud()
                global_match_pcd.points = o3d.utility.Vector3dVector(match_global_points)
                global_match_pcd.colors = o3d.utility.Vector3dVector(idx_color)
                global_match_mesh = keypoints_to_spheres(global_match_pcd, radius=0.2)
                pcds.append(global_match_mesh)

                local_match_pcd = o3d.geometry.PointCloud()
                local_match_pcd.points = o3d.utility.Vector3dVector(match_local_points + bias)
                local_match_pcd.colors = o3d.utility.Vector3dVector(idx_color)
                local_match_mesh = keypoints_to_spheres(local_match_pcd, radius=0.2)
                pcds.append(local_match_mesh)

                o3d.visualization.draw_geometries(pcds)

            else:
                raise NotImplementedError("Other visualization modes not supported")
