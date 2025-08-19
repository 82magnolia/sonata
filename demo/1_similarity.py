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
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

try:
    import flash_attn
except ImportError:
    flash_attn = None


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
    parser.add_argument("--local_pcd_path", help="Path to local point cloud .ply or .txt file", type=str)
    parser.add_argument("--global_pcd_path", help="Path to global point cloud .ply or .txt file", type=str)

    # Visualization configs
    parser.add_argument("--vis_margin", help="Amount of margins to apply for reference scene during visualization", default=10., type=float)
    parser.add_argument("--vis_every", help="Point cloud sampling rate for visualization", default=500, type=int)
    parser.add_argument("--vis_indices", help="Point indices for visualization", nargs="+", type=int)
    parser.add_argument("--save_vis", help="Save point cloud used for visualization", action="store_true")

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

    if args.global_pcd_path.endswith("txt"):
        global_points = np.loadtxt(args.global_pcd_path)
        global_points = global_points[:, :3]
        if global_points.shape[-1] > 3:
            global_colors = global_points[:, 3:6]
        else:
            global_colors = np.zeros_like(global_points)

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(global_points)
        o3d_pcd.estimate_normals()

        global_data = {
            "coord": global_points,
            "color": global_colors,
            "normal": np.asarray(o3d_pcd.normals)
        }

        local_points = np.loadtxt(args.local_pcd_path)
        local_points = local_points[:, :3]
        if local_points.shape[-1] > 3:
            local_colors = local_points[:, 3:6]
        else:
            local_colors = np.zeros_like(local_points)

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(local_points)
        o3d_pcd.estimate_normals()

        local_data = {
            "coord": local_points,
            "color": local_colors,
            "normal": np.asarray(o3d_pcd.normals)
        }
    else:
        global_pcd = o3d.io.read_point_cloud(args.global_pcd_path)
        global_points = np.asarray(global_pcd.points)

        if global_pcd.colors is not None:
            global_colors = np.asarray(global_pcd.colors)
        else:
            global_colors = np.zeros_like(global_points)

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(global_points)
        o3d_pcd.estimate_normals()

        global_data = {
            "coord": global_points,
            "color": global_colors,
            "normal": np.asarray(o3d_pcd.normals)
        }

        local_pcd = o3d.io.read_point_cloud(args.local_pcd_path)
        local_points = np.asarray(local_pcd.points)

        if local_pcd.colors is not None:
            local_colors = np.asarray(local_pcd.colors)
        else:
            local_colors = np.zeros_like(local_points)

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(local_points)
        o3d_pcd.estimate_normals()

        local_data = {
            "coord": local_points,
            "color": local_colors,
            "normal": np.asarray(o3d_pcd.normals)
        }

    global_data = transform(global_data)
    local_data = transform(local_data)

    # model forward:
    with torch.inference_mode():
        for key in global_data.keys():
            if isinstance(global_data[key], torch.Tensor):
                global_data[key] = global_data[key].cuda(non_blocking=True)
        for key in local_data.keys():
            if isinstance(local_data[key], torch.Tensor):
                local_data[key] = local_data[key].cuda(non_blocking=True)
        global_point = model(global_data)
        local_point = model(local_data)
        # upcast point feature
        # Point is a structure contains all the information during forward
        for _ in range(2):
            assert "pooling_parent" in global_point.keys()
            assert "pooling_inverse" in global_point.keys()
            parent = global_point.pop("pooling_parent")
            inverse = global_point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, global_point.feat[inverse]], dim=-1)
            global_point = parent
        while "pooling_parent" in global_point.keys():
            assert "pooling_inverse" in global_point.keys()
            parent = global_point.pop("pooling_parent")
            inverse = global_point.pop("pooling_inverse")
            parent.feat = global_point.feat[inverse]
            global_point = parent
        for _ in range(2):
            assert "pooling_parent" in local_point.keys()
            assert "pooling_inverse" in local_point.keys()
            parent = local_point.pop("pooling_parent")
            inverse = local_point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, local_point.feat[inverse]], dim=-1)
            local_point = parent
        while "pooling_parent" in local_point.keys():
            assert "pooling_inverse" in local_point.keys()
            parent = local_point.pop("pooling_parent")
            inverse = local_point.pop("pooling_inverse")
            parent.feat = local_point.feat[inverse]
            local_point = parent

        # Get point orderings
        local_points = local_point.coord.cpu().numpy()
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
            target = F.normalize(local_point.feat, p=2, dim=-1)
            refer = F.normalize(global_point.feat, p=2, dim=-1)
            inner_self = target[select_index] @ target.t()
            inner_cross = target[select_index] @ refer.t()

            oral = 0.02
            highlight = 0.1
            reject = 0.5
            cmap = plt.get_cmap("Spectral_r")
            sorted_inner = torch.sort(inner_cross, descending=True)[0]
            oral = sorted_inner[0, int(global_point.offset[0] * oral)]
            highlight = sorted_inner[0, int(global_point.offset[0] * highlight)]
            reject = sorted_inner[0, -int(global_point.offset[0] * reject)]

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
                coord=[global_point.coord, local_point.coord + bias],
                color=[global_heat_color, local_heat_color],
                verbose=False,
            )
            pcds.append(
                get_line_set(
                    coord=torch.cat(
                        [
                            local_point.coord[select_index] + bias,
                            global_point.coord[matched_index.unsqueeze(0)],
                        ]
                    ),
                    line=np.array([[0, 1]]),
                    color=np.array([0, 0, 0]) / 255,
                    verbose=False,
                )
            )

            global_match_pcd = o3d.geometry.PointCloud()
            global_match_pcd.points = o3d.utility.Vector3dVector((global_point.coord[matched_index.unsqueeze(0)]).cpu().numpy())
            global_match_pcd.paint_uniform_color((1., 0., 1.))
            global_match_mesh = keypoints_to_spheres(global_match_pcd, radius=0.2)
            pcds.append(global_match_mesh)

            local_match_pcd = o3d.geometry.PointCloud()
            local_match_pcd.points = o3d.utility.Vector3dVector((local_point.coord[select_index] + bias).cpu().numpy())
            local_match_pcd.paint_uniform_color((1., 0., 1.))
            local_match_mesh = keypoints_to_spheres(local_match_pcd, radius=0.2)
            pcds.append(local_match_mesh)

            if repeat_index == 0:
                video = None  # Initialize video for logging
                visualizer = None  # Initialize visualizer for logging

            video, visualizer = stream_geometry(args.log_dir, ["render_img", "render_video"], pcds, pcds,
                                                save_prefix="feat_dist", repeat_idx=repeat_index, num_repeats=len(view_indices), video=video, visualizer=visualizer)

            if args.save_vis:
                o3d.io.write_point_cloud(os.path.join(args.log_dir, f"similarity_{view_index}_global.ply"), pcds[0])
                o3d.io.write_point_cloud(os.path.join(args.log_dir, f"similarity_{view_index}_local.ply"), pcds[1])
                o3d.io.write_line_set(os.path.join(args.log_dir, f"similarity_{view_index}_line.ply"), pcds[2])
                o3d.io.write_triangle_mesh(os.path.join(args.log_dir, f"similarity_{view_index}_match_global.ply"), pcds[3])
                o3d.io.write_triangle_mesh(os.path.join(args.log_dir, f"similarity_{view_index}_match_local.ply"), pcds[4])
