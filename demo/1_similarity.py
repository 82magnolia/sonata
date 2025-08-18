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

try:
    import flash_attn
except ImportError:
    flash_attn = None


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

        # select_index = [[8706]]  # sofa arm
        # TODO: Below is for testing and making intermediate results (re-organize later)
        # select_index = [[8706]]  # sofa arm
        # select_index = [[1000]]  # second sofa arm
        # select_index = [[2000]]  # sofa arm
        select_index = [[4000]]  # sofa arm
        # select_index = [[20000]]  # cabinet
        # select_index = [[10000]]  # chair leg
        # select_index = [[30000]]

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

        o3d.visualization.draw_geometries(pcds)

        o3d.io.write_point_cloud(os.path.join(args.log_dir, "similarity_global.ply"), pcds[0])
        o3d.io.write_point_cloud(os.path.join(args.log_dir, "similarity_local.ply"), pcds[1])
        o3d.io.write_line_set(os.path.join(args.log_dir, "similarity_line.ply"), pcds[2])
        o3d.io.write_triangle_mesh(os.path.join(args.log_dir, "similarity_match_global.ply"), pcds[3])
        o3d.io.write_triangle_mesh(os.path.join(args.log_dir, "similarity_match_local.ply"), pcds[4])
