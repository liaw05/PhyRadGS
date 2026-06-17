import os
import sys
import os.path as osp
import numpy as np

sys.path.append("./")
from model.gaussian_model import GaussianModel
from model.params import ModelParams
from utils.graphics_utils import fetchPly
from utils.system_utils import searchForMaxIteration


def initialize_gaussian(gaussians: GaussianModel, args: ModelParams, loaded_iter=None, spatial_lr_scale=1.0):
    if loaded_iter:
        if loaded_iter == -1:
            loaded_iter = searchForMaxIteration(
                osp.join(args.model_path, "point_cloud")
            )
        ply_path = os.path.join(
            args.model_path,
            "point_cloud",
            "iteration_" + str(loaded_iter),
            "point_cloud.pickle",  # Pickle rather than ply
        )
        assert osp.exists(ply_path), f"Cannot find {ply_path} for loading."
        gaussians.load_ply(ply_path)
        print("Loading trained model at iteration {}".format(loaded_iter))
        gaussians.spatial_lr_scale = spatial_lr_scale
    elif args.random_init:
        print("Using grid initialization")
        num_min = min(args.nVoxel[0], args.nVoxel[1], args.nVoxel[2])
        base_num = (args.n_points*(num_min**3)/(args.nVoxel[0]*args.nVoxel[1]*args.nVoxel[2]))**(1/3)
        nx = int(base_num*args.nVoxel[0]/num_min)
        ny = int(base_num*args.nVoxel[1]/num_min)
        nz = int(base_num*args.nVoxel[2]/num_min)
        pts_gridx = (np.linspace(0, 1, nx) - 0.5) * args.sVoxel[0] + args.offOrigin[0]
        pts_gridy = (np.linspace(0, 1, ny) - 0.5) * args.sVoxel[1] + args.offOrigin[1]
        pts_gridz = (np.linspace(0, 1, nz) - 0.5) * args.sVoxel[2] + args.offOrigin[2]
        xyz = np.meshgrid(pts_gridx, pts_gridy, pts_gridz)
        xyz = np.stack(xyz, axis=-1).reshape(-1, 3)
        density = np.ones((xyz.shape[0], 1)) * 0.5
        gaussians.spatial_lr_scale = spatial_lr_scale
        gaussians.create_from_pcd(xyz, density, spatial_lr_scale)
        print(f"Generating grid point cloud ({xyz.shape[0]})...")
    else:
        if args.ply_path == "":
            if osp.exists(osp.join(args.source_path, "meta_data.json")):
                ply_path = osp.join(
                    args.source_path, "init_" + osp.basename(args.source_path) + ".npy"
                )
            elif args.source_path.split(".")[-1] in ["pickle", "pkl"]:
                ply_path = osp.join(
                    osp.dirname(args.source_path),
                    "init_" + osp.basename(args.source_path).split(".")[0] + ".npy",
                )
            else:
                raise ValueError("Could not recognize scene type!")
        else:
            ply_path = args.ply_path

        assert osp.exists(
            ply_path
        ), f"Cannot find {ply_path} for initialization. Please specify a valid ply_path or generate point cloud with initialize_pcd.py."

        print(f"Initialize Gaussians with {osp.basename(ply_path)}")
        ply_type = ply_path.split(".")[-1]
        if ply_type == "npy":
            point_cloud = np.load(ply_path)
            xyz = point_cloud[:, :3]
            density = point_cloud[:, 3:4]
        elif ply_type == ".ply":
            point_cloud = fetchPly(ply_path)
            xyz = np.asarray(point_cloud.points)
            density = np.asarray(point_cloud.colors[:, :1])

        gaussians.create_from_pcd(xyz, density, spatial_lr_scale)

    return loaded_iter
