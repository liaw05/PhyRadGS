import os
import sys
import os.path as osp
from argparse import ArgumentParser, Namespace

sys.path.append("./")
from utils.argument_utils import ParamGroup


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._model_path = ""
        self.data_device = "cuda"
        self.ply_path = ""  # Path to initialization point cloud (if None, we will try to find `init_*.npy`.)
        self.scale_min = 0.0005  # percent of volume size
        self.scale_max = 0.5  # percent of volume size
        self.eval = False
        self.recon_mode = "motion" # "motion", "motion_equivalent" or "static"
        self.rs_axis = "x" # "x" or "y"
        self.num_rolling_shutter_frames = 1
        self.num_exposure_frames = 1
        self.densify_grad_mode = "grad_2D" # "grad_2D" or "grad_3D"
        self.source_geometry = ""
        self.recon_type = "xray" # "xray"
        self.real_data = False
        # real phantom data param
        self.nDetector=[2104, 2104]
        self.dDetector=[0.0495, 0.0495]
        self.offOrigin = [0, 0, -0.5]
        self.nVoxel = [960, 960, 200]
        self.dVoxel = [0.015, 0.015, 0.015]
        self.chunk_size = [256, 256, 256]
        self.geo_idx = 0
        self.num_views = 32
        self.random_init = False
        self.n_points = 50000
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = osp.abspath(g.source_path)
        g.source_geometry = osp.abspath(g.source_geometry)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.compute_cov3D_python = False
        self.debug = False
        self.renderer = "gsplat-xray"  # Options: "r2_gaussian" or "gsplat-xray"
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 20_000
        self.position_lr_init = 0.0002
        self.position_lr_final = 0.00002
        self.density_lr_init = 0.01
        self.density_lr_final = 0.001
        self.scaling_lr_init = 0.005
        self.scaling_lr_final = 0.0005
        self.rotation_lr_init = 0.001
        self.rotation_lr_final = 0.0001
        self.density_scale_lr_init = 0.0005
        self.density_scale_lr_final = 0.00005
        self.lambda_dssim = 0.25 # 0.25, 0.05
        self.lambda_tv = 0.05 #0.05
        self.lambda_sparse = 0.01 # 0.001
        self.lambda_scale = 0.05 # 0.05
        self.tv_vol_size = 32
        self.density_min_threshold = 0.00001
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densify_grad_threshold = 5.0e-5 # 5.0e-5,2.0e-5
        self.densify_scale_threshold = 0.1  # percent of volume size
        self.max_screen_size = 1.0  # percent of screen size
        self.max_scale = 1.0  # percent of volume size
        self.max_num_gaussians = 500_000
        super().__init__(parser, "Optimization Parameters")