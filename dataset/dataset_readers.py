import os
import sys
from typing import NamedTuple
import numpy as np
import os.path as osp
import json
import torch
import pickle
import math
import cv2
import random
from scipy.spatial.transform import Rotation


mode_id = {
    "parallel": 0,
    "cone": 1,
}


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    angle: float
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mode: int
    scanner_cfg: dict


class SceneInfo(NamedTuple):
    train_cameras: list
    test_cameras: list
    vol: torch.tensor
    scanner_cfg: dict
    scene_scale: float


def readBlenderInfo(path, eval, args):
    """Read blender format CT data."""
    if args.real_data:
        if not args.source_geometry or not osp.isfile(args.source_geometry):
            raise ValueError(
                "--source_geometry must point to a valid geometry JSON file when --real_data is set.\n"
                f"  Got: {args.source_geometry}"
            )
        with open(args.source_geometry, "r") as handle:
            meta_data = json.load(handle)
        meta_data["scanner"] = {
            "nDetector": args.nDetector,
            "dDetector": args.dDetector,
            "sDetector": list(np.array(args.nDetector) * np.array(args.dDetector)),
            "offOrigin": args.offOrigin,
            "nVoxel": args.nVoxel,
            "dVoxel": args.dVoxel,
            "sVoxel": list(np.array(args.nVoxel) * np.array(args.dVoxel)),  
        }
        args.sVoxel = list(np.array(args.nVoxel) * np.array(args.dVoxel))
        cam_infos = readCTamerasGeometry(meta_data, path, eval, 1.0, args)
        vol_gt = None
        scene_scale = 1.0
    else:
        # Read meta data
        if args.recon_mode == "static":
            meta_data_path = osp.join(path, "meta_data.json")
            with open(meta_data_path, "r") as handle:
                meta_data = json.load(handle)
        else:
            meta_data_path = osp.join(path, "meta_data_motion_blur_rolling_shutter_fps18.json")
            with open(meta_data_path, "r") as handle:
                meta_data = json.load(handle)

        meta_data["vol"] = osp.join(path, meta_data["vol"])

        if not "dVoxel" in meta_data["scanner"]:
            meta_data["scanner"]["dVoxel"] = list(
                np.array(meta_data["scanner"]["sVoxel"])
                / np.array(meta_data["scanner"]["nVoxel"])
            )
        if not "dDetector" in meta_data["scanner"]:
            meta_data["scanner"]["dDetector"] = list(
                np.array(meta_data["scanner"]["sDetector"])
                / np.array(meta_data["scanner"]["nDetector"])
            )

        #! We will scale the scene so that the volume of interest is in [-1, 1]^3 cube.
        scene_scale = 2 / max(meta_data["scanner"]["sVoxel"])
        for key_to_scale in [
            "dVoxel",
            "sVoxel",
            "sDetector",
            "dDetector",
            "offOrigin",
            "offDetector",
            "DSD",
            "DSO",
        ]:
            meta_data["scanner"][key_to_scale] = (
                np.array(meta_data["scanner"][key_to_scale]) * scene_scale
            ).tolist()

        cam_infos = readCTameras(meta_data, path, eval, scene_scale, args)
        vol_gt = torch.from_numpy(np.load(meta_data["vol"])).float().cuda()

        # Set scan geometry on args for downstream use
        args.sVoxel = meta_data["scanner"]["sVoxel"]
        args.offOrigin = meta_data["scanner"]["offOrigin"]
        args.nVoxel = meta_data["scanner"]["nVoxel"]

    train_cam_infos = cam_infos["train"]
    test_cam_infos = cam_infos["test"]

    scene_info = SceneInfo(
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        scanner_cfg=meta_data["scanner"],
        vol=vol_gt,
        scene_scale=scene_scale,
    )
    return scene_info


def readCTamerasGeometry(meta_data, source_path, eval, scene_scale, args):
    """Read camera info from geometry file."""
    cam_cfg = meta_data["scanner"]
    # filenames
    fns = os.listdir(source_path)
    fns = [fn for fn in fns if '.tif' in fn or '.png' in fn]
    if '_' in fns[0]:
        try:
            fns.sort(key=lambda x: int(x.split('_')[0]))
        except:
            fns.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[1]))
    else:
        fns.sort(key=lambda x: int(os.path.splitext(x)[0]))
    
    # load image
    images = []
    for fn in fns:
        image_path = os.path.join(source_path, fn)
        image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
        image = np.array(image, dtype=np.float32)/40000.0
        image = cv2.medianBlur(image, 5)
        images.append(image)
    images = np.array(images)
    print(f"max value: {images.max()}, min value: {images.min()}")
    images = images / images.max()
    images = -np.log(images)
    images = images / images.max()

    index_list = list(range(len(fns)))
    if eval:
        splits = ["train", "test"]
        # random split
        random.shuffle(index_list)
        train_index = index_list[:int(len(index_list)*0.8)]
        test_index = index_list[int(len(index_list)*0.8):]
    else:
        splits = ["train"]
        train_index = index_list
        test_index = []
    
    def get_angle_step(geo, args):
        first_line_start = geo["first_row_start_angle"]
        first_line_end = geo["first_row_end_angle"]
        last_line_end = geo["last_row_end_angle"]
        last_line_start = geo["last_row_start_angle"]
        rolling_shutter_angle = last_line_end - first_line_end
        exposure_angle = first_line_end - first_line_start
        angle_step_rs = rolling_shutter_angle / args.num_rolling_shutter_frames
        angle_step_exposure = exposure_angle / args.num_exposure_frames
        return first_line_start, angle_step_rs, angle_step_exposure

    cam_infos = {"train": [], "test": []}
    view_geos = meta_data["views"]
    source_pos, detector_matrix = load_geometry_file(meta_data["views"], geo_idx=args.geo_idx, num_views=args.num_views)
    height, width = images[0].shape[:2]
    print(f"width: {width}, height: {height}")
    assert width == cam_cfg["nDetector"][1] and height == cam_cfg["nDetector"][0], f"width and height of image {fn} do not match the detector size"

    for idx in range(len(fns)):
        image = images[idx]
        if args.recon_mode == "motion":
            detector_center_xyz = meta_data["rolling_shutter_geometry"]["detector"]["center"]
            detector_radius_xy = meta_data["rolling_shutter_geometry"]["detector"]["radius"]
            source_center_xyz = meta_data["rolling_shutter_geometry"]["source"]["center"]
            source_radius_xy = meta_data["rolling_shutter_geometry"]["source"]["radius"]

            detector_geo = view_geos[idx]["rolling_shutter"]["detector"]
            detector_z = detector_geo["z"]
            detector_first_line_start, detector_angle_step_rs, detector_angle_step_exposure = get_angle_step(detector_geo, args)
            source_geo = view_geos[idx]["rolling_shutter"]["source"]
            source_z = source_geo["z"]
            source_first_line_start, source_angle_step_rs, source_angle_step_exposure = get_angle_step(source_geo, args)
                 
            assert not height%args.num_rolling_shutter_frames, "height must be divisible by num_rolling_shutter_frames"
            rs_height = height // args.num_rolling_shutter_frames
            c2w = detector_matrix[idx][:3,:3]
            for i in range(args.num_rolling_shutter_frames):
                for j in range(args.num_exposure_frames):
                    detector_angle = detector_first_line_start + detector_angle_step_rs*(i+0.5) + (j+0.5) * detector_angle_step_exposure
                    source_angle = source_first_line_start + source_angle_step_rs*(i+0.5) + (j+0.5) * source_angle_step_exposure
                    source_xyz = angle_to_xyz(source_angle, source_center_xyz, source_radius_xy, source_z)
                    detector_xyz = angle_to_xyz(detector_angle, detector_center_xyz, detector_radius_xy, detector_z)
                    source_T, FovY, FovX, source_in_detector = tansform_coordinate(c2w, detector_xyz, source_xyz, height, width, 
                        pixel_size=cam_cfg["dDetector"], rs_idx=i, rs_height=rs_height)
                    cam_info = CameraInfo(
                        uid=idx,
                        R=c2w,
                        T=source_T,
                        angle=detector_angle,
                        FovY=FovY,
                        FovX=FovX,
                        image=image,
                        image_path=os.path.join(source_path, fns[idx]),
                        image_name=os.path.splitext(fns[idx])[0],
                        width=width,
                        height=rs_height,
                        mode=1,
                        scanner_cfg=cam_cfg,
                    )
                    image = None
                    if idx in train_index:
                        cam_infos["train"].append(cam_info)
                    else:
                        cam_infos["test"].append(cam_info)
        else:
            c2w = detector_matrix[idx][:3,:3]
            detector_xyz = detector_matrix[idx][:3, 3]
            source_xyz = source_pos[idx]
            detector_angle = -1
            source_T, FovY, FovX, source_in_detector = tansform_coordinate(c2w, detector_xyz, source_xyz, height, width, 
                    pixel_size=cam_cfg["dDetector"], rs_idx=0, rs_height=height)
            cam_info = CameraInfo(
                        uid=idx,
                        R=c2w,
                        T=source_T,
                        angle=detector_angle,
                        FovY=FovY,
                        FovX=FovX,
                        image=image,
                        image_path=os.path.join(source_path, fns[idx]),
                        image_name=os.path.splitext(fns[idx])[0],
                        width=width,
                        height=height,
                        mode=1,
                        scanner_cfg=cam_cfg,
            )
            if idx in train_index:
                cam_infos["train"].append(cam_info)
            else:
                cam_infos["test"].append(cam_info)
    return cam_infos


def angle_to_xyz(angle_deg, center, radius, z):
    """
    将角度转换为三维坐标
    
    参数:
    - angle_deg: 角度(度）,以X轴正方向为0度,逆时针为正
    - center: 圆心坐标 [cx, cy] 或 [cx, cy, cz]
    - radius: 圆的半径(正值）
    - z: z坐标值
    
    返回:
    - [x, y, z] 三维坐标
    
    计算公式:
        x = center[0] + radius * cos(angle_deg * pi / 180)
        y = center[1] + radius * sin(angle_deg * pi / 180)
        z = z
    """
    angle_rad = angle_deg * np.pi / 180.0
    x = center[0] + radius * np.cos(angle_rad)
    y = center[1] + radius * np.sin(angle_rad)
    return [float(x), float(y), float(z)]


def calculate_angle_from_xy(x, y, center):
    dx = x - center[0]
    dy = y - center[1]
    angle = np.arctan2(dy, dx) * 180 / np.pi
    return angle


def load_geometry_file(views, geo_idx=0, num_views=128):
    """Load geometry file."""
    source_pos = np.array([view["source"] for view in views], dtype=np.float32)
    detector_matrix = np.array([view["detector"] for view in views], dtype=np.float32)

    if len(source_pos)%num_views:
        raise ValueError(f"num_views {num_views} should divide total views {len(source_pos)}")
    source_pos = source_pos.reshape(-1, num_views, 3)[geo_idx]
    detector_matrix = detector_matrix.reshape(-1, num_views, 4, 4)[geo_idx]
    print(f"Loaded idx {geo_idx} and {len(source_pos)} views of total {len(views)} views")
    return source_pos, detector_matrix


def tansform_coordinate(c2w, detector_pos, source_pos, height, width, pixel_size, rs_idx=0, rs_height=0):
    # get the world-to-camera transform and set R, T
    source_pos = np.array(source_pos, dtype=np.float32)
    detector_pos = np.array(detector_pos, dtype=np.float32)
    rot_world2detector = np.transpose(c2w)
    source_in_detector = np.dot(rot_world2detector, (source_pos - detector_pos))
    source_T = -np.dot(rot_world2detector, source_pos)

    fovy_max = np.arctan((-source_in_detector[1] + (-height/2 + (rs_idx+1)*rs_height) * pixel_size[1]) / -source_in_detector[2])  # in radians
    fovy_min = np.arctan((-source_in_detector[1] + (-height/2 + rs_idx*rs_height) * pixel_size[1]) / -source_in_detector[2])  # in radians
    FovY = [fovy_min, fovy_max]  # in radians
    fovx_max = np.arctan((-source_in_detector[0] + width * pixel_size[0]/2) / -source_in_detector[2])
    fovx_min = np.arctan((-source_in_detector[0] - width * pixel_size[0]/2) / -source_in_detector[2])
    FovX = [fovx_min, fovx_max]
    return source_T, FovY, FovX, source_in_detector


def readCTameras(meta_data, source_path, eval, scene_scale, args):
    """Read camera info."""
    cam_cfg = meta_data["scanner"]

    if eval:
        splits = ["train", "test"]
    else:
        splits = ["train"]

    def get_fov(cam_cfg):
        FovX = np.arctan2(cam_cfg["sDetector"][1] / 2, cam_cfg["DSD"]) * 2
        FovY = np.arctan2(cam_cfg["sDetector"][0] / 2, cam_cfg["DSD"]) * 2
        return FovX, FovY
    
    def get_transform(cam_cfg, frame_angle):
        c2w = angle2pose(cam_cfg["DSO"], frame_angle)  # c2w
        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        return R, T

    cam_infos = {"train": [], "test": []}
    for split in splits:
        if args.recon_mode != "static" and split == "train":
            prefix = "simulate_"
        else:
            prefix = "proj_"
        print(f"Reading {prefix + split} for {split} split")

        split_info = meta_data[prefix + split]
        n_split = len(split_info)
        if split == "test":
            uid_offset = len(meta_data[prefix + "train"])
        else:
            uid_offset = 0
        for i_split in range(n_split):
            sys.stdout.write("\r")
            sys.stdout.write(f"Reading camera {i_split + 1}/{n_split} for {split}")
            sys.stdout.flush()

            frame_info = split_info[i_split]
            frame_angle = frame_info["angle"]    
            image_path = osp.join(source_path, frame_info["file_path"])
            if not osp.exists(image_path):
                image_path = image_path.replace("_sim1", "")
            image = np.load(image_path) * scene_scale
            image = image.astype(np.float32)
            image = cv2.medianBlur(image, 3)
            mode = mode_id[cam_cfg["mode"]]
            FovX, FovY = get_fov(cam_cfg)
            width=cam_cfg["nDetector"][1]
            height=cam_cfg["nDetector"][0]

            if args.recon_mode == "motion" and split == "train":
                first_line_start = frame_info["first_line_start"]
                first_line_end = frame_info["first_line_end"]
                last_line_end = frame_info["last_line_end"]
                rolling_shutter_angle = last_line_end - first_line_end
                exposure_angle = first_line_end - first_line_start 
                angle_step_rs = rolling_shutter_angle / args.num_rolling_shutter_frames
                angle_step_exposure = exposure_angle / args.num_exposure_frames
                 
                angles = []
                FovXs = [] #rolling shutter FOV X
                FovYs = []
                assert not width%args.num_rolling_shutter_frames, "width must be divisible by num_rolling_shutter_frames"
                width = width//args.num_rolling_shutter_frames
                for i in range(args.num_rolling_shutter_frames):
                    x_min = -cam_cfg["sDetector"][1] / 2 + i*cam_cfg["sDetector"][1]/args.num_rolling_shutter_frames
                    x_max = -cam_cfg["sDetector"][1] / 2 + (i+1)*cam_cfg["sDetector"][1]/args.num_rolling_shutter_frames
                    fovx_min = np.arctan2(x_min, cam_cfg["DSD"])
                    fovx_max = np.arctan2(x_max, cam_cfg["DSD"])
                    for j in range(args.num_exposure_frames):
                        angles.append(first_line_start+angle_step_rs*(i+0.5) + (j+0.5) * angle_step_exposure)
                        FovXs.append([fovx_min, fovx_max])
                        FovYs.append([-FovY/2, FovY/2])
            else:
                angles = [frame_angle]
                FovXs = [[-FovX/2, FovX/2]]
                FovYs = [[-FovY/2, FovY/2]]

            for (frame_angle, FovX, FovY) in (zip(angles, FovXs, FovYs)):
                R, T = get_transform(cam_cfg, frame_angle)
                cam_info = CameraInfo(
                    uid=i_split + uid_offset,
                    R=R,
                    T=T,
                    angle=frame_angle,
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=osp.basename(image_path).split(".")[0],
                    width=width,
                    height=height,
                    mode=mode,
                    scanner_cfg=cam_cfg,
                )
                cam_infos[split].append(cam_info)
                image = None
        sys.stdout.write("\n")
    return cam_infos


def angle2pose(DSO, angle):
    """Transfer angle to pose (c2w) based on scanner geometry.
    1. rotate -90 degree around x-axis (fixed axis),
    2. rotate 90 degree around z-axis  (fixed axis),
    3. rotate angle degree around z axis  (fixed axis)"""

    phi1 = -np.pi / 2
    R1 = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(phi1), -np.sin(phi1)],
            [0.0, np.sin(phi1), np.cos(phi1)],
        ]
    )
    phi2 = np.pi / 2
    R2 = np.array(
        [
            [np.cos(phi2), -np.sin(phi2), 0.0],
            [np.sin(phi2), np.cos(phi2), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    R3 = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rot = np.dot(np.dot(R3, R2), R1)
    trans = np.array([DSO * np.cos(angle), DSO * np.sin(angle), 0])
    transform = np.eye(4)
    transform[:3, :3] = rot
    transform[:3, 3] = trans

    return transform


def readNAFInfo(path, eval):
    """Read blender format CT data."""
    # Read data
    with open(path, "rb") as f:
        data = pickle.load(f)
    # ! NAF scanner are measured in mm, but projections are measured in m. Therefore we need to / 1000.
    scanner_cfg = {
        "DSD": data["DSD"] / 1000,
        "DSO": data["DSO"] / 1000,
        "nVoxel": data["nVoxel"],
        "dVoxel": (np.array(data["dVoxel"]) / 1000).tolist(),
        "sVoxel": (np.array(data["nVoxel"]) * np.array(data["dVoxel"]) / 1000).tolist(),
        "nDetector": data["nDetector"],
        "dDetector": (np.array(data["dDetector"]) / 1000).tolist(),
        "sDetector": (
            np.array(data["nDetector"]) * np.array(data["dDetector"]) / 1000
        ).tolist(),
        "offOrigin": (np.array(data["offOrigin"]) / 1000).tolist(),
        "offDetector": (np.array(data["offDetector"]) / 1000).tolist(),
        "totalAngle": data["totalAngle"],
        "startAngle": data["startAngle"],
        "accuracy": data["accuracy"],
        "mode": data["mode"],
        "filter": None,
    }

    #! We will scale the scene so that the volume of interest is in [-1, 1]^3 cube.
    scene_scale = 2 / max(scanner_cfg["sVoxel"])
    for key_to_scale in [
        "dVoxel",
        "sVoxel",
        "sDetector",
        "dDetector",
        "offOrigin",
        "offDetector",
        "DSD",
        "DSO",
    ]:
        scanner_cfg[key_to_scale] = (
            np.array(scanner_cfg[key_to_scale]) * scene_scale
        ).tolist()

    # Generate camera infos
    if eval:
        splits = ["train", "test"]
    else:
        splits = ["train"]
    cam_infos = {"train": [], "test": []}
    for split in splits:
        if split == "test":
            uid_offset = data["numTrain"]
            n_split = data["numVal"]
        else:
            uid_offset = 0
            n_split = data["numTrain"]
        if split == "test" and "val" in data:
            data_split = data["val"]
        else:
            data_split = data[split]
        angles = data_split["angles"]
        projs = data_split["projections"]

        for i_split in range(n_split):
            sys.stdout.write("\r")
            sys.stdout.write(f"Reading camera {i_split + 1}/{n_split} for {split}")
            sys.stdout.flush()

            frame_angle = angles[i_split]
            c2w = angle2pose(scanner_cfg["DSO"], frame_angle)
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(
                w2c[:3, :3]
            )  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image = projs[i_split] * scene_scale

            # Note, dDetector is [v, u] not [u, v]
            FovX = np.arctan2(scanner_cfg["sDetector"][1] / 2, scanner_cfg["DSD"]) * 2
            FovY = np.arctan2(scanner_cfg["sDetector"][0] / 2, scanner_cfg["DSD"]) * 2

            mode = mode_id[scanner_cfg["mode"]]

            cam_info = CameraInfo(
                uid=i_split + uid_offset,
                R=R,
                T=T,
                angle=frame_angle,
                FovY=FovY,
                FovX=FovX,
                image=image,
                image_path=None,
                image_name=f"{i_split + uid_offset:04d}",
                width=scanner_cfg["nDetector"][1],
                height=scanner_cfg["nDetector"][0],
                mode=mode,
                scanner_cfg=scanner_cfg,
            )
            cam_infos[split].append(cam_info)
        sys.stdout.write("\n")

    # Store other data
    train_cam_infos = cam_infos["train"]
    test_cam_infos = cam_infos["test"]
    vol_gt = torch.from_numpy(data["image"]).float().cuda()
    scene_info = SceneInfo(
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        scanner_cfg=scanner_cfg,
        vol=vol_gt,
        scene_scale=scene_scale,
    )
    return scene_info


sceneLoadTypeCallbacks = {
    "Blender": readBlenderInfo,
    "NAF": readNAFInfo,
}
