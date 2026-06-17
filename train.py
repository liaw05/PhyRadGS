import os
import os.path as osp
import torch
from random import randint
import sys
import time
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml

sys.path.append("./")
from model.params import ModelParams, OptimizationParams, PipelineParams
from model import GaussianModel, initialize_gaussian
from algorithms import get_renderer
from algorithms.r2_gaussian.voxelizer import query
from algorithms.gsplat_xray import GSPLAT_AVAILABLE
from utils.general_utils import safe_state
from utils.cfg_utils import load_config
from utils.log_utils import prepare_output_and_logger
from dataset import Scene
from utils.loss_utils import l1_loss, ssim, tv_3d_loss, weighted_tv_3d_loss, hybrid_weighted_tv_3d_loss
from utils.image_utils import metric_vol, metric_proj
from utils.plot_utils import show_two_slice
from utils.write_nifti import write_nifti
from utils.motion_utils import merge_projection


def to_scalar(x):
    return x.item() if torch.is_tensor(x) else float(x)


def chunked_query(gaussians, center, nVoxel, sVoxel, pipe, chunk_size=128):
    """
    Perform voxelization in chunks to reduce memory usage.
    
    Args:
        gaussians: Gaussian model
        center: Volume center
        nVoxel: Number of voxels [nx, ny, nz]
        sVoxel: Voxel size [sx, sy, sz]
        pipe: Pipeline parameters
        chunk_size: Size of each chunk (default: 128)
    
    Returns:
        Combined volume from all chunks
    """
    nx, ny, nz = nVoxel
    sx, sy, sz = sVoxel
    cx, cy, cz = center
    if isinstance(chunk_size, int):
        chunk_size = [chunk_size, chunk_size, chunk_size]
    
    # Calculate chunk dimensions
    n_chunks_x = (nx + chunk_size[0] - 1) // chunk_size[0]
    n_chunks_y = (ny + chunk_size[1] - 1) // chunk_size[1]
    n_chunks_z = (nz + chunk_size[2] - 1) // chunk_size[2]
    
    # Initialize output volume
    vol_pred = torch.zeros((nx, ny, nz), dtype=torch.float32, device="cuda")
    print(f"Processing volume {nx}x{ny}x{nz} in {n_chunks_x}x{n_chunks_y}x{n_chunks_z} chunks")
    
    for iz in range(n_chunks_z):
        for iy in range(n_chunks_y):
            for ix in range(n_chunks_x):
                # Calculate chunk boundaries
                start_x = ix * chunk_size[0]
                end_x = min((ix + 1) * chunk_size[0], nx)
                start_y = iy * chunk_size[1]
                end_y = min((iy + 1) * chunk_size[1], ny)
                start_z = iz * chunk_size[2]
                end_z = min((iz + 1) * chunk_size[2], nz)
                
                # Calculate chunk dimensions
                chunk_nx = end_x - start_x
                chunk_ny = end_y - start_y
                chunk_nz = end_z - start_z
                
                # Calculate chunk center and size
                chunk_center_x = cx + ((start_x+end_x-1)/2 - (nx-1)/2) * sx / nx
                chunk_center_y = cy + ((start_y+end_y-1)/2 - (ny-1)/2) * sy / ny
                chunk_center_z = cz + ((start_z+end_z-1)/2 - (nz-1)/2) * sz / nz
                
                chunk_center = [chunk_center_x, chunk_center_y, chunk_center_z]
                chunk_nVoxel = [chunk_nx, chunk_ny, chunk_nz]
                chunk_sVoxel = [sx * chunk_nx / nx, sy * chunk_ny / ny, sz * chunk_nz / nz]
                
                # Process this chunk
                chunk_result = query(
                    gaussians,
                    chunk_center,
                    chunk_nVoxel,
                    chunk_sVoxel,
                    pipe,
                )
                
                # Place chunk result in the correct position
                vol_chunk = chunk_result["vol"]
                vol_pred[start_x:end_x, start_y:end_y, start_z:end_z] += vol_chunk

    return {"vol": vol_pred}
    

def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    save_proj_images=True,
    num_proj_to_save=5,
    args = None,
):
    first_iter = 0
    # Set up renderer based on pipeline configuration
    renderer_type = getattr(pipe, 'renderer', 'r2_gaussian')
    if renderer_type == "gsplat-xray" and not GSPLAT_AVAILABLE:
        print("Warning: gsplat not available, falling back to r2_gaussian renderer")
        renderer_type = "r2_gaussian"
    dataset.use_gsplat = renderer_type == "gsplat-xray"
    if dataset.recon_mode != "motion":
        dataset.num_rolling_shutter_frames = 1
        dataset.num_exposure_frames = 1

    # Set up dataset
    scene = Scene(dataset, shuffle=False)

    render_func = get_renderer(renderer_type)
    print(f"Using renderer: {renderer_type}")

    # Set up some parameters
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    print(f"max_scale: {max_scale}")
    screen_size = max(scanner_cfg["nDetector"])
    max_screen_size = opt.max_screen_size * screen_size if opt.max_screen_size else None
    print(f"max_screen_size: {max_screen_size}")
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world

    def queryfunc_chunked(x):
        return chunked_query(
                x,
                np.array(dataset.offOrigin),
                [dataset.nVoxel[0], dataset.nVoxel[1], dataset.nVoxel[2]],
                [dataset.sVoxel[0], dataset.sVoxel[1], dataset.sVoxel[2]],
                pipe,
                chunk_size=dataset.chunk_size
            )
    queryfunc = queryfunc_chunked

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound, args.opacity_activation)
    initialize_gaussian(gaussians, dataset, None, args.spatial_lr_scale)
    scene.gaussians = gaussians
    gaussians.training_setup(opt)
    if checkpoint is not None:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Load checkpoint {osp.basename(checkpoint)}.")

    # Set up loss
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, opt.iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1
    training_start_time = time.time()  # Record training start time
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # Get one camera for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_stack = list(viewpoint_stack.values())
        viewpoint_cams = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        if dataset.use_gsplat:
            render_pkg = render_func(viewpoint_cams, gaussians, pipe, dataset)
            render_images, viewspace_point_tensor, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
        else:
            render_images = []
            visibility_filters = []
            radiis = []
            viewspace_point_tensor = []
            for viewpoint_cam in viewpoint_cams:
                # Render X-ray projection
                render_pkg = render_func(viewpoint_cam, gaussians, pipe)
                image, viewspace_point_tensor_i, visibility_filter, radii = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                )
                render_images.append(image)
                visibility_filters.append(visibility_filter)
                radiis.append(radii)
                viewspace_point_tensor.append(viewspace_point_tensor_i)
            render_images = torch.stack(render_images, dim=0)
            # merge visibility filters and radii
            visibility_filter = torch.stack(visibility_filters, dim=0).any(dim=0)
            radii = torch.stack(radiis, dim=0).max(dim=0)[0]

        # Compute loss
        loss = {"total": 0.0, "render": 0.0, "dssim": 0.0, "tv": 0.0}
        if dataset.recon_mode == "motion":
            gt_image = viewpoint_cams[0].original_image.cuda().unsqueeze(0)
            render_images = merge_projection(render_images, dataset.num_exposure_frames, dataset.num_rolling_shutter_frames, dataset.rs_axis)
        else:
            gt_image = [viewpoint_cam.original_image.cuda() for viewpoint_cam in viewpoint_cams]
            gt_image = torch.stack(gt_image, dim=0)
        
        render_loss = l1_loss(render_images, gt_image)
        loss["render"] = render_loss
        loss["total"] += loss["render"]
        if opt.lambda_dssim > 0:
            for bi in range(render_images.shape[0]):
                loss_dssim = 1.0 - ssim(render_images[bi], gt_image[bi])
                loss["dssim"] += opt.lambda_dssim * loss_dssim
        loss["total"] += loss["dssim"]

        # 3D TV loss
        if use_tv:
            # Randomly get the tiny volume center
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            vol_pred = query(
                gaussians,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                pipe,
            )["vol"]
            if args.tv_delta is None:
                loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            else:
                loss_tv = hybrid_weighted_tv_3d_loss(vol_pred, delta=args.tv_delta, reduction="mean")
            loss["tv"] = opt.lambda_tv * loss_tv
            loss["total"] += loss["tv"]
        
        # Scale正则化损失：惩罚针状/过扁的高斯，可微分，使训练更平稳
        if iteration > opt.densify_from_iter and args.add_scale_regularization:  
            loss_scale_ratio = gaussians.scale_regularization_loss(
                target_ratio=10.0, mode="soft"
            )
            loss_scale_min = gaussians.scale_min_regularization(
                min_scale=0.004
            )
            
            loss["scale_reg"] = opt.lambda_scale*(loss_scale_ratio + loss_scale_min)
            loss["total"] += loss["scale_reg"]
        
        if iteration > opt.densify_from_iter and args.add_sparse_regularization:
            loss_sparse = opt.lambda_sparse * torch.mean(torch.abs(gaussians.get_density))
            loss["sparse"] = loss_sparse
            loss["total"] += loss["sparse"]

        loss["total"].backward()

        iter_end.record()
        torch.cuda.synchronize()

        with torch.no_grad():
            # Adaptive control
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            # Handle densification stats
            if dataset.densify_grad_mode == "grad_3D":
                gaussians.add_densification_stats(gaussians.get_xyz.grad, visibility_filter)
            else:
                if renderer_type == "gsplat-xray" and "meta" in render_pkg:
                    # For gsplat, use meta["means2d"].grad (retain_grad was called on it)
                    meta = render_pkg["meta"]
                    means2d_grad = meta["means2d"].grad
                    if means2d_grad.dim() == 3:
                        means2d_grad = means2d_grad.sum(dim=0)/dataset.num_exposure_frames  # [N, 2]
                    gaussians.add_densification_stats(means2d_grad, visibility_filter)
                elif renderer_type == "r2_gaussian_mr" or renderer_type == "r2_gaussian":
                    # For r2_gaussian, use standard .grad
                    means2d_grad = 0
                    for i in range(len(viewspace_point_tensor)):
                        means2d_grad += viewspace_point_tensor[i].grad/dataset.num_exposure_frames  # [N, 2]
                    gaussians.add_densification_stats(means2d_grad, visibility_filter)

            if iteration < opt.densify_until_iter:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                        renderer_type,
                    )
                    if renderer_type == "fastergs":
                        gaussians.reset_densification_info()

                    if args.add_soft_scale_filter:
                        gaussians.filter_gaussians_soft(
                            threshold_ratio=10.0, 
                            threshold_min=0.004, 
                            decay_factor=0.5
                        )

            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )

            # Optimization
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # Save gaussians
            if iteration in saving_iterations or iteration == opt.iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc)
                time_query = time.time()
                vol_pred = queryfunc(gaussians)["vol"]
                time_query = time.time() - time_query
                print(f"Time query: {time_query:.3f}s, shape: {vol_pred.shape}")
                vol_pred = vol_pred/vol_pred.max()
                write_nifti(vol_pred.cpu().numpy(),
                        affine=np.eye(4),
                        output_path=os.path.join(dataset.model_path, "pred_volumes_{}.nii.gz".format(iteration)),
                    )
                # save npy
                print(f"Saving pred_volumes_{iteration}.npy, shape: {vol_pred.cpu().numpy().shape}")
                np.save(os.path.join(dataset.model_path, "pred_volumes_{}.npy".format(iteration)), vol_pred.cpu().numpy())
                vol_gt = scene.vol_gt
                if vol_gt is not None and iteration == opt.iterations:
                    write_nifti(vol_gt.cpu().numpy(),
                            affine=np.eye(4),
                            output_path=os.path.join(dataset.model_path, "gt_volumes_{}.nii.gz".format(iteration)),
                            )          

            # Save checkpoints
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                )

            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "loss_rend": f"{to_scalar(loss['render']):.1e}",
                        "loss_dssim": f"{to_scalar(loss['dssim']):.1e}" if 'dssim' in loss else '0.0',
                        "loss_tv": f"{to_scalar(loss['tv']):.1e}" if 'tv' in loss else '0.0',
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Logging
            metrics = {}
            for l in loss:
                metrics["loss_" + l] = to_scalar(loss[l])
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]
            training_report(
                tb_writer,
                iteration,
                metrics,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                lambda x, y: render_func(x, y, pipe, dataset),
                queryfunc,
                training_start_time,
                save_proj_images=save_proj_images,
                num_proj_to_save=num_proj_to_save,
                args=dataset,
            )
    
    print(f"Training time: {time.time() - training_start_time:.3f}s")


def training_report(
    tb_writer,
    iteration,
    metrics_train,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    queryFunc,
    training_start_time=None,
    save_proj_images=True,  # Whether to save projection comparison images
    num_proj_to_save=5,     # Number of projection images to save
    args: ModelParams = None,
):
    # Add training statistics
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "train/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    # Calculate training time
    training_time_sec = time.time() - training_start_time if training_start_time else 0.0
    training_time_min = training_time_sec / 60.0

    if iteration in testing_iterations:
        print("max density:", scene.gaussians.get_density.max().item())
        # Evaluate 2D rendering performance
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        train_cameras_cams = list(scene.getTrainCameras().values())
        test_cameras_cams = list(scene.getTestCameras().values())
        validation_configs = [
            {"name": "render_train", "cameras": train_cameras_cams},
            {"name": "render_test", "cameras": test_cameras_cams},
        ]
        psnr_2d, ssim_2d = None, None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = []
                gt_images = []
                image_show_2d = []
                # Render projections
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                # Select indices for saving projection images
                save_idx = np.linspace(0, len(config["cameras"]) - 1, num_proj_to_save).astype(int) if save_proj_images else []
                
                # Create directory for projection comparison images
                if save_proj_images:
                    proj_save_dir = osp.join(eval_save_path, f"proj_{config['name']}")
                    os.makedirs(proj_save_dir, exist_ok=True)
                
                for idx, viewpoint in enumerate(config["cameras"]):
                    if args.use_gsplat:
                        render_images = renderFunc(viewpoint, scene.gaussians)["render"]
                    else:
                        render_images = []
                        for viewpoint_cam in viewpoint:
                            image = renderFunc(
                                viewpoint_cam,
                                scene.gaussians,
                            )["render"]
                            render_images.append(image)
                        render_images = torch.stack(render_images, dim=0)
                    if args.recon_mode == "motion":
                        gt_image = viewpoint[0].original_image.to("cuda").unsqueeze(0)
                        if render_images.shape[0] > 1:
                            render_images = merge_projection(render_images, args.num_exposure_frames, args.num_rolling_shutter_frames, args.rs_axis)
                    else:
                        gt_image = [viewpoint_cam.original_image.to("cuda") for viewpoint_cam in viewpoint]
                        gt_image = torch.stack(gt_image, dim=0)
                    image = render_images[0]
                    gt_image = gt_image[0]

                    images.append(image)
                    gt_images.append(gt_image)
                    
                    # Save projection comparison images to file (grayscale)
                    if save_proj_images and idx in save_idx:
                        import cv2
                        # Get numpy arrays
                        gt_np = gt_image[0].cpu().numpy()
                        render_np = image[0].detach().cpu().numpy()
                        diff_np = np.abs(gt_np - render_np)
                        
                        # Normalize to 0-255 range for saving
                        gt_norm = ((gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8) * 255).astype(np.uint8)
                        render_norm = ((render_np - render_np.min()) / (render_np.max() - render_np.min() + 1e-8) * 255).astype(np.uint8)
                        diff_norm = ((diff_np - diff_np.min()) / (diff_np.max() - diff_np.min() + 1e-8) * 255).astype(np.uint8)
                        
                        # Concatenate horizontally: GT | Render | Diff
                        comparison_img = np.concatenate([gt_norm, render_norm, diff_norm], axis=1)
                        
                        # Save as single-channel grayscale PNG
                        cv2.imwrite(
                            osp.join(proj_save_dir, f"{viewpoint[0].image_name}_comparison.png"),
                            comparison_img
                        )
                        cv2.imwrite(
                            osp.join(proj_save_dir, f"{viewpoint[0].image_name}_gt.png"),
                            gt_norm
                        )
                    
                    if tb_writer and idx in show_idx:
                        image_show_2d.append(
                            torch.from_numpy(
                                show_two_slice(
                                    gt_image[0],
                                    image[0],
                                    f"{viewpoint[0].image_name} gt",
                                    f"{viewpoint[0].image_name} render",
                                    vmin=gt_image[0].min() if iteration != 1 else None,
                                    vmax=gt_image[0].max() if iteration != 1 else None,
                                    save=True,
                                )
                            )
                        )
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                eval_dict_2d = {
                    "psnr_2d": psnr_2d,
                    "ssim_2d": ssim_2d,
                    "psnr_2d_projs": psnr_2d_projs,
                    "ssim_2d_projs": ssim_2d_projs,
                    "training_time_sec": training_time_sec,
                    "training_time_min": training_time_min,
                }
                with open(
                    osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        eval_dict_2d, f, default_flow_style=False, sort_keys=False
                    )

                if tb_writer:
                    image_show_2d = torch.from_numpy(
                        np.concatenate(image_show_2d, axis=0)
                    )[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        config["name"] + f"/{viewpoint[0].image_name}",
                        image_show_2d,
                        global_step=iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/psnr_2d", psnr_2d, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/ssim_2d", ssim_2d, iteration
                    )

        # Evaluate 3D reconstruction performance
        if scene.vol_gt is not None:
            vol_pred = queryFunc(scene.gaussians)["vol"]
            vol_gt = scene.vol_gt
            psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
            ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")

            # Save 3d slice comparison images and projection comparison images
            if save_proj_images:
                import cv2
                # Get numpy arrays
                gt_np = vol_gt[len(vol_gt)//2].cpu().numpy()
                render_np = vol_pred[len(vol_pred)//2].detach().cpu().numpy()
                diff_np = np.abs(gt_np - render_np)
                
                # Normalize to 0-255 range for saving
                gt_norm = ((gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8) * 255).astype(np.uint8)
                render_norm = ((render_np - render_np.min()) / (render_np.max() - render_np.min() + 1e-8) * 255).astype(np.uint8)
                diff_norm = ((diff_np - diff_np.min()) / (diff_np.max() - diff_np.min() + 1e-8) * 255).astype(np.uint8)
                
                # Concatenate horizontally: GT | Render | Diff
                comparison_img = np.concatenate([gt_norm, render_norm, diff_norm], axis=1)
                
                # Save as single-channel grayscale PNG
                cv2.imwrite(
                    osp.join(proj_save_dir, f"{viewpoint[0].image_name}_slice.png"),
                    comparison_img
                )
            
            eval_dict = {
                "psnr_3d": psnr_3d,
                "ssim_3d": ssim_3d,
                "ssim_3d_x": ssim_3d_axis[0],
                "ssim_3d_y": ssim_3d_axis[1],
                "ssim_3d_z": ssim_3d_axis[2],
                "training_time_sec": training_time_sec,
                "training_time_min": training_time_min,
                "iteration": iteration,
            }
            with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
                yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
            if tb_writer:
                image_show_3d = np.concatenate(
                    [
                        show_two_slice(
                            vol_gt[..., i],
                            vol_pred[..., i],
                            f"slice {i} gt",
                            f"slice {i} pred",
                            vmin=vol_gt[..., i].min(),
                            vmax=vol_gt[..., i].max(),
                            save=True,
                        )
                        for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]
                    ],
                    axis=0,
                )
                image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
                tb_writer.add_images(
                    "reconstruction/slice-gt_pred_diff",
                    image_show_3d,
                    global_step=iteration,
                )
                tb_writer.add_scalar("reconstruction/psnr_3d", psnr_3d, iteration)
                tb_writer.add_scalar("reconstruction/ssim_3d", ssim_3d, iteration)
                tb_writer.add_scalar("reconstruction/training_time_min", training_time_min, iteration)

            tqdm.write(
                f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d:.3f}, ssim3d {ssim_3d:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}, time {training_time_min:.2f}min"
            )
        
        else:
             tqdm.write(f"[ITER {iteration}] Evaluating: psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}, time {training_time_min:.2f}min"
        )

        # Record other metrics
        if tb_writer:
            tb_writer.add_histogram(
                "scene/density_histogram", scene.gaussians.get_density, iteration
            )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    # fmt: off
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 10_000, 15_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--save_proj_images", action="store_true", default=True, help="Save projection comparison images")
    parser.add_argument("--num_proj_to_save", type=int, default=5, help="Number of projection images to save")
    parser.add_argument("--spatial_lr_scale", type=float, default=1.0, help="Spatial learning rate scale")
    parser.add_argument("--tv_delta", type=float, default=None, help="TV delta")
    parser.add_argument("--opacity_activation", type=str, default="sigmoid", help="Opacity activation function: softplus, sigmoid, bounded_sigmoid")
    parser.add_argument("--add_scale_regularization", action="store_true", default=False, help="Add scale regularization")
    parser.add_argument("--add_sparse_regularization", action="store_true", default=False, help="Add sparse regularization")
    parser.add_argument("--add_soft_scale_filter", action="store_true", default=False, help="Add soft scale filter")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"Using GPU: {args.gpu}")
    # Load configuration files
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    # Set up logging writer
    tb_writer = prepare_output_and_logger(args)

    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        save_proj_images=args.save_proj_images,
        num_proj_to_save=args.num_proj_to_save,
        args=args,
    )

    # All done
    print("Training complete.")
