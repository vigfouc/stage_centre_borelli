import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import cv2
import os
from typing import Tuple
import csv
import json
from torus_diplay import display_mesh_with_flames, create_mesh_from_stl

MASK_CENTER = (720, 480) 
MASK_RADIUS = 250

PARAM_RANGES = {
    "cam_x": (1000, 5000),      # Camera X position
    "cam_y": (1500, 5000),      # Camera Y position
    "cam_z": (60000, 75000),    # Camera Z position
    "look_x": (1000, 5000),     # Look-at X position
    "look_y": (1000, 5000),     # Look-at Y position
    "look_z": (60000, 75000),   # Look-at Z position
}

def normalize_params(params):
    """Normalize 6 parameters: [cam_x, cam_y, cam_z, look_x, look_y, look_z]"""
    return np.array([
        (params[0] - PARAM_RANGES["cam_x"][0]) / (PARAM_RANGES["cam_x"][1] - PARAM_RANGES["cam_x"][0]),
        (params[1] - PARAM_RANGES["cam_y"][0]) / (PARAM_RANGES["cam_y"][1] - PARAM_RANGES["cam_y"][0]),
        (params[2] - PARAM_RANGES["cam_z"][0]) / (PARAM_RANGES["cam_z"][1] - PARAM_RANGES["cam_z"][0]),
        (params[3] - PARAM_RANGES["look_x"][0]) / (PARAM_RANGES["look_x"][1] - PARAM_RANGES["look_x"][0]),
        (params[4] - PARAM_RANGES["look_y"][0]) / (PARAM_RANGES["look_y"][1] - PARAM_RANGES["look_y"][0]),
        (params[5] - PARAM_RANGES["look_z"][0]) / (PARAM_RANGES["look_z"][1] - PARAM_RANGES["look_z"][0]),
    ])

def unnormalize_params(norm_params):
    """Unnormalize 6 parameters back to original scale"""
    return np.array([
        norm_params[0] * (PARAM_RANGES["cam_x"][1] - PARAM_RANGES["cam_x"][0]) + PARAM_RANGES["cam_x"][0],
        norm_params[1] * (PARAM_RANGES["cam_y"][1] - PARAM_RANGES["cam_y"][0]) + PARAM_RANGES["cam_y"][0],
        norm_params[2] * (PARAM_RANGES["cam_z"][1] - PARAM_RANGES["cam_z"][0]) + PARAM_RANGES["cam_z"][0],
        norm_params[3] * (PARAM_RANGES["look_x"][1] - PARAM_RANGES["look_x"][0]) + PARAM_RANGES["look_x"][0],
        norm_params[4] * (PARAM_RANGES["look_y"][1] - PARAM_RANGES["look_y"][0]) + PARAM_RANGES["look_y"][0],
        norm_params[5] * (PARAM_RANGES["look_z"][1] - PARAM_RANGES["look_z"][0]) + PARAM_RANGES["look_z"][0],
    ])

def hybrid_loss(synthetic_image, real_image, alpha=0.5, loss_type='combined', patch_size=7, save_debug=None, iteration=None):
    """
    Compute hybrid loss using edge detection with distance transform for robust matching
    
    Args:
        synthetic_image: Rendered 3D image
        real_image: Thermal camera image
        alpha: Weight for combining losses
        loss_type: 'combined', 'distance', 'chamfer', 'ssim', 'mse', 'correlation'
        patch_size: Size of dilation kernel for edge matching (makes edges thicker)
        save_debug: Whether to save debug images
        iteration: Current iteration number
    """
    h, w = real_image.shape[:2]

    # Convert to grayscale
    synthetic_gray = cv2.cvtColor(synthetic_image, cv2.COLOR_BGR2GRAY)
    real_gray = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY)
    
    # Extract edges with specified parameters
    real_edges = cv2.Canny(real_gray, threshold1=22, threshold2=10)
    mask_real_edges = cv2.Canny(real_gray, threshold1=50, threshold2=100)
    real_edges = real_edges - mask_real_edges
    synth_edges = cv2.Canny(synthetic_gray, threshold1=55, threshold2=110)
    
    # Normalize edges
    real_edges_norm = real_edges.astype(np.float32) / 255.0
    synth_edges_norm = synth_edges.astype(np.float32) / 255.0
    
    # Create dilated versions for more robust matching
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (patch_size, patch_size))
    real_edges_dilated = cv2.dilate(real_edges, kernel, iterations=1)
    synth_edges_dilated = cv2.dilate(synth_edges, kernel, iterations=1)
    
    real_edges_dilated_norm = real_edges_dilated.astype(np.float32) / 255.0
    synth_edges_dilated_norm = synth_edges_dilated.astype(np.float32) / 255.0
    
    # Compute distance transforms (distance from each pixel to nearest edge)
    real_dist_transform = cv2.distanceTransform((1 - (real_edges > 0).astype(np.uint8)), cv2.DIST_L2, 5)
    synth_dist_transform = cv2.distanceTransform((1 - (synth_edges > 0).astype(np.uint8)), cv2.DIST_L2, 5)
    
    # Normalize distance transforms
    real_dist_norm = cv2.normalize(real_dist_transform, None, 0, 1, cv2.NORM_MINMAX)
    synth_dist_norm = cv2.normalize(synth_dist_transform, None, 0, 1, cv2.NORM_MINMAX)
    
    # Save debug images
    if save_debug and iteration is not None:
        debug_folder = "Debug"
        os.makedirs(debug_folder, exist_ok=True)
        debug_subfolder = os.path.join(debug_folder, 'loss_debug')
        os.makedirs(debug_subfolder, exist_ok=True)
        
        # Save original edges
        cv2.imwrite(
            os.path.join(debug_subfolder, f'iter_{iteration:04d}_1_real_edges.png'),
            real_edges
        )
        cv2.imwrite(
            os.path.join(debug_subfolder, f'iter_{iteration:04d}_2_synth_edges.png'),
            synth_edges
        )
        
        # Save dilated edges
        cv2.imwrite(
            os.path.join(debug_subfolder, f'iter_{iteration:04d}_3_real_edges_dilated.png'),
            real_edges_dilated
        )
        cv2.imwrite(
            os.path.join(debug_subfolder, f'iter_{iteration:04d}_4_synth_edges_dilated.png'),
            synth_edges_dilated
        )
        
        # Save distance transforms (as heatmap)
        real_dist_colored = cv2.applyColorMap((real_dist_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        synth_dist_colored = cv2.applyColorMap((synth_dist_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(
            os.path.join(debug_subfolder, f'iter_{iteration:04d}_5_real_distance_transform.png'),
            real_dist_colored
        )
        cv2.imwrite(
            os.path.join(debug_subfolder, f'iter_{iteration:04d}_6_synth_distance_transform.png'),
            synth_dist_colored
        )
        
        # Save edge overlay (original thin edges)
        overlay_thin = np.zeros((h, w, 3), dtype=np.uint8)
        overlay_thin[real_edges > 0] = [0, 0, 255]  # Red: real edges
        overlay_thin[synth_edges > 0] = [255, 0, 0]  # Blue: synth edges
        overlay_thin[(real_edges > 0) & (synth_edges > 0)] = [0, 255, 0]  # Green: matching edges
        cv2.imwrite(
            os.path.join(debug_subfolder, f'iter_{iteration:04d}_7_edge_overlay_thin.png'),
            overlay_thin
        )
        
        # Save edge overlay (dilated edges)
        overlay_dilated = np.zeros((h, w, 3), dtype=np.uint8)
        overlay_dilated[real_edges_dilated > 0] = [0, 0, 255]  # Red: real edges
        overlay_dilated[synth_edges_dilated > 0] = [255, 0, 0]  # Blue: synth edges
        overlay_dilated[(real_edges_dilated > 0) & (synth_edges_dilated > 0)] = [0, 255, 0]  # Green: matching
        cv2.imwrite(
            os.path.join(debug_subfolder, f'iter_{iteration:04d}_8_edge_overlay_dilated.png'),
            overlay_dilated
        )
    
    # Compute loss based on selected type
    if loss_type == 'distance':
        # Distance transform loss - penalizes edges that are far apart
        # For each real edge pixel, measure distance to nearest synth edge
        real_edge_pixels = real_edges > 0
        synth_edge_pixels = synth_edges > 0
        
        if np.sum(real_edge_pixels) > 0:
            real_to_synth_dist = synth_dist_transform[real_edge_pixels]
            loss_real = np.mean(real_to_synth_dist)
        else:
            loss_real = 0
        
        if np.sum(synth_edge_pixels) > 0:
            synth_to_real_dist = real_dist_transform[synth_edge_pixels]
            loss_synth = np.mean(synth_to_real_dist)
        else:
            loss_synth = 0
        
        # Symmetric distance (Chamfer-like)
        loss = (loss_real + loss_synth) / 2.0
        
    elif loss_type == 'chamfer':
        # Proper Chamfer distance
        real_edge_pixels = real_edges > 0
        synth_edge_pixels = synth_edges > 0
        
        if np.sum(real_edge_pixels) > 0 and np.sum(synth_edge_pixels) > 0:
            # Distance from real edges to synth edges
            real_to_synth = np.mean(synth_dist_transform[real_edge_pixels] ** 2)
            # Distance from synth edges to real edges
            synth_to_real = np.mean(real_dist_transform[synth_edge_pixels] ** 2)
            
            loss = (real_to_synth + synth_to_real) / 2.0
        else:
            loss = 1000.0  # Large penalty if no edges detected
        
    elif loss_type == 'ssim':
        # SSIM on dilated edges (more robust)
        ssim_score = ssim(synth_edges_dilated_norm, real_edges_dilated_norm, win_size=9, gaussian_weights=True, sigma=8, full=False, data_range=1.0)
        loss = 1.0 - np.mean(ssim_score)
        
    elif loss_type == 'mse':
        # MSE on dilated edges
        loss = np.mean((synth_edges_dilated_norm - real_edges_dilated_norm) ** 2)
        
    elif loss_type == 'correlation':
        # Cross-correlation on dilated edges
        real_flat = real_edges_dilated_norm.flatten()
        synth_flat = synth_edges_dilated_norm.flatten()
        
        if np.std(real_flat) > 0 and np.std(synth_flat) > 0:
            correlation = np.corrcoef(real_flat, synth_flat)[0, 1]
            loss = 1.0 - max(0, correlation)
        else:
            loss = 1.0
            
    elif loss_type == 'combined':
        # Combined loss using dilated edges + distance transform
        
        # 1. SSIM on dilated edges
        ssim_score, ssim_map = ssim(synth_edges_dilated_norm, real_edges_dilated_norm, full=True, data_range=1.0)
        ssim_loss = 1.0 - np.mean(ssim_map)
        
        # 2. Distance transform loss (Chamfer-like)
        real_edge_pixels = real_edges > 0
        synth_edge_pixels = synth_edges > 0
        
        if np.sum(real_edge_pixels) > 0 and np.sum(synth_edge_pixels) > 0:
            real_to_synth = np.mean(synth_dist_transform[real_edge_pixels])
            synth_to_real = np.mean(real_dist_transform[synth_edge_pixels])
            distance_loss = (real_to_synth + synth_to_real) / 2.0
            # Normalize distance loss to [0, 1] range (assuming max distance ~50 pixels)
            distance_loss = min(distance_loss / 50.0, 1.0)
        else:
            distance_loss = 1.0
        
        # 3. IoU on dilated edges
        intersection = np.sum((real_edges_dilated_norm > 0.5) & (synth_edges_dilated_norm > 0.5))
        union = np.sum((real_edges_dilated_norm > 0.5) | (synth_edges_dilated_norm > 0.5))
        
        if union > 0:
            iou_loss = 1.0 - (intersection / union)
        else:
            iou_loss = 1.0
        
        # Combine: emphasize distance loss (most important for alignment)
        loss = 0.5 * distance_loss + 0.3 * ssim_loss + 0.2 * iou_loss
    
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'combined', 'distance', 'chamfer', 'ssim', 'mse', or 'correlation'")

    return loss

def compute_gradients(norm_params, real_image, stl_path, epsilons, alpha, loss_type='combined', verbose=False):
    """Compute gradients for all 6 camera parameters using finite differences"""
    gradients = np.zeros(6)
    param_names = ['cam_x', 'cam_y', 'cam_z', 'look_x', 'look_y', 'look_z']

    for i in range(6):
        offset = np.zeros_like(norm_params)
        offset[i] = epsilons[i]

        # Forward step
        params_plus = unnormalize_params(norm_params + offset)
        camera_position_plus = params_plus[:3]
        look_at_position_plus = params_plus[3:6]
        image_plus = display_mesh_with_flames(
            stl_path, 
            [], 
            camera_position=camera_position_plus,
            look_at_position=look_at_position_plus,
            show_window=False
        )
        loss_plus = hybrid_loss(image_plus, real_image, alpha, loss_type=loss_type)

        # Backward step
        params_minus = unnormalize_params(norm_params - offset)
        camera_position_minus = params_minus[:3]
        look_at_position_minus = params_minus[3:6]
        image_minus = display_mesh_with_flames(
            stl_path, 
            [], 
            camera_position=camera_position_minus,
            look_at_position=look_at_position_minus,
            show_window=False
        )
        loss_minus = hybrid_loss(image_minus, real_image, alpha, loss_type=loss_type)

        gradients[i] = (loss_plus - loss_minus) / (2 * epsilons[i])
        
        if verbose:
            print(f"  {param_names[i]}: loss_plus={loss_plus:.6f}, loss_minus={loss_minus:.6f}, grad={gradients[i]:.6f}")

    # Store raw gradient norm
    raw_grad_norm = np.linalg.norm(gradients)
    
    # DON'T normalize gradients - let Adam handle the scaling
    # Normalization was removing important magnitude information
    
    if verbose:
        print(f"  Raw gradient norm: {raw_grad_norm:.8f}")
        print(f"  Individual gradients: {gradients}")

    return gradients, raw_grad_norm

def adam_optimizer(real_image, stl_path, init_params, epsilons, iterations=200, lr=0.1, alpha=0.5, early_stop_patience=20, step_folder=None, loss_type='combined'):
    """Adam optimizer for camera parameter optimization with step-by-step logging"""
    m = np.zeros(6)
    v = np.zeros(6)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    norm_params = normalize_params(init_params)

    best_loss = float('inf')
    best_params = norm_params.copy()
    patience_counter = 0
    losses = []
    
    # Create step tracking lists
    step_data = []
    raw_grad_norms = []
    
    # Create CSV file for step-by-step tracking
    if step_folder:
        step_csv_path = os.path.join(step_folder, 'optimization_steps.csv')
        with open(step_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Iteration', 'Loss', 
                'Cam_X', 'Cam_Y', 'Cam_Z',
                'Look_X', 'Look_Y', 'Look_Z',
                'Raw_Grad_Norm'
            ])

    for t in range(1, iterations + 1):
        # Verbose output for first 3 iterations
        verbose = (t <= 3)
        
        if verbose:
            print(f"\n=== Iteration {t} - Gradient Computation ===")
        
        gradients, raw_grad_norm = compute_gradients(norm_params, real_image, stl_path, epsilons, alpha, loss_type= loss_type, verbose=verbose)
        raw_grad_norms.append(raw_grad_norm)

        m = beta1 * m + (1 - beta1) * gradients
        v = beta2 * v + (1 - beta2) * (gradients ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        norm_params -= lr * m_hat / (np.sqrt(v_hat) + eps)
        norm_params = np.clip(norm_params, 0, 1)

        params = unnormalize_params(norm_params)
        camera_position = params[:3]
        look_at_position = params[3:6]
        
        # Generate image for current parameters
        full_res_image = display_mesh_with_flames(
            stl_path, 
            [], 
            camera_position=camera_position,
            look_at_position=look_at_position,
            show_window=False
        )
        loss = hybrid_loss(full_res_image, real_image, alpha, save_debug=True, iteration=t, loss_type=loss_type)
        losses.append(loss)
        
        # Save step data
        if step_folder:
            # Save image
            image_filename = f'step_{t:04d}.png'
            image_path = os.path.join(step_folder, image_filename)
            cv2.imwrite(image_path, cv2.cvtColor(full_res_image, cv2.COLOR_RGB2BGR))
            
            # Append to CSV
            with open(step_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    t, f'{loss:.6f}',
                    f'{params[0]:.2f}', f'{params[1]:.2f}', f'{params[2]:.2f}',
                    f'{params[3]:.2f}', f'{params[4]:.2f}', f'{params[5]:.2f}',
                    f'{raw_grad_norm:.8f}'
                ])
            
            # Store step data for JSON
            step_data.append({
                'iteration': t,
                'loss': float(loss),
                'camera_position': {
                    'x': float(params[0]),
                    'y': float(params[1]),
                    'z': float(params[2])
                },
                'look_at_position': {
                    'x': float(params[3]),
                    'y': float(params[4]),
                    'z': float(params[5])
                },
                'raw_gradient_norm': float(raw_grad_norm),
                'image_file': image_filename
            })

        if loss < best_loss:
            best_loss = loss
            best_params = norm_params.copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stop_patience:
            print(f"Early stopping at iteration {t}")
            break

        print(f"Iter {t}: Loss={loss:.6f}, Raw_Grad_Norm={raw_grad_norm:.8f}")
        print(f"  Camera: [{params[0]:.2f}, {params[1]:.2f}, {params[2]:.2f}]")
        print(f"  LookAt: [{params[3]:.2f}, {params[4]:.2f}, {params[5]:.2f}]")
        
        # Warning if gradient is too small
        if raw_grad_norm < 1e-6:
            print(f"  WARNING: Very small gradient norm! Loss surface may be flat.")
    
    # Save complete step data as JSON
    if step_folder:
        json_path = os.path.join(step_folder, 'optimization_steps.json')
        with open(json_path, 'w') as f:
            json.dump({
                'total_iterations': len(step_data),
                'best_loss': float(best_loss),
                'best_iteration': int(np.argmin(losses) + 1),
                'steps': step_data
            }, f, indent=2)
        
        # Plot raw gradient norm over iterations
        plt.figure(figsize=(10, 6))
        plt.semilogy(range(1, len(raw_grad_norms) + 1), raw_grad_norms, linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Raw Gradient Norm (log scale)', fontsize=12)
        plt.title('Gradient Magnitude vs Iteration', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(step_folder, 'gradient_norm_plot.png'), dpi=150)
        plt.close()

    return unnormalize_params(best_params), losses

def optimize_camera_registration(real_image_path, stl_path, initial_params, alpha=0.5, learning_rate=0.01, iterations=200, loss_type='combined', result_folder=None):
    """Main optimization function for camera registration"""
    real_image = cv2.imread(real_image_path)
    #real_image = cv2.resize(real_image, (1280, 960))

    # Set epsilon values for finite differences (normalized space)
    epsilons = np.array([
        50.0 / (PARAM_RANGES["cam_x"][1] - PARAM_RANGES["cam_x"][0]),
        50.0 / (PARAM_RANGES["cam_y"][1] - PARAM_RANGES["cam_y"][0]),
        100.0 / (PARAM_RANGES["cam_z"][1] - PARAM_RANGES["cam_z"][0]),
        50.0 / (PARAM_RANGES["look_x"][1] - PARAM_RANGES["look_x"][0]),
        50.0 / (PARAM_RANGES["look_y"][1] - PARAM_RANGES["look_y"][0]),
        100.0 / (PARAM_RANGES["look_z"][1] - PARAM_RANGES["look_z"][0]),
    ])

    if result_folder is None:
        result_folder = os.path.join("results", os.path.splitext(os.path.basename(real_image_path))[0])
    os.makedirs(result_folder, exist_ok=True)
    
    # Create subfolder for step-by-step images and data
    step_folder = os.path.join(result_folder, 'optimization_steps')
    os.makedirs(step_folder, exist_ok=True)

    optimized_params, losses = adam_optimizer(
        real_image,
        stl_path,
        initial_params,
        epsilons,
        iterations=iterations,
        lr=learning_rate,
        alpha=alpha,
        early_stop_patience=20,
        step_folder=step_folder,
        loss_type=loss_type
    )

    # Plot and save loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Camera Registration Loss vs Iteration', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(result_folder, 'camera_optimization_loss.png'), dpi=150)
    plt.close()

    # Save final optimized image
    final_image = display_mesh_with_flames(
        stl_path,
        [],
        camera_position=optimized_params[:3],
        look_at_position=optimized_params[3:6],
        show_window=False
    )
    cv2.imwrite(
        os.path.join(result_folder, 'optimized_camera_view.png'),
        cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)
    )
    
    # Copy the target image to results folder for comparison
    real_image = cv2.imread(real_image_path)
    real_image_resized = cv2.resize(real_image, (1280, 960))
    cv2.imwrite(
        os.path.join(result_folder, 'target_image.png'),
        real_image_resized
    )
    
    # Create side-by-side comparison
    comparison = np.hstack([real_image_resized, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR)])
    cv2.imwrite(
        os.path.join(result_folder, 'comparison.png'),
        comparison
    )

    return optimized_params

def save_camera_params_to_csv(real_image: str, initial_params: np.ndarray, optimized_params: np.ndarray, csv_filename: str):
    """Save camera parameters to CSV file"""
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                'Real Image', 
                'Initial Cam X', 'Initial Cam Y', 'Initial Cam Z',
                'Initial Look X', 'Initial Look Y', 'Initial Look Z',
                'Optimized Cam X', 'Optimized Cam Y', 'Optimized Cam Z',
                'Optimized Look X', 'Optimized Look Y', 'Optimized Look Z'
            ])
        writer.writerow([real_image, *initial_params, *optimized_params])

if __name__ == '__main__':
    stl_path = "four_version_finale.STL"
    real_image_path = "avg_image.png"  # Your thermal image
    csv_filename = "optimized_camera_parameters.csv"
    
    # Initial camera parameters: [cam_x, cam_y, cam_z, look_x, look_y, look_z]
    #initial_params = np.array([3917.11, 2260.72, 65871.05, 4330.44, 2106.89, 66768.55]). #good initial params
   
    initial_params = np.array([3525.20, 2027.26, 67277.01, 4200.07, 1558.07, 70268.76])

    print("Starting camera registration optimization...")
    print(f"Initial camera position: {initial_params[:3]}")
    print(f"Initial look-at position: {initial_params[3:6]}")
    
    optimized_params = optimize_camera_registration(
        real_image_path,
        stl_path,
        initial_params,
        alpha=0.5,
        learning_rate=0.00001,
        iterations=20,
        loss_type="ssim"
    )
    
    save_camera_params_to_csv(
        os.path.basename(real_image_path),
        initial_params,
        optimized_params,
        csv_filename
    )
    
    print("\nOptimization complete!")
    print(f"Optimized camera position: {optimized_params[:3]}")
    print(f"Optimized look-at position: {optimized_params[3:6]}")
    print(f"\nResults saved in: results/{os.path.splitext(os.path.basename(real_image_path))[0]}/")