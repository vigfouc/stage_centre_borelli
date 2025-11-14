import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import cv2
import os
import csv
from typing import Tuple

# Parameter ranges for torus (R = major radius, r = minor radius)
PARAM_RANGES = {
    "R": (100, 1000),  # Major radius range
    "r": (50, 500),     # Minor radius range
}

def create_torus(R, r, radial_segments=100, tubular_segments=50, color=[1.0, 0.0, 0.0], name="Torus", translation=[0, 0, 0]):
    vertices = []
    triangles = []

    for i in range(radial_segments):
        theta = 2.0 * np.pi * i / radial_segments
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        for j in range(tubular_segments):
            phi = 2.0 * np.pi * j / tubular_segments
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)

            x = (R + r * cos_phi) * cos_theta
            y = (R + r * cos_phi) * sin_theta
            z = r * sin_phi
            vertices.append([x, y, z])

    for i in range(radial_segments):
        for j in range(tubular_segments):
            next_i = (i + 1) % radial_segments
            next_j = (j + 1) % tubular_segments

            idx0 = i * tubular_segments + j
            idx1 = next_i * tubular_segments + j
            idx2 = next_i * tubular_segments + next_j
            idx3 = i * tubular_segments + next_j

            triangles.append([idx0, idx1, idx2])
            triangles.append([idx0, idx2, idx3])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.array(triangles))
    mesh.compute_vertex_normals()
    mesh.translate(translation)
    mesh.paint_uniform_color(color)

    return {"name": name, "geometry": mesh}

def create_mesh_from_stl(stl_path, color=[0.5, 0.5, 0.5], name="Mesh"):
    """Load kiln mesh from STL file"""
    mesh = o3d.io.read_triangle_mesh(stl_path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return {"name": name, "geometry": mesh}

def display_mesh_with_torus(stl_path, torus, save_filename=None, show_window=False, save_images=False):
    
    mesh_from_stl = create_mesh_from_stl(stl_path)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Open3D: Mesh and Torus", width=1280, height=960, visible=show_window)

    vis.add_geometry(mesh_from_stl["geometry"])
    
    if torus is not None:
        vis.add_geometry(torus["geometry"])

    ctr = vis.get_view_control()
    camera_params = ctr.convert_to_pinhole_camera_parameters()

    # Camera settings matching your flame rendering
    #initial_params = np.array([3917.11, 2260.72, 65871.05, 4330.44, 2106.89, 66768.55]). #good initial params

    camera_position = np.array([3917.11, 2260.72, 65871.05], dtype=np.float64)
    look_at_position = np.array([4330.44, 2106.89, 66768.55], dtype=np.float64)
    up_vector = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    forward = (camera_position - look_at_position)
    forward = forward / np.linalg.norm(forward)

    right = np.cross(up_vector, forward)
    right = right / np.linalg.norm(right)

    true_up = np.cross(forward, right)

    R = np.stack([right, true_up, forward], axis=1)
    t = -R.T @ camera_position

    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = R.T
    extrinsic[:3, 3] = t

    camera_params.extrinsic = extrinsic
    ctr.convert_from_pinhole_camera_parameters(camera_params)

    vis.poll_events()
    vis.update_renderer()
    
    if save_images == True:
        output_path = os.path.join(os.getcwd(), save_filename)
        vis.capture_screen_image(output_path)
        print(f"Scene saved to: {output_path}")
        
    if show_window:
        vis.run()
        vis.destroy_window()
        return None
    else:
        image = vis.capture_screen_float_buffer(do_render=True)
        image_array = np.asarray(image) * 255
        image_array = image_array.astype(np.uint8)
        vis.destroy_window()
        return image_array

def normalize_params(params):
    return np.array([
        (params[0] - PARAM_RANGES["R"][0]) / (PARAM_RANGES["R"][1] - PARAM_RANGES["R"][0]),
        (params[1] - PARAM_RANGES["r"][0]) / (PARAM_RANGES["r"][1] - PARAM_RANGES["r"][0])
    ])

def unnormalize_params(norm_params):
    return np.array([
        norm_params[0] * (PARAM_RANGES["R"][1] - PARAM_RANGES["R"][0]) + PARAM_RANGES["R"][0],
        norm_params[1] * (PARAM_RANGES["r"][1] - PARAM_RANGES["r"][0]) + PARAM_RANGES["r"][0]
    ])

def extract_torus_mask(synthetic_image, kiln_mask_image):
    
    if len(synthetic_image.shape) == 3:
        synthetic_gray = cv2.cvtColor(synthetic_image, cv2.COLOR_RGB2GRAY)
    else:
        synthetic_gray = synthetic_image.copy()
    
    if len(kiln_mask_image.shape) == 3:
        kiln_gray = cv2.cvtColor(kiln_mask_image, cv2.COLOR_RGB2GRAY)
    else:
        kiln_gray = kiln_mask_image.copy()
    
    if synthetic_gray.shape != kiln_gray.shape:
        kiln_gray = cv2.resize(kiln_gray, (synthetic_gray.shape[1], synthetic_gray.shape[0]))
    
    
    diff = np.abs(synthetic_gray.astype(np.float32) - kiln_gray.astype(np.float32))
    torus_mask = diff > 10  # Threshold for detecting torus pixels
    
    return torus_mask

def compute_loss(synthetic_image, target_image, kiln_mask, torus_mask, alpha=0.5):
    
    if len(target_image.shape) == 3:
        target_gray = cv2.cvtColor(target_image, cv2.COLOR_RGB2GRAY)
    else:
        target_gray = target_image.copy()
    
    flame_mask = (target_gray > 0).astype(np.float32)
    torus_mask_float = torus_mask.astype(np.float32)
    
    if torus_mask_float.shape != flame_mask.shape:
        torus_mask_float = cv2.resize(torus_mask_float, (flame_mask.shape[1], flame_mask.shape[0]))
    
    if np.sum(torus_mask) == 0:
        print("Warning: Empty torus mask!")
        return 1.0, 0.0
    
    intersection = np.sum(torus_mask_float * flame_mask)
    union = np.sum(torus_mask_float) + np.sum(flame_mask) - intersection
    
    if union == 0:
        iou = 0.0
    else:
        iou = intersection / union
    
    dice = (2.0 * intersection) / (np.sum(torus_mask_float) + np.sum(flame_mask) + 1e-8)
    
    eps = 1e-7  
    torus_prob = np.clip(torus_mask_float, eps, 1 - eps)
    bce = -np.mean(flame_mask * np.log(torus_prob) + (1 - flame_mask) * np.log(1 - torus_prob))
    
   
    iou_loss = 1.0 - iou
    
    combined_loss = alpha * bce + (1 - alpha) * iou_loss
    
    return combined_loss, iou

def compute_gradients(norm_params, target_image, stl_path, kiln_mask, 
                     torus_translation, epsilons, alpha, flame_mask):
    gradients = np.zeros(2)
    
    for i in range(2):
        offset = np.zeros_like(norm_params)
        offset[i] = epsilons[i]
        
        params_plus = unnormalize_params(norm_params + offset)
        torus_plus = create_torus(params_plus[0], params_plus[1], 
                                  color=[1.0, 0.0, 0.0], 
                                  translation=torus_translation)
        image_plus = display_mesh_with_torus(stl_path, torus_plus, show_window=False)
        torus_mask_plus = extract_torus_mask(image_plus, kiln_mask)
        loss_plus, _ = compute_loss(image_plus, target_image, kiln_mask, torus_mask_plus, alpha)
        
        params_minus = unnormalize_params(norm_params - offset)
        torus_minus = create_torus(params_minus[0], params_minus[1], 
                                   color=[1.0, 0.0, 0.0], 
                                   translation=torus_translation)
        image_minus = display_mesh_with_torus(stl_path, torus_minus, show_window=False)
        torus_mask_minus = extract_torus_mask(image_minus, kiln_mask)
        loss_minus, _ = compute_loss(image_minus, target_image, kiln_mask, torus_mask_minus, alpha)
        
        gradients[i] = (loss_plus - loss_minus) / (2 * epsilons[i])
    
    grad_norm = np.linalg.norm(gradients)
    if grad_norm > 0:
        gradients /= grad_norm
    
    return gradients

def adam_optimizer(target_image, stl_path, kiln_mask,
                  init_params, torus_translation, epsilons, flame_mask,
                  iterations=200, lr=0.1, alpha=0.5, early_stop_patience=20,
                  debug_folder=None):
    m = np.zeros(2)
    v = np.zeros(2)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    norm_params = normalize_params(init_params)
    
    best_loss = float('inf')
    best_params = norm_params.copy()
    patience_counter = 0
    losses = []
    iou_metrics = []
    
    if debug_folder is not None:
        os.makedirs(debug_folder, exist_ok=True)
        
        target_normalized = ((target_image - target_image.min()) / 
                           (target_image.max() - target_image.min()) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_folder, 'target_flame.png'), target_normalized)
        
        flame_mask_viz = (flame_mask * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(debug_folder, 'flame_mask.png'), flame_mask_viz)
    
    for t in range(1, iterations + 1):
        gradients = compute_gradients(norm_params, target_image, stl_path, 
                                     kiln_mask, torus_translation, 
                                     epsilons, alpha, flame_mask)
        
        m = beta1 * m + (1 - beta1) * gradients
        v = beta2 * v + (1 - beta2) * (gradients ** 2)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        norm_params -= lr * m_hat / (np.sqrt(v_hat) + eps)
        norm_params = np.clip(norm_params, 0, 1)
        
        params = unnormalize_params(norm_params)
        torus = create_torus(params[0], params[1], 
                           color=[1.0, 0.0, 0.0], 
                           translation=torus_translation)
        image = display_mesh_with_torus(stl_path, torus, show_window=False)
        
        torus_mask = extract_torus_mask(image, kiln_mask)
        
        loss, iou_metric = compute_loss(image, target_image, kiln_mask, torus_mask, alpha)
        losses.append(loss)
        iou_metrics.append(iou_metric)
        
        if debug_folder is not None:
            debug_image = image.copy()
            
            if len(debug_image.shape) == 3:
                overlay = debug_image.copy()
                overlay[torus_mask] = [0, 255, 0]  # Green for torus
                overlay[flame_mask] = [255, 0, 0]   # Red for target flame
                debug_image = cv2.addWeighted(debug_image, 0.7, overlay, 0.3, 0)
            
            text = f"Iter {t}: R={params[0]:.1f}, r={params[1]:.1f}"
            text2 = f"Loss={loss:.6f}, IoU={iou_metric:.4f}"
            cv2.putText(debug_image, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(debug_image, text2, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            filename = os.path.join(debug_folder, f'iter_{t:04d}.png')
            if len(debug_image.shape) == 3:
                cv2.imwrite(filename, cv2.cvtColor(debug_image, cv2.COLOR_RGB2BGR))
            else:
                cv2.imwrite(filename, debug_image)
            
            torus_mask_viz = (torus_mask * 255).astype(np.uint8)
            mask_filename = os.path.join(debug_folder, f'torus_mask_{t:04d}.png')
            cv2.imwrite(mask_filename, torus_mask_viz)
        
        if loss < best_loss:
            best_loss = loss
            best_params = norm_params.copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at iteration {t}")
            break
        
        print(f"Iter {t}: Loss={loss:.6f}, IoU={iou_metric:.4f}, R={params[0]:.2f}, r={params[1]:.2f}")
    
    return unnormalize_params(best_params), losses, iou_metrics

def optimize_torus(thermal_image, kiln_mask_image, kiln_stl_path, 
                  initial_R=400, initial_r=100, 
                  torus_translation=[0, 0, 0],
                  iterations=100, lr=0.1, alpha=0.5, 
                  result_folder="results",
                  save_debug_images=True):
   
    os.makedirs(result_folder, exist_ok=True)
    
    debug_folder = os.path.join(result_folder, 'debug_iterations') if save_debug_images else None
    
    flame_mask = thermal_image > 0
    
    target_normalized = (thermal_image - thermal_image.min()) / (thermal_image.max() - thermal_image.min())
    target_normalized = (target_normalized * 255).astype(np.uint8)
    
    init_params = np.array([initial_R, initial_r])
    epsilons = np.array([
        10.0 / (PARAM_RANGES["R"][1] - PARAM_RANGES["R"][0]),
        5.0 / (PARAM_RANGES["r"][1] - PARAM_RANGES["r"][0])
    ])
    
    optimized_params, losses, iou_metrics = adam_optimizer(
        target_normalized,
        kiln_stl_path,
        kiln_mask_image,
        init_params,
        torus_translation,
        epsilons,
        flame_mask,
        iterations=iterations,
        lr=lr,
        alpha=alpha,
        early_stop_patience=20,
        debug_folder=debug_folder
    )
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(losses)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Combined Loss')
    ax1.set_title('Combined Loss vs Iteration (Lower is Better)')
    ax1.grid(True)
    
    ax2.plot(iou_metrics, color='green')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('IoU (Intersection over Union)')
    ax2.set_title('IoU Metric vs Iteration (Higher is Better)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_folder, 'loss_plots.png'), dpi=150)
    plt.close()
    
    final_torus = create_torus(optimized_params[0], optimized_params[1], 
                               color=[1.0, 0.0, 0.0], 
                               translation=torus_translation)
    final_image = display_mesh_with_torus(kiln_stl_path, final_torus, show_window=False)
    final_torus_mask = extract_torus_mask(final_image, kiln_mask_image)
    final_loss, final_iou = compute_loss(final_image, target_normalized, kiln_mask_image, final_torus_mask, alpha)
    
    cv2.imwrite(os.path.join(result_folder, 'final_render.png'), 
                cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
    
    cv2.imwrite(os.path.join(result_folder, 'final_torus_mask.png'), 
                (final_torus_mask * 255).astype(np.uint8))
    
    comparison = np.hstack([
        target_normalized,
        cv2.cvtColor(final_image, cv2.COLOR_RGB2GRAY)
    ])
    cv2.imwrite(os.path.join(result_folder, 'comparison.png'), comparison)
    
    with open(os.path.join(result_folder, 'optimized_params.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Parameter', 'Value'])
        writer.writerow(['Initial Major Radius (R)', initial_R])
        writer.writerow(['Initial Minor Radius (r)', initial_r])
        writer.writerow(['Optimized Major Radius (R)', optimized_params[0]])
        writer.writerow(['Optimized Minor Radius (r)', optimized_params[1]])
        writer.writerow(['Final Combined Loss', final_loss])
        writer.writerow(['Final IoU', final_iou])
        writer.writerow(['Number of Iterations', len(losses)])
    
    with open(os.path.join(result_folder, 'iteration_metrics.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'Combined Loss', 'IoU'])
        for i, (loss, iou) in enumerate(zip(losses, iou_metrics), 1):
            writer.writerow([i, loss, iou])
    
    print(f"\n{'='*60}")
    print(f"Optimization Complete!")
    print(f"{'='*60}")
    print(f"Initial parameters: R={initial_R:.2f}, r={initial_r:.2f}")
    print(f"Optimized R (major radius): {optimized_params[0]:.2f}")
    print(f"Optimized r (minor radius): {optimized_params[1]:.2f}")
    print(f"Final combined loss: {final_loss:.6f}")
    print(f"Final IoU (torus vs flame): {final_iou:.4f}")
    print(f"Total iterations: {len(losses)}")
    print(f"\nResults saved to: {result_folder}")
    if save_debug_images:
        print(f"Debug images saved to: {debug_folder}")
    print(f"{'='*60}")
    
    return optimized_params, losses, iou_metrics

def create_thermal_binary_mask(images_folder, threshold):
    files = sorted([f for f in os.listdir(images_folder) if f.endswith(".npy")])
    combined_mask = None
    
    for f_name in files:
        std_image_path = os.path.join(images_folder, f_name)
        std_image = np.load(std_image_path)
        
        mask = (std_image > threshold)
        
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = np.logical_or(combined_mask, mask)
    
    return combined_mask

def display_std_top_percentage(image, percent, title="Top % Pixels", visu=False):
    assert 0 < percent < 100, "Percent must be between 0 and 100"
    
    threshold = np.percentile(image, 100 - percent)  
    mask = image >= threshold
    
    masked_image = np.zeros_like(image)
    masked_image[mask] = image[mask]
    
    if visu == True:
        plt.figure()
        plt.imshow(masked_image, cmap='hot')
        plt.colorbar(label="Pixel value")
        plt.title(f"{title} (Top {percent}%)")
        plt.show()
        
    return masked_image

if __name__ == "__main__":
    
    # Thermal Images
    # ---------------------
    avg_path = "avg_images_10_sample_rate"
    std_path = "std_images_10_sample_rate"
    
    avg_image_path = os.path.join(avg_path, "avg_frame_0400.npy")
    std_image_path = os.path.join(std_path, "std_frame_0400.npy")
    
    # --------------------------------------------
    # Undistort images
    camera_matrix = np.array([
        [1.05e+03, 0.00000000e+00, 6.40000000e+02],
        [0.00000000e+00, 1.05e+03, 4.80000000e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ], dtype=np.float32)

    dist_coeffs = np.array([
        -0.35036623, 0.23745768, 0.00211392, 0.00222737, -0.29409797
    ], dtype=np.float32)

    new_camera_matrix = np.array([
        [827.14155814, 0.0, 655.42672321],
        [0.0, 918.87433136, 484.15324826],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    avg_mask = create_thermal_binary_mask(avg_path, 1600)
    
    avg_image_distorted = np.load(avg_image_path)
    std_image_distorted = np.load(std_image_path)
    
    avg_image = cv2.undistort(avg_image_distorted, camera_matrix, dist_coeffs, None, new_camera_matrix)
    std_image = cv2.undistort(std_image_distorted, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    masked_std_img = np.where(avg_mask, std_image, 0)
    
    plt.imsave("std_image.png", std_image)
    plt.imsave("avg_image.png", avg_image)
    
    flame_torus = display_std_top_percentage(masked_std_img, percent=0.7, title="Average Image", visu=False)
    
    # Kiln Images
    # ---------------------
    mask_path = "kiln_mask.png"
    kiln_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Run optimization
    optimized_params, losses, mse_metrics = optimize_torus(
        thermal_image=flame_torus,
        kiln_mask_image=kiln_mask,
        kiln_stl_path="four_version_finale.STL",
        initial_R=400,
        initial_r=100,
        torus_translation=[3083.51, 3036.26, 64129.9],
        iterations=100,
        lr=0.1,
        alpha=0.1,
        result_folder="torus_optimization_results",
        save_debug_images=True
    )