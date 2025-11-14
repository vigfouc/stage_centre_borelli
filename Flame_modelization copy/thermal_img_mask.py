import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

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
    
    #Thermal Images
    #---------------------

    avg_path = "avg_images_10_sample_rate"
    std_path = "std_images_10_sample_rate"
    
    avg_image_path = os.path.join(avg_path, "avg_frame_0400.npy")
    std_image_path = os.path.join(std_path, "std_frame_0400.npy")
    
    #--------------------------------------------
    #undistord images
    
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
    
    avg_image_distored = np.load(avg_image_path)
    
    std_image_distored = np.load(std_image_path)
    
    avg_image =  cv2.undistort(avg_image_distored, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    std_image  = cv2.undistort(std_image_distored, camera_matrix, dist_coeffs, None, new_camera_matrix)
    
    masked_std_img = np.where(avg_mask, std_image, 0)
    
    plt.imsave("std_image.png", std_image)
    plt.imsave("avg_image.png", avg_image)
    
    flame_torus = display_std_top_percentage(masked_std_img, percent=0.7, title="Average Image", visu=True)
    
    #Kiln Images
    #---------------------

    mask_path = "3d_kiln_mask.png"
    torus_path = "torus_400_100.png"
    
    mask = cv2.imread(mask_path)[:,:,0]
    torus = cv2.imread(torus_path)[:,:,0]
    diff = np.abs(torus - mask)
    
    binary_mask = np.zeros_like(diff)
    binary_mask[diff>0] = 255
    
    #-------------------------
    #torus and kiln as mask
    
    
    binary_mask_bool = binary_mask > 0

    final_mask = np.logical_and(avg_mask, binary_mask_bool)

    final_masked_std_img = np.where(final_mask, std_image, 0)
    
    #--------------------------------
    #Concat
    
    concat = np.concat((std_image_distored, std_image, torus), axis=1)
    
    diff = std_image_distored - std_image
    
    plt.imshow(concat)
    plt.show()
    
    

    







