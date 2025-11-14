import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


MASK_CENTER = (720, 480) 
MASK_RADIUS = 250

def test_edge_detection(image, image_name, canny_low=18, canny_high=9):
    """Test different edge detection methods on thermal images"""
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float = gray.astype(np.float32)

    edges_canny = cv2.Canny(gray, threshold1=canny_low, threshold2=canny_high)
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f'Edge Detection Tests - {image_name}', fontsize=16)
    
    # 1. Original grayscale
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Grayscale')
    axes[0, 0].axis('off')
    
    # 2. Canny edges
    edges_canny = cv2.Canny(gray, threshold1=canny_low, threshold2=canny_high)
    axes[0, 1].imshow(edges_canny, cmap='gray')
    axes[0, 1].set_title(f'Canny ({canny_low}, {canny_high})')
    axes[0, 1].axis('off')
    
    # 3. Sobel gradient magnitude
    grad_x = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=5)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    gradient_norm = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    axes[0, 2].imshow(gradient_norm, cmap='hot')
    axes[0, 2].set_title('Sobel Gradient Magnitude')
    axes[0, 2].axis('off')
    
    # 4. Adaptive threshold (Gaussian)
    thresh_adaptive_gauss = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 21, -5
    )
    axes[1, 0].imshow(thresh_adaptive_gauss, cmap='gray')
    axes[1, 0].set_title('Adaptive Threshold (Gaussian)')
    axes[1, 0].axis('off')
    
    # 5. Adaptive threshold (Mean)
    thresh_adaptive_mean = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_MEAN_C, 
        cv2.THRESH_BINARY, 21, -5
    )
    axes[1, 1].imshow(thresh_adaptive_mean, cmap='gray')
    axes[1, 1].set_title('Adaptive Threshold (Mean)')
    axes[1, 1].axis('off')
    
    # 6. Otsu's thresholding
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    axes[1, 2].imshow(thresh_otsu, cmap='gray')
    axes[1, 2].set_title('Otsu Threshold')
    axes[1, 2].axis('off')
    
    # 7. Combined: Gradient + Adaptive
    combined = cv2.addWeighted(gradient_norm, 0.7, thresh_adaptive_gauss, 0.3, 0)
    axes[2, 0].imshow(combined, cmap='hot')
    axes[2, 0].set_title('Combined (70% Grad + 30% Adaptive)')
    axes[2, 0].axis('off')
    
    # 8. Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_norm = cv2.normalize(np.abs(laplacian), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    axes[2, 1].imshow(laplacian_norm, cmap='gray')
    axes[2, 1].set_title('Laplacian')
    axes[2, 1].axis('off')
    
    # 9. With circular mask
    h, w = gray.shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - MASK_CENTER[0])**2 + (Y - MASK_CENTER[1])**2)
    mask = dist_from_center <= MASK_RADIUS
    
    masked_combined = combined.copy()
    masked_combined[~mask] = 0
    
    # Add circle overlay
    display_img = cv2.cvtColor(masked_combined, cv2.COLOR_GRAY2BGR)
    cv2.circle(display_img, MASK_CENTER, MASK_RADIUS, (0, 255, 0), 2)
    
    axes[2, 2].imshow(display_img)
    axes[2, 2].set_title('Combined + Mask (Green Circle)')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    return fig

def interactive_threshold_tuning(image, image_name):
    """Interactive window to tune Canny thresholds"""
    
    def update_canny(val):
        low = cv2.getTrackbarPos('Low Threshold', 'Canny Tuning')
        high = cv2.getTrackbarPos('High Threshold', 'Canny Tuning')
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1=low, threshold2=high)
        
        # Create side-by-side display
        display = np.hstack([gray, edges])
        
        # Add text
        cv2.putText(display, f'Low: {low}, High: {high}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Canny Tuning', display)
    
    cv2.namedWindow('Canny Tuning')
    cv2.createTrackbar('Low Threshold', 'Canny Tuning', 50, 500, update_canny)
    cv2.createTrackbar('High Threshold', 'Canny Tuning', 150, 500, update_canny)
    
    # Initial display
    update_canny(0)
    
    print(f"\nInteractive tuning for {image_name}")
    print("Adjust trackbars to find optimal thresholds")
    print("Press any key to close and continue...")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Get final values
    low = cv2.getTrackbarPos('Low Threshold', 'Canny Tuning')
    high = cv2.getTrackbarPos('High Threshold', 'Canny Tuning')
    
    return low, high


if __name__ == "__main__":

    # avg_thermal_img_path = "avg_image.png"
    # std_thermal_img_path = "std_image.png"

    # # Check if files exist
    # if not os.path.exists(avg_thermal_img_path):
    #     print(f"Error: {avg_thermal_img_path} not found!")
    #     exit(1)
    # if not os.path.exists(std_thermal_img_path):
    #     print(f"Error: {std_thermal_img_path} not found!")
    #     exit(1)

    # avg_img = cv2.imread(avg_thermal_img_path)
    # std_img = cv2.imread(std_thermal_img_path)

    # print("=" * 60)
    # print("THERMAL IMAGE THRESHOLD TESTING TOOL")
    # print("=" * 60)
    
    # # Test avg image
    # print("\n1. Testing AVG image with multiple methods...")
    # fig_avg = test_edge_detection(avg_img, "AVG Image", canny_low=18, canny_high=9)
    # plt.show()
    
    # # Test std image
    # print("\n2. Testing STD image with multiple methods...")
    # fig_std = test_edge_detection(std_img, "STD Image", canny_low=40, canny_high=90)
    # plt.show()
    
    # # Interactive tuning
    # print("\n" + "=" * 60)
    # print("INTERACTIVE THRESHOLD TUNING")
    # print("=" * 60)
    
    # response = input("\nDo you want to interactively tune Canny thresholds? (y/n): ")
    # if response.lower() == 'y':
    #     print("\nTuning AVG image...")
    #     low_avg, high_avg = interactive_threshold_tuning(avg_img, "AVG Image")
        
    #     print("\nTuning STD image...")
    #     low_std, high_std = interactive_threshold_tuning(std_img, "STD Image")
        
    #     print("\n" + "=" * 60)
    #     print("RECOMMENDED THRESHOLDS:")
    #     print("=" * 60)
    #     print(f"AVG Image: Canny({low_avg}, {high_avg})")
    #     print(f"STD Image: Canny({low_std}, {high_std})")
    #     print("\nUse these values in your optimization code!")
    
    kiln_img_path = "Kiln_img.png"

    # Check if files exist
    if not os.path.exists(kiln_img_path):
        print(f"Error: {kiln_img_path} not found!")
        exit(1)

    kiln_img = cv2.imread(kiln_img_path)

    print("=" * 60)
    print("Kiln IMAGE THRESHOLD TESTING TOOL")
    print("=" * 60)
    
    # Test avg image
    print("\n1. Testing image with multiple methods...")
    fig_avg = test_edge_detection(kiln_img, "AVG Image", canny_low=18, canny_high=9)
    plt.show()
    
    # Interactive tuning
    print("\n" + "=" * 60)
    print("INTERACTIVE THRESHOLD TUNING")
    print("=" * 60)
    
    response = input("\nDo you want to interactively tune Canny thresholds? (y/n): ")
    if response.lower() == 'y':
        print("\nTuning kiln image...")
        low_avg, high_avg = interactive_threshold_tuning(kiln_img, "AVG Image")
        
        
        print("\n" + "=" * 60)
        print("RECOMMENDED THRESHOLDS:")
        print("=" * 60)
        print(f"Kiln Image: Canny({low_avg}, {high_avg})")
        print("\nUse these values in your optimization code!")