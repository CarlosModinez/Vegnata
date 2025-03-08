#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced Vegetation Analysis with Advanced Visualization
Version: 2.0
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import os
from datetime import datetime

# Enhanced Configuration Parameters
CONFIG = {
    'standard_image_size': (720, 720),
    'vegetation_detection': {
        'hue_range': (30, 90),
        'sat_range': (20, 255),
        'val_range': (20, 255),
        'contrast_enhancement': 1.5
    },
    'clustering': {
        'num_clusters': 5,
        'min_area': 100,
        'smoothing_kernel': 5
    },
    'difference_detection': {
        'window_size': 15,
        'threshold': 0.3,
        'min_diff_area': 50,
        'smoothing': 3
    },
    'visualization': {
        'overlay_alpha': 0.5,
        'boundary_thickness': 2,
        'text_scale': 0.7,
        'border': {
            'color': (0, 255, 255),  # Yellow color for main border
            'thickness': 3,           # Border thickness
            'alpha': 0.8             # Border transparency
        },
        'colors': {
            'healthy': (0, 100, 0),      # Dark green
            'medium': (0, 180, 0),       # Medium green
            'light': (180, 255, 100),    # Light green
            'stressed': (200, 200, 150),  # Very light green
            'unplantable': (180, 180, 180), # Gray
            'difference': (200, 100, 255)   # Purple
        }
    }
}
CONFIG['visualization'].update({
    'cluster_borders': {
            'main_color': (0, 255, 255),     # Bright yellow
            'secondary_color': (255, 200, 0), # Golden yellow
            'thickness': {
                        'main': 4,        # Main polygon border
                        'cluster': 2,     # Individual cluster borders
                        'highlight': 3    # Highlight effects
                    },
            'effects': {
                        'double_line': True,    # Draw double-line borders
                        'glow': True,           # Add glow effect
                        'alpha': 0.85           # Border transparency
                    }
        }
})

def enhance_contrast(image, factor=1.5):
    """Enhance image contrast"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=factor, tileGridSize=(8,8))
    l = clahe.apply(l)
    enhanced = cv2.merge((l,a,b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def preprocess_image(image):
    """Enhanced preprocessing with contrast adjustment"""
    enhanced = enhance_contrast(image, CONFIG['vegetation_detection']['contrast_enhancement'])
    hsv_image = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
    
    lower_thresh = np.array([
        CONFIG['vegetation_detection']['hue_range'][0],
        CONFIG['vegetation_detection']['sat_range'][0],
        CONFIG['vegetation_detection']['val_range'][0]
    ])
    upper_thresh = np.array([
        CONFIG['vegetation_detection']['hue_range'][1],
        CONFIG['vegetation_detection']['sat_range'][1],
        CONFIG['vegetation_detection']['val_range'][1]
    ])
    
    field_mask = cv2.inRange(hsv_image, lower_thresh, upper_thresh)
    kernel = np.ones((CONFIG['clustering']['smoothing_kernel'], 
                     CONFIG['clustering']['smoothing_kernel']), np.uint8)
    field_mask = cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel)
    
    return enhanced, field_mask, hsv_image

'''
[continued in next part due to length...]

Would you like me to continue with the rest of the code? The remaining sections include:

1. Clustering and Analysis Functions
2. Visualization Functions
3. Report Generation
4. Main Execution Flow
[Previous code remains the same, continuing with new sections...]
'''
def cluster_vegetation(hsv_image, field_mask):
    """Perform clustering on the vegetation"""
    valid_mask = field_mask > 0
    valid_pixels = hsv_image[valid_mask]
    
    features = np.column_stack((
        valid_pixels[:, 0] * 0.5,
        valid_pixels[:, 1] * 1.5,
        valid_pixels[:, 2] * 2.0
    ))
    
    features = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0))
    
    kmeans = KMeans(n_clusters=CONFIG['clustering']['num_clusters'], random_state=42)
    labels = kmeans.fit_predict(features)
    
    cluster_map = np.zeros(field_mask.shape, dtype=int)
    cluster_map[valid_mask] = labels
    
    centers = kmeans.cluster_centers_
    vegetation_scores = centers[:, 1] - centers[:, 2] * 0.5
    sorted_indices = np.argsort(vegetation_scores)[::-1]
    
    remapped_cluster_map = np.zeros_like(cluster_map)
    for new_idx, old_idx in enumerate(sorted_indices):
        remapped_cluster_map[cluster_map == old_idx] = new_idx
    
    return remapped_cluster_map

def calculate_area_metrics(mask, pixel_size=0.000247105):
    """Calculate area metrics from mask"""
    total_pixels = np.sum(mask > 0)
    area_acres = total_pixels * pixel_size
    return {
        'pixel_count': total_pixels,
        'acres': area_acres
    }

def detect_anomalies(cluster_map, hsv_image, polygon_mask):
    """Enhanced anomaly detection"""
    diff_mask = np.zeros_like(cluster_map, dtype=np.uint8)
    window_size = CONFIG['difference_detection']['window_size']
    threshold = CONFIG['difference_detection']['threshold']
    
    gradient_x = cv2.Sobel(hsv_image[:,:,1], cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(hsv_image[:,:,1], cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_mag = cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX)
    
    valid_area = polygon_mask > 0
    pad = window_size // 2
    padded_hsv = np.pad(hsv_image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
    padded_gradient = np.pad(gradient_mag, pad, mode='edge')
    
    h, w = cluster_map.shape
    for i in range(h):
        for j in range(w):
            if valid_area[i, j]:
                neighborhood_hsv = padded_hsv[i:i+window_size, j:j+window_size]
                neighborhood_gradient = padded_gradient[i:i+window_size, j:j+window_size]
                
                center_hsv = neighborhood_hsv[window_size//2, window_size//2]
                mean_hsv = np.mean(neighborhood_hsv, axis=(0,1))
                gradient_score = neighborhood_gradient[window_size//2, window_size//2]
                
                hsv_diff = np.abs(center_hsv - mean_hsv) / [180, 255, 255]
                diff_score = np.mean(hsv_diff) + gradient_score/255
                
                if diff_score > threshold:
                    diff_mask[i, j] = 255
    
    kernel = np.ones((CONFIG['difference_detection']['smoothing'],
                     CONFIG['difference_detection']['smoothing']), np.uint8)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_OPEN, kernel)
    diff_mask = cv2.morphologyEx(diff_mask, cv2.MORPH_CLOSE, kernel)
    
    return diff_mask

def analyze_vegetation_health(cluster_map, hsv_image, mask):
    """Enhanced vegetation health analysis with more metrics"""
    health_metrics = {}
    valid_mask = mask > 0
    valid_hsv = hsv_image[valid_mask]
    
    normalized_health = (valid_hsv[:,1] * valid_hsv[:,0]) / (255 * 180)
    
    health_metrics['mean_health'] = np.mean(normalized_health)
    health_metrics['health_std'] = np.std(normalized_health)
    health_metrics['health_percentiles'] = np.percentile(normalized_health, [10, 25, 50, 75, 90])
    health_metrics['health_min'] = np.min(normalized_health)
    health_metrics['health_max'] = np.max(normalized_health)
    
    density = np.sum(valid_hsv[:,1] > 50) / valid_hsv.shape[0]
    health_metrics['density'] = density
    
    uniformity = 1 - (health_metrics['health_std'] / health_metrics['mean_health'])
    health_metrics['uniformity'] = max(0, min(1, uniformity))
    
    return health_metrics

def draw_cluster_boundaries(image, cluster_map, contour_mask):
    """Draw boundaries around clusters with enhanced border visualization using a mask."""
    result_image = image.copy()
    overlay = image.copy()
    
    colors = list(CONFIG['visualization']['colors'].values())
    border_config = CONFIG['visualization']['cluster_borders']
    
    glow_layer = np.zeros_like(image, dtype=np.float32)
    
    # Draw clusters with enhanced borders
    for cluster_idx in range(CONFIG['clustering']['num_clusters']):
        cluster_mask = np.zeros_like(cluster_map, dtype=np.uint8)
        # Use the contour_mask to isolate cluster areas
        cluster_mask[(cluster_map == cluster_idx) & (contour_mask > 0)] = 255
        
        # Smooth and find contours in the cluster mask
        kernel = np.ones((3, 3), np.uint8)
        cluster_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Fill clusters and apply color from the configuration
        overlay[cluster_mask > 0] = colors[cluster_idx]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > CONFIG['clustering']['min_area']:
                # Draw filled cluster area
                cv2.drawContours(result_image, [contour], -1, colors[cluster_idx], -1)
                
                # Apply glow effect if enabled
                if border_config['effects']['glow']:
                    glow_mask = np.zeros_like(cluster_mask)
                    cv2.drawContours(glow_mask, [contour], -1, 255, border_config['thickness']['highlight'])
                    blur_radius = 5
                    glow_mask = cv2.GaussianBlur(glow_mask, (blur_radius*2+1, blur_radius*2+1), 0)
                    glow_layer[glow_mask > 0] += border_config['main_color']
                
                # Draw cluster borders based on config
                if border_config['effects']['double_line']:
                    cv2.drawContours(result_image, [contour], -1, border_config['main_color'], border_config['thickness']['cluster'])
                    cv2.drawContours(result_image, [contour], -1, border_config['secondary_color'], max(1, border_config['thickness']['cluster'] - 1))
                else:
                    cv2.drawContours(result_image, [contour], -1, border_config['main_color'], border_config['thickness']['cluster'])
    
    # Overlay clusters with transparency
    cv2.addWeighted(overlay, CONFIG['visualization']['overlay_alpha'], result_image, 1 - CONFIG['visualization']['overlay_alpha'], 0, result_image)
    
    # Apply glow effect
    if border_config['effects']['glow']:
        glow_layer = cv2.normalize(glow_layer, None, 0, 1, cv2.NORM_MINMAX)
        result_image = cv2.addWeighted(result_image, 1, glow_layer.astype(np.uint8), border_config['effects']['alpha'], 0)
    
    return result_image


def analyze_clusters(image, cluster_map, mask, hsv_image):
    """Analyze clusters and detect differences"""

    # Detect anomalies
    diff_mask = detect_anomalies(cluster_map, hsv_image, mask)
    
    # Draw clusters and borders
    result_image = draw_cluster_boundaries(image, cluster_map, mask)
    
    cluster_labels = [
        'Healthy (Dark Green)',
        'Medium Health',
        'Light Green',
        'Stressed (Very Light)',
        'Unplantable'
    ]
    
    areas = []
    for cluster_idx in range(CONFIG['clustering']['num_clusters']):
        cluster_mask = (cluster_map == cluster_idx) & (mask > 0)
        metrics = calculate_area_metrics(cluster_mask.astype(np.uint8))
        areas.append((cluster_labels[cluster_idx], metrics['pixel_count'], metrics['acres']))
    
    # Process difference areas
    diff_metrics = calculate_area_metrics(diff_mask)
    
    # Highlight differences
    diff_contours, _ = cv2.findContours(diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_diff_contours = [cnt for cnt in diff_contours 
                               if cv2.contourArea(cnt) > CONFIG['difference_detection']['min_diff_area']]
    cv2.drawContours(result_image, significant_diff_contours, -1,
                    CONFIG['visualization']['colors']['difference'], 2)
    
    return result_image, areas, (diff_metrics['pixel_count'], diff_metrics['acres'])
def draw_enhanced_visualization(result_image, areas, diff_area, health_metrics):
    """Create enhanced visualization with more detailed metrics and improved styling"""
    h, w = result_image.shape[:2]
    
    # Enhanced panel parameters
    panel_width = 350
    panel_start = 10
    panel_color = (0, 0, 0)
    panel_alpha = 0.85
    
    # Create panel overlay with correct broadcasting
    overlay = result_image.copy()
    gradient = np.linspace(0, 1, h-2*panel_start)
    panel = np.zeros((h-2*panel_start, panel_width, 3), dtype=np.uint8)
    
    # Properly broadcast gradient to all color channels
    for i in range(3):
        panel[:, :, i] = (panel_color[i] * (0.7 + 0.3 * gradient[:, np.newaxis])).astype(np.uint8)
    
    # Apply panel to overlay
    overlay[panel_start:h-panel_start, panel_start:panel_start+panel_width] = panel
    cv2.addWeighted(overlay, panel_alpha, result_image, 1 - panel_alpha, 0, result_image)
    
    # Initialize text position
    y_pos = 40
    line_height = 25
    text_color = (255, 255, 255)
    
    # Add title with enhanced styling
    cv2.putText(result_image, "Vegetation Analysis Report", 
                (panel_start + 10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    y_pos += line_height * 2
    
    # Add separator line
    cv2.line(result_image, 
             (int(panel_start + 10), int(y_pos)), 
             (int(panel_start + panel_width - 10), int(y_pos)),
             text_color, 1)
    y_pos += line_height
    
    # Calculate total areas
    total_acres = sum(acres for _, _, acres in areas)
    plantable_acres = sum(acres for label, _, acres in areas if 'Unplantable' not in label)
    
    # Area Analysis Section
    cv2.putText(result_image, "Area Analysis:", 
                (panel_start + 10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    y_pos += line_height
    
    for label, _, acres in areas:
        percentage = (acres / total_acres) * 100
        text = f"{label}: {acres:.1f}ac ({percentage:.1f}%)"
        cv2.putText(result_image, text,
                   (panel_start + 20, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        y_pos += line_height
    
    # Add separator
    y_pos += line_height//2
    cv2.line(result_image, 
             (int(panel_start + 10), int(y_pos)), 
             (int(panel_start + panel_width - 10), int(y_pos)),
             text_color, 1)
    y_pos += line_height
    
    # Health Metrics Section
    cv2.putText(result_image, "Health Metrics:", 
                (panel_start + 10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    y_pos += line_height
    
    # Add health index bar
    health_percentage = health_metrics['mean_health'] * 100
    bar_width = panel_width - 40
    bar_height = 15
    
    # Draw background bar
    cv2.rectangle(result_image, 
                 (int(panel_start + 20), int(y_pos)), 
                 (int(panel_start + 20 + bar_width), int(y_pos + bar_height)),
                 (150, 150, 150), 1)
    
    # Draw filled portion of health bar
    filled_width = int(bar_width * health_metrics['mean_health'])
    cv2.rectangle(result_image,
                 (int(panel_start + 20), int(y_pos)),
                 (int(panel_start + 20 + filled_width), int(y_pos + bar_height)),
                 (0, 255, 0), -1)
    y_pos += bar_height + 5
    
    cv2.putText(result_image, f"Health Index: {health_percentage:.1f}%",
                (panel_start + 20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += line_height
    
    # Add uniformity bar
    uniformity_percentage = health_metrics['uniformity'] * 100
    cv2.rectangle(result_image, 
                 (int(panel_start + 20), int(y_pos)), 
                 (int(panel_start + 20 + bar_width), int(y_pos + bar_height)),
                 (150, 150, 150), 1)
    
    filled_width = int(bar_width * health_metrics['uniformity'])
    cv2.rectangle(result_image,
                 (int(panel_start + 20), int(y_pos)),
                 (int(panel_start + 20 + filled_width), int(y_pos + bar_height)),
                 (255, 200, 0), -1)
    y_pos += bar_height + 5
    
    cv2.putText(result_image, f"Uniformity: {uniformity_percentage:.1f}%",
                (panel_start + 20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += line_height
    
    # Add density information
    density_percentage = health_metrics['density'] * 100
    cv2.putText(result_image, f"Vegetation Density: {density_percentage:.1f}%",
                (panel_start + 20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    y_pos += line_height
    
    # Add separator
    y_pos += line_height//2
    cv2.line(result_image, 
             (int(panel_start + 10), int(y_pos)), 
             (int(panel_start + panel_width - 10), int(y_pos)),
             text_color, 1)
    y_pos += line_height
    
    # Difference Analysis Section
    cv2.putText(result_image, "Difference Analysis:", 
                (panel_start + 10, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    y_pos += line_height
    
    diff_percentage = (diff_area[1] / total_acres) * 100
    cv2.putText(result_image, f"Anomaly Area: {diff_area[1]:.1f}ac ({diff_percentage:.1f}%)",
                (panel_start + 20, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    return result_image
#[Previous visualization and report generation functions remain the same]
def generate_enhanced_report(areas, diff_area, health_metrics, output_path):
    """Generate enhanced analysis report with more detailed statistics"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_path, f"detailed_analysis_{timestamp}.txt")

    with open(report_path, 'w') as f:
        f.write("Detailed Vegetation Analysis Report\n")
        f.write("================================\n\n")

        # Area Analysis
        f.write("Area Analysis\n")
        f.write("-------------\n")
        total_acres = sum(acres for _, _, acres in areas)
        for label, pixels, acres in areas:
            percentage = (acres / total_acres) * 100
            f.write(f"\n{label}:\n")
            f.write(f"  Area: {acres:.2f} acres\n")
            f.write(f"  Coverage: {percentage:.1f}%\n")
            f.write(f"  Pixel Count: {pixels:,}\n")

        # Health Analysis
        f.write("\nVegetation Health Analysis\n")
        f.write("--------------------------\n")
        f.write(f"Mean Health Index: {health_metrics['mean_health']:.3f}\n")
        f.write(f"Health Variation: {health_metrics['health_std']:.3f}\n")
        f.write(f"Vegetation Density: {health_metrics['density']*100:.1f}%\n")
        f.write(f"Uniformity Score: {health_metrics['uniformity']*100:.1f}%\n")

        f.write("\nHealth Percentiles:\n")
        percentile_labels = ['10th', '25th', '50th', '75th', '90th']
        for label, value in zip(percentile_labels, health_metrics['health_percentiles']):
            f.write(f"  {label}: {value:.3f}\n")

        # Difference Analysis
        f.write("\nDifference Analysis\n")
        f.write("-------------------\n")
        diff_percentage = (diff_area[1] / total_acres) * 100
        f.write(f"Anomaly Area: {diff_area[1]:.2f} acres ({diff_percentage:.1f}%)\n")
        f.write(f"Anomaly Pixel Count: {diff_area[0]:,}\n")

        # Summary Statistics
        f.write("\nSummary Statistics\n")
        f.write("-----------------\n")
        plantable_acres = sum(acres for label, _, acres in areas if 'Unplantable' not in label)
        f.write(f"Total Area: {total_acres:.2f} acres\n")
        f.write(f"Plantable Area: {plantable_acres:.2f} acres\n")
        f.write(f"Planting Efficiency: {(plantable_acres/total_acres)*100:.1f}%\n")

def save_analysis_outputs(image, cluster_map, mask, diff_mask, result_image, output_dir):
    """Save all analysis outputs"""
    os.makedirs(output_dir, exist_ok=True)

    # Save masks and visualizations
    cv2.imwrite(os.path.join(output_dir, 'original.png'),
                cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(output_dir, 'cluster_mask.png'),
                cluster_map.astype(np.uint8) * 50)
    cv2.imwrite(os.path.join(output_dir, 'area_mask.png'), mask)
    cv2.imwrite(os.path.join(output_dir, 'difference_mask.png'), diff_mask)
    cv2.imwrite(os.path.join(output_dir, 'final_analysis.png'),
                cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))

    # Create masked versions
    for cluster_idx in range(CONFIG['clustering']['num_clusters']):
        cluster_mask = np.zeros_like(cluster_map, dtype=np.uint8)
        cluster_mask[cluster_map == cluster_idx] = 255
        cluster_only = cv2.bitwise_and(image, image, mask=cluster_mask)
        cv2.imwrite(os.path.join(output_dir, f'cluster_{cluster_idx}.png'),
                   cv2.cvtColor(cluster_only, cv2.COLOR_RGB2BGR))

def resize_image(image, size=CONFIG['standard_image_size']):
    """Resizes the image to a standard size for processing."""
    return cv2.resize(image, size)


def preprocess_image_with_saturation(image, saturation_factor=1.5):
    """Applies saturation adjustment and thresholding to create a binary mask."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[..., 1] = np.clip(hsv_image[..., 1] * saturation_factor, 0, 255)
    mask = cv2.inRange(hsv_image, np.array([30, 20, 20]), np.array([90, 255, 255]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return np.ones_like(mask) if np.array_equal(np.unique(mask), [0]) else mask


def get_exg(image: np.ndarray) -> np.ndarray:
    """Computes the Excess Green (ExG) index to enhance green vegetation in the image."""
    R, G, B = image[:,:,0]/np.max(image[:,:,0]), image[:,:,1]/np.max(image[:,:,1]), image[:,:,2]/np.max(image[:,:,2])
    exg_image = (2*G - R - G)*-1
    exg_image = ((exg_image + np.abs(exg_image.min())) / (np.abs(exg_image.min()) + exg_image.max()) * 255).astype(np.uint8)
    return exg_image


def get_cive(image: np.ndarray) -> np.ndarray:
    """Calculates the CIVE (Color Index of Vegetation Extraction) to highlight green vegetation areas."""
    R, G, B = image[:,:,0]/np.max(image[:,:,0]), image[:,:,1]/np.max(image[:,:,1]), image[:,:,2]/np.max(image[:,:,2])
    cive_img = (0.441 * R) - (0.811 * G) + (0.385 * B)
    cive_img = ((cive_img + np.abs(cive_img.min())) / (np.abs(cive_img.min()) + cive_img.max()) * 255).astype(np.uint8)
    return cive_img


def create_hyperspectral_image(image):
    """Creates a hyperspectral image with CIVE, excess green, and Gabor filter."""
    blur_kernel_size = 9
    image = cv2.bilateralFilter(image, blur_kernel_size, 75, 75)
    cive = cv2.bilateralFilter(get_cive(image), blur_kernel_size, 75, 75)
    exg = cv2.bilateralFilter(get_exg(image), blur_kernel_size, 75, 75)
    
    gabor_kernel = cv2.getGaborKernel((5, 5), sigma=1.0, theta=np.pi/4, lambd=10.0, gamma=0.5)
    gabor_texture = cv2.filter2D(image, cv2.CV_8UC3, gabor_kernel)
    hyperspectral_image = np.dstack((image, cive, exg, gabor_texture[:, :, 0]))

    return hyperspectral_image


def cluster_unplanted_area(image, mask, n_clusters=2, similarity_threshold=50):
    """Clusters the unplanted area and creates a binary mask based on the minority class.
       Ignores clustering if the clusters are too similar."""
       
    pixel_values = image[mask > 0].reshape((-1, image.shape[-1])).astype(np.float32)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixel_values)

    cluster_centers = kmeans.cluster_centers_
    if euclidean(cluster_centers[0], cluster_centers[1]) < similarity_threshold:
        print("Clusters are too similar; returning an empty mask.")
        return np.zeros(image.shape[:2], dtype=np.uint8)  # Return an empty mask if clusters are similar

    segmented_image = np.zeros(image.shape[:2], dtype=np.uint8)
    segmented_image[mask > 0] = (labels + 1)

    class_1_count = np.sum(segmented_image == 1)
    class_2_count = np.sum(segmented_image == 2)

    if class_1_count > class_2_count:
        segmented_image[segmented_image == 1] = 255
        segmented_image[segmented_image == 2] = 0
    else:
        segmented_image[segmented_image == 2] = 255
        segmented_image[segmented_image == 1] = 0

    _, binary_mask = cv2.threshold(segmented_image, 127, 255, cv2.THRESH_BINARY)
    
    return binary_mask


def find_contours_and_create_mask(binary_image):
    """Finds and filters contours, creating a mask with filled contours based on minimum area threshold."""
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(binary_image) 

    image_area = binary_image.shape[0] * binary_image.shape[1]
    min_contour_area = 0.01 * image_area ## exclude contours with area lower than 1% of image area

    for contour in contours:
        if cv2.contourArea(contour) >= min_contour_area:
            cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)

    contour_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_ERODE, np.ones((15, 15), dtype=np.uint8))
    contour_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_CLOSE, np.ones((11, 11), dtype=np.uint8))
    contour_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_DILATE, np.ones((11, 11), dtype=np.uint8))

    return np.ones_like(contour_mask) if np.array_equal(np.unique(contour_mask), [0]) else contour_mask


def apply_mask_to_hyperspectral(hyperspectral_image, mask):
    """Applies a binary mask to a hyperspectral image."""
    masked_hyperspectral = np.zeros_like(hyperspectral_image)
    for i in range(hyperspectral_image.shape[2]):
        masked_hyperspectral[:, :, i] = cv2.bitwise_and(hyperspectral_image[:, :, i], hyperspectral_image[:, :, i], mask=mask)
    return masked_hyperspectral


def separete_planted_and_unplanted_area(image_path):
    """Separates planted and unplanted areas, returning the masked vegetation image and contour mask."""
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    original_size = original_image.shape[:2][::-1]
    image = resize_image(original_image)
        
    mask = preprocess_image_with_saturation(image)
    hyperspectral_image = create_hyperspectral_image(image)
    binary_mask = cluster_unplanted_area(hyperspectral_image, mask)
    contour_mask = find_contours_and_create_mask(binary_mask)
    contour_mask = cv2.resize(contour_mask, original_size, interpolation=cv2.INTER_NEAREST)

    only_vegetation_image = apply_mask_to_hyperspectral(original_image, contour_mask)

    return only_vegetation_image, contour_mask


def main():
    try:
        image_path = "images/train/sample_019.png"  # Update with your image path
        output_dir = "cluster_images"
        os.makedirs(output_dir, exist_ok=True)
        
        print("Getting planted area...")
        image, mask = separete_planted_and_unplanted_area(image_path)

        print("Processing image...")
        image, _, hsv_image = preprocess_image(image)
        
        print("Performing vegetation analysis...")
        cluster_map = cluster_vegetation(hsv_image, mask)
            
        # Perform analysis with enhanced visualization
        result_image, areas, diff_area = analyze_clusters(image, cluster_map, mask, hsv_image)
        health_metrics = analyze_vegetation_health(cluster_map, hsv_image, mask)
        
        # Generate visualizations and report
        result_image = draw_enhanced_visualization(result_image, areas, diff_area, health_metrics)
        generate_enhanced_report(areas, diff_area, health_metrics, output_dir)
        
        # Save outputs
        diff_mask = detect_anomalies(cluster_map, hsv_image, mask)
        save_analysis_outputs(image, cluster_map, mask, diff_mask, result_image, output_dir)
            
        # Display results
        cv2.namedWindow('Vegetation Analysis', cv2.WINDOW_NORMAL)
        cv2.imshow('Vegetation Analysis', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise e

if __name__ == "__main__":
    main()


