import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
from typing import Tuple, List, Dict
import json
from datetime import datetime

class BeachSedimentAnalyzer:
    def __init__(self, aruco_size_mm=5.0):
        """
        Initialize the Beach Sediment Analyzer
        
        Args:
            aruco_size_mm (float): Size of the ArUco marker in millimeters (default: 5.0mm for 5x5mm marker)
        """
        self.aruco_size_mm = aruco_size_mm
        self.scale_mm_per_pixel = None
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        
        # Wentworth grain size classification (in mm)
        self.wentworth_classes = {
            'Boulder': (256, float('inf')),
            'Cobble': (64, 256),
            'Very_Coarse_Gravel': (32, 64),
            'Coarse_Gravel': (16, 32),
            'Medium_Gravel': (8, 16),
            'Fine_Gravel': (4, 8),
            'Very_Fine_Gravel': (2, 4),
            'Very_Coarse_Sand': (1, 2),
            'Coarse_Sand': (0.5, 1),
            'Medium_Sand': (0.25, 0.5),
            'Fine_Sand': (0.125, 0.25),
            'Very_Fine_Sand': (0.0625, 0.125),
            'Coarse_Silt': (0.031, 0.0625),
            'Medium_Silt': (0.016, 0.031),
            'Fine_Silt': (0.008, 0.016),
            'Very_Fine_Silt': (0.004, 0.008),
            'Clay': (0, 0.004)
        }
    
    def detect_aruco_and_calibrate(self, image: np.ndarray) -> Tuple[bool, float, np.ndarray]:
        """
        Detect ArUco marker and calculate scale calibration
        
        Args:
            image: Input image
            
        Returns:
            success: Whether ArUco marker was detected
            scale: Scale in mm per pixel
            annotated_image: Image with ArUco marker annotated
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if len(corners) > 0:
            # Use the first detected marker
            marker_corners = corners[0][0]
            
            # Calculate marker size in pixels (average of all sides)
            side_lengths = []
            for i in range(4):
                p1 = marker_corners[i]
                p2 = marker_corners[(i + 1) % 4]
                side_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                side_lengths.append(side_length)
            
            avg_marker_size_pixels = np.mean(side_lengths)
            scale = self.aruco_size_mm / avg_marker_size_pixels
            self.scale_mm_per_pixel = scale
            
            # Annotate image
            annotated = image.copy()
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            
            # Add scale information
            text = f"Scale: {scale:.6f} mm/px"
            cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            return True, scale, annotated
        
        return False, 0.0, image
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for grain detection
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        return binary
    
    def segment_grains_watershed(self, binary: np.ndarray) -> np.ndarray:
        """
        Advanced grain segmentation using watershed algorithm
        """
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        
        # Find local maxima (seeds for watershed)
        local_maxima = ndimage.maximum_filter(dist_transform, size=20) == dist_transform
        local_maxima = local_maxima & (dist_transform > 0.3 * dist_transform.max())
        
        # Label seeds
        markers = ndimage.label(local_maxima)[0]
        
        # Apply watershed
        labels = cv2.watershed(cv2.cvtColor(opening, cv2.COLOR_GRAY2BGR), markers)
        
        return labels
    
    def extract_grain_features(self, labels: np.ndarray, min_area_pixels: int = 20) -> pd.DataFrame:
        """
        Extract grain features from segmented image
        """
        features = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            if label <= 0:  # Skip background and watershed lines
                continue
                
            # Create mask for current grain
            mask = (labels == label).astype(np.uint8)
            area_pixels = cv2.countNonZero(mask)
            
            if area_pixels < min_area_pixels:
                continue
            
            # Find contour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
                
            contour = max(contours, key=cv2.contourArea)
            
            # Calculate features
            area_pixels = cv2.contourArea(contour)
            equivalent_diameter_pixels = 2 * np.sqrt(area_pixels / np.pi)
            
            # Fit ellipse if possible
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                major_axis_pixels = max(ellipse[1])
                minor_axis_pixels = min(ellipse[1])
                aspect_ratio = major_axis_pixels / minor_axis_pixels
            else:
                major_axis_pixels = equivalent_diameter_pixels
                minor_axis_pixels = equivalent_diameter_pixels
                aspect_ratio = 1.0
            
            # Bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            features.append({
                'grain_id': int(label),
                'area_pixels': area_pixels,
                'equivalent_diameter_pixels': equivalent_diameter_pixels,
                'major_axis_pixels': major_axis_pixels,
                'minor_axis_pixels': minor_axis_pixels,
                'aspect_ratio': aspect_ratio,
                'centroid_x': x + w/2,
                'centroid_y': y + h/2,
                'bbox_x': x,
                'bbox_y': y,
                'bbox_w': w,
                'bbox_h': h
            })
        
        return pd.DataFrame(features)
    
    def convert_to_real_world_units(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert pixel measurements to real-world units (mm)
        """
        if self.scale_mm_per_pixel is None:
            raise ValueError("Scale not calibrated. Run detect_aruco_and_calibrate first.")
        
        df_converted = df.copy()
        
        # Convert measurements to mm
        pixel_columns = ['area_pixels', 'equivalent_diameter_pixels', 'major_axis_pixels', 
                        'minor_axis_pixels', 'centroid_x', 'centroid_y', 'bbox_x', 'bbox_y', 
                        'bbox_w', 'bbox_h']
        
        for col in pixel_columns:
            if col == 'area_pixels':
                df_converted[col.replace('_pixels', '_mm2')] = df[col] * (self.scale_mm_per_pixel ** 2)
            else:
                df_converted[col.replace('_pixels', '_mm')] = df[col] * self.scale_mm_per_pixel
        
        # Calculate phi values
        df_converted['phi'] = -np.log2(df_converted['equivalent_diameter_mm'])
        
        return df_converted
    
    def classify_grains(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classify grains according to Wentworth scale
        """
        df_classified = df.copy()
        classifications = []
        
        for diameter_mm in df['equivalent_diameter_mm']:
            classification = 'Unclassified'
            for class_name, (min_size, max_size) in self.wentworth_classes.items():
                if min_size <= diameter_mm < max_size:
                    classification = class_name
                    break
            classifications.append(classification)
        
        df_classified['classification'] = classifications
        return df_classified
    
    def compute_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute sediment statistics
        """
        diameters_mm = df['equivalent_diameter_mm'].values
        phi_values = df['phi'].values
        
        stats = {
            'total_grains': len(df),
            'diameter_mm_mean': np.mean(diameters_mm),
            'diameter_mm_median': np.median(diameters_mm),  # D50
            'diameter_mm_std': np.std(diameters_mm),
            'diameter_mm_min': np.min(diameters_mm),
            'diameter_mm_max': np.max(diameters_mm),
            'phi_mean': np.mean(phi_values),
            'phi_median': np.median(phi_values),
            'phi_std': np.std(phi_values),  # Sorting coefficient
            'D10': np.percentile(diameters_mm, 10),
            'D50': np.percentile(diameters_mm, 50),
            'D90': np.percentile(diameters_mm, 90)
        }
        
        # Calculate uniformity coefficient and curvature coefficient
        stats['uniformity_coefficient'] = stats['D60'] / stats['D10'] if 'D60' in stats else None
        stats['D60'] = np.percentile(diameters_mm, 60)
        stats['D30'] = np.percentile(diameters_mm, 30)
        stats['uniformity_coefficient'] = stats['D60'] / stats['D10']
        stats['curvature_coefficient'] = (stats['D30'] ** 2) / (stats['D10'] * stats['D60'])
        
        # Classification distribution
        class_counts = df['classification'].value_counts()
        stats['classification_distribution'] = class_counts.to_dict()
        
        # Dominant sediment type
        stats['dominant_type'] = class_counts.index[0] if len(class_counts) > 0 else 'Unknown'
        
        return stats
    
    def create_visualizations(self, df: pd.DataFrame, stats: Dict, output_dir: str):
        """
        Create visualization plots
        """
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Grain size distribution histogram
        axes[0, 0].hist(df['equivalent_diameter_mm'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(stats['D50'], color='red', linestyle='--', label=f'D50: {stats["D50"]:.3f} mm')
        axes[0, 0].set_xlabel('Grain Diameter (mm)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Grain Size Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Phi scale distribution
        axes[0, 1].hist(df['phi'], bins=30, alpha=0.7, edgecolor='black', color='orange')
        axes[0, 1].axvline(stats['phi_median'], color='red', linestyle='--', 
                          label=f'Median φ: {stats["phi_median"]:.2f}')
        axes[0, 1].set_xlabel('Phi (φ)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Phi Scale Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative size distribution
        sorted_diameters = np.sort(df['equivalent_diameter_mm'])
        cumulative_percent = np.arange(1, len(sorted_diameters) + 1) / len(sorted_diameters) * 100
        axes[1, 0].semilogx(sorted_diameters, cumulative_percent)
        axes[1, 0].set_xlabel('Grain Diameter (mm)')
        axes[1, 0].set_ylabel('Cumulative Percent Finer (%)')
        axes[1, 0].set_title('Cumulative Size Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Classification pie chart
        class_counts = stats['classification_distribution']
        if class_counts:
            axes[1, 1].pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%')
            axes[1, 1].set_title('Sediment Classification Distribution')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'grain_analysis_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def annotate_image(self, original_image: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """
        Annotate original image with detected grains
        """
        annotated = original_image.copy()
        
        for _, grain in df.iterrows():
            # Draw bounding box
            x, y, w, h = int(grain['bbox_x']), int(grain['bbox_y']), int(grain['bbox_w']), int(grain['bbox_h'])
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 1)
            
            # Add grain ID and size
            if 'equivalent_diameter_mm' in grain:
                text = f"{grain['grain_id']}: {grain['equivalent_diameter_mm']:.2f}mm"
            else:
                text = f"{grain['grain_id']}"
            
            cv2.putText(annotated, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
        
        return annotated
    
    def process_image(self, image_path: str, output_dir: str, gps_coords: Tuple[float, float] = None) -> Dict:
        """
        Complete processing pipeline for a single image
        
        Args:
            image_path: Path to input image
            output_dir: Directory for output files
            gps_coords: Optional GPS coordinates (latitude, longitude)
            
        Returns:
            Dictionary containing analysis results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Detect ArUco marker and calibrate
        aruco_success, scale, aruco_annotated = self.detect_aruco_and_calibrate(image)
        
        if not aruco_success:
            print("Warning: ArUco marker not detected. Analysis will be in pixels only.")
        
        cv2.imwrite(os.path.join(output_dir, 'aruco_detection.jpg'), aruco_annotated)
        
        # Step 2: Preprocess image
        binary = self.preprocess_image(image)
        cv2.imwrite(os.path.join(output_dir, 'preprocessed.jpg'), binary)
        
        # Step 3: Segment grains
        labels = self.segment_grains_watershed(binary)
        
        # Step 4: Extract features
        df_pixels = self.extract_grain_features(labels)
        
        if len(df_pixels) == 0:
            return {"error": "No grains detected"}
        
        # Step 5: Convert to real-world units if calibrated
        if aruco_success:
            df_processed = self.convert_to_real_world_units(df_pixels)
            df_classified = self.classify_grains(df_processed)
            
            # Compute statistics
            stats = self.compute_statistics(df_classified)
            
            # Create visualizations
            self.create_visualizations(df_classified, stats, output_dir)
            
        else:
            df_classified = df_pixels
            stats = {
                'total_grains': len(df_pixels),
                'diameter_pixels_mean': np.mean(df_pixels['equivalent_diameter_pixels']),
                'diameter_pixels_median': np.median(df_pixels['equivalent_diameter_pixels']),
                'note': 'Measurements in pixels only - ArUco marker not detected'
            }
        
        # Step 6: Annotate original image
        annotated_image = self.annotate_image(image, df_classified)
        cv2.imwrite(os.path.join(output_dir, 'annotated_result.jpg'), annotated_image)
        
        # Step 7: Save results
        df_classified.to_csv(os.path.join(output_dir, 'grain_measurements.csv'), index=False)
        
        # Add metadata
        results = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'gps_coordinates': gps_coords,
            'scale_mm_per_pixel': self.scale_mm_per_pixel,
            'aruco_detected': aruco_success,
            'statistics': stats,
            'output_directory': output_dir
        }
        
        # Save metadata
        with open(os.path.join(output_dir, 'analysis_metadata.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results

# Example usage
if __name__ == "__main__":
    # Initialize analyzer with 5x5mm ArUco marker
    analyzer = BeachSedimentAnalyzer(aruco_size_mm=5.0)
    
    # Process single image
    # results = analyzer.process_image('path/to/beach_image.jpg', 'output_folder', gps_coords=(lat, lon))
    
    print("Beach Sediment Analyzer initialized successfully!")
    print("Usage:")
    print("1. Include a 5x5mm ArUco marker in your beach sediment photos")
    print("2. Call analyzer.process_image(image_path, output_dir, gps_coords)")
    print("3. Results will be saved in the output directory")
