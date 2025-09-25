#!/usr/bin/env python3
"""
Beach Sediment Mapping System - Quick Deployment Script
Usage: python deploy_beach_mapping.py [options]
"""

import os
import sys
import argparse
from pathlib import Path
import json
import cv2

def install_requirements():
    """Install required packages"""
    requirements = [
        'opencv-python',
        'numpy', 
        'pandas',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'folium',
        'Pillow',
        'exifread'
    ]
    
    print("Installing required packages...")
    for package in requirements:
        os.system(f"pip install {package}")
    print("✓ All packages installed successfully!")

def create_config_file(config_path: str):
    """Create configuration file with default settings"""
    config = {
        "system_settings": {
            "aruco_marker_size_mm": 5.0,
            "min_grain_area_pixels": 20,
            "image_extensions": [".jpg", ".jpeg", ".png", ".tiff", ".bmp"],
            "output_dpi": 300
        },
        "processing_parameters": {
            "clahe_clip_limit": 3.0,
            "clahe_tile_size": [8, 8],
            "gaussian_blur_kernel": [5, 5],
            "adaptive_threshold_block_size": 11,
            "adaptive_threshold_c": 2,
            "morphology_kernel_size": [3, 3],
            "morphology_iterations": 2,
            "distance_transform_threshold": 0.3,
            "watershed_seed_size": 20
        },
        "classification": {
            "wentworth_scale": "standard",
            "phi_scale": true,
            "include_aspect_ratio": true,
            "statistical_measures": ["D10", "D50", "D90", "mean", "std", "skewness"]
        },
        "output_settings": {
            "save_intermediate_images": true,
            "create_visualizations": true,
            "generate_spatial_maps": true,
            "include_validation": true,
            "map_tile_layer": "OpenStreetMap"
        },
        "field_deployment": {
            "recommended_camera_height_cm": 25,
            "recommended_lighting": "ring_light_or_diffused",
            "gps_accuracy_threshold_m": 5.0,
            "min_grains_per_image": 50,
            "sampling_interval_m": 20
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ Configuration file created: {config_path}")
    return config

def generate_aruco_markers(output_dir: str, marker_size_mm: float = 5.0):
    """Generate printable ArUco markers"""
    import cv2
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate ArUco dictionary
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
    
    # Create markers
    marker_ids = [0, 1, 2, 3, 4]  # Generate 5 different markers
    marker_size_pixels = 200  # High resolution for printing
    
    for marker_id in marker_ids:
        # Generate marker image
        marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size_pixels)
        
        # Add border and size information
        border_size = 50
        bordered_img = cv2.copyMakeBorder(marker_img, border_size, border_size, 
                                        border_size, border_size, cv2.BORDER_CONSTANT, 
                                        value=255)
        
        # Add text with size information
        text = f"ArUco ID: {marker_id} | Size: {marker_size_mm}x{marker_size_mm}mm"
        cv2.putText(bordered_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 0), 2)
        
        # Save marker
        marker_path = os.path.join(output_dir, f"aruco_marker_{marker_id}_{marker_size_mm}mm.png")
        cv2.imwrite(marker_path, bordered_img)
    
    # Create printing instructions
    instructions = f"""
# ArUco Marker Printing Instructions

## Important Notes:
- Print at actual size (100% scale, no scaling)
- Use high-quality printer (minimum 300 DPI)
- Print on waterproof material or laminate after printing
- Verify printed size with ruler: {marker_size_mm}mm x {marker_size_mm}mm
- Keep markers flat and clean during field work

## Field Usage:
1. Place marker on sediment surface
2. Ensure marker is clearly visible and in focus
3. Position camera 20-30cm above surface
4. Include diverse grain population in frame
5. Avoid shadows on marker

## Quality Check:
- Marker edges should be sharp and clear
- Black and white squares should be distinct
- No reflections or glare on marker surface
"""
    
    with open(os.path.join(output_dir, "printing_instructions.txt"), 'w') as f:
        f.write(instructions)
    
    print(f"✓ ArUco markers generated: {output_dir}")
    print(f"  - {len(marker_ids)} markers created")
    print(f"  - Marker size: {marker_size_mm}mm x {marker_size_mm}mm")
    print(f"  - Print at 100% scale for accurate calibration")

def create_sample_gps_file(output_path: str):
    """Create sample GPS data file"""
    sample_data = """filename,latitude,longitude,site_description,collection_time
beach_sample_001.jpg,19.0760,72.8777,Juhu Beach North,2024-03-15 09:30:00
beach_sample_002.jpg,19.0750,72.8780,Juhu Beach Center,2024-03-15 09:45:00
beach_sample_003.jpg,19.0740,72.8783,Juhu Beach South,2024-03-15 10:00:00
beach_sample_004.jpg,19.0845,72.8258,Chowpatty Beach East,2024-03-15 11:15:00
beach_sample_005.jpg,19.0840,72.8255,Chowpatty Beach West,2024-03-15 11:30:00
"""
    
    with open(output_path, 'w') as f:
        f.write(sample_data)
    
    print(f"✓ Sample GPS file created: {output_path}")

def create_sample_validation_file(output_path: str):
    """Create sample validation data file"""
    sample_data = """filename,lab_D50_mm,lab_classification,lab_D10_mm,lab_D90_mm,lab_mean_mm,lab_std_mm,notes
beach_sample_001.jpg,0.245,Medium_Sand,0.125,0.420,0.267,0.089,Sieve analysis - well sorted
beach_sample_002.jpg,0.180,Fine_Sand,0.090,0.315,0.195,0.067,Sieve analysis - moderately sorted
beach_sample_003.jpg,0.320,Medium_Sand,0.160,0.580,0.345,0.125,Sieve analysis - poorly sorted
beach_sample_004.jpg,0.425,Coarse_Sand,0.250,0.710,0.445,0.142,Sieve analysis - well sorted
beach_sample_005.jpg,0.380,Coarse_Sand,0.210,0.680,0.405,0.138,Sieve analysis - moderately sorted
"""
    
    with open(output_path, 'w') as f:
        f.write(sample_data)
    
    print(f"✓ Sample validation file created: {output_path}")

def setup_project_structure(project_dir: str):
    """Create complete project structure"""
    dirs_to_create = [
        "input_images",
        "output_results", 
        "aruco_markers",
        "config",
        "validation_data",
        "documentation",
        "scripts"
    ]
    
    for dir_name in dirs_to_create:
        dir_path = os.path.join(project_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"✓ Project structure created in: {project_dir}")
    return {name: os.path.join(project_dir, name) for name in dirs_to_create}

def create_run_script(project_dir: str):
    """Create easy-to-use run script"""
    script_content = f"""#!/usr/bin/env python3
\"\"\"
Quick run script for Beach Sediment Mapping System
Usage: python run_analysis.py
\"\"\"

import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from beach_sediment_analyzer import BeachSedimentAnalyzer
from beach_mapping_system import BeachMappingSystem

def main():
    # Configuration
    PROJECT_DIR = Path(__file__).parent
    INPUT_DIR = PROJECT_DIR / "input_images"
    OUTPUT_DIR = PROJECT_DIR / "output_results"
    GPS_FILE = PROJECT_DIR / "validation_data" / "gps_coordinates.csv"
    VALIDATION_FILE = PROJECT_DIR / "validation_data" / "lab_validation.csv"
    
    # Check if input directory has images
    if not any(INPUT_DIR.glob("*.jpg")) and not any(INPUT_DIR.glob("*.png")):
        print("❌ No images found in input_images directory!")
        print("   Please add your beach sediment images to:", INPUT_DIR)
        return
    
    print("🏖️  Starting Beach Sediment Analysis...")
    print(f"   Input directory: {{INPUT_DIR}}")
    print(f"   Output directory: {{OUTPUT_DIR}}")
    
    # Initialize system
    analyzer = BeachSedimentAnalyzer(aruco_size_mm=5.0)
    mapping_system = BeachMappingSystem(analyzer)
    
    # Process images
    try:
        results = mapping_system.process_batch_images(
            str(INPUT_DIR),
            str(OUTPUT_DIR),
            str(GPS_FILE) if GPS_FILE.exists() else None,
            str(VALIDATION_FILE) if VALIDATION_FILE.exists() else None
        )
        
        print("\\n✅ Analysis Complete!")
        print(f"   📊 Processed {{len(results['spatial_data'])}} images")
        print(f"   🗺️  Spatial map: {{OUTPUT_DIR}}/beach_sediment_map.html")
        print(f"   📈 Results database: {{OUTPUT_DIR}}/spatial_database.csv")
        
        if results['validation_comparison']:
            print(f"   🔬 Validation report: {{OUTPUT_DIR}}/validation_report.png")
            
    except Exception as e:
        print(f"❌ Error during analysis: {{e}}")
        print("   Check that:")
        print("   - Images contain visible ArUco markers")
        print("   - Images are in focus and well-lit")
        print("   - Required packages are installed")

if __name__ == "__main__":
    main()
"""
    
    script_path = os.path.join(project_dir, "run_analysis.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':  # Not Windows
        os.chmod(script_path, 0o755)
    
    print(f"✓ Run script created: {script_path}")

def main():
    parser = argparse.ArgumentParser(description='Deploy Beach Sediment Mapping System')
    parser.add_argument('--project_dir', default='beach_mapping_project',
                       help='Project directory name (default: beach_mapping_project)')
    parser.add_argument('--install_deps', action='store_true',
                       help='Install required Python packages')
    parser.add_argument('--aruco_size', type=float, default=5.0,
                       help='ArUco marker size in mm (default: 5.0)')
    parser.add_argument('--skip_examples', action='store_true',
                       help='Skip creating example data files')
    
    args = parser.parse_args()
    
    print("🏖️  Beach Sediment Mapping System - Deployment Script")
    print("=" * 55)
    
    # Install dependencies if requested
    if args.install_deps:
        install_requirements()
    
    # Create project structure
    project_path = os.path.abspath(args.project_dir)
    print(f"📁 Setting up project in: {project_path}")
    
    project_dirs = setup_project_structure(project_path)
    
    # Create configuration
    config_path = os.path.join(project_dirs['config'], 'system_config.json')
    config = create_config_file(config_path)
    
    # Generate ArUco markers
    generate_aruco_markers(project_dirs['aruco_markers'], args.aruco_size)
    
    # Create sample data files if not skipped
    if not args.skip_examples:
        create_sample_gps_file(os.path.join(project_dirs['validation_data'], 'gps_coordinates.csv'))
        create_sample_validation_file(os.path.join(project_dirs['validation_data'], 'lab_validation.csv'))
    
    # Create run script
    create_run_script(project_path)
    
    # Create README
    readme_content = f"""# Beach Sediment Mapping System

## Quick Start
1. Print ArUco markers from `aruco_markers/` directory
2. Take photos of beach sediments with ArUco markers
3. Place photos in `input_images/` directory
4. Run: `python run_analysis.py`
5. View results in `output_results/` directory

## Project Structure
- `input_images/`: Place your beach sediment photos here
- `output_results/`: Analysis results and visualizations
- `aruco_markers/`: Printable ArUco markers for calibration
- `config/`: System configuration files
- `validation_data/`: GPS coordinates and lab validation data
- `documentation/`: Field guides and technical documentation
- `scripts/`: Utility scripts

## Requirements
- Python 3.7+
- OpenCV, NumPy, Pandas, SciPy, Matplotlib, Folium
- Camera with macro capability
- 5x5mm ArUco markers (print from aruco_markers/ folder)

## Field Data Collection
1. Include ArUco marker in each photo for scale calibration
2. Position camera 20-30cm above sediment surface  
3. Ensure good lighting and sharp focus
4. Record GPS coordinates for each location
5. Cover representative areas of beach (dune, berm, intertidal)

## Outputs
- Interactive spatial maps (HTML)
- Grain size distribution plots
- Statistical summaries (D10, D50, D90)
- Wentworth classification
- Validation reports (if lab data provided)

## Configuration
Edit `config/system_config.json` to adjust:
- Processing parameters
- Classification thresholds
- Output settings
- Field deployment parameters

## Support
For issues or questions, refer to the documentation in the `documentation/` folder.
"""

    readme_path = os.path.join(project_path, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    print(f"✓ README created: {readme_path}")
    
    # Create requirements.txt
    requirements_content = """opencv-python>=4.5.0
numpy>=1.19.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
folium>=0.12.0
Pillow>=8.0.0
exifread>=2.3.0
"""
    
    requirements_path = os.path.join(project_path, 'requirements.txt')
    with open(requirements_path, 'w') as f:
        f.write(requirements_content)
    
    print(f"✓ Requirements file created: {requirements_path}")
    
    # Summary
    print("\n" + "=" * 55)
    print("🎉 DEPLOYMENT COMPLETE!")
    print("=" * 55)
    print(f"📍 Project location: {project_path}")
    print(f"🖨️  Print markers from: {project_dirs['aruco_markers']}")
    print(f"📸 Add photos to: {project_dirs['input_images']}")
    print(f"🚀 Run analysis: cd {args.project_dir} && python run_analysis.py")
    print("\nNext steps:")
    print("1. Print ArUco markers on waterproof material")
    print("2. Collect beach sediment photos with markers")
    print("3. Run the analysis script")
    print("4. View results in output_results/ directory")
    
    if not args.install_deps:
        print("\n⚠️  Don't forget to install dependencies:")
        print(f"   cd {args.project_dir} && pip install -r requirements.txt")

if __name__ == "__main__":
    main()