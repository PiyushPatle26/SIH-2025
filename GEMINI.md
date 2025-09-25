# 🌊 Beach Sediment Mapping System

## 📌 Problem Statement
Grain size is a fundamental property used to classify beach types and plays a critical role in influencing coastal morphodynamic processes.  
However, measuring grain size is particularly challenging because:
- Scientists must visit each site to collect sediment samples.  
- Samples undergo **time-consuming physical processing** in laboratories.  
- Sediments change dynamically with **waves, tides, and weather events**, requiring **frequent re-measurements**.  

Thus, there is a need for a **low-cost, automated, and scalable system** to measure and classify beach sediments in the field without heavy manual processing.

---

## 💡 Proposed Solution
We propose a **camera-based automated mapping system** with **ArUco marker calibration** and **GNSS/GPS integration** that can:
- Estimate **grain size distribution** of sandy beach regions (berm, intertidal, dune).  
- Classify beaches based on **Wentworth grain size scale**.  
- Provide **georeferenced spatial maps** of sediment properties.  

This approach reduces fieldwork time, eliminates repeated lab processing, and provides **real-time monitoring capability**.

---

## ⚙️ System Workflow

### 1. Calibration & Setup
- Use a **5×5 mm ArUco marker** for scale calibration.  
- Automatic detection ensures **pixel-to-millimeter conversion** for accurate measurement.  

### 2. Grain Analysis
- Image pre-processing: grayscale, thresholding, noise removal.  
- **Watershed segmentation** for separating overlapping grains.  
- Extract morphological parameters:  
  - Equivalent diameter  
  - Aspect ratio  
  - Roundness, solidity, etc.  
- Convert measurements into **Wentworth classes** and **Phi scale values**.  
- Compute **D10, D50, D90, sorting coefficient** for statistical characterization.  

### 3. GPS Integration
- Capture coordinates via:  
  - **EXIF metadata** (if available in camera image).  
  - **NMEA stream** from external GPS device.  
  - **Manual entry** if GPS unavailable.  
- Save spatially referenced results in **GeoJSON/CSV** for GIS use.  

### 4. Validation & Accuracy
- Compare field-based results with **lab sieve analysis**.  
- Metrics:  
  - Mean Absolute Error (MAE)  
  - Root Mean Square Error (RMSE)  
  - R² correlation  
- **Bland-Altman plots** for method comparison.  

### 5. Deployment
- Real-time capture using field laptop + USB camera.  
- Batch processing of collected images.  
- Output includes:  
  - Annotated images  
  - Histograms, pie charts  
  - CSV + JSON metadata  
  - Interactive maps  

---

## 📂 What’s Inside the Code

### `beach_analyzer.py`
Core grain analysis pipeline:
- `BeachSedimentAnalyzer` class for:
  - **ArUco detection** (`cv2.aruco`)  
  - **Pixel-to-mm calibration**  
  - **Pre-processing & segmentation**  
  - **Grain size measurement & classification**  
  - **Statistical computation**  
- Saves results: annotated images, plots, CSVs, metadata.  

### `deploy_beach_mapping.py`
- Project setup script.  
- Installs dependencies.  
- Creates project directory with config + markers.  

### `field_data_collector.py`
- Connects to **GPS receiver** (via serial/NMEA).  
- Captures images + coordinates.  
- Stores raw field data for later analysis.  

### `run_analysis.py`
- Batch processing of collected field data.  
- Generates histograms, validation metrics, and spatial maps.  

### Outputs
- **Images**: `aruco_detection.jpg`, `annotated_result.jpg`  
- **Plots**: Histograms, Wentworth pie chart, cumulative distribution  
- **Data**: `grain_measurements.csv`, `analysis_metadata.json`  
- **Maps**: Folium-based interactive spatial maps  

---

## 🚀 Getting Started
```bash
# Deploy the system
python deploy_beach_mapping.py --project_dir my_beach_project --install_deps

# Collect field data
cd my_beach_project
python field_data_collector.py --gps_port COM3

# Run analysis
python run_analysis.py
