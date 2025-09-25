#!/usr/bin/env python3
"""
Real-time Beach Sediment Field Data Collector
Captures photos with GPS coordinates and immediate analysis feedback
"""

import cv2
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
import time
import threading
import queue
from pathlib import Path

try:
    import serial
    import pynmea2
    NMEA_AVAILABLE = True
except ImportError:
    NMEA_AVAILABLE = False
    print("Warning: pynmea2 and pyserial not available. GPS functionality limited.")

try:
    from geopy.geocoders import Nominatim
    GEOCODING_AVAILABLE = True
except ImportError:
    GEOCODING_AVAILABLE = False
    print("Warning: geopy not available. Reverse geocoding disabled.")

class GPSLogger:
    def __init__(self, port=None, baudrate=9600):
        """
        GPS Logger for real-time coordinate capture
        
        Args:
            port: Serial port for GPS device (e.g., '/dev/ttyUSB0', 'COM3')
            baudrate: GPS device baud rate
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.running = False
        self.current_coords = None
        self.coord_queue = queue.Queue()
        
        if NMEA_AVAILABLE and port:
            self._connect_gps()
    
    def _connect_gps(self):
        """Connect to GPS device"""
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"✓ Connected to GPS on {self.port}")
        except Exception as e:
            print(f"❌ Failed to connect to GPS: {e}")
            self.serial_connection = None
    
    def start_logging(self):
        """Start GPS logging in background thread"""
        if not self.serial_connection:
            print("⚠️  GPS not connected, using manual coordinate entry")
            return
        
        self.running = True
        self.gps_thread = threading.Thread(target=self._gps_loop)
        self.gps_thread.daemon = True
        self.gps_thread.start()
        print("✓ GPS logging started")
    
    def _gps_loop(self):
        """GPS reading loop"""
        while self.running:
            try:
                if self.serial_connection.in_waiting:
                    line = self.serial_connection.readline().decode('utf-8', errors='ignore')
                    if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                        msg = pynmea2.parse(line)
                        if hasattr(msg, 'latitude') and msg.latitude:
                            self.current_coords = (msg.latitude, msg.longitude)
                            self.coord_queue.put(self.current_coords)
            except Exception as e:
                print(f"GPS read error: {e}")
            time.sleep(0.1)
    
    def get_current_coordinates(self, timeout=30):
        """
        Get current GPS coordinates
        
        Args:
            timeout: Maximum time to wait for GPS fix (seconds)
            
        Returns:
            tuple: (latitude, longitude) or None if unavailable
        """
        if self.current_coords:
            return self.current_coords
        
        if not self.serial_connection:
            return self._manual_coordinate_entry()
        
        print(f"🛰️  Waiting for GPS fix (timeout: {timeout}s)...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                coords = self.coord_queue.get(timeout=1)
                print(f"✓ GPS fix acquired: {coords[0]:.6f}, {coords[1]:.6f}")
                return coords
            except queue.Empty:
                print(".", end="", flush=True)
        
        print("\n⚠️  GPS timeout, falling back to manual entry")
        return self._manual_coordinate_entry()
    
    def _manual_coordinate_entry(self):
        """Manual coordinate entry fallback"""
        print("\n📍 Manual GPS Coordinate Entry")
        try:
            lat = float(input("Enter latitude (decimal degrees): "))
            lon = float(input("Enter longitude (decimal degrees): "))
            return (lat, lon)
        except ValueError:
            print("❌ Invalid coordinates entered")
            return None
    
    def stop_logging(self):
        """Stop GPS logging"""
        self.running = False
        if self.serial_connection:
            self.serial_connection.close()

class FieldDataCollector:
    def __init__(self, project_dir: str, gps_port=None):
        """
        Field data collection system
        
        Args:
            project_dir: Project directory path
            gps_port: GPS device serial port (optional)
        """
        self.project_dir = Path(project_dir)
        self.session_data = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize GPS logger
        self.gps_logger = GPSLogger(gps_port)
        
        # Create session directories
        self.session_dir = self.project_dir / "field_sessions" / f"session_{self.session_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize camera
        self.camera = None
        self._init_camera()
        
        # Load analyzer if available
        try:
            from beach_sediment_analyzer import BeachSedimentAnalyzer
            self.analyzer = BeachSedimentAnalyzer(aruco_size_mm=5.0)
            self.analysis_enabled = True
        except ImportError:
            print("⚠️  Beach sediment analyzer not available - photos only mode")
            self.analysis_enabled = False
        
        print(f"📁 Session directory: {self.session_dir}")
    
    def _init_camera(self):
        """Initialize camera connection"""
        # Try different camera indices
        for i in range(3):
            self.camera = cv2.VideoCapture(i)
            if self.camera.isOpened():
                print(f"✓ Camera connected on index {i}")
                # Set camera properties for better quality
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                return
        
        print("❌ No camera found")
        self.camera = None
    
    def start_field_session(self):
        """Start interactive field data collection session"""
        print("\n🏖️  BEACH SEDIMENT FIELD DATA COLLECTION")
        print("=" * 50)
        print(f"Session ID: {self.session_id}")
        print(f"Session directory: {self.session_dir}")
        
        # Get session metadata
        session_meta = self._get_session_metadata()
        
        # Start GPS logging
        self.gps_logger.start_logging()
        
        # Main collection loop
        photo_count = 0
        
        try:
            while True:
                print(f"\n📸 Photo {photo_count + 1}")
                print("-" * 30)
                
                # Get location information
                location_info = self._get_location_info()
                
                # Capture photo
                photo_data = self._capture_photo(photo_count)
                
                if photo_data:
                    # Combine all data
                    sample_data = {
                        **session_meta,
                        **location_info,
                        **photo_data,
                        'sample_id': f"{self.session_id}_{photo_count:03d}",
                        'photo_number': photo_count + 1
                    }
                    
                    # Quick analysis if enabled
                    if self.analysis_enabled:
                        analysis_result = self._quick_analysis(sample_data['photo_path'])
                        sample_data.update(analysis_result)
                    
                    # Save data
                    self.session_data.append(sample_data)
                    self._save_session_data()
                    
                    # Display summary
                    self._display_sample_summary(sample_data)
                    
                    photo_count += 1
                
                # Continue or finish
                continue_choice = input("\n[C]ontinue, [F]inish session, [Q]uit without saving: ").lower()
                
                if continue_choice == 'f':
                    self._finalize_session()
                    break
                elif continue_choice == 'q':
                    print("Session terminated without final save")
                    break
                    
        except KeyboardInterrupt:
            print("\n\n⚠️  Session interrupted by user")
            save_choice = input("Save current data? [y/N]: ").lower()
            if save_choice == 'y':
                self._finalize_session()
        
        finally:
            self.gps_logger.stop_logging()
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
    
    def _get_session_metadata(self):
        """Collect session-level metadata"""
        print("\n📋 Session Information")
        metadata = {
            'session_id': self.session_id,
            'start_time': datetime.now().isoformat(),
            'operator': input("Operator name: "),
            'site_name': input("Site/Beach name: "),
            'weather': input("Weather conditions: "),
            'tide_condition': input("Tide condition (high/mid/low): "),
            'equipment': input("Camera/equipment used: "),
            'notes': input("General notes: ")
        }
        return metadata
    
    def _get_location_info(self):
        """Get GPS and location information for current sample"""
        print("🛰️  Getting GPS coordinates...")
        coords = self.gps_logger.get_current_coordinates()
        
        location_data = {
            'timestamp': datetime.now().isoformat(),
            'latitude': coords[0] if coords else None,
            'longitude': coords[1] if coords else None,
            'gps_accuracy': None,  # Could be enhanced with GPS quality info
            'zone_description': input("Zone description (dune/berm/intertidal/other): "),
            'substrate_notes': input("Visual substrate notes: ")
        }
        
        # Reverse geocoding if available
        if GEOCODING_AVAILABLE and coords:
            try:
                geolocator = Nominatim(user_agent="beach_sediment_mapper")
                location = geolocator.reverse(f"{coords[0]}, {coords[1]}")
                location_data['address'] = location.address if location else None
            except Exception as e:
                print(f"Geocoding failed: {e}")
        
        return location_data
    
    def _capture_photo(self, photo_count):
        """Capture and save photo"""
        if not self.camera:
            # Manual file input mode
            photo_path = input("Enter photo file path (or 'skip'): ")
            if photo_path.lower() == 'skip':
                return None
            
            if not os.path.exists(photo_path):
                print(f"❌ File not found: {photo_path}")
                return None
            
            # Copy to session directory
            import shutil
            filename = f"sample_{photo_count:03d}_{datetime.now().strftime('%H%M%S')}.jpg"
            dest_path = self.session_dir / filename
            shutil.copy2(photo_path, dest_path)
            
            return {
                'photo_path': str(dest_path),
                'photo_filename': filename,
                'capture_method': 'manual_file'
            }
        
        # Live camera capture
        print("📷 Position camera over sediment with ArUco marker visible")
        print("Press SPACE to capture, ESC to skip, 'p' for preview")
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("❌ Failed to read from camera")
                return None
            
            # Display preview
            preview = cv2.resize(frame, (800, 600))
            cv2.putText(preview, "SPACE: Capture, ESC: Skip, P: Preview", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Beach Sediment Capture", preview)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space to capture
                filename = f"sample_{photo_count:03d}_{datetime.now().strftime('%H%M%S')}.jpg"
                photo_path = self.session_dir / filename
                cv2.imwrite(str(photo_path), frame)
                print(f"✓ Photo saved: {filename}")
                cv2.destroyAllWindows()
                
                return {
                    'photo_path': str(photo_path),
                    'photo_filename': filename,
                    'capture_method': 'live_camera',
                    'image_width': frame.shape[1],
                    'image_height': frame.shape[0]
                }
            
            elif key == 27:  # ESC to skip
                cv2.destroyAllWindows()
                print("Photo capture skipped")
                return None
    
    def _quick_analysis(self, photo_path):
        """Perform quick analysis for immediate feedback"""
        try:
            print("🔍 Performing quick analysis...")
            
            # Load image
            image = cv2.imread(photo_path)
            if image is None:
                return {'analysis_error': 'Could not load image'}
            
            # Check for ArUco marker
            aruco_success, scale, _ = self.analyzer.detect_aruco_and_calibrate(image)
            
            if not aruco_success:
                return {
                    'aruco_detected': False,
                    'analysis_error': 'ArUco marker not detected',
                    'recommendation': 'Retake photo with visible ArUco marker'
                }
            
            # Quick grain count estimation
            binary = self.analyzer.preprocess_image(image)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by size
            min_area = 20
            grain_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            return {
                'aruco_detected': True,
                'scale_mm_per_pixel': scale,
                'estimated_grain_count': len(grain_contours),
                'analysis_quality': 'good' if len(grain_contours) > 50 else 'fair' if len(grain_contours) > 20 else 'poor'
            }
        
        except Exception as e:
            return {'analysis_error': f'Analysis failed: {e}'}
    
    def _display_sample_summary(self, sample_data):
        """Display summary of collected sample"""
        print("\n✅ SAMPLE SUMMARY")
        print("-" * 20)
        print(f"Sample ID: {sample_data['sample_id']}")
        print(f"GPS: {sample_data['latitude']:.6f}, {sample_data['longitude']:.6f}")
        print(f"Zone: {sample_data['zone_description']}")
        
        if 'estimated_grain_count' in sample_data:
            print(f"Estimated grains: {sample_data['estimated_grain_count']}")
            print(f"Quality: {sample_data['analysis_quality']}")
        
        if sample_data.get('analysis_error'):
            print(f"⚠️  Analysis issue: {sample_data['analysis_error']}")
            if sample_data.get('recommendation'):
                print(f"💡 Recommendation: {sample_data['recommendation']}")
    
    def _save_session_data(self):
        """Save current session data"""
        # Save as CSV
        df = pd.DataFrame(self.session_data)
        csv_path = self.session_dir / "field_data.csv"
        df.to_csv(csv_path, index=False)
        
        # Save as JSON with full metadata
        json_path = self.session_dir / "field_data.json"
        with open(json_path, 'w') as f:
            json.dump(self.session_data, f, indent=2, default=str)
    
    def _finalize_session(self):
        """Finalize field session"""
        print(f"\n🎉 Finalizing session with {len(self.session_data)} samples")
        
        # Final save
        self._save_session_data()
        
        # Create session summary
        summary = {
            'session_id': self.session_id,
            'total_samples': len(self.session_data),
            'start_time': self.session_data[0]['start_time'] if self.session_data else None,
            'end_time': datetime.now().isoformat(),
            'operator': self.session_data[0]['operator'] if self.session_data else None,
            'site_name': self.session_data[0]['site_name'] if self.session_data else None,
            'gps_coverage': sum(1 for s in self.session_data if s.get('latitude')),
            'analysis_coverage': sum(1 for s in self.session_data if s.get('estimated_grain_count')),
        }
        
        summary_path = self.session_dir / "session_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"✅ Session data saved to: {self.session_dir}")
        print(f"📊 CSV file: field_data.csv")
        print(f"📄 JSON file: field_data.json")
        print(f"📋 Summary: session_summary.json")

def main():
    """Main function with command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Beach Sediment Field Data Collector')
    parser.add_argument('--project_dir', required=True, help='Project directory path')
    parser.add_argument('--gps_port', help='GPS serial port (e.g., COM3, /dev/ttyUSB0)')
    parser.add_argument('--camera_test', action='store_true', help='Test camera connection')
    
    args = parser.parse_args()
    
    if args.camera_test:
        print("🔍 Testing camera connection...")
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"✓ Camera found on index {i}")
                ret, frame = cap.read()
                if ret:
                    print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
                cap.release()
            else:
                print(f"❌ No camera on index {i}")
        return
    
    # Initialize collector
    collector = FieldDataCollector(args.project_dir, args.gps_port)
    
    # Start field session
    collector.start_field_session()

if __name__ == "__main__":
    main()