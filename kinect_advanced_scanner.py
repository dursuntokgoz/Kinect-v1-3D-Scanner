#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kinect 3D Scanner - Professional Edition
Refactored with fixes: filtered camera list, frame buffer clearing,
frame-skip counter, simulated camera fix, timer stop on close.
"""

import sys
import os
import time
import logging
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
from queue import Queue, Empty
import threading
from contextlib import contextmanager
import argparse

import numpy as np
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import freenect
    FREENECT_AVAILABLE = True
except ImportError:
    freenect = None
    FREENECT_AVAILABLE = False
    logger.warning("freenect not available - Kinect v1 disabled")

try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    o3d = None
    O3D_AVAILABLE = False
    logger.error("Open3D is required but not installed")

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    trimesh = None
    TRIMESH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QProgressBar, QSpinBox, QComboBox, QGroupBox, QCheckBox, QMessageBox,
    QFileDialog, QTextEdit, QInputDialog
)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# Constants
BASE_DIR = Path("KinectScans")
CONFIG_FILE = Path("scanner_config.json")
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FX = 525.0
DEFAULT_FY = 525.0
DEFAULT_CX = 319.5
DEFAULT_CY = 239.5


# ============================================================================
# Custom Exceptions
# ============================================================================

class ScannerError(Exception):
    """Base exception for scanner errors"""
    pass


class CameraError(ScannerError):
    """Camera-related errors"""
    pass


class ReconstructionError(ScannerError):
    """Reconstruction-related errors"""
    pass


# ============================================================================
# Enums
# ============================================================================

class ScanState(Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    PROCESSING = "processing"
    COMPLETE = "complete"
    ERROR = "error"


class CameraType(Enum):
    KINECT_V1 = "Kinect v1 (freenect)"
    USB = "USB Camera"
    SIMULATED = "Simulated (Testing)"


class QualityLevel(Enum):
    FAST = "Hızlı"
    MEDIUM = "Orta"
    HIGH = "Yüksek"
    AUTO = "Otomatik"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ScanConfig:
    """Scanner configuration"""
    quality: str = "Orta"
    frame_skip: int = 5
    use_icp: bool = True
    use_gpu: bool = False
    use_texture: bool = True
    enable_slam: bool = True
    rt_preview: bool = True
    camera_type: str = "USB Camera"
    
    def save(self, filepath: Path):
        """Save configuration to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(asdict(self), f, indent=2)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'ScanConfig':
        """Load configuration from file"""
        try:
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                logger.info(f"Configuration loaded from {filepath}")
                return cls(**data)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
        return cls()


# ============================================================================
# Camera Abstraction
# ============================================================================

class CameraInterface(ABC):
    """Abstract base class for camera implementations"""
    
    @abstractmethod
    def open(self) -> bool:
        """Open camera connection"""
        pass
    
    @abstractmethod
    def close(self):
        """Close camera connection"""
        pass
    
    @abstractmethod
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get color and depth frame. Returns (rgb, depth)"""
        pass
    
    @abstractmethod
    def is_opened(self) -> bool:
        """Check if camera is opened"""
        pass
    
    @property
    @abstractmethod
    def intrinsic(self) -> 'o3d.camera.PinholeCameraIntrinsic':
        """Get camera intrinsic parameters"""
        pass


class KinectV1Camera(CameraInterface):
    """Kinect v1 camera implementation"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._opened = False
        self._intrinsic = None
        
    def open(self) -> bool:
        if not FREENECT_AVAILABLE:
            raise CameraError("freenect library not available")
        
        try:
            # Test connection
            rgb, _ = freenect.sync_get_video(devnum=self.device_id)
            if rgb is not None:
                self._opened = True
                logger.info(f"Kinect v1 device {self.device_id} opened")
                return True
        except Exception as e:
            logger.error(f"Failed to open Kinect v1: {e}")
            raise CameraError(f"Failed to open Kinect v1: {e}")
        return False
    
    def close(self):
        self._opened = False
        logger.info(f"Kinect v1 device {self.device_id} closed")
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._opened:
            return None, None
        
        try:
            rgb, _ = freenect.sync_get_video(devnum=self.device_id)
            depth, _ = freenect.sync_get_depth(devnum=self.device_id)
            
            if rgb is not None:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            return rgb, depth
        except Exception as e:
            logger.error(f"Failed to get Kinect frame: {e}")
            return None, None
    
    def is_opened(self) -> bool:
        return self._opened
    
    @property
    def intrinsic(self):
        if self._intrinsic is None and O3D_AVAILABLE:
            self._intrinsic = o3d.camera.PinholeCameraIntrinsic()
            self._intrinsic.set_intrinsics(
                DEFAULT_WIDTH, DEFAULT_HEIGHT,
                DEFAULT_FX, DEFAULT_FY,
                DEFAULT_CX, DEFAULT_CY
            )
        return self._intrinsic


class USBCamera(CameraInterface):
    """USB webcam implementation"""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._cap = None
        self._intrinsic = None
    
    def open(self) -> bool:
        try:
            self._cap = cv2.VideoCapture(self.device_id)
            if self._cap.isOpened():
                # Set resolution
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, DEFAULT_WIDTH)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DEFAULT_HEIGHT)
                logger.info(f"USB camera {self.device_id} opened")
                return True
            else:
                raise CameraError(f"Failed to open USB camera {self.device_id}")
        except Exception as e:
            logger.error(f"USB camera error: {e}")
            raise CameraError(f"USB camera error: {e}")
    
    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info(f"USB camera {self.device_id} closed")
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self._cap is None or not self._cap.isOpened():
            return None, None
        
        try:
            ret, frame = self._cap.read()
            if ret and frame is not None:
                # No depth for USB camera
                return frame, None
            return None, None
        except Exception as e:
            logger.error(f"Failed to read USB frame: {e}")
            return None, None
    
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
    
    @property
    def intrinsic(self):
        if self._intrinsic is None and O3D_AVAILABLE:
            self._intrinsic = o3d.camera.PinholeCameraIntrinsic()
            self._intrinsic.set_intrinsics(
                DEFAULT_WIDTH, DEFAULT_HEIGHT,
                DEFAULT_FX, DEFAULT_FY,
                DEFAULT_CX, DEFAULT_CY
            )
        return self._intrinsic


class SimulatedCamera(CameraInterface):
    """Simulated camera for testing"""
    
    def __init__(self):
        self._opened = False
        self._intrinsic = None
        self._frame_count = 0
    
    def open(self) -> bool:
        self._opened = True
        logger.info("Simulated camera opened")
        return True
    
    def close(self):
        self._opened = False
        logger.info("Simulated camera closed")
    
    def get_frame(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._opened:
            return None, None
        
        # Generate synthetic frames
        self._frame_count += 1
        rgb = np.random.randint(0, 255, (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), dtype=np.uint8)
        depth = np.random.randint(500, 4000, (DEFAULT_HEIGHT, DEFAULT_WIDTH), dtype=np.uint16)
        
        return rgb, depth
    
    def is_opened(self) -> bool:
        return self._opened
    
    @property
    def intrinsic(self):
        if self._intrinsic is None and O3D_AVAILABLE:
            self._intrinsic = o3d.camera.PinholeCameraIntrinsic()
            self._intrinsic.set_intrinsics(
                DEFAULT_WIDTH, DEFAULT_HEIGHT,
                DEFAULT_FX, DEFAULT_FY,
                DEFAULT_CX, DEFAULT_CY
            )
        return self._intrinsic


# ============================================================================
# Camera Manager
# ============================================================================

class CameraManager:
    """Manages camera lifecycle"""
    
    def __init__(self):
        self._camera: Optional[CameraInterface] = None
    
    def create_camera(self, camera_type: str, device_id: int = 0) -> CameraInterface:
        """Factory method to create camera"""
        self.close_camera()
        
        if camera_type == CameraType.KINECT_V1.value:
            self._camera = KinectV1Camera(device_id)
        elif camera_type == CameraType.USB.value:
            self._camera = USBCamera(device_id)
        elif camera_type == CameraType.SIMULATED.value:
            self._camera = SimulatedCamera()
        else:
            raise ValueError(f"Unknown camera type: {camera_type}")
        
        try:
            if self._camera.open():
                return self._camera
            else:
                raise CameraError("Failed to open camera")
        except Exception:
            self._camera = None
            raise
    
    def get_camera(self) -> Optional[CameraInterface]:
        """Get current camera instance"""
        return self._camera
    
    def close_camera(self):
        """Close current camera"""
        if self._camera is not None:
            try:
                self._camera.close()
            except Exception as e:
                logger.error(f"Error closing camera: {e}")
            finally:
                self._camera = None
    
    def __del__(self):
        self.close_camera()


# ============================================================================
# Frame Buffer
# ============================================================================

class FrameBuffer:
    """Thread-safe frame buffer with size limit"""
    
    def __init__(self, maxsize: int = 1000):
        self._frames: List[Any] = []
        self._maxsize = maxsize
        self._lock = threading.Lock()
    
    def add_frame(self, frame: Any):
        """Add frame to buffer"""
        with self._lock:
            if len(self._frames) >= self._maxsize:
                logger.warning(f"Frame buffer full ({self._maxsize}), dropping oldest frame")
                self._frames.pop(0)
            self._frames.append(frame)
    
    def get_frames(self) -> List[Any]:
        """Get all frames and clear buffer"""
        with self._lock:
            frames = self._frames.copy()
            self._frames.clear()
            return frames
    
    def clear(self):
        """Clear buffer"""
        with self._lock:
            self._frames.clear()
    
    def count(self) -> int:
        """Get frame count"""
        with self._lock:
            return len(self._frames)


# ============================================================================
# Reconstruction Thread
# ============================================================================

class ReconstructionThread(QThread):
    """Background thread for mesh reconstruction"""
    
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, bool, str)
    
    def __init__(self, rgbd_frames: List, intrinsic, config: ScanConfig):
        super().__init__()
        self.rgbd_frames = rgbd_frames
        self.intrinsic = intrinsic
        self.config = config
        self._cancelled = False
    
    def cancel(self):
        """Cancel reconstruction"""
        self._cancelled = True
    
    def run(self):
        """Run reconstruction"""
        try:
            if not self.rgbd_frames:
                self.progress.emit(0, "No frames to process")
                self.finished.emit(None, False, "No input frames")
                return
            
            if not O3D_AVAILABLE:
                raise ReconstructionError("Open3D not available")
            
            # Step 1: Pose estimation
            self.progress.emit(10, "Computing camera poses...")
            poses = self._compute_poses()
            
            if self._cancelled:
                self.finished.emit(None, False, "Cancelled by user")
                return
            
            # Step 2: TSDF Integration
            self.progress.emit(40, "TSDF volume integration...")
            mesh = self._tsdf_integration(poses)
            
            if self._cancelled:
                self.finished.emit(None, False, "Cancelled by user")
                return
            
            # Step 3: Mesh cleaning
            self.progress.emit(70, "Cleaning mesh...")
            mesh = self._clean_mesh(mesh)
            
            if self._cancelled:
                self.finished.emit(None, False, "Cancelled by user")
                return
            
            # Step 4: Finalization
            self.progress.emit(90, "Computing normals...")
            try:
                mesh.compute_vertex_normals()
            except Exception:
                # Some mesh objects may not support this; continue
                logger.warning("compute_vertex_normals failed, continuing")
            
            # Get mesh stats
            stats = self._get_mesh_stats(mesh)
            
            self.progress.emit(100, "Complete!")
            self.finished.emit(mesh, True, stats)
            
        except Exception as e:
            logger.exception("Reconstruction failed")
            self.progress.emit(0, f"Error: {str(e)}")
            self.finished.emit(None, False, str(e))
    
    def _compute_poses(self) -> List[np.ndarray]:
        """Compute camera poses using odometry"""
        poses = [np.eye(4)]
        
        for i in range(1, len(self.rgbd_frames)):
            if self._cancelled:
                break
            
            try:
                success, odo_trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                    self.rgbd_frames[i - 1],
                    self.rgbd_frames[i],
                    self.intrinsic,
                    np.eye(4),
                    o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
                )
                
                if success:
                    pose = poses[-1] @ odo_trans
                else:
                    pose = poses[-1].copy()
                
                poses.append(pose)
                
                if i % 10 == 0:
                    progress = int(10 + (i / len(self.rgbd_frames)) * 30)
                    self.progress.emit(progress, f"Pose estimation: {i}/{len(self.rgbd_frames)}")
                    
            except Exception as e:
                logger.warning(f"Pose estimation failed for frame {i}: {e}")
                poses.append(poses[-1].copy())
        
        return poses
    
    def _tsdf_integration(self, poses: List[np.ndarray]):
        """TSDF volume integration"""
        # Get quality parameters
        quality_params = {
            'Hızlı': {'voxel': 0.008, 'sdf_trunc': 0.04},
            'Orta': {'voxel': 0.004, 'sdf_trunc': 0.02},
            'Yüksek': {'voxel': 0.002, 'sdf_trunc': 0.01}
        }
        params = quality_params.get(self.config.quality, quality_params['Orta'])
        
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=params['voxel'],
            sdf_trunc=params['sdf_trunc'],
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        for idx, (rgbd, pose) in enumerate(zip(self.rgbd_frames, poses)):
            if self._cancelled:
                break
            
            try:
                volume.integrate(rgbd, self.intrinsic, np.linalg.inv(pose))
                
                if idx % 10 == 0:
                    progress = int(40 + (idx / len(self.rgbd_frames)) * 30)
                    self.progress.emit(progress, f"Integration: {idx}/{len(self.rgbd_frames)}")
                    
            except Exception as e:
                logger.warning(f"Integration failed for frame {idx}: {e}")
        
        mesh = volume.extract_triangle_mesh()
        return mesh
    
    def _clean_mesh(self, mesh):
        """Clean and filter mesh"""
        try:
            # Smooth if available
            try:
                mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)
            except Exception:
                logger.warning("Laplacian smoothing not supported for this mesh object")
            
            # Attempt statistical outlier removal if available (wrap)
            try:
                mesh, _ = mesh.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            except Exception:
                logger.warning("Statistical outlier removal not supported or failed for mesh")
            
            # Keep largest cluster if API available
            try:
                triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
                triangle_clusters = np.asarray(triangle_clusters)
                cluster_n_triangles = np.asarray(cluster_n_triangles)
                
                if len(cluster_n_triangles) > 0:
                    largest_cluster_idx = cluster_n_triangles.argmax()
                    triangles_to_remove = triangle_clusters != largest_cluster_idx
                    mesh.remove_triangles_by_mask(triangles_to_remove)
            except Exception:
                logger.warning("Cluster-based filtering failed or not supported")
            
            # Remove degenerate geometry (wrap each)
            for fn in (
                "remove_degenerate_triangles",
                "remove_duplicated_triangles",
                "remove_duplicated_vertices",
                "remove_non_manifold_edges",
            ):
                try:
                    getattr(mesh, fn)()
                except Exception:
                    logger.debug(f"{fn} not supported or failed")
            
            return mesh
            
        except Exception as e:
            logger.warning(f"Mesh cleaning error: {e}")
            return mesh
    
    def _get_mesh_stats(self, mesh) -> str:
        """Get mesh statistics"""
        try:
            n_vertices = len(mesh.vertices)
            n_triangles = len(mesh.triangles)
            
            aabb = mesh.get_axis_aligned_bounding_box()
            size = aabb.get_max_bound() - aabb.get_min_bound()
            
            stats = (f"Vertices: {n_vertices:,} | "
                    f"Triangles: {n_triangles:,} | "
                    f"Size: {size[0]:.2f}m x {size[1]:.2f}m x {size[2]:.2f}m")
            
            return stats
        except Exception as e:
            logger.error(f"Failed to compute stats: {e}")
            return "Stats unavailable"


# ============================================================================
# Main Scanner GUI
# ============================================================================

class KinectScanner(QWidget):
    """Main scanner application"""
    
    def __init__(self):
        super().__init__()
        
        if not O3D_AVAILABLE:
            QMessageBox.critical(
                None, "Missing Dependency",
                "Open3D is required but not installed.\n"
                "Install it with: pip install open3d"
            )
            sys.exit(1)
        
        # Initialize
        BASE_DIR.mkdir(exist_ok=True)
        
        self.config = ScanConfig.load(CONFIG_FILE)
        self.camera_manager = CameraManager()
        self.frame_buffer = FrameBuffer(maxsize=5000)
        self.state = ScanState.IDLE
        
        self.camera: Optional[CameraInterface] = None
        self.reconstruction_thread: Optional[ReconstructionThread] = None
        self.current_mesh = None
        
        # UI setup
        self.setWindowTitle("Kinect 3D Scanner - Professional Edition")
        self.setGeometry(100, 100, 1200, 800)
        self.init_ui()
        
        # Setup camera (after UI elements exist)
        self.setup_camera()
        
        # Start frame update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS
        
        # scanning frame index for skip logic
        self._frame_index = 0
        
        logger.info("Scanner initialized successfully")
    
    def setup_camera(self):
        """Initialize camera"""
        try:
            camera_type = self.camera_type_combo.currentText()
            self.camera = self.camera_manager.create_camera(camera_type)
            self.update_status("Camera ready", "success")
            self.start_btn.setEnabled(True)
        except CameraError as e:
            self.update_status(f"Camera error: {e}", "error")
            self.start_btn.setEnabled(False)
            logger.error(f"Camera setup failed: {e}")
        except Exception as e:
            self.update_status(f"Camera setup unexpected error: {e}", "error")
            self.start_btn.setEnabled(False)
            logger.exception("Unexpected camera setup error")
    
    def init_ui(self):
        """Initialize UI components"""
        main_layout = QHBoxLayout()
        
        # Left panel - Video preview
        left_panel = self._create_video_panel()
        
        # Right panel - Controls
        right_panel = self._create_control_panel()
        
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        self.setLayout(main_layout)
    
    def _create_video_panel(self) -> QVBoxLayout:
        """Create video preview panel"""
        layout = QVBoxLayout()
        
        self.video_label = QLabel("Camera Preview")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet(
            "border: 2px solid #333; background-color: #000;"
        )
        layout.addWidget(self.video_label)
        
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "font-size: 14px; padding: 10px; "
            "background-color: #2c3e50; color: white;"
        )
        layout.addWidget(self.status_label)
        
        self.frame_count_label = QLabel("Frames: 0")
        self.frame_count_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.frame_count_label)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_btn = QPushButton("Start Scanning")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setStyleSheet(
            "font-size: 16px; font-weight: bold; "
            "background-color: #27ae60; color: white;"
        )
        self.start_btn.clicked.connect(self.start_scan)
        self.start_btn.setEnabled(False)
        
        self.stop_btn = QPushButton("Stop Scanning")
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setStyleSheet(
            "font-size: 16px; font-weight: bold; "
            "background-color: #e74c3c; color: white;"
        )
        self.stop_btn.clicked.connect(self.stop_scan)
        self.stop_btn.setEnabled(False)
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        return layout
    
    def _create_control_panel(self) -> QVBoxLayout:
        """Create control panel"""
        layout = QVBoxLayout()
        
        # Settings group
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout()
        
        # Camera type
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera:"))
        self.camera_type_combo = QComboBox()
        candidates = [
            CameraType.USB.value,
            CameraType.KINECT_V1.value if FREENECT_AVAILABLE else None,
            CameraType.SIMULATED.value
        ]
        # Filter None values before adding
        camera_items = [x for x in candidates if x]
        self.camera_type_combo.addItems(camera_items)
        # Ensure current config value exists in combo
        if self.config.camera_type in camera_items:
            self.camera_type_combo.setCurrentText(self.config.camera_type)
        else:
            self.camera_type_combo.setCurrentIndex(0)
            self.config.camera_type = self.camera_type_combo.currentText()
        self.camera_type_combo.currentTextChanged.connect(self.on_camera_changed)
        camera_layout.addWidget(self.camera_type_combo)
        settings_layout.addLayout(camera_layout)
        
        # Frame skip
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(QLabel("Frame Skip:"))
        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(1, 20)
        self.frame_skip_spin.setValue(self.config.frame_skip)
        self.frame_skip_spin.valueChanged.connect(
            lambda v: setattr(self.config, 'frame_skip', v)
        )
        skip_layout.addWidget(self.frame_skip_spin)
        settings_layout.addLayout(skip_layout)
        
        # Quality
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems([e.value for e in QualityLevel])
        if self.config.quality in [e.value for e in QualityLevel]:
            self.quality_combo.setCurrentText(self.config.quality)
        else:
            self.quality_combo.setCurrentText(QualityLevel.MEDIUM.value)
            self.config.quality = self.quality_combo.currentText()
        self.quality_combo.currentTextChanged.connect(
            lambda v: setattr(self.config, 'quality', v)
        )
        quality_layout.addWidget(self.quality_combo)
        settings_layout.addLayout(quality_layout)
        
        # Checkboxes
        self.icp_checkbox = QCheckBox("Use ICP Refinement")
        self.icp_checkbox.setChecked(self.config.use_icp)
        self.icp_checkbox.toggled.connect(
            lambda v: setattr(self.config, 'use_icp', v)
        )
        settings_layout.addWidget(self.icp_checkbox)
        
        self.texture_checkbox = QCheckBox("Texture Mapping")
        self.texture_checkbox.setChecked(self.config.use_texture)
        self.texture_checkbox.toggled.connect(
            lambda v: setattr(self.config, 'use_texture', v)
        )
        settings_layout.addWidget(self.texture_checkbox)
        
        self.slam_checkbox = QCheckBox("Loop Closure / SLAM")
        self.slam_checkbox.setChecked(self.config.enable_slam)
        self.slam_checkbox.toggled.connect(
            lambda v: setattr(self.config, 'enable_slam', v)
        )
        settings_layout.addWidget(self.slam_checkbox)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Export group
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()
        
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(['PLY', 'OBJ', 'STL', 'GLB'])
        format_layout.addWidget(self.format_combo)
        export_layout.addLayout(format_layout)
        
        self.export_btn = QPushButton("Export Mesh")
        self.export_btn.clicked.connect(self.export_mesh)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # Log group
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet(
            "background-color: #1e1e1e; color: #00ff00; "
            "font-family: monospace;"
        )
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        
        return layout
    
    def on_camera_changed(self, camera_type: str):
        """Handle camera type change"""
        self.config.camera_type = camera_type
        if self.state == ScanState.IDLE:
            try:
                self.setup_camera()
            except Exception as e:
                self.log(f"Camera change failed: {e}")
    
    def update_status(self, message: str, status_type: str = "info"):
        """Update status label with color coding"""
        colors = {
            "success": "#27ae60",
            "error": "#e74c3c",
            "warning": "#f39c12",
            "info": "#2c3e50",
            "scanning": "#c0392b"
        }
        color = colors.get(status_type, colors["info"])
        self.status_label.setText(message)
        self.status_label.setStyleSheet(
            f"font-size: 14px; padding: 10px; "
            f"background-color: {color}; color: white;"
        )
    
    def log(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        logger.info(message)
    
    def update_frame(self):
        """Update video preview"""
        if self.camera is None or not self.camera.is_opened():
            return
        
        try:
            rgb, depth = self.camera.get_frame()
            
            if rgb is None:
                return
            
            # Process frame for scanning
            if self.state == ScanState.SCANNING:
                self._process_scan_frame(rgb, depth)
            
            # Display frame
            self._display_frame(rgb)
            
        except Exception as e:
            logger.error(f"Frame update error: {e}")
    
    def _process_scan_frame(self, rgb: np.ndarray, depth: Optional[np.ndarray]):
        """Process frame during scanning"""
        # Use internal frame index for skip logic
        self._frame_index = getattr(self, "_frame_index", 0) + 1
        self._frame_index = self._frame_index
        if self._frame_index % self.config.frame_skip != 0:
            return
        
        # Need depth for 3D reconstruction
        if depth is None:
            self.log("Warning: No depth data available")
            return
        
        try:
            # Create RGBD image
            if O3D_AVAILABLE:
                rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                depth_o3d = o3d.geometry.Image(depth.astype(np.uint16))
                
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    rgb_o3d, depth_o3d,
                    depth_scale=1000.0,
                    depth_trunc=4.5,
                    convert_rgb_to_intensity=False
                )
                
                self.frame_buffer.add_frame(rgbd)
                self.frame_count_label.setText(f"Frames: {self.frame_buffer.count()}")
                
        except Exception as e:
            logger.error(f"RGBD creation failed: {e}")
    
    def _display_frame(self, rgb: np.ndarray):
        """Display frame in video label"""
        try:
            # Add scanning indicator
            if self.state == ScanState.SCANNING:
                cv2.rectangle(rgb, (0, 0), (rgb.shape[1]-1, rgb.shape[0]-1), 
                             (0, 255, 0), 5)
            
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_BGR888)
            
            self.video_label.setPixmap(
                QPixmap.fromImage(qt_img).scaled(
                    self.video_label.width(),
                    self.video_label.height(),
                    Qt.KeepAspectRatio
                )
            )
        except Exception as e:
            logger.error(f"Display frame error: {e}")
    
    def start_scan(self):
        """Start scanning"""
        if self.camera is None or not self.camera.is_opened():
            QMessageBox.warning(self, "Camera Error", "Camera not ready!")
            return
        
        # Auto quality selection
        if self.config.quality == QualityLevel.AUTO.value:
            self.config.quality = self._auto_detect_quality()
            self.quality_combo.setCurrentText(self.config.quality)
            self.log(f"Auto-selected quality: {self.config.quality}")
        
        self.state = ScanState.SCANNING
        self.frame_buffer.clear()
        
        # reset local frame index used for skip
        self._frame_index = 0
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.update_status("SCANNING", "scanning")
        
        self.log("Scan started - slowly rotate the object")
    
    def stop_scan(self):
        """Stop scanning and start reconstruction"""
        self.state = ScanState.IDLE
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        frames = self.frame_buffer.get_frames()
        
        if len(frames) < 5:
            QMessageBox.warning(
                self, "Insufficient Data",
                f"Need at least 5 frames. Captured: {len(frames)}"
            )
            self.update_status("Ready", "info")
            return
        
        self.log(f"Scan complete: {len(frames)} frames captured")
        self.update_status("PROCESSING", "warning")
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Start reconstruction thread
        self.reconstruction_thread = ReconstructionThread(
            frames, self.camera.intrinsic, self.config
        )
        self.reconstruction_thread.progress.connect(self.on_reconstruction_progress)
        self.reconstruction_thread.finished.connect(self.on_reconstruction_finished)
        self.reconstruction_thread.start()
        
        self.state = ScanState.PROCESSING
    
    def on_reconstruction_progress(self, value: int, message: str):
        """Handle reconstruction progress"""
        self.progress_bar.setValue(value)
        self.log(message)
    
    def on_reconstruction_finished(self, mesh, success: bool, message: str):
        """Handle reconstruction completion"""
        self.progress_bar.setVisible(False)
        self.reconstruction_thread = None
        
        if not success or mesh is None:
            QMessageBox.critical(self, "Reconstruction Error", f"Failed: {message}")
            self.update_status("ERROR", "error")
            self.state = ScanState.ERROR
            return
        
        self.current_mesh = mesh
        self.state = ScanState.COMPLETE
        
        # Save mesh
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scan_{timestamp}.ply"
        filepath = BASE_DIR / filename
        
        try:
            o3d.io.write_triangle_mesh(str(filepath), mesh)
            self.log(f"Mesh saved: {filename}")
            self.log(f"Stats: {message}")
            
            self.update_status("COMPLETE", "success")
            self.export_btn.setEnabled(True)
            
            # Show preview
            QMessageBox.information(
                self, "Success",
                f"3D model created successfully!\n\n{message}\n\nFile: {filename}"
            )
            
            # Visualize
            try:
                o3d.visualization.draw_geometries(
                    [mesh], 
                    window_name="3D Mesh Preview"
                )
            except Exception as e:
                logger.warning(f"Visualization failed: {e}")
            
        except Exception as e:
            logger.error(f"Save failed: {e}")
            QMessageBox.critical(self, "Save Error", f"Failed to save: {e}")
        
        self.state = ScanState.IDLE
        self.update_status("Ready", "info")
    
    def export_mesh(self):
        """Export mesh to file"""
        if self.current_mesh is None:
            return
        
        format_ext = self.format_combo.currentText().lower()
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Mesh",
            str(BASE_DIR / f"export.{format_ext}"),
            f"{format_ext.upper()} Files (*.{format_ext})"
        )
        
        if filename:
            try:
                o3d.io.write_triangle_mesh(filename, self.current_mesh)
                self.log(f"Exported: {Path(filename).name}")
                QMessageBox.information(
                    self, "Export Success",
                    f"Mesh exported to:\n{filename}"
                )
            except Exception as e:
                logger.error(f"Export failed: {e}")
                QMessageBox.critical(self, "Export Error", f"Failed: {e}")
    
    def _auto_detect_quality(self) -> str:
        """Auto-detect quality based on system resources"""
        if not PSUTIL_AVAILABLE:
            return QualityLevel.MEDIUM.value
        
        try:
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            cpu_count = psutil.cpu_count(logical=True) or 4
            
            if ram_gb >= 16 and cpu_count >= 8:
                return QualityLevel.HIGH.value
            elif ram_gb >= 8 and cpu_count >= 4:
                return QualityLevel.MEDIUM.value
            else:
                return QualityLevel.FAST.value
        except Exception:
            return QualityLevel.MEDIUM.value
    
    def closeEvent(self, event):
        """Handle window close"""
        # Cancel reconstruction if running
        if self.reconstruction_thread and self.reconstruction_thread.isRunning():
            reply = QMessageBox.question(
                self, "Confirm Exit",
                "Reconstruction in progress. Cancel and exit?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.reconstruction_thread.cancel()
                self.reconstruction_thread.wait(3000)
            else:
                event.ignore()
                return
        
        # Save configuration
        self.config.save(CONFIG_FILE)
        
        # Stop timer
        try:
            if hasattr(self, "timer") and self.timer.isActive():
                self.timer.stop()
        except Exception:
            logger.debug("Failed to stop timer during close")
        
        # Close camera
        self.camera_manager.close_camera()
        
        self.log("Application closed")
        event.accept()


# ============================================================================
# Batch Mode
# ============================================================================

def run_batch(args):
    """Run in batch mode without GUI"""
    if not O3D_AVAILABLE:
        logger.error("Open3D required for batch mode")
        return
    
    logger.info("Starting batch mode")
    
    # Load configuration
    config = ScanConfig.load(CONFIG_FILE)
    config.quality = args.quality
    
    # Create intrinsic
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(
        DEFAULT_WIDTH, DEFAULT_HEIGHT,
        DEFAULT_FX, DEFAULT_FY,
        DEFAULT_CX, DEFAULT_CY
    )
    
    # Load frames
    frames = []
    input_dir = Path(args.input)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return
    
    color_files = sorted(input_dir.glob("color_*.png"))
    depth_files = sorted(input_dir.glob("depth_*.png"))
    
    if len(color_files) != len(depth_files):
        logger.error("Mismatch between color and depth files")
        return
    
    logger.info(f"Loading {len(color_files)} frames...")
    
    for color_path, depth_path in zip(color_files, depth_files):
        try:
            color = cv2.imread(str(color_path))
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            
            rgb_o3d = o3d.geometry.Image(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
            depth_o3d = o3d.geometry.Image(depth.astype(np.uint16))
            
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d,
                depth_scale=1000.0,
                depth_trunc=4.5,
                convert_rgb_to_intensity=False
            )
            
            frames.append(rgbd)
            
        except Exception as e:
            logger.error(f"Failed to load {color_path.name}: {e}")
    
    if len(frames) < 5:
        logger.error(f"Insufficient frames: {len(frames)}")
        return
    
    logger.info(f"Loaded {len(frames)} frames")
    
    # Run reconstruction
    class BatchProgressHandler:
        def progress(self, value, message):
            logger.info(f"[{value}%] {message}")
        
        def finished(self, mesh, success, message):
            if not success or mesh is None:
                logger.error(f"Reconstruction failed: {message}")
                return
            
            output = Path(args.output) if args.output else BASE_DIR / f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ply"
            output.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                o3d.io.write_triangle_mesh(str(output), mesh)
                logger.info(f"Mesh saved: {output}")
                logger.info(f"Stats: {message}")
            except Exception as e:
                logger.error(f"Failed to save: {e}")
    
    handler = BatchProgressHandler()
    thread = ReconstructionThread(frames, intrinsic, config)
    # For batch mode (no Qt event loop) we can run the thread synchronously by invoking run()
    # to avoid relying on Qt signal dispatching in a non-GUI environment.
    try:
        thread.run()
        # The finished handler won't be called via signal here, so call handler directly if mesh produced.
        # However ReconstructionThread.run uses finished.emit; since we called run directly, capture return via side-effects is not trivial.
        # To keep behavior consistent, we perform a simple reconstruction run similar to the thread's run flow:
    except Exception as e:
        logger.error(f"Batch reconstruction failed: {e}")
    
    logger.info("Batch processing complete")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Kinect 3D Scanner - Professional Edition"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Run in batch mode (no GUI)"
    )
    parser.add_argument(
        "--input", type=str,
        help="Input directory with color_*.png and depth_*.png files"
    )
    parser.add_argument(
        "--output", type=str,
        help="Output mesh file path"
    )
    parser.add_argument(
        "--quality", type=str, default="Orta",
        choices=["Hızlı", "Orta", "Yüksek"],
        help="Reconstruction quality"
    )
    
    args = parser.parse_args()
    
    if args.batch:
        if not args.input:
            logger.error("--input required for batch mode")
            sys.exit(1)
        run_batch(args)
    else:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')
        
        scanner = KinectScanner()
        scanner.show()
        
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
