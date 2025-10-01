#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
from datetime import datetime
from pathlib import Path
import threading
from collections import namedtuple
import argparse
import importlib
import subprocess
import shlex

import numpy as np
import cv2

# Optional native Kinect v1 binding
try:
    import freenect
    FREENECT_AVAILABLE = True
except Exception:
    freenect = None
    FREENECT_AVAILABLE = False

# Open3D and trimesh (may be installed by ensure_dependencies)
try:
    import open3d as o3d
    O3D_AVAILABLE = True
except Exception:
    o3d = None
    O3D_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except Exception:
    trimesh = None
    TRIMESH_AVAILABLE = False

# Optional tensor/GPU backend for Open3D T
try:
    import open3d.core as o3c
    import open3d.t as o3t
    O3D_T_AVAILABLE = True
except Exception:
    O3D_T_AVAILABLE = False

# Optional torch (for AI segmentation / MiDaS)
TORCH_AVAILABLE = False
TORCHVISION_AVAILABLE = False
SEGMENT_MODEL = None
MIDAS_MODEL = None
MIDAS_TRANSFORMS = None
MI_DAS_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    try:
        import torchvision
        from torchvision import transforms
        TORCHVISION_AVAILABLE = True
    except Exception:
        TORCHVISION_AVAILABLE = False
except Exception:
    TORCH_AVAILABLE = False

# MiDaS availability flag (separate import try to avoid blocking)
try:
    import torch.hub  # if torch exists, hub likely available
    MI_DAS_AVAILABLE = TORCH_AVAILABLE
except Exception:
    MI_DAS_AVAILABLE = False

# Optional system info (auto quality)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except Exception:
    PSUTIL_AVAILABLE = False

# ArUco marker detection (marker-based alignment)
ARUCO_AVAILABLE = False
try:
    aruco = cv2.aruco
    ARUCO_AVAILABLE = True
except Exception:
    ARUCO_AVAILABLE = False

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QProgressBar, QSpinBox, QComboBox, QGroupBox, QCheckBox, QMessageBox,
    QFileDialog, QTextEdit, QInputDialog, QProgressDialog
)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

BASE_DIR = "KinectScans"
os.makedirs(BASE_DIR, exist_ok=True)

FramePack = namedtuple("FramePack", ["rgbd_frames", "poses", "color_np"])

# Camera mode constants
CAM_KINECT_V1 = "Kinect v1 (freenect)"
CAM_KINECT_V2 = "Kinect v2 (libfreenect2)"
CAM_AZURE_K = "Kinect v3 / Azure Kinect"
CAM_USB = "USB Kamera (OpenCV)"
CAM_OPTIONS = [CAM_KINECT_V1, CAM_KINECT_V2, CAM_AZURE_K, CAM_USB]


def camera_matrix_from_intrinsic(intrinsic: 'o3d.camera.PinholeCameraIntrinsic'):
    fx, fy = intrinsic.get_focal_length()
    cx, cy = intrinsic.get_principal_point()
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float32)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def check_lib_freenect2_available():
    try:
        import pylibfreenect2  # noqa: F401
        return True
    except Exception:
        return False


def check_lib_azure_available():
    try:
        import pyk4a  # common python wrapper for azure kinect
        return True
    except Exception:
        return False


def init_midas_model(device=None):
    global MIDAS_MODEL, MIDAS_TRANSFORMS, MI_DAS_AVAILABLE
    if not MI_DAS_AVAILABLE:
        return False
    if MIDAS_MODEL is not None and MIDAS_TRANSFORMS is not None:
        return True
    try:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        # Use MiDaS small for speed by default
        MIDAS_MODEL = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        MIDAS_TRANSFORMS = torch.hub.load("intel-isl/MiDaS", "transforms")
        MIDAS_MODEL.to(device).eval()
        return True
    except Exception:
        MIDAS_MODEL = None
        MIDAS_TRANSFORMS = None
        return False


def ensure_dependencies(parent=None):
    """
    Eksik paketleri tespit edip kullanıcı onayıyla pip ile yükler.
    parent: opsiyonel Qt parent (QWidget) -- dialoglar için.
    Returns: (ok: bool, report)
    """
    checks = [
        ("open3d", "open3d", True, ""),
        ("cv2", "opencv-python", True, ""),
        ("PyQt5", "PyQt5", True, ""),
        ("freenect", "freenect", False, "Kinect v1 için; bazı platformlarda manuel kurulum gerekebilir"),
        ("torch", "torch", False, "MiDaS / AI için; GPU versiyonu tercih edilebilir"),
        ("torchvision", "torchvision", False, "AI segmentasyon için"),
        ("psutil", "psutil", False, ""),
        ("trimesh", "trimesh", False, ""),
    ]

    missing = []
    for imp_name, pip_name, mandatory, note in checks:
        try:
            importlib.import_module(imp_name)
        except Exception:
            missing.append((imp_name, pip_name, mandatory, note))

    if not missing:
        return True, []

    mandatory_missing = [m for m in missing if m[2]]
    optional_missing = [m for m in missing if not m[2]]

    msg_lines = []
    if mandatory_missing:
        msg_lines.append("Gerekli paketler eksik:")
        for imp, pipn, man, note in mandatory_missing:
            msg_lines.append(f"  - {imp} (pip: {pipn}) {note}")
    if optional_missing:
        msg_lines.append("")
        msg_lines.append("İsteğe bağlı paketler eksik (AI/ek özellikler için):")
        for imp, pipn, man, note in optional_missing:
            msg_lines.append(f"  - {imp} (pip: {pipn}) {note}")
    msg_lines.append("")
    msg_lines.append("Bu paketler otomatik olarak pip ile yüklenecek. Devam edilsin mi?")

    reply = QMessageBox.question(parent or None, "Eksik Paketler", "\n".join(msg_lines),
                                 QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
    if reply != QMessageBox.Yes:
        return False, missing

    installed = []
    failures = []
    try:
        dialog = QProgressDialog("Paketler yükleniyor...", "İptal", 0, len(missing), parent)
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.setMinimumDuration(200)
    except Exception:
        dialog = None

    for idx, (imp, pipn, man, note) in enumerate(missing):
        if dialog:
            dialog.setValue(idx)
            dialog.setLabelText(f"{pipn} yükleniyor...")
            QApplication.processEvents()
        cmd = f"{shlex.quote(sys.executable)} -m pip install --upgrade {pipn}"
        try:
            proc = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60*15)
            if proc.returncode == 0:
                try:
                    importlib.import_module(imp)
                    installed.append((imp, pipn))
                except Exception:
                    failures.append((imp, pipn, proc.stderr.strip() or proc.stdout.strip()))
            else:
                failures.append((imp, pipn, proc.stderr.strip() or proc.stdout.strip()))
        except Exception as e:
            failures.append((imp, pipn, str(e)))

        if dialog and dialog.wasCanceled():
            failures.append(("__canceled__", "__canceled__", "Kullanıcı iptal etti"))
            break

    if dialog:
        dialog.setValue(len(missing))

    native_notes = []
    native_notes.append("Not: libfreenect2 ve Azure Kinect SDK gibi native bağımlılıklar pip ile kurulmaz; lütfen ilgili SDK kurulum rehberini takip edin.")

    report = {"installed": installed, "failures": failures, "native_notes": native_notes}
    return (len(failures) == 0), report


class ReconstructionThread(QThread):
    """Arka planda mesh oluşturma ve birleştirme."""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, bool)

    def __init__(self, rgbd_frames, intrinsic, quality_level, use_icp,
                 use_gpu=False, use_texture=True, ai_model='Kapalı',
                 enable_slam=True, use_marker_alignment=False,
                 marker_size_m=0.05, noise_aware=True, rt_simplify=True):
        super().__init__()
        try:
            if not rgbd_frames:
                self.input_scans = []
            elif isinstance(rgbd_frames[0], list):
                self.input_scans = rgbd_frames
            else:
                self.input_scans = [rgbd_frames]
        except Exception:
            self.input_scans = []

        self.intrinsic = intrinsic
        self.quality_level = quality_level
        self.use_icp = use_icp
        self.use_gpu = use_gpu and O3D_T_AVAILABLE
        self.use_texture = use_texture
        self.ai_model = ai_model
        self.enable_slam = enable_slam
        self.use_marker_alignment = use_marker_alignment and ARUCO_AVAILABLE
        self.marker_size_m = marker_size_m
        self.noise_aware = noise_aware
        self.rt_simplify = rt_simplify

    def run(self):
        try:
            if not self.input_scans:
                self.progress.emit(0, "Girdi taraması boş; işleme sonlandırıldı.")
                self.finished.emit(None, False)
                return

            self.progress.emit(10, "Pose estimation başlıyor...")
            scan_packs = self.compute_poses_multi()

            if self.enable_slam:
                self.progress.emit(25, "Loop closure ve pose graph optimizasyonu...")
                scan_packs = [self.pose_graph_optimize(p) for p in scan_packs]

            self.progress.emit(35, "Global hizalama ve paket birleştirme...")
            merged_pack = self.global_merge(scan_packs)

            self.progress.emit(50, "TSDF integration yapılıyor...")
            mesh = self.tsdf_integration(merged_pack)

            self.progress.emit(68, "Mesh temizleniyor ve düzgünleştiriliyor...")
            mesh = self.clean_mesh(mesh)

            if self.use_texture:
                self.progress.emit(78, "Color map optimizasyonu ile texture mapping...")
                mesh = self.texture_map(mesh, merged_pack)

            self.progress.emit(86, "AI tamamlama uygulanıyor...")
            mesh = self.ai_complete(mesh)

            self.progress.emit(92, "Ölçüm ve analiz hesaplanıyor...")
            size_text = self.measure_and_log(mesh)

            self.progress.emit(96, "Son rötuşlar...")
            mesh.compute_vertex_normals()

            self.progress.emit(100, f"Tamamlandı! {size_text}")
            self.finished.emit(mesh, True)
        except Exception as e:
            self.progress.emit(0, f"Hata: {str(e)}")
            self.finished.emit(None, False)

    def detect_marker_pose(self, color_np):
        if not self.use_marker_alignment or color_np is None:
            return None
        try:
            gray = cv2.cvtColor(color_np, cv2.COLOR_BGR2GRAY)
            dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters_create()
            corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=parameters)
            if ids is None or len(corners) == 0:
                return None
            pts_img = corners[0][0].astype(np.float32)
            if pts_img.shape[0] < 4:
                return None
            objp = np.array([
                [-self.marker_size_m/2, self.marker_size_m/2, 0],
                [self.marker_size_m/2, self.marker_size_m/2, 0],
                [self.marker_size_m/2, -self.marker_size_m/2, 0],
                [-self.marker_size_m/2, -self.marker_size_m/2, 0]
            ], dtype=np.float32)
            cam_mtx = camera_matrix_from_intrinsic(self.intrinsic)
            dist = np.zeros((5,), dtype=np.float32)
            ok, rvec, tvec = False, None, None
            try:
                res = cv2.solvePnP(objp, pts_img, cam_mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                if res is not None and len(res) >= 3:
                    ok = True
                    rvec, tvec = res[1], res[2]
            except Exception:
                ok = False
            if not ok or rvec is None or tvec is None:
                return None
            R, _ = cv2.Rodrigues(rvec)
            T_cam_marker = np.eye(4, dtype=np.float64)
            T_cam_marker[:3, :3] = R
            T_cam_marker[:3, 3] = tvec.reshape(3)
            T_world_cam = np.linalg.inv(T_cam_marker)
            return T_world_cam
        except Exception:
            return None

    def compute_poses_multi(self):
        packs = []
        for s_idx, frames in enumerate(self.input_scans):
            if not frames:
                packs.append(FramePack([], [], []))
                continue
            pose = np.eye(4)
            poses = [pose.copy()]
            success_count = 0
            total_drift = 0.0
            color_np_frames = []
            for i in range(0, len(frames)):
                color_np = None
                try:
                    color_img = getattr(frames[i], 'color', None)
                    if color_img is not None:
                        color_np = cv2.cvtColor(np.asarray(color_img), cv2.COLOR_RGB2BGR)
                except Exception:
                    color_np = None
                color_np_frames.append(color_np)

            for i in range(1, len(frames)):
                marker_pose = None
                try:
                    marker_pose = self.detect_marker_pose(color_np_frames[i])
                except Exception:
                    marker_pose = None

                if marker_pose is not None:
                    try:
                        odo_trans = marker_pose @ np.linalg.inv(poses[i - 1])
                        use_odo = True
                    except Exception:
                        use_odo = False
                        odo_trans = np.eye(4)
                else:
                    try:
                        success_odo, odo_trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                            frames[i - 1], frames[i], self.intrinsic, np.eye(4),
                            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
                        )
                        use_odo = success_odo
                    except Exception:
                        use_odo = False
                        odo_trans = np.eye(4)

                if use_odo:
                    if self.use_icp:
                        try:
                            src = o3d.geometry.PointCloud.create_from_rgbd_image(frames[i], self.intrinsic)
                            tgt = o3d.geometry.PointCloud.create_from_rgbd_image(frames[i - 1], self.intrinsic)
                            icp_result = o3d.pipelines.registration.registration_icp(
                                src, tgt, 0.05, odo_trans,
                                o3d.pipelines.registration.TransformationEstimationPointToPoint()
                            )
                            if icp_result.fitness > 0.3:
                                odo_trans = icp_result.transformation
                        except Exception:
                            pass
                    translation = np.linalg.norm(odo_trans[:3, 3])
                    if translation < 0.5:
                        pose = pose @ odo_trans
                        success_count += 1
                        total_drift += translation

                poses.append(pose.copy())
                if i % 10 == 0:
                    progress = int(10 + (i / max(1, len(frames))) * 15)
                    avg_drift = total_drift / max(success_count, 1)
                    self.progress.emit(progress, f"Paket {s_idx + 1}: Pose {i}/{len(frames)} - Başarı: {success_count}/{i} - Drift: {avg_drift:.4f}m")
            packs.append(FramePack(frames, poses, color_np_frames))
        return packs

    def pose_graph_optimize(self, pack: FramePack):
        frames, poses, _ = pack.rgbd_frames, pack.poses, pack.color_np
        if len(frames) < 10:
            return pack
        pg = o3d.pipelines.registration.PoseGraph()
        pg.nodes.append(o3d.pipelines.registration.PoseGraphNode(np.eye(4)))
        for i in range(1, len(frames)):
            try:
                transformation = np.linalg.inv(poses[i - 1]) @ poses[i]
            except Exception:
                transformation = np.eye(4)
            info = np.eye(6)
            pg.edges.append(
                o3d.pipelines.registration.PoseGraphEdge(i - 1, i, transformation, info, uncertain=False)
            )
        step = max(30, len(frames) // 10)
        for i in range(0, len(frames), step):
            for j in range(i + step, len(frames), step):
                try:
                    src = o3d.geometry.PointCloud.create_from_rgbd_image(frames[i], self.intrinsic)
                    tgt = o3d.geometry.PointCloud.create_from_rgbd_image(frames[j], self.intrinsic)
                    src.estimate_normals()
                    tgt.estimate_normals()
                    result = o3d.pipelines.registration.registration_icp(
                        src, tgt, 0.05, np.eye(4),
                        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30)
                    )
                    if result.fitness > 0.5:
                        pg.edges.append(
                            o3d.pipelines.registration.PoseGraphEdge(i, j, result.transformation, np.eye(6), uncertain=True)
                        )
                except Exception:
                    continue
        opt_option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=0.05,
            edge_prune_threshold=0.25,
            reference_node=0
        )
        try:
            o3d.pipelines.registration.global_optimization(
                pg,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                opt_option
            )
        except Exception:
            pass

        new_poses = []
        try:
            for node in pg.nodes:
                try:
                    new_poses.append(node.pose)
                except Exception:
                    new_poses.append(np.eye(4))
        except Exception:
            new_poses = poses.copy()
        return FramePack(frames, new_poses, pack.color_np)

    def global_merge(self, scan_packs):
        if len(scan_packs) == 1:
            return scan_packs[0]

        def pcd_from_pack(pack):
            pcd = o3d.geometry.PointCloud()
            step = max(1, len(pack.rgbd_frames) // 30) if pack.rgbd_frames else 1
            for rgbd in pack.rgbd_frames[::step]:
                try:
                    p = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)
                    pcd += p.voxel_down_sample(0.01)
                except Exception:
                    continue
            return pcd

        base = scan_packs[0]
        base_pcd = pcd_from_pack(base)
        if len(base_pcd.points) == 0:
            try:
                if base.rgbd_frames:
                    base_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(base.rgbd_frames[0], self.intrinsic).voxel_down_sample(0.01)
            except Exception:
                pass
        base_pcd.estimate_normals()
        merged_frames = base.rgbd_frames.copy()
        merged_poses = base.poses.copy()
        merged_colors = base.color_np.copy()

        for k in range(1, len(scan_packs)):
            cur = scan_packs[k]
            cur_pcd = pcd_from_pack(cur)
            if len(cur_pcd.points) == 0:
                self.progress.emit(45, f"Paket {k + 1} boş nokta bulutu, atlandı.")
                continue
            cur_pcd.estimate_normals()

            radius = 0.05
            try:
                base_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    base_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))
                cur_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                    cur_pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=100))

                ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                    cur_pcd, base_pcd, cur_fpfh, base_fpfh, True, 0.05,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                    4,
                    [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                     o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(0.05)],
                    o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 1000))
                T_kb = ransac.transformation

                icp = o3d.pipelines.registration.registration_icp(
                    cur_pcd, base_pcd, 0.03, T_kb,
                    o3d.pipelines.registration.TransformationEstimationPointToPlane())
                T_kb = icp.transformation
            except Exception:
                T_kb = np.eye(4)

            for i, pose in enumerate(cur.poses):
                try:
                    merged_poses.append(T_kb @ pose)
                    merged_frames.append(cur.rgbd_frames[i])
                    merged_colors.append(cur.color_np[i])
                except Exception:
                    continue

            self.progress.emit(45, f"Paket {k + 1} global hizalama tamamlandı.")
        return FramePack(merged_frames, merged_poses, merged_colors)

    def derive_quality_params(self, base_level, depth_noise_std=None):
        quality_params = {
            'Hızlı': {'voxel': 0.008, 'sdf_trunc': 0.04},
            'Orta': {'voxel': 0.004, 'sdf_trunc': 0.02},
            'Yüksek': {'voxel': 0.002, 'sdf_trunc': 0.01}
        }
        params = quality_params.get(base_level, quality_params['Orta']).copy()
        if self.noise_aware and depth_noise_std is not None:
            noise = float(depth_noise_std)
            scale = np.clip(noise / 50.0, 0.5, 2.0)
            params['voxel'] *= scale
            params['sdf_trunc'] *= scale
        return params

    def tsdf_integration(self, merged_pack):
        try:
            depth_samples = []
            step = max(1, len(merged_pack.rgbd_frames) // 20) if merged_pack.rgbd_frames else 1
            for rgbd in merged_pack.rgbd_frames[::step]:
                try:
                    d = np.asarray(rgbd.depth)
                    m = d[d > 0]
                    if m.size > 0:
                        depth_samples.append(np.std(m))
                except Exception:
                    continue
            depth_noise_std = float(np.median(depth_samples)) if depth_samples else None
        except Exception:
            depth_noise_std = None

        params = self.derive_quality_params(self.quality_level, depth_noise_std)

        if self.use_gpu:
            try:
                voxel_size = params['voxel']
                sdf_trunc = params['sdf_trunc']
                tsdf = o3t.geometry.TSDFVoxelGrid(
                    voxel_size=voxel_size, sdf_trunc=sdf_trunc,
                    block_resolution=16, block_count=12000,
                    device=o3c.cuda.Device(0)
                )
                fx, fy = self.intrinsic.get_focal_length()
                cx, cy = self.intrinsic.get_principal_point()
                K = o3c.Tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=o3c.Dtype.Float32)
                for idx, (rgbd, T) in enumerate(zip(merged_pack.rgbd_frames, merged_pack.poses)):
                    color = o3t.geometry.Image.from_legacy(rgbd.color)
                    depth = o3t.geometry.Image.from_legacy(rgbd.depth)
                    extr = o3c.Tensor(np.linalg.inv(T), dtype=o3c.Dtype.Float32)
                    tsdf.integrate(depth, color, K, extr)
                    if idx % 10 == 0:
                        progress = int(50 + (idx / max(1, len(merged_pack.rgbd_frames))) * 18)
                        self.progress.emit(progress, f"GPU Integration {idx}/{len(merged_pack.rgbd_frames)}")
                mesh = tsdf.extract_triangle_mesh().to_legacy()
                mesh.compute_vertex_normals()
                return mesh
            except Exception as e:
                self.progress.emit(52, f"GPU TSDF başarısız, CPU moduna düşüldü: {e}")
                self.use_gpu = False

        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=params['voxel'],
            sdf_trunc=params['sdf_trunc'],
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        for idx, (rgbd, T) in enumerate(zip(merged_pack.rgbd_frames, merged_pack.poses)):
            try:
                volume.integrate(rgbd, self.intrinsic, np.linalg.inv(T))
            except Exception:
                continue
            if self.rt_simplify and idx % 50 == 0 and idx > 0:
                try:
                    temp_mesh = volume.extract_triangle_mesh()
                    if len(temp_mesh.triangles) > 50000:
                        temp_mesh = temp_mesh.simplify_quadric_decimation(50000)
                    temp_mesh.compute_vertex_normals()
                except Exception:
                    pass
            if idx % 10 == 0:
                progress = int(50 + (idx / max(1, len(merged_pack.rgbd_frames))) * 18)
                self.progress.emit(progress, f"Integration {idx}/{len(merged_pack.rgbd_frames)}")

        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    def clean_mesh(self, mesh):
        mesh_clean = mesh.filter_smooth_laplacian(number_of_iterations=3)
        mesh_clean.compute_vertex_normals()

        try:
            mesh_clean, _ = mesh_clean.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        except Exception:
            pass

        try:
            triangle_clusters, cluster_n_triangles, _ = mesh_clean.cluster_connected_triangles()
            triangle_clusters = np.asarray(triangle_clusters)
            cluster_n_triangles = np.asarray(cluster_n_triangles)
            if len(cluster_n_triangles) > 0:
                largest_cluster_idx = cluster_n_triangles.argmax()
                triangles_to_remove = triangle_clusters != largest_cluster_idx
                mesh_clean.remove_triangles_by_mask(triangles_to_remove)
        except Exception:
            pass

        try:
            mesh_clean.remove_degenerate_triangles()
            mesh_clean.remove_duplicated_triangles()
            mesh_clean.remove_duplicated_vertices()
            mesh_clean.remove_non_manifold_edges()
        except Exception:
            pass

        if self.rt_simplify and len(mesh_clean.triangles) > 150000:
            try:
                mesh_clean = mesh_clean.simplify_quadric_decimation(150000)
            except Exception:
                pass

        mesh_clean.compute_vertex_normals()
        return mesh_clean

    def texture_map(self, mesh, merged_pack):
        try:
            option = o3d.pipelines.color_map.ColorMapOptimizationOption()
            option.max_iteration = 300 if self.quality_level == 'Yüksek' else 150
            o3d.pipelines.color_map.color_map_optimization(
                mesh,
                merged_pack.rgbd_frames,
                merged_pack.poses,
                self.intrinsic,
                option
            )
            return mesh
        except Exception as e:
            self.progress.emit(82, f"Texture mapping hatası: {e}, devam ediliyor.")
            return mesh

    def ai_complete(self, mesh):
        if self.ai_model == 'Kapalı' or not TRIMESH_AVAILABLE:
            return mesh
        try:
            m = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
            if self.ai_model == 'PCN':
                try:
                    m = m.subdivide()
                except Exception:
                    pass
            elif self.ai_model == 'SnowflakeNet':
                try:
                    m = m.smoothed()
                    m = m.subdivide()
                except Exception:
                    pass
            if self.quality_level == 'Yüksek':
                target_faces = int(len(m.faces) * 0.9)
            elif self.quality_level == 'Orta':
                target_faces = int(len(m.faces) * 0.7)
            else:
                target_faces = int(len(m.faces) * 0.5)
            if len(m.faces) > target_faces:
                try:
                    m = m.simplify_quadratic_decimation(target_faces)
                except Exception:
                    pass
            mesh_ai = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(m.vertices),
                o3d.utility.Vector3iVector(m.faces)
            )
            if mesh.has_vertex_colors():
                mesh_ai.vertex_colors = mesh.vertex_colors
            mesh_ai.compute_vertex_normals()
            return mesh_ai
        except Exception as e:
            self.progress.emit(89, f"AI tamamlama hatası: {e}, orijinal mesh ile devam.")
            return mesh

    def measure_and_log(self, mesh):
        try:
            aabb = mesh.get_axis_aligned_bounding_box()
            min_bound = aabb.get_min_bound()
            max_bound = aabb.get_max_bound()
            size = max_bound - min_bound
            hull = mesh.compute_convex_hull()[0]
            volume = hull.get_volume()
            roughness = float(np.std(np.asarray(mesh.vertex_normals)))
            txt = f"Boyut: {size[0]:.3f}m x {size[1]:.3f}m x {size[2]:.3f}m | Hacim: {volume:.6f} m^3 | Pürüzlülük: {roughness:.4f}"
            self.progress.emit(93, txt)
            return txt
        except Exception:
            return "Ölçüm hesaplanamadı."


class KinectScanner(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinect 3D Tarama Sistemi - Pro Edition")
        self.setGeometry(100, 100, 1100, 750)

        # Başlangıç: bağımlılıkları doğrula ve yükle
        ok, report = ensure_dependencies(parent=self)
        if not ok:
            if isinstance(report, list) and report:
                QMessageBox.warning(self, "Paket Eksik", "Bazı paketler eksik. Uygulama bazı özellikleri desteklemeyebilir.")
            else:
                failures = report.get("failures", []) if isinstance(report, dict) else []
                lines = ["Bazı paketler yüklenemedi:"]
                for f in failures:
                    lines.append(f" - {f[0]} (pip: {f[1]}) -> {f[2]}")
                lines += report.get("native_notes", []) if isinstance(report, dict) else []
                QMessageBox.warning(self, "Kurulum Hatası", "\n".join(lines))

        # Kamera modu seçimi (kullanıcıya sor)
        self.camera_mode = self.ask_camera_mode()
        # Kinect durumu
        self.kinect_connected = False
        self.device_count = 1
        self.device_ids = [0]
        self.check_kinect_connection()

        # Webcam ve MiDaS flags
        self.webcam = None
        self.use_midas_for_usb = MI_DAS_AVAILABLE  # GUI ile değiştirilebilir
        self.rgb_only_mode = False

        # Tarama durumu
        self.scanning = False
        self.rgbd_frames = []
        self.color_np_frames = []
        self.last_frame_time = 0
        self.frame_skip = 5
        self.frame_counter = 0
        self.reconstruction_thread = None

        # Çoklu tarama
        self.scans = []

        # Kamera intrinsic parametreleri (Kinect v1 default, USB için de kullanılacak)
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic() if O3D_AVAILABLE else None
        if self.intrinsic:
            self.intrinsic.set_intrinsics(640, 480, 525.0, 525.0, 319.5, 239.5)

        # RT preview
        self.preview_vis = None
        self.preview_pcd = None
        self.preview_lock = threading.Lock()

        self.init_ui()

        # Canlı görüntü timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        self.log("Sistem hazır. Kamera modu: " + self.camera_mode + " | Kinect durumu: " + ("BAĞLI ✅" if self.kinect_connected else "BAĞLI DEĞİL ❌"))

    def ask_camera_mode(self):
        try:
            mode, ok = QInputDialog.getItem(
                self, "Kamera Seçimi", "Hangi kamerayı kullanmak istersiniz?", CAM_OPTIONS, 0, False
            )
            if ok and mode:
                if mode == CAM_KINECT_V2 and not check_lib_freenect2_available():
                    QMessageBox.warning(self, "Lib eksik", "libfreenect2 yüklü değil; Kinect v2 seçildi ancak fallback USB kullanılacak.")
                    return CAM_USB
                if mode == CAM_AZURE_K and not check_lib_azure_available():
                    QMessageBox.warning(self, "Lib eksik", "Azure Kinect SDK bağlayıcıları yüklü değil; fallback USB kullanılacak.")
                    return CAM_USB
                return mode
        except Exception:
            pass
        return CAM_USB

    def check_kinect_connection(self):
        if self.camera_mode == CAM_KINECT_V1:
            if not FREENECT_AVAILABLE:
                self.kinect_connected = False
                return
            try:
                rgb, _ = freenect.sync_get_video(devnum=0)
                self.kinect_connected = rgb is not None
            except Exception:
                self.kinect_connected = False
        elif self.camera_mode == CAM_KINECT_V2:
            try:
                import pylibfreenect2 as lf2
                devices = lf2.Freenect2().enumerateDevices()
                self.kinect_connected = len(devices) > 0
            except Exception:
                self.kinect_connected = False
        elif self.camera_mode == CAM_AZURE_K:
            try:
                import pyk4a
                self.kinect_connected = True
            except Exception:
                self.kinect_connected = False
        else:
            try:
                cap = cv2.VideoCapture(0)
                ok, _ = cap.read()
                cap.release()
                self.kinect_connected = bool(ok)
            except Exception:
                self.kinect_connected = False

    def init_ui(self):
        main_layout = QHBoxLayout()

        left_panel = QVBoxLayout()
        self.video_label = QLabel("Kamera Görüntüsü")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        left_panel.addWidget(self.video_label)

        self.status_label = QLabel("Hazır" if self.kinect_connected else "⚠️ Kamera Bağlı Değil!")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #2c3e50; color: white;")
        left_panel.addWidget(self.status_label)

        self.frame_count_label = QLabel("Toplanan Frame: 0")
        self.frame_count_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.frame_count_label)

        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("🎬 Taramaya Başla")
        self.start_btn.setMinimumHeight(50)
        self.start_btn.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #27ae60; color: white;")
        self.start_btn.clicked.connect(self.start_scan)
        self.start_btn.setEnabled(self.kinect_connected)

        self.stop_btn = QPushButton("⏹ Taramayı Bitir")
        self.stop_btn.setMinimumHeight(50)
        self.stop_btn.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #e74c3c; color: white;")
        self.stop_btn.clicked.connect(self.stop_scan)
        self.stop_btn.setEnabled(False)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        left_panel.addLayout(button_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_panel.addWidget(self.progress_bar)

        right_panel = QVBoxLayout()

        settings_group = QGroupBox("⚙️ Tarama Ayarları")
        settings_layout = QVBoxLayout()

        frame_skip_layout = QHBoxLayout()
        frame_skip_layout.addWidget(QLabel("Frame Aralığı:"))
        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(1, 20)
        self.frame_skip_spin.setValue(5)
        self.frame_skip_spin.setToolTip("Her kaç frame'de bir kayıt yapılacak (düşük = daha fazla veri)")
        self.frame_skip_spin.valueChanged.connect(lambda v: setattr(self, 'frame_skip', v))
        frame_skip_layout.addWidget(self.frame_skip_spin)
        settings_layout.addLayout(frame_skip_layout)

        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Kalite Seviyesi:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(['Otomatik', 'Hızlı', 'Orta', 'Yüksek'])
        self.quality_combo.setCurrentText('Orta')
        self.quality_combo.setToolTip("Otomatik donanımına göre seçer")
        quality_layout.addWidget(self.quality_combo)
        settings_layout.addLayout(quality_layout)

        self.icp_checkbox = QCheckBox("ICP Refinement Kullan")
        self.icp_checkbox.setChecked(True)
        self.icp_checkbox.setToolTip("Daha doğru hizalama için (biraz yavaşlatır)")
        settings_layout.addWidget(self.icp_checkbox)

        self.rt_preview_checkbox = QCheckBox("Gerçek Zamanlı 3D Önizleme")
        self.rt_preview_checkbox.setChecked(True)
        self.rt_preview_checkbox.setToolTip("Tarama sırasında nokta bulutu önizlemesi gösterilir")
        settings_layout.addWidget(self.rt_preview_checkbox)

        self.gpu_checkbox = QCheckBox("GPU Hızlandırma (Deneysel)")
        self.gpu_checkbox.setChecked(False)
        self.gpu_checkbox.setToolTip("Open3D Tensor/SLAM varsa kullanır (deneysel)")
        settings_layout.addWidget(self.gpu_checkbox)

        ai_layout = QHBoxLayout()
        ai_layout.addWidget(QLabel("AI Tamamlama:"))
        self.ai_combo = QComboBox()
        self.ai_combo.addItems(['Kapalı', 'PCN', 'SnowflakeNet'])
        self.ai_combo.setToolTip("Eksik bölgeler için AI destekli nokta tamamlama (placeholder)")
        ai_layout.addWidget(self.ai_combo)
        settings_layout.addLayout(ai_layout)

        self.texture_checkbox = QCheckBox("Gerçek Zamanlı Texture Mapping")
        self.texture_checkbox.setChecked(True)
        self.texture_checkbox.setToolTip("TSDF sonrası RGB projeksiyonu ile renk optimizasyonu")
        settings_layout.addWidget(self.texture_checkbox)

        self.slam_checkbox = QCheckBox("Loop Closure / SLAM")
        self.slam_checkbox.setChecked(True)
        self.slam_checkbox.setToolTip("Uzun taramalarda drift düzeltme")
        settings_layout.addWidget(self.slam_checkbox)

        self.segment_checkbox = QCheckBox("AI Destekli Segmentasyon")
        self.segment_checkbox.setChecked(False)
        self.segment_checkbox.setToolTip("Derinlik + AI segmentasyon ile arka planı maskele")
        settings_layout.addWidget(self.segment_checkbox)

        self.hybrid_checkbox = QCheckBox("Hibrit Kamera (Webcam RGB)")
        self.hybrid_checkbox.setChecked(False)
        self.hybrid_checkbox.setToolTip("Kinect derinlik + webcam renk")
        settings_layout.addWidget(self.hybrid_checkbox)

        self.marker_checkbox = QCheckBox("Marker Tabanlı Alignment (ArUco)")
        self.marker_checkbox.setChecked(False)
        self.marker_checkbox.setToolTip("ArUco marker ile hizalama (marker_size=5cm)")
        settings_layout.addWidget(self.marker_checkbox)

        # USB / MiDaS options
        self.midas_checkbox = QCheckBox("USB için MiDaS Monocular Depth kullan")
        self.midas_checkbox.setChecked(self.use_midas_for_usb and MI_DAS_AVAILABLE)
        self.midas_checkbox.setEnabled(MI_DAS_AVAILABLE)
        self.midas_checkbox.toggled.connect(lambda v: setattr(self, 'use_midas_for_usb', v))
        settings_layout.addWidget(self.midas_checkbox)

        self.rgb_only_checkbox = QCheckBox("Sadece RGB kaydet (photogrammetry)")
        self.rgb_only_checkbox.setChecked(False)
        self.rgb_only_checkbox.toggled.connect(lambda v: setattr(self, 'rgb_only_mode', v))
        settings_layout.addWidget(self.rgb_only_checkbox)

        md_layout = QHBoxLayout()
        md_layout.addWidget(QLabel("Kinect Sayısı:"))
        self.device_spin = QSpinBox()
        self.device_spin.setRange(1, 4)
        self.device_spin.setValue(1)
        self.device_spin.setToolTip("Aynı anda kullanılacak Kinect sayısı")
        self.device_spin.valueChanged.connect(self.update_device_count)
        md_layout.addWidget(self.device_spin)
        settings_layout.addLayout(md_layout)

        settings_group.setLayout(settings_layout)
        right_panel.addWidget(settings_group)

        multi_group = QGroupBox("📷 Multi-Scan Kontrolü")
        multi_layout = QHBoxLayout()
        self.new_scan_btn = QPushButton("Yeni Tarama Ekle")
        self.new_scan_btn.setToolTip("Mevcut taramayı paketleyip yeni taramaya başla")
        self.new_scan_btn.clicked.connect(self.pack_current_scan)
        self.new_scan_btn.setEnabled(False)
        multi_layout.addWidget(self.new_scan_btn)

        self.merge_scans_btn = QPushButton("Taramaları Birleştir ve İşle")
        self.merge_scans_btn.setToolTip("Eklenmiş taramaları tek modelde birleştir")
        self.merge_scans_btn.clicked.connect(self.merge_and_reconstruct)
        self.merge_scans_btn.setEnabled(False)
        multi_layout.addWidget(self.merge_scans_btn)
        multi_group.setLayout(multi_layout)
        right_panel.addWidget(multi_group)

        export_group = QGroupBox("💾 Export Ayarları")
        export_layout = QVBoxLayout()
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(['OBJ', 'STL', 'PLY', 'GLB', 'GLTF'])
        format_layout.addWidget(self.format_combo)
        export_layout.addLayout(format_layout)

        self.export_btn = QPushButton("📁 Farklı Kaydet")
        self.export_btn.clicked.connect(self.export_mesh)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        export_group.setLayout(export_layout)
        right_panel.addWidget(export_group)

        log_group = QGroupBox("📋 İşlem Günlüğü")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(230)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: monospace;")
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        right_panel.addWidget(log_group)

        right_panel.addStretch()

        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        self.setLayout(main_layout)

    def update_device_count(self, v):
        self.device_count = v
        self.device_ids = list(range(v))
        self.log(f"Kinect cihaz sayısı: {v}")

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")

    def init_preview(self):
        if self.preview_vis is not None:
            return
        try:
            self.preview_vis = o3d.visualization.Visualizer()
            self.preview_vis.create_window(window_name="RT Nokta Bulutu Önizleme", width=640, height=480, visible=True)
            self.preview_pcd = o3d.geometry.PointCloud()
            self.preview_vis.add_geometry(self.preview_pcd)
        except Exception:
            self.preview_vis = None
            self.preview_pcd = None

    def update_preview(self, rgbd):
        if not self.rt_preview_checkbox.isChecked():
            return
        if self.preview_vis is None:
            self.init_preview()
            if self.preview_vis is None:
                return
        with self.preview_lock:
            try:
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, self.intrinsic)
                pcd.transform([[1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]])
                self.preview_pcd += pcd.voxel_down_sample(voxel_size=0.01)
                try:
                    if len(self.preview_pcd.points) > 200000:
                        self.preview_pcd = self.preview_pcd.voxel_down_sample(voxel_size=0.02)
                except Exception:
                    pass
                try:
                    self.preview_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                except Exception:
                    pass
                self.preview_pcd.estimate_normals()
                self.preview_vis.update_geometry(self.preview_pcd)
                self.preview_vis.poll_events()
                self.preview_vis.update_renderer()
            except Exception:
                pass

    def ai_segment_mask(self, rgb_bgr):
        if not self.segment_checkbox.isChecked():
            return None
        if not (TORCH_AVAILABLE and TORCHVISION_AVAILABLE):
            return None
        try:
            global SEGMENT_MODEL
            if SEGMENT_MODEL is None:
                SEGMENT_MODEL = torchvision.models.segmentation.deeplabv3_resnet50(weights="DEFAULT").eval()
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            rgb_rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
            inp = transform(rgb_rgb).unsqueeze(0)
            with torch.no_grad():
                out = SEGMENT_MODEL(inp)["out"][0].softmax(0)
                conf, cls = torch.max(out, dim=0)
                mask = (conf > 0.5).cpu().numpy().astype(np.uint8)
            return mask
        except Exception:
            return None

    def apply_segment(self, depth, rgb=None):
        d = depth.copy().astype(np.uint16)
        d[d == 0] = 0
        d[d > 4500] = 0
        if rgb is not None:
            mask = self.ai_segment_mask(rgb)
            if mask is not None:
                try:
                    h_d, w_d = d.shape
                    if mask.shape[0] == h_d and mask.shape[1] == w_d:
                        mask_u16 = mask.astype(np.uint16)
                    else:
                        mask_rs = cv2.resize(mask, (w_d, h_d), interpolation=cv2.INTER_NEAREST)
                        mask_u16 = mask_rs.astype(np.uint16)
                    d = d * mask_u16
                except Exception:
                    pass
        return d

    def get_webcam_frame_and_depth(self):
        if self.webcam is None:
            try:
                self.webcam = cv2.VideoCapture(0)
            except Exception:
                self.webcam = None
        ret, frame = (False, None)
        try:
            if self.webcam is not None:
                ret, frame = self.webcam.read()
        except Exception:
            ret = False
        if not ret or frame is None:
            return None, None

        depth = None
        if self.use_midas_for_usb and MI_DAS_AVAILABLE:
            try:
                if not init_midas_model():
                    depth = None
                else:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    input_transform = MIDAS_TRANSFORMS.small_transform if MIDAS_TRANSFORMS is not None and hasattr(MIDAS_TRANSFORMS, 'small_transform') else MIDAS_TRANSFORMS.default_transform if MIDAS_TRANSFORMS is not None else None
                    if input_transform is None:
                        depth = None
                    else:
                        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        inp = input_transform(img).to(device)
                        with torch.no_grad():
                            prediction = MIDAS_MODEL(inp.unsqueeze(0))
                            prediction = torch.nn.functional.interpolate(
                                prediction.unsqueeze(1),
                                size=img.shape[:2],
                                mode="bilinear",
                                align_corners=False
                            ).squeeze()
                            depth_map = prediction.cpu().numpy()
                            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
                            depth = (depth_map * 4500).astype(np.uint16)
            except Exception:
                depth = None
        return frame, depth

    def get_rgb_frames(self):
        rgbs = []
        if self.camera_mode == CAM_KINECT_V1:
            if not FREENECT_AVAILABLE:
                for _ in self.device_ids:
                    rgbs.append(None)
                return rgbs
            for dev in self.device_ids:
                try:
                    rgb, _ = freenect.sync_get_video(devnum=dev)
                except Exception:
                    rgb = None
                if rgb is None:
                    rgbs.append(None)
                    continue
                try:
                    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                except Exception:
                    pass
                if self.hybrid_checkbox.isChecked():
                    if self.webcam is None:
                        try:
                            self.webcam = cv2.VideoCapture(0)
                        except Exception:
                            self.webcam = None
                    if self.webcam is not None:
                        ret, wrgb = self.webcam.read()
                        if ret and wrgb is not None:
                            try:
                                wrgb = cv2.resize(wrgb, (rgb.shape[1], rgb.shape[0]))
                                rgbs.append(wrgb)
                                continue
                            except Exception:
                                pass
                rgbs.append(rgb)
            return rgbs

        elif self.camera_mode == CAM_KINECT_V2:
            try:
                import pylibfreenect2 as lf2
                for _ in self.device_ids:
                    rgbs.append(None)
                return rgbs
            except Exception:
                self.log("Kinect v2 kütüphanesi bulunamadı; USB kameraya fallback.")
                self.camera_mode = CAM_USB
                return self.get_rgb_frames()

        elif self.camera_mode == CAM_AZURE_K:
            try:
                import pyk4a
                for _ in self.device_ids:
                    rgbs.append(None)
                return rgbs
            except Exception:
                self.log("Azure Kinect SDK bulunamadı; USB kameraya fallback.")
                self.camera_mode = CAM_USB
                return self.get_rgb_frames()

        else:  # CAM_USB
            for dev in self.device_ids:
                try:
                    cap = cv2.VideoCapture(dev)
                    ret, frame = cap.read()
                    cap.release()
                except Exception:
                    ret, frame = False, None
                if not ret or frame is None:
                    rgbs.append(None)
                else:
                    rgbs.append(frame)
            return rgbs

    def update_frame(self):
        try:
            rgbs = self.get_rgb_frames()
            primary_rgb = rgbs[0] if rgbs and rgbs[0] is not None else None

            # If no primary rgb from chosen device, try webcam fallback for USB mode
            if primary_rgb is None and self.camera_mode == CAM_USB:
                frame, depth = self.get_webcam_frame_and_depth()
                if frame is None:
                    if self.kinect_connected:
                        self.kinect_connected = False
                        self.status_label.setText("⚠️ Kamera Bağlantısı Kesildi!")
                        self.log("HATA: Kamera bağlantısı kesildi")
                    return
                primary_rgb = frame
                if self.scanning:
                    if depth is not None and not self.rgb_only_mode:
                        try:
                            rgb_o3d = o3d.geometry.Image(cv2.cvtColor(primary_rgb, cv2.COLOR_BGR2RGB))
                            depth_o3d = o3d.geometry.Image(depth.astype(np.uint16))
                            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                                rgb_o3d, depth_o3d,
                                depth_scale=1000.0,
                                depth_trunc=4.5,
                                convert_rgb_to_intensity=False
                            )
                            self.rgbd_frames.append(rgbd)
                            self.color_np_frames.append(primary_rgb)
                            self.frame_count_label.setText(f"Toplanan Frame: {len(self.rgbd_frames)}")
                            cv2.rectangle(primary_rgb, (0, 0), (639, 479), (0, 255, 0), 5)
                            if self.rt_preview_checkbox.isChecked():
                                self.update_preview(rgbd)
                        except Exception as e:
                            self.log(f"USB RGBD oluşturma hatası: {e}")
                    else:
                        self.color_np_frames.append(primary_rgb)
                        self.frame_count_label.setText(f"Toplanan Renk Frame: {len(self.color_np_frames)}")

            if primary_rgb is None:
                if self.kinect_connected:
                    self.kinect_connected = False
                    self.status_label.setText("⚠️ Kamera Bağlantısı Kesildi!")
                    self.log("HATA: Kamera bağlantısı kesildi")
                return

            if not self.kinect_connected:
                self.kinect_connected = True
                self.status_label.setText("Hazır")
                self.log("Kamera bağlantısı yeniden kuruldu")
                self.start_btn.setEnabled(True)

            if self.scanning:
                self.frame_counter += 1
                if self.frame_counter % self.frame_skip == 0:
                    for dev_index, rgb in enumerate(rgbs):
                        if rgb is None:
                            # USB branch handled above
                            continue
                        depth = None
                        if self.camera_mode == CAM_KINECT_V1:
                            try:
                                depth, _ = freenect.sync_get_depth(devnum=dev_index)
                            except Exception:
                                depth = None
                        elif self.camera_mode == CAM_USB:
                            if self.use_midas_for_usb and MI_DAS_AVAILABLE:
                                try:
                                    _, depth = self.get_webcam_frame_and_depth()
                                except Exception:
                                    depth = None
                            else:
                                depth = None
                        else:
                            depth = None

                        if depth is not None:
                            depth = self.apply_segment(depth, rgb=rgb)
                            try:
                                rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                            except Exception:
                                rgb_o3d = o3d.geometry.Image(np.asarray(rgb))
                            try:
                                depth_o3d = o3d.geometry.Image(depth.astype(np.uint16))
                            except Exception:
                                depth_o3d = o3d.geometry.Image(depth)
                            try:
                                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                                    rgb_o3d, depth_o3d,
                                    depth_scale=1000.0,
                                    depth_trunc=4.5,
                                    convert_rgb_to_intensity=False
                                )
                                self.rgbd_frames.append(rgbd)
                                self.color_np_frames.append(rgb)
                                self.frame_count_label.setText(f"Toplanan Frame: {len(self.rgbd_frames)}")
                                cv2.rectangle(rgb, (0, 0), (639, 479), (0, 255, 0), 5)
                                if dev_index == 0:
                                    self.update_preview(rgbd)
                            except Exception as e:
                                self.log(f"RGBD oluşturma hatası: {e}")
                        else:
                            # sadece renk kaydı (photogrammetry) veya depth yok
                            if self.rgb_only_mode:
                                self.color_np_frames.append(rgb)
                                self.frame_count_label.setText(f"Toplanan Renk Frame: {len(self.color_np_frames)}")

            if primary_rgb is not None:
                h, w, ch = primary_rgb.shape
                bytes_per_line = ch * w
                qt_img = QImage(primary_rgb.data, w, h, bytes_per_line, QImage.Format_BGR888)
                self.video_label.setPixmap(
                    QPixmap.fromImage(qt_img).scaled(
                        self.video_label.width(),
                        self.video_label.height(),
                        Qt.KeepAspectRatio
                    )
                )

        except Exception as e:
            self.log(f"Frame güncelleme hatası: {str(e)}")

    def start_scan(self):
        if not self.kinect_connected:
            QMessageBox.warning(self, "Bağlantı Hatası", "Kamera bağlı değil!")
            return

        self.scanning = True
        self.rgbd_frames = []
        self.color_np_frames = []
        self.frame_counter = 0

        if self.quality_combo.currentText() == 'Otomatik':
            self.quality_combo.setCurrentText(self.auto_quality())

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("🔴 TARAMA AKTIF")
        self.status_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #c0392b; color: white;")
        self.new_scan_btn.setEnabled(True)

        self.log("Tarama başladı! Nesneyi yavaşça döndürün...")

    def stop_scan(self):
        self.scanning = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if self.preview_vis is not None:
            try:
                self.preview_vis.destroy_window()
            except Exception:
                pass
            self.preview_vis = None
            self.preview_pcd = None

        if not self.rgbd_frames and not self.color_np_frames:
            QMessageBox.warning(self, "Yetersiz Veri", "En az 5 frame gerekli! Toplanan: 0")
            self.status_label.setText("Hazır")
            return

        if len(self.rgbd_frames) < 5 and not self.rgb_only_mode:
            QMessageBox.warning(self, "Yetersiz Veri",
                                f"En az 5 RGBD frame gerekli! Toplanan RGBD: {len(self.rgbd_frames)}")
            self.status_label.setText("Hazır")
            return

        self.log(f"Tarama tamamlandı. RGBD frame: {len(self.rgbd_frames)}, RGB-only frame: {len(self.color_np_frames)}")
        self.status_label.setText("⚙️ İŞLENİYOR...")
        self.status_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f39c12; color: white;")

        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        if len(self.scans) >= 1:
            self.log("Tek paket veya birleştirilmemiş tarama işleniyor...")
            rgbd_data = self.rgbd_frames.copy()
            scans_input = self.scans.copy() + ([rgbd_data] if rgbd_data else [])
        else:
            scans_input = [self.rgbd_frames.copy()]

        self.reconstruction_thread = ReconstructionThread(
            scans_input,
            self.intrinsic,
            self.quality_combo.currentText(),
            self.icp_checkbox.isChecked(),
            use_gpu=self.gpu_checkbox.isChecked(),
            use_texture=self.texture_checkbox.isChecked(),
            ai_model=self.ai_combo.currentText(),
            enable_slam=self.slam_checkbox.isChecked(),
            use_marker_alignment=self.marker_checkbox.isChecked(),
            marker_size_m=0.05,
            noise_aware=True,
            rt_simplify=True
        )
        self.reconstruction_thread.progress.connect(self.update_progress)
        self.reconstruction_thread.finished.connect(self.reconstruction_finished)
        self.reconstruction_thread.start()

    def update_progress(self, value, message):
        self.progress_bar.setValue(value)
        self.log(message)

    def reconstruction_finished(self, mesh, success):
        self.progress_bar.setVisible(False)

        if not success or mesh is None:
            QMessageBox.critical(self, "Hata", "Mesh oluşturulamadı!")
            self.status_label.setText("❌ HATA")
            self.status_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #e74c3c; color: white;")
            return

        self.current_mesh = mesh

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scan_{timestamp}.obj"
        filepath = os.path.join(BASE_DIR, filename)

        try:
            o3d.io.write_triangle_mesh(filepath, mesh)
            self.log(f"✅ Mesh kaydedildi: {filename}")
            self.log(f"   Vertices: {len(mesh.vertices)}, Faces: {len(mesh.triangles)}")

            self.status_label.setText("✅ TAMAMLANDI")
            self.status_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #27ae60; color: white;")

            self.export_btn.setEnabled(True)

            self.log("Mesh önizlemesi açılıyor...")
            try:
                o3d.visualization.draw_geometries([mesh], window_name="3D Mesh Önizleme")
            except Exception:
                pass

            QMessageBox.information(self, "Başarılı",
                                    f"3D model başarıyla oluşturuldu!\n\nDosya: {filename}\nKonum: {BASE_DIR}")

        except Exception as e:
            self.log(f"❌ Kaydetme hatası: {str(e)}")
            QMessageBox.critical(self, "Kayıt Hatası", f"Mesh kaydedilemedi:\n{str(e)}")

        self.scans = []
        self.merge_scans_btn.setEnabled(False)
        self.new_scan_btn.setEnabled(False)

    def export_mesh(self):
        if not hasattr(self, 'current_mesh'):
            return

        format_ext = self.format_combo.currentText().lower()
        filename, _ = QFileDialog.getSaveFileName(
            self, "Mesh Kaydet",
            os.path.join(BASE_DIR, f"scan.{format_ext}"),
            f"{format_ext.upper()} Files (*.{format_ext})"
        )

        if filename:
            try:
                o3d.io.write_triangle_mesh(filename, self.current_mesh)
                self.log(f"✅ Export başarılı: {os.path.basename(filename)}")
                QMessageBox.information(self, "Başarılı", f"Mesh kaydedildi:\n{filename}")

                if format_ext in ('glb', 'gltf'):
                    self.launch_webxr_viewer(filename)

            except Exception as e:
                self.log(f"❌ Export hatası: {str(e)}")
                QMessageBox.critical(self, "Hata", f"Export başarısız:\n{str(e)}")

    def launch_webxr_viewer(self, model_path):
        html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>WebXR Model Viewer</title>
<style>body,html{{margin:0;height:100%;background:#111}} canvas{{display:block}}</style>
</head>
<body>
<script src="https://unpkg.com/three@0.159.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.159.0/examples/js/loaders/GLTFLoader.js"></script>
<script src="https://unpkg.com/three@0.159.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://unpkg.com/three@0.159.0/examples/jsm/webxr/VRButton.js"></script>
<script>
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.01, 100);
camera.position.set(0.5, 0.5, 1.2);
const renderer = new THREE.WebGLRenderer({{antialias:true}});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true;
document.body.appendChild(renderer.domElement);
document.body.appendChild(THREE.VRButton.createButton(renderer));
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
const light = new THREE.HemisphereLight(0xffffff, 0x444444, 1.0);
scene.add(light);
const loader = new THREE.GLTFLoader();
loader.load("{model_path.replace('\\', '/')}", gltf => {{
    const m = gltf.scene;
    scene.add(m);
}}, undefined, err => console.error(err));
window.addEventListener('resize', () => {{
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}});
function animate(){{
    renderer.setAnimationLoop(() => {{
        controls.update();
        renderer.render(scene, camera);
    }});
}}
animate();
</script>
</body>
</html>
"""
        html_path = os.path.join(BASE_DIR, "viewer_xr.html")
        try:
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html)
            if os.name == 'nt':
                os.startfile(html_path)
            else:
                if sys.platform == 'darwin':
                    os.system(f"open '{html_path}'")
                else:
                    os.system(f"xdg-open '{html_path}'")
        except Exception:
            pass

    def pack_current_scan(self):
        if not self.rgbd_frames and not self.color_np_frames:
            QMessageBox.warning(self, "Boş Tarama", "Paketlenecek veri yok.")
            return
        # Eğer RGBD varsa paketle, yoksa RGB-only paketle
        if self.rgbd_frames:
            self.scans.append(self.rgbd_frames.copy())
        else:
            self.scans.append(self.color_np_frames.copy())
        self.log(f"Tarama paketi eklendi. Toplam paket: {len(self.scans)}")
        self.rgbd_frames = []
        self.color_np_frames = []
        self.frame_count_label.setText("Toplanan Frame: 0")
        self.merge_scans_btn.setEnabled(len(self.scans) >= 2)

    def merge_and_reconstruct(self):
        if len(self.scans) < 2:
            QMessageBox.warning(self, "Yetersiz Paket", "En az iki tarama paketi gerekli.")
            return
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.reconstruction_thread = ReconstructionThread(
            self.scans.copy(),
            self.intrinsic,
            self.quality_combo.currentText(),
            self.icp_checkbox.isChecked(),
            use_gpu=self.gpu_checkbox.isChecked(),
            use_texture=self.texture_checkbox.isChecked(),
            ai_model=self.ai_combo.currentText(),
            enable_slam=self.slam_checkbox.isChecked(),
            use_marker_alignment=self.marker_checkbox.isChecked(),
            marker_size_m=0.05,
            noise_aware=True,
            rt_simplify=True
        )
        self.reconstruction_thread.progress.connect(self.update_progress)
        self.reconstruction_thread.finished.connect(self.reconstruction_finished)
        self.reconstruction_thread.start()

    def auto_quality(self):
        if not PSUTIL_AVAILABLE:
            return 'Orta'
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        cpu_count = psutil.cpu_count(logical=True)
        if ram_gb >= 16 and cpu_count >= 8:
            return 'Yüksek'
        elif ram_gb >= 8 and cpu_count >= 4:
            return 'Orta'
        else:
            return 'Hızlı'


def run_gui():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    scanner = KinectScanner()
    scanner.show()
    sys.exit(app.exec_())


def run_batch(args):
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(640, 480, 525.0, 525.0, 319.5, 239.5)

    def load_scan_dir(d):
        color_paths = sorted(Path(d).glob("color_*.png"))
        depth_paths = sorted(Path(d).glob("depth_*.png"))
        frames = []
        for cp, dp in zip(color_paths, depth_paths):
            color = cv2.cvtColor(cv2.imread(str(cp)), cv2.COLOR_BGR2RGB)
            depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
            rgb_o3d = o3d.geometry.Image(color)
            depth_o3d = o3d.geometry.Image(depth.astype(np.uint16))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_o3d, depth_o3d,
                depth_scale=1000.0, depth_trunc=4.5,
                convert_rgb_to_intensity=False
            )
            frames.append(rgbd)
        return frames

    scans = []
    for d in args.inputs:
        scans.append(load_scan_dir(d))

    def on_finish(mesh, ok):
        if not ok or mesh is None:
            print("Batch: Reconstruction failed.")
            return
        out = args.output if args.output else os.path.join(BASE_DIR, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.ply")
        o3d.io.write_triangle_mesh(out, mesh)
        print(f"Batch: Saved mesh -> {out}")

    rt = ReconstructionThread(
        scans, intrinsic, args.quality, use_icp=True,
        use_gpu=args.gpu, use_texture=args.texture,
        ai_model=args.ai, enable_slam=args.slam,
        use_marker_alignment=args.marker, marker_size_m=0.05,
        noise_aware=True, rt_simplify=True
    )
    rt.progress.connect(lambda v, m: print(f"[{v}%] {m}"))
    rt.finished.connect(on_finish)
    rt.start()
    rt.wait()


def main():
    parser = argparse.ArgumentParser(description="Kinect 3D Tarama Sistemi - Pro Edition")
    parser.add_argument("--nogui", action="store_true", help="GUI olmadan batch mode çalıştır")
    parser.add_argument("--inputs", nargs="*", help="Batch mode için giriş dizinleri (color_*.png, depth_*.png)")
    parser.add_argument("--quality", default="Orta", choices=["Hızlı", "Orta", "Yüksek"], help="Reconstruction kalite seviyesi")
    parser.add_argument("--gpu", action="store_true", help="GPU hızlandırma (deneysel)")
    parser.add_argument("--texture", action="store_true", help="Texture mapping aktif")
    parser.add_argument("--ai", default="Kapalı", choices=["Kapalı", "PCN", "SnowflakeNet"], help="AI tamamlama modu")
    parser.add_argument("--slam", action="store_true", help="Loop closure / SLAM aktif")
    parser.add_argument("--marker", action="store_true", help="Marker tabanlı alignment (ArUco) aktif")
    parser.add_argument("--output", help="Çıkış mesh dosyası yolu")
    args, unknown = parser.parse_known_args()

    if args.nogui:
        if not args.inputs:
            print("Batch mode için --inputs dizinleri gereklidir.")
            sys.exit(1)
        run_batch(args)
    else:
        run_gui()


if __name__ == "__main__":
    main()
