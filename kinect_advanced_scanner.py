import sys
import os
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import freenect
import open3d as o3d
import trimesh
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                              QVBoxLayout, QHBoxLayout, QProgressBar, 
                              QSpinBox, QComboBox, QGroupBox, QCheckBox,
                              QMessageBox, QFileDialog, QTextEdit)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont

BASE_DIR = "KinectScans"
os.makedirs(BASE_DIR, exist_ok=True)

class ReconstructionThread(QThread):
    """Arka planda mesh oluşturma için thread"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, bool)
    
    def __init__(self, rgbd_frames, intrinsic, quality_level, use_icp):
        super().__init__()
        self.rgbd_frames = rgbd_frames
        self.intrinsic = intrinsic
        self.quality_level = quality_level
        self.use_icp = use_icp
        
    def run(self):
        try:
            self.progress.emit(10, "Pose estimation başlıyor...")
            poses = self.compute_poses()
            
            self.progress.emit(40, "TSDF integration yapılıyor...")
            mesh = self.tsdf_integration(poses)
            
            self.progress.emit(70, "Mesh temizleniyor...")
            mesh = self.clean_mesh(mesh)
            
            self.progress.emit(90, "AI iyileştirme uygulanıyor...")
            mesh = self.ai_enhance(mesh)
            
            self.progress.emit(100, "Tamamlandı!")
            self.finished.emit(mesh, True)
            
        except Exception as e:
            self.progress.emit(0, f"Hata: {str(e)}")
            self.finished.emit(None, False)
    
    def compute_poses(self):
        """Gelişmiş pose estimation (Odometry + ICP)"""
        pose = np.eye(4)
        poses = [pose.copy()]
        
        success_count = 0
        total_drift = 0.0
        
        for i in range(1, len(self.rgbd_frames)):
            # RGB-D Odometry
            success_odo, odo_trans, info = o3d.pipelines.odometry.compute_rgbd_odometry(
                self.rgbd_frames[i-1], 
                self.rgbd_frames[i], 
                self.intrinsic,
                np.eye(4),
                o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()
            )
            
            if success_odo:
                # ICP ile refine et (eğer aktifse)
                if self.use_icp:
                    source_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                        self.rgbd_frames[i], self.intrinsic
                    )
                    target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                        self.rgbd_frames[i-1], self.intrinsic
                    )
                    
                    icp_result = o3d.pipelines.registration.registration_icp(
                        source_pcd, target_pcd, 0.05, odo_trans,
                        o3d.pipelines.registration.TransformationEstimationPointToPoint()
                    )
                    
                    if icp_result.fitness > 0.3:  # ICP başarılıysa
                        odo_trans = icp_result.transformation
                
                # Drift kontrolü
                translation = np.linalg.norm(odo_trans[:3, 3])
                if translation < 0.5:  # Aşırı büyük hareket reddedilir
                    pose = pose @ odo_trans
                    success_count += 1
                    total_drift += translation
                else:
                    print(f"[WARN] Frame {i}: Aşırı büyük hareket tespit edildi ({translation:.3f}m)")
            
            poses.append(pose.copy())
            
            if i % 10 == 0:
                progress = int(10 + (i / len(self.rgbd_frames)) * 30)
                avg_drift = total_drift / max(success_count, 1)
                self.progress.emit(progress, 
                    f"Pose {i}/{len(self.rgbd_frames)} - Başarı: {success_count}/{i} - Drift: {avg_drift:.4f}m")
        
        print(f"[INFO] Toplam {success_count}/{len(self.rgbd_frames)-1} pose başarılı")
        return poses
    
    def tsdf_integration(self, poses):
        """Optimized TSDF integration"""
        # Kalite seviyesine göre parametreler
        quality_params = {
            'Hızlı': {'voxel': 0.008, 'sdf_trunc': 0.04},
            'Orta': {'voxel': 0.004, 'sdf_trunc': 0.02},
            'Yüksek': {'voxel': 0.002, 'sdf_trunc': 0.01}
        }
        
        params = quality_params.get(self.quality_level, quality_params['Orta'])
        
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=params['voxel'],
            sdf_trunc=params['sdf_trunc'],
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        for idx, (rgbd, T) in enumerate(zip(self.rgbd_frames, poses)):
            volume.integrate(rgbd, self.intrinsic, np.linalg.inv(T))
            
            if idx % 10 == 0:
                progress = int(40 + (idx / len(self.rgbd_frames)) * 30)
                self.progress.emit(progress, f"Integration {idx}/{len(self.rgbd_frames)}")
        
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh
    
    def clean_mesh(self, mesh):
        """Mesh temizleme ve filtreleme"""
        # Statistical outlier removal
        mesh_clean, _ = mesh.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        # Küçük bağlantısız parçaları kaldır
        triangle_clusters, cluster_n_triangles, _ = mesh_clean.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        
        # En büyük cluster'ı tut
        if len(cluster_n_triangles) > 0:
            largest_cluster_idx = cluster_n_triangles.argmax()
            triangles_to_remove = triangle_clusters != largest_cluster_idx
            mesh_clean.remove_triangles_by_mask(triangles_to_remove)
        
        mesh_clean.remove_degenerate_triangles()
        mesh_clean.remove_duplicated_triangles()
        mesh_clean.remove_duplicated_vertices()
        mesh_clean.remove_non_manifold_edges()
        
        return mesh_clean
    
    def ai_enhance(self, mesh):
        """AI destekli mesh iyileştirme"""
        try:
            # Trimesh'e dönüştür
            m = trimesh.Trimesh(
                vertices=np.asarray(mesh.vertices),
                faces=np.asarray(mesh.triangles)
            )
            
            # Kalite seviyesine göre işlem
            if self.quality_level == 'Yüksek':
                # Subdivision ile detay artır
                m = m.subdivide()
                target_faces = int(len(m.faces) * 0.9)
            elif self.quality_level == 'Orta':
                target_faces = int(len(m.faces) * 0.7)
            else:
                target_faces = int(len(m.faces) * 0.5)
            
            # Quadratic decimation
            if len(m.faces) > target_faces:
                m = m.simplify_quadratic_decimation(target_faces)
            
            # Smoothing
            m = m.smoothed()
            
            # Open3D mesh'e geri dönüştür
            mesh_ai = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(m.vertices),
                o3d.utility.Vector3iVector(m.faces)
            )
            mesh_ai.compute_vertex_normals()
            
            # Renkleri koru
            if mesh.has_vertex_colors():
                mesh_ai.vertex_colors = mesh.vertex_colors
            
            return mesh_ai
            
        except Exception as e:
            print(f"[WARN] AI enhance hatası: {e}, orijinal mesh döndürülüyor")
            return mesh


class KinectScanner(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinect 3D Tarama Sistemi - Pro Edition")
        self.setGeometry(100, 100, 1000, 700)
        
        # Kinect durumu
        self.kinect_connected = False
        self.check_kinect_connection()
        
        # Tarama durumu
        self.scanning = False
        self.rgbd_frames = []
        self.last_frame_time = 0
        self.frame_skip = 5  # Her 5 frame'den 1'ini al
        self.frame_counter = 0
        self.reconstruction_thread = None
        
        # Kamera intrinsic parametreleri
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
        self.intrinsic.set_intrinsics(640, 480, 525.0, 525.0, 319.5, 239.5)
        
        self.init_ui()
        
        # Canlı görüntü timer'ı
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
    
    def check_kinect_connection(self):
        """Kinect bağlantısını kontrol et"""
        try:
            rgb, _ = freenect.sync_get_video()
            if rgb is not None:
                self.kinect_connected = True
                print("[INFO] ✅ Kinect bağlantısı başarılı")
            else:
                self.kinect_connected = False
                print("[ERROR] ❌ Kinect bağlanamadı")
        except Exception as e:
            self.kinect_connected = False
            print(f"[ERROR] Kinect hatası: {e}")
    
    def init_ui(self):
        """Gelişmiş UI oluştur"""
        main_layout = QHBoxLayout()
        
        # Sol panel - Video ve kontroller
        left_panel = QVBoxLayout()
        
        # Video label
        self.video_label = QLabel("Kinect Görüntüsü")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        left_panel.addWidget(self.video_label)
        
        # Durum bilgisi
        self.status_label = QLabel("Hazır" if self.kinect_connected else "⚠️ Kinect Bağlı Değil!")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #2c3e50; color: white;")
        left_panel.addWidget(self.status_label)
        
        # Frame sayacı
        self.frame_count_label = QLabel("Toplanan Frame: 0")
        self.frame_count_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.frame_count_label)
        
        # Kontrol butonları
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
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_panel.addWidget(self.progress_bar)
        
        # Sağ panel - Ayarlar ve log
        right_panel = QVBoxLayout()
        
        # Ayarlar grubu
        settings_group = QGroupBox("⚙️ Tarama Ayarları")
        settings_layout = QVBoxLayout()
        
        # Frame skip ayarı
        frame_skip_layout = QHBoxLayout()
        frame_skip_layout.addWidget(QLabel("Frame Aralığı:"))
        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(1, 20)
        self.frame_skip_spin.setValue(5)
        self.frame_skip_spin.setToolTip("Her kaç frame'de bir kayıt yapılacak (düşük = daha fazla veri)")
        self.frame_skip_spin.valueChanged.connect(lambda v: setattr(self, 'frame_skip', v))
        frame_skip_layout.addWidget(self.frame_skip_spin)
        settings_layout.addLayout(frame_skip_layout)
        
        # Kalite ayarı
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Kalite Seviyesi:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(['Hızlı', 'Orta', 'Yüksek'])
        self.quality_combo.setCurrentText('Orta')
        self.quality_combo.setToolTip("Yüksek = daha iyi kalite ama yavaş")
        quality_layout.addWidget(self.quality_combo)
        settings_layout.addLayout(quality_layout)
        
        # ICP refinement
        self.icp_checkbox = QCheckBox("ICP Refinement Kullan")
        self.icp_checkbox.setChecked(True)
        self.icp_checkbox.setToolTip("Daha doğru hizalama için (biraz yavaşlatır)")
        settings_layout.addWidget(self.icp_checkbox)
        
        # Mesh önizleme
        self.preview_checkbox = QCheckBox("İşlem Sonrası Önizleme Göster")
        self.preview_checkbox.setChecked(True)
        settings_layout.addWidget(self.preview_checkbox)
        
        settings_group.setLayout(settings_layout)
        right_panel.addWidget(settings_group)
        
        # Export grubu
        export_group = QGroupBox("💾 Export Ayarları")
        export_layout = QVBoxLayout()
        
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(['OBJ', 'STL', 'PLY'])
        format_layout.addWidget(self.format_combo)
        export_layout.addLayout(format_layout)
        
        self.export_btn = QPushButton("📁 Farklı Kaydet")
        self.export_btn.clicked.connect(self.export_mesh)
        self.export_btn.setEnabled(False)
        export_layout.addWidget(self.export_btn)
        
        export_group.setLayout(export_layout)
        right_panel.addWidget(export_group)
        
        # Log alanı
        log_group = QGroupBox("📋 İşlem Günlüğü")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: monospace;")
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        right_panel.addWidget(log_group)
        
        right_panel.addStretch()
        
        # Ana layout'a ekle
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(right_panel, 1)
        
        self.setLayout(main_layout)
        
        self.log("Sistem hazır. Kinect durumu: " + ("BAĞLI ✅" if self.kinect_connected else "BAĞLI DEĞİL ❌"))
    
    def log(self, message):
        """Log mesajı ekle"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
    
    def update_frame(self):
        """Canlı görüntü güncelleme"""
        try:
            rgb, _ = freenect.sync_get_video()
            
            if rgb is None:
                if self.kinect_connected:
                    self.kinect_connected = False
                    self.status_label.setText("⚠️ Kinect Bağlantısı Kesildi!")
                    self.log("HATA: Kinect bağlantısı kesildi")
                return
            
            # Bağlantı yeniden kurulduysa
            if not self.kinect_connected:
                self.kinect_connected = True
                self.status_label.setText("Hazır")
                self.log("Kinect bağlantısı yeniden kuruldu")
                self.start_btn.setEnabled(True)
            
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Tarama sırasında frame kaydet
            if self.scanning:
                self.frame_counter += 1
                
                if self.frame_counter % self.frame_skip == 0:
                    depth, _ = freenect.sync_get_depth()
                    
                    if depth is not None:
                        # Depth filtreleme (0 ve çok uzak değerleri temizle)
                        depth = depth.astype(np.uint16)
                        depth[depth == 0] = 0
                        depth[depth > 4500] = 0  # 4.5m'den uzakları kes
                        
                        rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                        depth_o3d = o3d.geometry.Image(depth)
                        
                        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                            rgb_o3d, depth_o3d, 
                            depth_scale=1000.0,
                            depth_trunc=4.5,
                            convert_rgb_to_intensity=False
                        )
                        
                        self.rgbd_frames.append(rgbd)
                        self.frame_count_label.setText(f"Toplanan Frame: {len(self.rgbd_frames)}")
                        
                        # Yeşil çerçeve ekle (tarama aktif göstergesi)
                        cv2.rectangle(rgb, (0, 0), (639, 479), (0, 255, 0), 5)
            
            # PyQt görüntüsüne dönüştür
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
            self.log(f"Frame güncelleme hatası: {str(e)}")
    
    def start_scan(self):
        """Taramayı başlat"""
        if not self.kinect_connected:
            QMessageBox.warning(self, "Bağlantı Hatası", "Kinect bağlı değil!")
            return
        
        self.scanning = True
        self.rgbd_frames = []
        self.frame_counter = 0
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("🔴 TARAMA AKTIF")
        self.status_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #c0392b; color: white;")
        
        self.log("Tarama başladı! Nesneyi yavaşça döndürün...")
    
    def stop_scan(self):
        """Taramayı durdur ve reconstruction başlat"""
        self.scanning = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if len(self.rgbd_frames) < 5:
            QMessageBox.warning(self, "Yetersiz Veri", 
                f"En az 5 frame gerekli! Toplanan: {len(self.rgbd_frames)}")
            self.status_label.setText("Hazır")
            return
        
        self.log(f"Tarama tamamlandı. {len(self.rgbd_frames)} frame toplandı.")
        self.status_label.setText("⚙️ İŞLENİYOR...")
        self.status_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f39c12; color: white;")
        
        # Reconstruction thread'i başlat
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.reconstruction_thread = ReconstructionThread(
            self.rgbd_frames.copy(),
            self.intrinsic,
            self.quality_combo.currentText(),
            self.icp_checkbox.isChecked()
        )
        
        self.reconstruction_thread.progress.connect(self.update_progress)
        self.reconstruction_thread.finished.connect(self.reconstruction_finished)
        self.reconstruction_thread.start()
    
    def update_progress(self, value, message):
        """Progress güncelleme"""
        self.progress_bar.setValue(value)
        self.log(message)
    
    def reconstruction_finished(self, mesh, success):
        """Reconstruction tamamlandı"""
        self.progress_bar.setVisible(False)
        
        if not success or mesh is None:
            QMessageBox.critical(self, "Hata", "Mesh oluşturulamadı!")
            self.status_label.setText("❌ HATA")
            self.status_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #e74c3c; color: white;")
            return
        
        self.current_mesh = mesh
        
        # Otomatik kaydet
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
            
            # Önizleme göster
            if self.preview_checkbox.isChecked():
                self.log("Mesh önizlemesi açılıyor...")
                o3d.visualization.draw_geometries([mesh], window_name="3D Mesh Önizleme")
            
            QMessageBox.information(self, "Başarılı", 
                f"3D model başarıyla oluşturuldu!\n\nDosya: {filename}\nKonum: {BASE_DIR}")
            
        except Exception as e:
            self.log(f"❌ Kaydetme hatası: {str(e)}")
            QMessageBox.critical(self, "Kayıt Hatası", f"Mesh kaydedilemedi:\n{str(e)}")
    
    def export_mesh(self):
        """Mesh'i farklı formatta kaydet"""
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
            except Exception as e:
                self.log(f"❌ Export hatası: {str(e)}")
                QMessageBox.critical(self, "Hata", f"Export başarısız:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    
    # Uygulama stili
    app.setStyle('Fusion')
    
    scanner = KinectScanner()
    scanner.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
