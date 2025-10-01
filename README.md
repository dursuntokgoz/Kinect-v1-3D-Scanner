# Kinect 3D Scanner

Profesyonel Open3D tabanlı Kinect / USB 3B tarayıcı uygulaması — renk ve derinlik verisinden TSDF tabanlı yüzey rekonstrüksiyonu yapan PyQt5 GUI ve batch aracı.

---

## Özellikler
- Kamera desteği: Kinect v1 (freenect), USB webcam, Simulated.  
- Gerçek zamanlı önizleme ve tarama göstergesi.  
- Frame buffering ve ayarlanabilir frame-skip.  
- Arka plan reconstruct thread: RGB-D odometri, TSDF entegrasyonu, mesh temizleme, normal hesaplama.  
- Export formatları: PLY, OBJ, STL, GLB.  
- Batch modu: color_*.png ve depth_*.png dosyalarından offline rekonstrüksiyon.  
- Otomatik kalite seçimi sistem kaynaklarına göre.  
- Kapsamlı hata yakalama ve kullanıcıya yönelik log/mesajlar.

---

## Gereksinimler
- Python 3.8+  
- open3d, pyqt5, opencv-python  
- Opsiyonel: freenect (Kinect v1), psutil (auto-quality), trimesh, torch

Kurulum örneği:
```bash
pip install open3d pyqt5 opencv-python
# Kinect v1 desteği için (platforma göre farklılık gösterebilir)
pip install freenect
# İsteğe bağlı ek paketler
pip install psutil trimesh torch
