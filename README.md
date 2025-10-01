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
```



Dosya yapısı
kinect_advanced_scanner.py — Ana uygulama (GUI + batch).

scanner_config.json — Konfigürasyon dosyası (otomatik oluşturulur).

KinectScans/ — Kaydedilen tarama çıktılarının bulunduğu dizin.

Hızlı başlangıç — GUI
Gereksinimleri yükleyin.

Uygulamayı çalıştırın:

```bash
python kinect_advanced_scanner.py
```
Sağ panelden kamera seçin.

Start Scanning ile taramayı başlatın, nesneyi yavaşça çevirin.

Stop Scanning ile yakalamayı bitirin ve işleme başlatın. Sonuç KinectScans/ dizinine kaydedilir ve önizleme gösterilir.

Batch modu
Renk ve derinlik dosyaları eşleşmeli adlandırma ile aynı klasörde olmalıdır: color_0001.png, depth_0001.png vb.

Örnek:

bash
```
python kinect_advanced_scanner.py --batch --input /path/to/frames --output /path/to/out.ply --quality Orta
```
--quality seçenekleri: Hızlı, Orta, Yüksek

Konfigürasyon
scanner_config.json içindeki ana alanlar: quality, frame_skip, use_icp, use_gpu, use_texture, enable_slam, rt_preview, camera_type. Ayarlar uygulama kapanışında otomatik kaydedilir. Manuel düzenleme için JSON formatını koruyun.

İpuçları
Kinect v1 kullanırken freenect kurulumu ve USB izinlerine dikkat edin.

Kaliteli mesh için 20+ çeşitli frame önerilir.

Yüksek kalite daha fazla bellek ve işlem gücü gerektirir.

USB kameralar doğal derinlik verisi sağlamaz; gerçek 3B rekonstrüksiyon için derinlik sensörü kullanın.

Uzun işlemler için geçici dosya kullanıp tamamlandığında son dosya adına taşıma stratejisi kullanın.

Sık karşılaşılan sorunlar
Open3D bulunamadı hatası: aynı Python ortamında Open3D kurulu olduğundan emin olun.

freenect bulunamadı: Kinect v1 desteği opsiyoneldir; freenect yükleyin veya USB/Simulated kullanın.

Kamera görüntüsü yok: doğru cihaz seçili mi, başka uygulama kamerayı kullanıyor mu, OS izinleri doğru mu kontrol edin.

Batch modunda "Mismatch between color and depth files": color ve depth dosya sayıları eşit olmalıdır.

Geliştirici notları
Modüler yapı: Camera abstraction, FrameBuffer, ReconstructionThread.

Önerilen geliştirmeler: producer/consumer pipeline, tmp->final atomic kaydetme, birim testleri, CI entegrasyonu, Open3D sürüm uyumluluk kontrolleri.




