Kinect 3D Scanner — README
Profesyonel Open3D tabanlı Kinect / USB 3B tarayıcı uygulaması
Kısa açıklama Kinect 3D Scanner, Kinect v1, USB kamera veya simüle edilmiş kaynaklardan renk ve derinlik verisi alıp Open3D ile TSDF tabanlı yüzey yeniden yapılandırma yapan masaüstü bir PyQt5 uygulamasıdır. GUI, batch işleme ve yapılandırılabilir kalite/ayar seçenekleri içerir.

Özellikler
Kamera soyutlaması: Kinect v1 (freenect), USB webcam, Simulated.

Gerçek zamanlı önizleme ve akış görüntüleme.

Frame buffering ve frame-skip ile kontrol edilebilir yakalama.

Arka plan reconstruct thread: RGB-D odometri, TSDF entegrasyonu, mesh temizleme ve normal hesaplama.

Mesh export: PLY, OBJ, STL, GLB.

Batch modu: klasörden color_.png ve depth_.png okuyup offline rekonstrüksiyon.

Otomatik kalite seçimi (sistem kaynaklarına göre).

Hata yakalama ve kullanıcı yönlendiren log/mesaj kutuları.

Gereksinimler
Python 3.8+

Open3D

PyQt5

OpenCV (cv2)

opsiyonel: freenect (Kinect v1 için), psutil (otomatik kalite tespiti), trimesh, torch

Örnek pip komutları

bash
pip install open3d pyqt5 opencv-python
# Kinect v1 kullanacaksanız:
pip install cython
pip install freenect  # platforma göre farklılık gösterebilir
# isteğe bağlı:
pip install psutil trimesh torch
Dosya yapısı
kinect_advanced_scanner.py — Ana uygulama (GUI + batch).

scanner_config.json — Uygulama konfigürasyonu (otomatik oluşturulur).

KinectScans/ — Kaydedilen tarama çıktılarının bulunduğu dizin.

Hızlı başlangıç — GUI modu
Gereksinimleri yükleyin.

Terminalde çalıştırın:

bash
python kinect_advanced_scanner.py
Uygulama açıldığında sağ panelden kamera tipini seçin.

"Start Scanning" ile taramayı başlatın, nesneyi yavaşça çevirin.

"Stop Scanning" ile yakalamayı bitirin; işleme arka planda yapılacaktır.

İşlem tamamlandığında dosya KinectScans/ dizinine kaydedilir ve önizleme gösterilir.

Batch modu (komut satırı)
Renk ve derinlik görüntüleri eşleşmeli isimlendirme ile bir klasörde olmalı: color_0001.png, depth_0001.png vb. Örnek:

bash
python kinect_advanced_scanner.py --batch --input /path/to/frames --output /path/to/out.ply --quality Orta
--quality seçenekleri: Hızlı, Orta, Yüksek

Konfigürasyon
scanner_config.json içinde şu ana alanlar bulunur: quality, frame_skip, use_icp, use_gpu, use_texture, enable_slam, rt_preview, camera_type.

Uygulama kapatılırken ayarlar otomatik kaydedilir. Elle düzenlemek isterseniz JSON formatını koruyun.

İpuçları ve en iyi uygulamalar
Kinect v1 kullanıyorsanız sistemde freenect kurulumu ve USB izinlerine dikkat edin.

Yeterli sayıda ve çeşitlilikte frame (en az 20 önerilir) alın; kısa taramalar zayıf mesh üretir.

Kalite arttıkça bellek ve işlem süresi yükselir; 8GB+ RAM için Orta, 16GB+ ve çok çekirdekli CPU için Yüksek tercih edin.

USB kamerada derinlik yoksa gerçek 3B rekonstrüksiyon mümkün değildir; simüle edilmiş veya Kinect/derinlik kaynakları kullanın.

Uzun süren işlemlerde çıktı dosyalarının .tmp veya geçici adla yazılmasını sağlayıp tamamlandığında son isme taşımak güvenlidir.

Sık rastlanan sorunlar
Uygulama Open3D bulunamadı hatası: Open3D kurulu mu kontrol edin ve aynı Python ortamını kullanın.

freenect bulunamadı: Kinect v1 desteği opsiyoneldir; yoksa USB veya Simulated seçin veya freenect kurulumunu gerçekleştirin.

GUI açılıyor ama kamera görüntüsü yok: doğru kamera seçili mi, başka uygulama kamerayı kullanıyor mu, cihaz izinleri doğru mu kontrol edin.

Batch modunda "Mismatch between color and depth files": color_* ve depth_* sayıları eşit olmalıdır.

Geliştirici notları
Frame buffering, frame-skip ve reconstruction pipeline ayrı katmanlarda tutulmuştur; test yazımı kolaydır.

Önerilen geliştirmeler: producer/consumer mimarisi ile frame oluşturma ayrıştırması, daha sağlam sürüm uyumluluk kontrolleri, otomatik tmp->final kaydetme, birim testleri ve CI entegrasyonu.
