🖥️ Donanım Gereksinimleri
Kinect v1 sensör (Xbox 360 Kinect veya Kinect for Windows v1)

Kinect USB adaptörü (sensörü PC’ye bağlamak için)

Windows / Linux bilgisayar (Python ve gerekli kütüphaneleri çalıştırabilecek)

(Opsiyonel) Webcam → Hibrit kamera modu için

(Opsiyonel) VR gözlük (Oculus Quest, HoloLens vs.) → WebXR önizleme için

⚙️ Yazılım Gereksinimleri
Python 3.8–3.10 (3.11 ve üstünde bazı kütüphaneler sorun çıkarabiliyor)

Gerekli kütüphaneler:

bash
pip install open3d==0.17.0
pip install opencv-python
pip install freenect
pip install trimesh
pip install PyQt5
pip install psutil
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
libfreenect driver (Kinect v1 için):

Windows: libfreenect Windows binaries indirip kur.

Linux: sudo apt-get install freenect

▶️ Çalıştırma
Kodunu scanner.py gibi bir dosyaya kaydet.

Terminalden çalıştır:

bash
python scanner.py
→ GUI açılacak, Kinect görüntüsü geliyorsa her şey yolunda.

Batch mode (GUI olmadan):

bash
python scanner.py --nogui --inputs ./scan1 ./scan2 --quality Orta --gpu --texture --ai PCN --slam --marker
→ scan1 ve scan2 klasörlerinde color_*.png ve depth_*.png dosyaları olmalı.

🔧 Olası Sorunlar ve Çözümler
Kinect tanınmıyor → libfreenect driver’ı doğru kuruldu mu kontrol et.

Open3D GPU hatası → GPU seçeneğini kapat (--gpu kullanma).

PyQt5 GUI açılmıyor → Windows’ta pip install pyqt5-tools ile ek GUI bileşenlerini kurabilirsin.

AI segmentasyon çok yavaş → AI Destekli Segmentasyon kutusunu kapatabilirsin.

🎯 Önerim
İlk denemede:

Orta kalite

ICP açık

Texture açık

SLAM açık

GPU kapalı

Böylece sistemin stabil çalıştığını görürsün. Sonra GPU ve AI segmentasyonu açarak performansı test edebilirsin.




-------------------------------------



🎯 Eklenen Özellikler
1. Hata Yönetimi ve Güvenilirlik

✅ Kinect bağlantı kontrolü ve otomatik yeniden bağlanma
✅ Odometry başarı oranı takibi
✅ Drift (kayma) kontrolü ve aşırı hareket tespiti
✅ Try-catch blokları ile hata yakalama
✅ Kullanıcıya anlamlı hata mesajları

2. Performans Optimizasyonları

✅ Adaptif TSDF parametreleri (kaliteye göre değişken voxel boyutu)
✅ Memory-efficient frame saklama
✅ Outlier removal ve mesh temizleme

3. Gelişmiş 3D İşleme

✅ ICP Refinement: Odometry sonuçlarını ICP ile iyileştirme
✅ Hybrid Jacobian: Hem renk hem geometri bilgisi kullanılıyor
✅ Statistical Outlier Removal: Gürültülü noktaları temizleme
✅ Connected Component Analysis: Sadece en büyük nesneyi tutma
✅ Manifold Edge Kontrolü: Topolojik hataları düzeltme
✅ Adaptive Decimation: Kaliteye göre mesh sadeleştirme

4. Kullanıcı Arayüzü İyileştirmeleri

✅ Modern, profesyonel tasarım
✅ Gerçek zamanlı progress bar
✅ Detaylı log sistemi (timestamp'li)
✅ Frame sayacı
✅ Durum göstergeleri (renkli)
✅ Ayarlanabilir parametreler
✅ Export format seçimi (OBJ/STL/PLY)
✅ "Farklı Kaydet" özelliği
✅ Mesh önizleme (Open3D visualizer)

5. Akıllı Tarama Özellikleri

✅ Depth değer filtreleme (0 ve >4.5m reddedilir)
✅ Minimum frame kontrolü (en az 5 frame)
✅ Otomatik timestamped kayıt
✅ Mesh istatistikleri (vertex/face sayıları)



🚀 Kullanım Kılavuzu
Kurulum:

bashpip install freenect open3d numpy trimesh opencv-python pyqt5

Optimal Tarama İpuçları:
1. Hazırlık:

Kinect'i masaya sabit yerleştirin
Nesneyi döner tabla üzerine koyun (veya etrafında dolanın)
İyi aydınlatma sağlayın (doğal ışık ideal)

2. Ayar Önerileri:
SenaryoFrame AralığıKaliteICPHızlı test10Hızlı❌Normal kullanım5Orta✅Yüksek detay2-3Yüksek✅Büyük nesne5Orta✅Küçük nesne3Yüksek✅
3. Tarama Tekniği:

30-60 saniye boyunca yavaşça dönün (360°)
Yukarı ve aşağı açılardan da görüntüler alın
Ani hareketlerden kaçının
En az 40-50 frame toplayın

4. Sorun Giderme:
SorunÇözümMesh delikliDaha yavaş dönün, daha fazla frameÇok gürültülüICP'yi aktif edin, kaliteyi artırınİşlem çok yavaşFrame aralığını artırın, kaliteyi düşürünKinect bağlanamıyorUSB port değiştirin, driver kontrol edinDrift var (kayıyor)Daha yavaş dönün, ICP kullanın
