# Face Detection
Cara mudah membuat pendeteksi wajah (tanpa fitur anti spoofing)<br />
![Demo File](https://github.com/fajarlabs/absensi_wajah/blob/master/demo.gif)

# How To Install
Saya menggunakan RaspberryPi3 untuk instalasi menggunakan cara dibawah ini :<br />

``````
sudo apt install cmake build-essential pkg-config git
sudo apt install python3-opencv
sudo apt-get install raspberrypi-ui-mods
sudo apt install ffmpeg python3-opencv
sudo apt install libxcb-shm0 libcdio-paranoia-dev libsdl2-2.0-0 libxv1  libtheora0 libva-drm2 libva-x11-2 libvdpau1 libharfbuzz0b libbluray2 libatlas-base-dev libhdf5-103 libgtk-3-0 libdc1394-22 libopenexr23
sudo apt install python3-dev python3-pip python3-numpy
``````

<br />
Didalam praktek ini menggunakan python versi 3.7.3 <br />
Selanjutnya <i>install</i> paket <i>library</i> python seperti dibawah ini : <br />

``````
pip3 install -r requirements.txt
``````

Untuk instalasi <b>opencv-contrib-python</b> akan memakan waktu lebih lama, jadi ketika instalasi biarkan beberapa jam, mungkin berbeda pada versi raspbery4 keatas. <br />
Jika masih terjadi kendala, dimungkinkan ada kendala library yang kurang, dibawah ini : <br />

Library yang harus di install di raspberry PI seperti dibawah ini : <br />
``````
sudo apt install cmake build-essential pkg-config git
sudo apt install libjpeg-dev libtiff-dev libjasper-dev libpng-dev libwebp-dev libopenexr-dev
sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libdc1394-22-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev
sudo apt install libgtk-3-dev libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5
sudo apt install libatlas-base-dev liblapacke-dev gfortran
sudo apt install libhdf5-dev libhdf5-103
sudo apt install python3-dev python3-pip python3-numpy
``````
<br />
<h2>ATTENTION!!!</h2>

Jika instalasi paket ada yang gagal atau paket repository tidak lengkap atau tidak mendukung seperti libgtk,libtif dll gunakan cara kedua : <br />

<br />
<h2>HOW TO INSTALL OPENCV</h2>
Jika menginstall opencv lewat script silahkan ikuti tahapan dibawah ini : <br />

sudo nano /etc/dphys-swapfile<br />
Ubah <b>CONF_SWAPSIZE=100 </b> menjadi <b>CONF_SWAPSIZE=2048 </b><br />
Restart swabfile <b>sudo systemctl restart dphys-swapfile</b><br />

<b>Cloning source code opencv</b><br />
git clone https://github.com/opencv/opencv.git<br />
git clone https://github.com/opencv/opencv_contrib.git<br /><br />

``````
cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
-D ENABLE_NEON=ON \
-D ENABLE_VFPV3=ON \
-D BUILD_TESTS=OFF \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D OPENCV_ENABLE_NONFREE=ON \
-D CMAKE_SHARED_LINKER_FLAGS=-latomic \
-D BUILD_EXAMPLES=OFF ..
``````
<br />
make -j$(nproc) (proses ini memakan waktu 1 jam lebih )<br />
sudo make install <br />
sudo ldconfig <br />
<br />

Setelah proses diatas selesai kembalikan swabfile seperti semula : <br />
sudo nano /etc/dphys-swapfile (ubah 2048 menjadi 100) nilai dari CONF_SWAPSIZE <br />
<br />
<b> Restart SwabFile service </b> <br />
sudo systemctl restart dphys-swapfile <br />
<br />

Jika tahap-tahap diatas sudah selesai, selanjutnya silahkan install library face-recognitionnya : <br />
``````
pip install face-recognition
pip install imutils
``````
<br />
Jika ada permasalahan instalasi dlib lewat PIP, alternatif lainnya install lewat source code seperti langkah-langkah di bawah ini : <br />

<h2>HOW TO INSTALL DLIB (USING GPU)</h2>

Cara dibawah ini di instruksikan agar DLIB menggunakan GPU, jika tidak ada GPU dan hanya menggunakan CPU gunakan cara instalasi biasa<br />

``````
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build
cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
cmake --build .
cd ..
python setup.py install --set USE_AVX_INSTRUCTIONS=1 --set DLIB_USE_CUDA=1 --no DLIB_GIF_SUPPORT 
``````

<H2> How to install DLIB using conda

``````
conda install -c conda-forge dlib
``````

Mulai cloning programnya disini : <b> git clone https://github.com/fajarlabs/absensi_wajah.git </b> <br />
Jalankan file <b>python test_camera_basic.py</b> untuk memeriksa apakah sudah bisa dijalankan OPENCVnya. <br />

# How to capture model photo
Jika sudah ok selanjutnya silahkan buat folder nama didalam folder "DATASET", didalam folder nama tersebut akan di isi foto wajah untuk ditraining. <br />
<a href="https://ibb.co/09sLjHm"><img src="https://i.ibb.co/6tXSRjr/dataset-1.jpg" alt="dataset-1" border="0"></a><br />
Untuk melakukan pengisian dataset foto silahkan edit skrip <b>ambil_foto_wajah.py</b> di line 3 sesuaikan dengan lokasi nama folder foto didalam folder "DATASET".<br /> 
<a href="https://imgbb.com/"><img src="https://i.ibb.co/yYTn6bk/sesi-ambil-foto.jpg" alt="sesi-ambil-foto" border="0"></a> <br />
Jalankan <b>python ambil_foto_wajah.py</b> untuk melakukan sesi pengambilkan dan tekan <b>SPACE</b> untuk capture dan tekan <b>ESC</b> untuk keluar program. <br />

# How to train
Selanjutnya jika DATASET foto sudah OK selanjutnya melakukan training model dengan menjalankan skrip <b>python train_model.py</b> dan menghasilkan file "encodings.pickel" 

# How to detection
Selanjutkan untuk menjalankan deteksi wajah silahkan jalankan skrip <br />
``````
python deteksi_wajah.py
``````
<br />
<a href="https://imgbb.com/"><img src="https://i.ibb.co/M1sKvQ3/Capture.jpg" alt="Capture" border="0"></a>

# Literature
https://yunusmuhammad007.medium.com/2-raspberry-pi-install-opencv-pada-python-3-7-menggunakan-pip3-a2504dffd984 <br />
https://programmersought.com/article/57453207651/ <br />
https://www.tomshardware.com/how-to/raspberry-pi-facial-recognition <br />
