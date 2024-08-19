import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import sys

class Plant_Disease_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 38)

    def forward(self, xb):
        out = self.network(xb)
        return out

class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui', self)
        self.image = None
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.pushButton_4.clicked.connect(self.loadClicked)
        self.pushButton_5.clicked.connect(self.predictFromImage)
        self.pushButton_2.clicked.connect(self.stopCam)
        self.pushButton.clicked.connect(self.startCam)
        self.pushButton_3.clicked.connect(self.reset)

    def displayImage(self, image, label):
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:  # rows[0], cols[1], channels[2]
            if image.shape[2] == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        img = img.rgbSwapped()
        pixmap = QPixmap.fromImage(img)
        label.setPixmap(pixmap)
        label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        label.setScaledContents(True)
    
    # membuat prosedur button clicked untuk load
    def loadClicked(self):
        # Open a file dialog to select an image file
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an Image File",
            r"D:\projek\project fix\app-gui-plant-deases\dataset\test",
            "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)",
            options=options
        )

        if file_path:
            self.loaded_image_path = file_path
            # Read the selected image file
            self.image = cv2.imread(file_path)

            if self.image is not None:
                print("Image read successfully")
                # Display the image using the existing method
                self.displayImage(self.image, self.label_2)
            else:
                print("Failed to read image")
        else:
            print("No file selected")

    def startCam(self):
        self.cap = cv2.VideoCapture(0)
        self.timer.start(20)  # Update frame every 20ms

    def stopCam(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        self.label.clear()  # Clear the label content
        self.label.setText("Camera Off")  # Set default text

    def reset(self):
        self.image = None
        self.label.clear()  # Clear the label content
        self.label.setText("No Image")  # Set default text
        self.label_2.clear()
        self.label_2.setText("No Image")
        self.label_7.clear()
        self.textEdit.clear()

    def update_frame(self): 
        ret, frame = self.cap.read()
        if ret:
            prediction = predict_image(frame)
            # Draw the prediction text on the frame
            cv2.putText(frame, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # Display the frame with the prediction text
            self.displayImage(frame, self.label)
            self.label_2.clear() 

    def predictFromImage(self):
        if self.image is not None:
            prediction = predict_image(self.image)
            description = disease_dic.get(prediction, "Description not available.")
            self.label_7.setText(prediction)
            self.textEdit.setText(description)

        else:
            self.label_2.setText("No image loaded.")
            self.label_6.clear()


transform = transforms.Compose([
    transforms.Resize(size=128),
    transforms.ToTensor()
])

num_classes = [
    'Daun_Apel_Apel_scab', 'Daun_Apel_Busuk_hitam', 'Daun_Apel_Karangan_apel', 'Daun_Apel_sehat', 'Daun_Blueberry_sehat', 
    'Daun_Ceri_(termasuk_asam)_Jamur_pudarnya', 'Daun_Ceri_(termasuk_asam)_sehat', 
    'Daun_Jagung_Bercak_daun_Cercospora_Bercak_daun_abu', 'Daun_Jagung_Jamur_umum', 
    'Daun_Jagung_Bercak_daun_ular', 'Daun_Jagung_sehat', 'Daun_Anggur_Busuk_hitam', 
    'Daun_Anggur_Esca_(Jamur_Hitam)', 'Daun_Anggur_Bercak_daun_(Isariopsis_Leaf_Spot)', 'Daun_Anggur_sehat', 
    'Daun_Jeruk_Haunglongbing_(Penyakit_greening_Citrus)', 'Daun_Persik_Bercak_bakteri', 'Daun_Persik_sehat',
    'Daun_Paprika_bell_Bercak_bakteri', 'Daun_Paprika_bell_sehat', 'Daun_Kentang_Bercak_awal', 
    'Daun_Kentang_Bercak_akhir', 'Daun_Kentang_sehat', 'Daun_Raspberry_sehat', 'Daun_Kedelai_sehat', 
    'Daun_Labuh_Jamur_pudarnya', 'Daun_Stroberi_Bercak_daun', 'Daun_Stroberi_sehat', 'Daun_Tomat_Bercak_bakteri', 
    'Daun_Tomat_Bercak_awal', 'Daun_Tomat_Bercak_akhir', 'Daun_Tomat_Jamur_daun', 'Daun_Tomat_Bercak_Septoria', 
    'Daun_Tomat_Kutu_spider_Dua_titik', 'Daun_Tomat_Bercak_Target', 
    'Daun_Tomat_Virus_Kerut_Daun_Kuning_Tomat', 'Daun_Tomat_Virus_mosaik_Tomat', 'Daun_Tomat_sehat'
]

disease_dic = {
    'Daun_Apel_Apel_scab': "Jamur *Venturia inaequalis* menyebabkan penyakit apple scab, yang mengakibatkan lesi gelap dan cekung pada daun, buah, dan batang.",
    
    'Daun_Apel_Busuk_hitam': "Busuk hitam pada apel disebabkan oleh jamur *Diplodia seriata*, yang menyebabkan lesi gelap dan cekung yang akhirnya menyebabkan pembusukan buah.",
    
    'Daun_Apel_Karangan_apel': "Kanker apel disebabkan oleh jamur *Neonectria ditissima*, yang menyebabkan lesi cekung dan kasar pada batang dan cabang.",
    
    'Daun_Apel_sehat': "Daun apel yang sehat tidak menunjukkan tanda-tanda penyakit dan bebas dari lesi atau perubahan warna.",
    
    'Daun_Blueberry_sehat': "Daun blueberry yang sehat berwarna cerah dan bebas dari gejala penyakit seperti bercak atau perubahan warna.",
    
    'Daun_Ceri_(termasuk_asam)_Jamur_pudarnya': "Jamur bubuk pada ceri, disebabkan oleh *Podosphaera clandestina*, menghasilkan pertumbuhan jamur putih seperti bubuk pada daun dan tunas.",
    
    'Daun_Ceri_(termasuk_asam)_sehat': "Daun ceri yang sehat bebas dari gejala penyakit dan terlihat hijau dan segar.",
    
    'Daun_Jagung_Bercak_daun_Cercospora_Bercak_daun_abu': "Bercak daun abu pada jagung, disebabkan oleh *Cercospora zeae-maydis*, mengakibatkan lesi abu-abu dengan halo kuning pada daun.",
    
    'Daun_Jagung_Jamur_umum': "Penyakit jamur umum pada jagung termasuk karat dan jamur, yang menyebabkan berbagai gejala seperti perubahan warna dan deformasi.",
    
    'Daun_Jagung_Bercak_daun_ular': "Bercak daun jagung akibat *Helminthosporium turcicum* menghasilkan lesi gelap memanjang pada daun jagung.",
    
    'Daun_Jagung_sehat': "Daun jagung yang sehat menunjukkan warna hijau segar dan bebas dari gejala penyakit.",
    
    'Daun_Anggur_Busuk_hitam': "Busuk hitam pada anggur disebabkan oleh jamur *Guignardia bidwellii*, yang menyebabkan lesi hitam pada buah dan daun.",
    
    'Daun_Anggur_Esca_(Jamur_Hitam)': "Esca pada anggur disebabkan oleh jamur *Phaeomoniella chlamydospora* dan *Phaeoacremonium aleophilum*, yang menyebabkan pembusukan dan kematian jaringan.",
    
    'Daun_Anggur_Bercak_daun_(Isariopsis_Leaf_Spot)': "Bercak daun anggur disebabkan oleh jamur *Isariopsis griseola*, menghasilkan bercak coklat pada daun.",
    
    'Daun_Anggur_sehat': "Daun anggur yang sehat bebas dari lesi atau gejala penyakit dan terlihat hijau dan segar.",
    
    'Daun_Jeruk_Haunglongbing_(Penyakit_greening_Citrus)': "Penyakit greening pada jeruk disebabkan oleh bakteri *Candidatus Liberibacter asiaticus*, menyebabkan daun menguning dan buah kecil dan cacat.",
    
    'Daun_Persik_Bercak_bakteri': "Bercak bakteri pada persik disebabkan oleh *Xanthomonas campestris*, yang menyebabkan lesi basah pada daun dan buah.",
    
    'Daun_Persik_sehat': "Daun persik yang sehat bebas dari gejala penyakit dan tampak segar dan hijau.",
    
    'Daun_Paprika_bell_Bercak_bakteri': "Bercak bakteri pada paprika bell disebabkan oleh *Xanthomonas campestris*, yang menyebabkan bercak basah pada daun.",
    
    'Daun_Paprika_bell_sehat': "Daun paprika bell yang sehat bebas dari lesi atau gejala penyakit dan tampak hijau dan segar.",
    
    'Daun_Kentang_Bercak_awal': "Bercak awal pada kentang disebabkan oleh jamur *Alternaria solani*, yang menyebabkan bercak gelap dengan tepi kuning pada daun.",
    
    'Daun_Kentang_Bercak_akhir': "Bercak akhir pada kentang adalah stadium lanjut dari penyakit yang disebabkan oleh jamur *Alternaria solani*, yang menyebabkan lesi gelap besar pada daun.",
    
    'Daun_Kentang_sehat': "Daun kentang yang sehat bebas dari gejala penyakit dan tampak hijau dan segar.",
    
    'Daun_Raspberry_sehat': "Daun raspberry yang sehat menunjukkan warna hijau cerah dan bebas dari lesi atau gejala penyakit.",
    
    'Daun_Kedelai_sehat': "Daun kedelai yang sehat bebas dari bercak atau gejala penyakit dan terlihat hijau dan segar.",
    
    'Daun_Labuh_Jamur_pudarnya': "Jamur pudarnya pada labuh disebabkan oleh *Colletotrichum orbiculare*, yang menyebabkan bercak hitam dan kerusakan pada daun.",
    
    'Daun_Stroberi_Bercak_daun': "Bercak daun pada stroberi disebabkan oleh *Mycosphaerella fragariae*, yang menyebabkan bercak coklat dengan tepi kuning pada daun.",
    
    'Daun_Stroberi_sehat': "Daun stroberi yang sehat bebas dari gejala penyakit dan tampak hijau dan segar.",
    
    'Daun_Tomat_Bercak_bakteri': "Bercak bakteri pada tomat disebabkan oleh *Xanthomonas campestris*, yang menyebabkan bercak kecil dan lesi basah pada daun.",
    
    'Daun_Tomat_Bercak_awal': "Bercak awal pada tomat disebabkan oleh *Alternaria solani*, yang menyebabkan bercak gelap dengan tepi kuning pada daun.",
    
    'Daun_Tomat_Bercak_akhir': "Bercak akhir pada tomat adalah stadium lanjut dari penyakit yang disebabkan oleh *Alternaria solani*, yang menyebabkan lesi besar pada daun.",
    
    'Daun_Tomat_Jamur_daun': "Jamur daun pada tomat disebabkan oleh *Cladosporium fulvum*, yang menyebabkan bercak hitam pada daun.",
    
    'Daun_Tomat_Bercak_Septoria': "Bercak Septoria pada tomat disebabkan oleh *Septoria lycopersici*, menghasilkan bercak hitam dengan tepi kuning pada daun.",
    
    'Daun_Tomat_Kutu_spider_Dua_titik': "Kutu laba-laba dua titik pada tomat disebabkan oleh *Tetranychus urticae*, menyebabkan noda kuning dan kerusakan daun.",
    
    'Daun_Tomat_Bercak_Target': "Bercak target pada tomat disebabkan oleh *Alternaria solani*, yang menyebabkan bercak hitam dengan pola konsentris pada daun.",
    
    'Daun_Tomat_Virus_Kerut_Daun_Kuning_Tomat': "Virus kerut daun kuning tomat disebabkan oleh virus *Tomato yellow leaf curl virus* (TYLCV), menyebabkan daun mengerut dan menguning.",
    
    'Daun_Tomat_Virus_mosaik_Tomat': "Virus mosaik tomat disebabkan oleh *Tomato mosaic virus* (ToMV), yang menyebabkan pola mosaik dan perubahan warna pada daun.",
    
    'Daun_Tomat_sehat': "Daun tomat yang sehat bebas dari gejala penyakit dan tampak hijau dan segar."
}




model = Plant_Disease_Model()
model.load_state_dict(torch.load('./Models/plantDisease-resnet34.pth', map_location=torch.device('cpu')))
model.eval()

def predict_image(img):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tensor = transform(img_pil)
    xb = tensor.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return num_classes[preds[0].item()]

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ShowImage()
    window.setWindowTitle('Sistem Pendeteksi Penyakit Daun')
    window.show()
    sys.exit(app.exec_())
