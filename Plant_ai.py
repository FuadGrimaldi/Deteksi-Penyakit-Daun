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
import speech_recognition as sr  # recognise speech
import playsound  # to play an audio file
from gtts import gTTS  # google text to speech
import random
from time import ctime  # get time details
import webbrowser  # open browser
import threading

class person:
    name = 'Mr'

    def setName(self, name):
        self.name = name


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
    voice_signal = QtCore.pyqtSignal(str, str)  # Define the signal
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('GUI.ui', self)
        self.image = None
        self.cap = None
        self.recording = False
        self.voice_thread = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.pushButton_4.clicked.connect(self.loadClicked)
        self.pushButton_5.clicked.connect(self.predictFromImage)
        self.pushButton_2.clicked.connect(self.stopCam)
        self.pushButton.clicked.connect(self.startCam)
        self.pushButton_6.clicked.connect(self.toggleVoiceAi)
        self.pushButton_3.clicked.connect(self.reset)
        self.voice_signal.connect(self.update_text_edit)

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
    
    def update_text_edit(self, user_text, assistant_text):
        self.textEdit_2.append(f"You: {user_text}")
        self.textEdit_2.append(f"Assistant: {assistant_text}")
    
    def toggleVoiceAi(self):
        if self.recording:
            self.stopVoiceAi()
        else:
            self.startVoiceAi()

    def startVoiceAi(self):
        self.recording = True
        self.pushButton_6.setText("Stop Recording")
        self.voice_thread = threading.Thread(target=self.record_voice)
        self.voice_thread.start()

    def stopVoiceAi(self):
        self.recording = False
        self.pushButton_6.setText("Start Recording")
        if self.voice_thread:
            self.voice_thread.join()  # Wait for the thread to finish

    def record_voice(self):
        while self.recording:
            voice_data = record_audio()
            if voice_data:
                response = self.generate_response(voice_data)
                self.voice_signal.emit(voice_data, response)
                speak(response)

    def generate_response(self, voice_data):
        response = ""  # Initialize response with an empty string

        if there_exists(['hey', 'hi', 'hello'], voice_data):
            greetings = [
                f"Hey {person_obj.name}, how can I help you?",
                f"Hey, what's up {person_obj.name}?",
                f"I'm listening {person_obj.name}",
                f"How can I assist you? {person_obj.name}",
                f"Hello {person_obj.name}"
            ]
            response = greetings[random.randint(0, len(greetings) - 1)]
        
        elif there_exists(["what is your name", "what's your name", "tell me your name"], voice_data):
            if person_obj.name:
                response = f"My name is PlantAI, {person_obj.name}."
            else:
                response = "My name is PlantAI. What's your name?"
        
        elif there_exists(["my name is", "i am"], voice_data):
            person_name = voice_data.split("is")[-1].strip()
            response = f"Okay, I will remember that {person_name}."
            person_obj.setName(person_name)
        
        elif there_exists(["what is alternaria solani", "alernaria solani"], voice_data):
            response = "Alternaria solani is a fungus that causes early blight in tomatoes and potatoes. It is known for producing dark, concentric lesions on the leaves and stems of these plants."
        
        elif there_exists(["who are you"], voice_data):
            response = "I am your assistant in the field of plants. I am PlantAI."
        elif there_exists(["Tell me about tomato blight"], voice_data):
            response = "Tomato blight is a serious disease that affects tomato plants. It is caused by a fungus called Phytophthora infestans. This disease can cause the leaves to turn brown and wither, and in severe cases, it can kill the plant."
        elif there_exists(["describe seed plants"], voice_data):
            response = "a group of plants that have seeds as a means of reproduction. Seeds are the result of fertilization between egg cells and pollen, which function as a means of spreading and maintaining the survival of plant species."
        elif there_exists(["thanks", "thank you", "thank u"], voice_data):
            response = "You're welcome! If you need anything else, just let me know."
        elif there_exists(["open webcam"], voice_data):
            self.cap = cv2.VideoCapture(0)
            self.timer.start(20) 
            response = "Webcam has been opened."
        elif there_exists(["close webcam"], voice_data):
            self.timer.stop()
            if self.cap is not None:
                self.cap.release()
            self.label.clear()  # Clear the label content
            self.label.setText("Camera Off")  # Set default text
            response = "Webcam has been closed"
        
        else:
            response = "I'm not sure how to respond to that."

        return response
        

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

    def predictFromImage(self):
        if self.image is not None:
            prediction = predict_image(self.image)
            description = disease_dic.get(prediction, "Description not available.")
            self.label_7.setText(prediction)
            self.textEdit.setText(description)

        else:
            self.label_2.setText("No image loaded.")
            self.label_6.clear()

person_obj = person()

def there_exists(terms, voice_data):
    for term in terms:
        if term in voice_data:
            return True
    return False


def record_audio(ask=False):
    r = sr.Recognizer()  # initialise a recogniser
    with sr.Microphone() as source:  # microphone as source
        print("Say something...")
        if ask:
            speak(ask)
        r.adjust_for_ambient_noise(source)
        try:
            audio = r.listen(source)  # listen for the audio with a timeout  
        except sr.WaitTimeoutError:
            speak('Sorry, I did not hear anything. Please try again.')
            return ""
        print("Recognizing now...")
        
        voice_data = ''
        try:
            voice_data = r.recognize_google(audio)  # convert audio to text
        except sr.UnknownValueError:  # error: recognizer does not understand
            speak('I did not get that')
            print("Recognition failed: UnknownValueError")
        except sr.RequestError:
            speak('Sorry, the service is down')  # error: recognizer is not connected
            print("Recognition failed: RequestError")
        
        if voice_data:
            print(f">> {voice_data.lower()}")  # print what user said
        else:
            print("No voice data captured")

        return voice_data.lower()


def speak(audio_string):
    if not audio_string:
        print("No text to speak")
        return
    tts = gTTS(text=audio_string, lang='en')  # text to speech(voice)
    r = random.randint(1, 20000000)
    audio_file = 'audio' + str(r) + '.mp3'
    tts.save(audio_file)  # save as mp3
    playsound.playsound(audio_file)  # play the audio file
    print(f"PlantAI: {audio_string}")  # print what app said
    os.remove(audio_file)  # remove the audio file after playback


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
