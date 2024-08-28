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

from database.disease_data import num_classes, disease_dic

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

        elif there_exists(["tell me about tomato blight"], voice_data):
            response = "Tomato blight is a serious disease that affects tomato plants. It is caused by a fungus called Phytophthora infestans. This disease can cause the leaves to turn brown and wither, and in severe cases, it can kill the plant."
        
        elif there_exists(["describe seed plants"], voice_data):
            response = "Seed plants are a group of plants that reproduce using seeds. These seeds are formed after the fertilization of egg cells by pollen, ensuring the survival and spread of the plant species."

        elif there_exists(["what is powdery mildew"], voice_data):
            response = "Powdery mildew is a fungal disease that appears as a white or gray powdery substance on the leaves, stems, and flowers of plants. It thrives in warm, dry conditions and can weaken the plant if left untreated."

        elif there_exists(["tell me about rust disease"], voice_data):
            response = "Rust disease is a fungal infection that causes orange, yellow, or brown pustules on the undersides of leaves. It can lead to leaf drop and reduced plant vigor, especially in crops like wheat, beans, and roses."

        elif there_exists(["what is downy mildew"], voice_data):
            response = "Downy mildew is a fungal disease that causes yellow or brown spots on the upper surface of leaves, while the undersides develop a fuzzy, grayish growth. It's commonly found in humid, cool environments."

        elif there_exists(["tell me about leaf spot disease"], voice_data):
            response = "Leaf spot disease is caused by various fungi or bacteria and results in small, dark, circular spots on leaves. Over time, the spots may enlarge and cause the leaves to yellow and fall off."

        elif there_exists(["what is botrytis cinerea"], voice_data):
            response = "Botrytis cinerea, also known as gray mold, is a fungus that affects many plants, especially in humid conditions. It causes grayish-brown lesions on flowers, fruits, and leaves, leading to rot."

        elif there_exists(["tell me something interesting about plants"], voice_data):
            response = "Did you know that plants can communicate with each other? Some plants release chemicals to warn neighboring plants of insect attacks, prompting them to produce defensive compounds!"

        elif there_exists(["what is plant respiration"], voice_data):
            response = "Plant respiration is the process by which plants convert oxygen and glucose into energy, releasing carbon dioxide and water as byproducts. It's the opposite of photosynthesis and occurs all the time."

        elif there_exists(["tell me a plant joke", "tell me something funny about plants"], voice_data):
            jokes = [
                "Why do plants hate math? Because it gives them square roots!",
                "What did the tree say to the wind? Leaf me alone!",
                "Why don't plants go to school? Because they’re rooted to the spot!"
            ]
            response = jokes[random.randint(0, len(jokes) - 1)]

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
            response = "Webcam has been closed."
        
        elif there_exists(["do plants sleep"], voice_data):
            response = "Yes, plants do have a form of 'sleep'! At night, they undergo a process called nyctinasty, where their leaves close or droop in response to the lack of light."

        elif there_exists(["how do plants drink water"], voice_data):
            response = "Plants absorb water through their roots via a process called osmosis. The water travels up through the plant's xylem vessels to the leaves and other parts of the plant."

        elif there_exists(["what is chlorophyll"], voice_data):
            response = "Chlorophyll is the green pigment in plants that allows them to absorb sunlight and convert it into energy through photosynthesis."

        elif there_exists(["what is photosynthesis"], voice_data):
            response = "Photosynthesis is the process by which plants use sunlight to convert carbon dioxide and water into glucose and oxygen. It's how plants produce the energy they need to grow."

        elif there_exists(["tell me about plant roots"], voice_data):
            response = "Roots anchor plants to the ground and absorb water and nutrients from the soil. They also store food for the plant and sometimes produce new plants through asexual reproduction."

        elif there_exists(["how do plants grow"], voice_data):
            response = "Plants grow by using sunlight, water, and nutrients from the soil to produce energy through photosynthesis. This energy is used to build new cells, allowing the plant to grow taller and develop leaves, flowers, and fruits."

        elif there_exists(["what are plant hormones"], voice_data):
            response = "Plant hormones, also known as phytohormones, are chemicals that regulate various aspects of plant growth and development, such as cell division, flowering, and response to environmental stimuli."

        elif there_exists(["what is plant transpiration"], voice_data):
            response = "Plant transpiration is the process by which water evaporates from the leaves, stems, and flowers of plants. This helps cool the plant and also creates a negative pressure that pulls water and nutrients up from the roots."

        elif there_exists(["tell me about plant pollination"], voice_data):
            response = "Pollination is the process by which pollen is transferred from the male part of a flower (the stamen) to the female part (the pistil). This leads to fertilization and the production of seeds."

        elif there_exists(["what are succulents"], voice_data):
            response = "Succulents are plants that have thick, fleshy leaves or stems adapted to store water. They are often found in arid environments and are popular as low-maintenance houseplants."

        elif there_exists(["how do plants protect themselves"], voice_data):
            response = "Plants have various defense mechanisms, such as thorns, toxic chemicals, and the ability to close their leaves when touched. Some plants even produce chemicals that attract predators of the insects eating them."

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
