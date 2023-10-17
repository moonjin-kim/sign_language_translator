from PyQt5.QtWidgets import *
from PyQt5 import uic
import cv2
import threading
from PyQt5.QtWidgets import  QWidget, QLabel, QApplication
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QMainWindow, QCheckBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtGui
import numpy as np
from time import sleep
import numpy as np
from tensorflow import keras
import mediapipe as mp
from collections import deque

from tensorflow import keras
import tensorflow as tf
import timeit

from tensorflow import keras 
from PIL import ImageFont ,ImageDraw ,Image

import urllib.request
import json

form_traiswindow = uic.loadUiType("./ui2/translation.ui")[0] #번역 창
deque1 = deque()
# classes = ['자극', '당뇨병', '면역', '감기', '변비', '붕대', '설사', '성병', '소화제', '수면제',
#        '회복', '입원', '진단서', '치료', '퇴원', '빈혈', '화상', '술', '커피', '의사', '간호사',
#        '금식', '금연', '금주', '식도염', '숨차다', '통증', '가렵다', '답답', '건강', '불안',
#        '검사', '팔', '아프다', '춥다', '머리', '충혈', '왼쪽', '오른쪽', '떨다', '몸', '전염',
#        '병원', '병', '상처', '병원', '붓다', '피곤', '중독', '치매', '환자', '충격', '노화',
#        '가루약', '물약', '약효', '무기력', '체온']
classes = ['가렵다', '건강', '걷다', '검사', '겨드랑이', '고민', '과로', '금식', '답답', '당하다',
       '떨다', '마르다', '머리', '목', '몸', '밥', '병', '병명', '부족', '불안', '붓다',
       '붕대', '상담', '상처', '손', '숨차다', '실수', '싫다', '아래', '아프다', '얼굴', '엉덩이',
       '예전', '오른쪽', '오줌', '온도', '왼쪽', '일어나다', '입', '입원', '잊어버리다', '자다',
       '체온', '춥다', '치료', '침착', '커피', '코', '팔', '피곤', '회복']
## 네이버 번역 api
client_id = "LhKWjcKjXATQJxqDlyKi" # 개발자센터에서 발급받은 Client ID 값
client_secret = "VbhZhly6JW" # 개발자센터에서 발급받은 Client Secret 값\
pose_state = False

url = "https://openapi.naver.com/v1/papago/n2mt"
request = urllib.request.Request(url)
request.add_header("X-Naver-Client-Id",client_id)
request.add_header("X-Naver-Client-Secret",client_secret)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

##keypoint 좌표 반화
def extract_keypoints(results):
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([pose, lh, rh])

def extract_keypoints_pose(results):
    pose = np.array([[res.x,res.y,res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    return np.concatenate([pose])

def extract_keypoints_hand(results):
    lh = np.array([[res.x,res.y,res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x,res.y,res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    return np.concatenate([lh, rh])   

#랜드마크 그려주기
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) #POSE 랜드마크
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) #왼손 랜드마크
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) #오른손 랜드마크

sequence_lenght = 30

label_map = {label:num for num, label in enumerate(classes)}
new_model = tf.keras.models.load_model('./model/best.h5')

## mediapipe로 관절 포인트 표시
def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image, results

def listToString(str_list):
    result = ""
    for s in str_list:
        result += s + " "
    result += '\n\n'
    return result.strip()

def request_papago(text):
        data = "source=ko&target=en&text="
        data2 = "source=en&target=ko&text="
        data += text
        response = urllib.request.urlopen(request, data=data.encode("utf-8"))
        rescode = response.getcode()
        if(rescode==200):
            response_body = response.read()
            translation_en = json.loads(response_body.decode('utf-8'))
            data2 += translation_en.get("message").get("result").get('translatedText')
            response = urllib.request.urlopen(request, data=data2.encode("utf-8"))
            rescode = response.getcode()
            if(rescode==200):
                response_body = response.read()
                translation_ko = json.loads(response_body.decode('utf-8'))
                return translation_ko.get("message").get("result").get('translatedText')
            return 0 
        else:
            return 0

def get_state():
    return pose_state

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    print_sentense = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        # capture from web cam
        sequence = []
        sentence = []
        threshold = 0.7
        i = 2
        state = 0
        draw_state = False
        cap = cv2.VideoCapture(0)
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self._run_flag:
                # if self._run_flag == False:
                #     break
                # feed 읽기
                ret, frame = cap.read()
                start_t = timeit.default_timer()
                img, results = mediapipe_detection(frame, holistic)
                
                #랜드마크 그리기
                if len(deque1) != 0:
                    draw_state = deque1.popleft()
                    
                if draw_state  == True:
                    draw_landmarks(img,results)
                
        #       keypoints = extract_keypoints_pose(result)
                
                keypoint = extract_keypoints_pose(results)
                # #예측
                # print(keypoint[65])
                if keypoint[65]<0.85 or keypoint[61]<0.85:
                    keypoints = extract_keypoints(results)
                    keypoints = np.concatenate([keypoints[0:99], keypoints[132:258]])
                    sequence.insert(0,keypoints)
                    sequence = sequence[:30]
                    if i%30 == 0:
                        res = new_model.predict(np.expand_dims(sequence, axis=0))[0]
                        print(classes[np.argmax(res)])
                        print(res[np.argmax(res)])
                        if res[np.argmax(res)] > 0.3:
                            state = 1
                            sentence.append(classes[np.argmax(res)])
                else:
                    if state == 1:
                        print(state)
                        text = listToString(sentence)
                        text1 = text
                        if len(sentence)>1:
                            text1 = request_papago(text)
                        if text1 == 0:
                            text1 = text
                        self.print_sentense.emit(text1)
                        state = 0
                    i=0
                    sentence = []
                    sequence=[]
                fontpath = "fonts/gulim.ttc"
                font = ImageFont.truetype(fontpath, 50)
                img_pil = Image.fromarray(img)
                draw = ImageDraw.Draw(img_pil)
                

                cv2.rectangle(img,(0,0),(640,40),(245,117,16),-1)
                #cv2.putText(img, ''.join(sentence),,
                        ##cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255,0),2,cv2.LINE_AA)
                # draw.text((3,30),  ''.join(sentence), font=font, fill=(0,0,0,0))
                draw.text((550,30),  str(i), font=font, fill=(0,255,0,0))
                img = np.array(img_pil)
                self.change_pixmap_signal.emit(img)
                terminate_t = timeit.default_timer()
                FPS = int(1./(terminate_t - start_t ))
                i+=1

        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()

class TranslationWindow(QDialog, QWidget, form_traiswindow):

    def __init__(self):
        super(TranslationWindow,self).__init__()
        self.disply_width = 640
        self.display_height = 480
        self.cam = QLabel(self)
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.print_sentense.connect(self.update_text)
        # start the thread
        self.thread.start()
        self.initUI()
        

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.cam.setPixmap(qt_img)
    
    @pyqtSlot(str)
    def update_text(self,sequnce):
        self.result_text.append(sequnce)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)


    def initUI(self):
        self.setupUi(self)
        self.onoff_btn.clicked.connect(self.changeState)
        self.home_btn.clicked.connect(self.Home)
        self.show()

    def changeState(self, state):
        if self.onoff_btn.isChecked() :
            pose_state = True
        else:
            pose_state = False
        print(pose_state)
        deque1.append(pose_state)

    def Home(self):
        self.thread.stop()
        self.close()