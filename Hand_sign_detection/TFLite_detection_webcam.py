

# Gerekli kütüphanelerin import edilmesi
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import lirc

client = lirc.Client( #LIRC client oluşturulması
  connection=lirc.LircdConnection(
    #address="/var/run/lirc/lircd",
    #socket=socket.socket(socket.AF_UNIX, socket.SOCK_STREAM),
    timeout = None
  )
)

class VideoStream:
    # Pi kamera modülünün aktifleştirilmesi ve görüntü akışının sağlanması
    def __init__(self,resolution=(640,480),framerate=30):
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Kameradan gelen ilk karenin okunması
        (self.grabbed, self.frame) = self.stream.read()

# Kamerayı durdurmak için kullanılan değişken
        self.stopped = False

    def start(self):
# Kareleri okuyan thread in başlatılması
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Thread durana kadar dönen sonsuz döngü
        while True:
            # Kamera durduysa thread i durdur
            if self.stopped:
                self.stream.release()
                return

            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# En yeni kareyi dön
        return self.frame

    def stop(self):
	# Kamera ve thread i durdurmak için kullanılan fonksiyon
        self.stopped = True

# Giriş argümanlarını tanımla ve böl
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.7)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)

# Tenserflow kütüphanelerinin importlanması
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter

else:
    from tensorflow.lite.python.interpreter import Interpreter

# Açık olan klasörün dosya konumunun bulunması
CWD_PATH = os.getcwd()

# İçerisinde obje tespit modelini barındıran .tflite dosya konumuna ulaşılması
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Label map dosyasının bulunması
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Label map dosyasının yüklenmesi
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Label mapde ufak bir düeltme
if labels[0] == '???':
    del(labels[0])

# Tensorflow Lite modelinin yüklenmesi
interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Model detaylarının alınması
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Video yayının başlaması
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)


while True:

    # Video yayınından karelerin alınması
    frame1 = videostream.read()

    # Yayından alınan karelerin istenilen boyuta getirilmesi
    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # Pixel değerlerinin normalize edilmesi
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    #Karenin girdi olarak verilip hareket tespitinin yapılması
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Tespit sonuçlarının alınması
    boxes = interpreter.get_tensor(output_details[0]['index'])[0] # El hareketinin kordinatlarının bulunması
    classes = interpreter.get_tensor(output_details[1]['index'])[0] # Tespit edilen hareketin sınıfı
    scores = interpreter.get_tensor(output_details[2]['index'])[0] # Tespit edilen objenin doğruluğu

    # Tespit edilen tüm hareketlere bak ve en yüksek doğruluğu olanı bul
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            

            # Etiketi çiz
            object_name = labels[int(classes[i])] # Obje isimlerini label lara bakarak bul
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Label ve doğruluğu belirle
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Font boyutu
            label_ymin = max(ymin, labelSize[1] + 10) 
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) 
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Label ı ekrana yazdır
            
            if object_name == 'volumeup': # Volumeup hareketi algılanırsa Volumeup sinyalini gönder
                client.send_start("mamur", "KEY_VOLUMEUP") 
                client.send_stop("mamur", "KEY_VOLUMEUP")
                time.sleep(0.5)
        
            if object_name == 'volumedown': # Volumedown hareketi algılanırsa Volumedown sinyalini gönder
                client.send_start("mamur", "KEY_VOLUMEDOWN")
                client.send_stop("mamur", "KEY_VOLUMEDOWN")
                time.sleep(0.5)
        
            if object_name == 'channelnext': # channelnext hareketi algılanırsa channelup sinyalini gönder
                client.send_start("mamur", "KEY_CHANNELUP")
                client.send_stop("mamur", "KEY_CHANNELUP")
                time.sleep(0.5)
            
            if object_name == 'channelpre': # channelpre hareketi algılanırsa channeldown sinyalini gönder
                client.send_start("mamur", "KEY_CHANNELDOWN")
                client.send_stop("mamur", "KEY_CHANNELDOWN")
                time.sleep(0.5)
        
            if object_name == 'n1': # n1 hareketi algılanırsa 1 sinyalini gönder
                client.send_start("mamur", "BTN_1")
                client.send_stop("mamur", "BTN_1")
                time.sleep(0.5)
        
            if object_name == 'n2': # n2 hareketi algılanırsa 2 sinyalini gönder
                client.send_start("mamur", "BTN_2")
                client.send_stop("mamur", "BTN_2")
                time.sleep(0.5)
        
            if object_name == 'n3': # n3 hareketi algılanırsa 3 sinyalini gönder
                client.send_start("mamur", "BTN_3")
                client.send_stop("mamur", "BTN_3")
                time.sleep(0.5)
        
            if object_name == 'n4': # n4 hareketi algılanırsa 4 sinyalini gönder
                client.send_start("mamur", "BTN_4")
                client.send_stop("mamur", "BTN_4")
                time.sleep(0.5)
        
            if object_name == 'n5': # n5 hareketi algılanırsa 5 sinyalini gönder
                client.send_start("mamur", "BTN_5")
                client.send_stop("mamur", "BTN_5")
                time.sleep(0.5)
    
            if object_name == 'n6': # n6 hareketi algılanırsa 6 sinyalini gönder
                client.send_start("mamur", "BTN_6")
                client.send_stop("mamur", "BTN_6")
                time.sleep(0.5)
        
            if object_name == 'n7': # n7 hareketi algılanırsa 7 sinyalini gönder
                client.send_start("mamur", "BTN_7")
                client.send_stop("mamur", "BTN_7")
                time.sleep(0.5)
        
            if object_name == 'n8': # n8 hareketi algılanırsa 8 sinyalini gönder
                client.send_start("mamur", "BTN_8")
                client.send_stop("mamur", "BTN_8")
                time.sleep(0.5)
        
            if object_name == 'n9': # n9 hareketi algılanırsa 9 sinyalini gönder
                client.send_start("mamur", "BTN_9")
                client.send_stop("mamur", "BTN_9")
                time.sleep(0.5)
        
            if object_name == 'n0': # n0 hareketi algılanırsa 0 sinyalini gönder
                client.send_start("mamur", "BTN_0")
                client.send_stop("mamur", "BTN_0")
                time.sleep(0.5)
        
            if object_name == 'mute': # mute hareketi algılanırsa mute sinyalini gönder
                client.send_start("mamur", "KEY_MUTE")
                client.send_stop("mamur", "KEY_MUTE")
                time.sleep(0.5)
        
            if object_name == 'menu': # menu hareketi algılanırsa menu sinyalini gönder
                client.send_start("mamur", "KEY_MENU")
                client.send_stop("mamur", "KEY_MENU")
                time.sleep(0.5)

            if object_name == 'ok': # ok hareketi algılanırsa ok sinyalini gönder
                client.send_start("mamur", "KEY_OK")
                client.send_stop("mamur", "KEY_OK")
                time.sleep(0.5)
        
            if object_name == 'exit': # exit hareketi algılanırsa exit sinyalini gönder
                client.send_start("mamur", "KEY_EXIT")
                client.send_stop("mamur", "KEY_EXIT")
                time.sleep(0.5)
                
            if object_name == 'power': # power hareketi algılanırsa power sinyalini gönder
                client.send_start("mamur", "KEY_POWER")
                client.send_stop("mamur", "KEY_POWER")
                time.sleep(0.5)
            
    # Hesaplanan sonuçların kareye çizdirilmesi
    cv2.imshow('Object detector', frame)
    

         
#    time.sleep(1)
    # Kapatmak için q tuşuna basılır
    if cv2.waitKey(1) == ord('q'):
        break

# Pencerelerin kapatılması
cv2.destroyAllWindows()
videostream.stop()
