import numpy as np
import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
#from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from os import listdir
import mediapipe as mp
from utils import *
from scipy.spatial.distance import cosine
import mtcnn
import time

#=======Parameters===========#
encoder_model = 'Facenet\model/facenet_keras.h5'
people_dir = "E:\College-Project\Rice-ATM-Face-Verification\Facenet\database_face/"
encodings_path = 'E:\College-Project\Rice-ATM-Face-Verification\Facenet\encodings/encodings.pkl'
required_size = (160, 160)
start = 0
threshold =6.5 #tuned threshold for l2 disabled euclidean distance
cap = cv2.VideoCapture(0)
start = 0
Count = 0
encoding_dict = load_pickle(encodings_path)
#================================#

#=======Euclidean=Distance=======#
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance
#================================#

def preprocess_image(img):
    img = cv2.resize(img, (160,160))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

#========LOAD=MODEL==============#
from tensorflow import keras
face_encoder = keras.models.load_model('E:\College-Project\Rice-ATM-Face-Verification\Facenet\model/facenet_keras.h5')
print("model built")
#================================#

#==========DETECTOR==============#
mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_facedetector.FaceDetection(min_detection_confidence=0.7)
#face_detection = mtcnn.MTCNN()
#================================#

#=================LOOP=====================#
while cap.isOpened():
        ret, image = cap.read()
        start = time.time()
        # Convert the BGR image to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Process the image and find faces
        results = face_detection.process(image)
        # Convert the image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for id, detection in enumerate(results.detections):
                #mp_draw.draw_detection(image, detection)
                bBox = detection.location_data.relative_bounding_box
                h, w, c = image.shape
                boundBox = int(bBox.xmin * w), int(bBox.ymin * h*0.8), \
                           int(bBox.width * w), int(bBox.height * h*1.25)
                detected_face = image[boundBox[1]:boundBox[1]+boundBox[3], boundBox[0]:boundBox[0]+boundBox[2]]
                detected_face = cv2.resize(detected_face, (160, 160)) #resize to 224x224
                img_pixels = np.asarray(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis = 0)
                #employee dictionary is using preprocess_image and it normalizes in scale of [-1, +1]
                img_pixels  = preprocess_input(img_pixels )
                captured_representation = face_encoder.predict(img_pixels)[0,:]
                distances = []
                for i in encoding_dict: #O(n)
                    employee_name = i
                    source_representation = encoding_dict[i]
                    
                    distance = findEuclideanDistance(captured_representation, source_representation)
                    
                    print(employee_name,": ",distance)
                    distances.append(distance)
                
                label_name = 'unknown'
                index = 0
                for i in encoding_dict:  #O(n)
                    person_name = i
                    if index == np.argmin(distances):
                        if distances[index] <= threshold:
                            print("detected: ",person_name)
                            label_name = "%s" % (person_name)
                            break
                    index = index + 1

                if label_name == 'unknown':
                    out = "Accept"
                    Color = (0, 255, 0)
                else:
                    out = "Reject"
                    Color = (0,0, 255)

                cv2.rectangle(image,boundBox,Color,2)
                cv2.putText(image,out, (boundBox[0], boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, Color, 2)

        #======FPS=CALC======#
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        start = end
        #====================#

        cv2.putText(image, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
        cv2.imshow('img',image)

        #Process control
        k = cv2.waitKey(10)
        if k == ord('q'):# Bam q de thoat
            with open(encodings_path, 'bw') as file:
                pickle.dump(encoding_dict, file)
            break
        elif k == ord('u'):
            id =len(encoding_dict) + 1
            face = image[boundBox[1]:boundBox[1]+boundBox[3], boundBox[0]:boundBox[0]+boundBox[2]]
            #cv2.imwrite(people_dir+"{}.jpg".format(str(id)),face)
            face = preprocess_image(face)
            encode = face_encoder.predict(face)[0]
            encoding_dict[id] = encode
#==========================================================#

#kill open cv things
cap.release()
cv2.destroyAllWindows()