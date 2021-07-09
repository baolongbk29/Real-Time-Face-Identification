import mediapipe as mp
import numpy as np
import cv2
import os
from os import listdir
import mediapipe as mp
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from utils import extract_face, get_encode, l2_normalizer, normalize
import pickle
import mtcnn

#=====hyper=parameters===========#
encoder_model = 'Facenet\model/facenet_keras.h5'
people_dir = "E:\College-Project\Rice-ATM-Face-Verification\Facenet\database_face/"
encodings_path = 'E:\College-Project\Rice-ATM-Face-Verification\Facenet\encodings/encodings.pkl'
required_size = (160, 160)
encoding_dict=dict()
encodes = []
#================================#

#==========DETECTOR==============#
mp_facedetector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detector = mp_facedetector.FaceDetection(min_detection_confidence=0.7)
#face_detector = mtcnn.MTCNN()
#================================#


#========LOAD=MODEL==============#
from tensorflow import keras
face_encoder = keras.models.load_model('E:\College-Project\Rice-ATM-Face-Verification\Facenet\model/facenet_keras.h5')
print("model built")
#================================#

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(160, 160))
    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

#==========ENCODING=DICT========#
encoding_dict=dict()
encodes = []
for file in listdir(people_dir):
    person, extension = file.split(".")
    face= preprocess_image('E:\College-Project\Rice-ATM-Face-Verification\Facenet\database_face/%s.jpg' % (person))
    encode = face_encoder.predict(face)[0]
    encoding_dict[person] = encode

for key in encoding_dict.keys():
    print(key)

with open(encodings_path, 'bw') as file:
    pickle.dump(encoding_dict, file)
#================================#
