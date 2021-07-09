import cv2
import mediapipe as mp
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer

def extract_face(image_path):
    mp_facedetector = mp.solutions.face_detection
    mp_draw = mp.solutions.drawing_utils
    with mp_facedetector.FaceDetection(min_detection_confidence=0.7) as face_detection:

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Process the image and find faces
        results = face_detection.process(image)
        # Convert the image color back so it can be displayed
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for id, detection in enumerate(results.detections):
                # mp_draw.draw_detection(image, detection)
                bBox = detection.location_data.relative_bounding_box
                h, w, c = image.shape
                boundBox = int(bBox.xmin * w), int(bBox.ymin * h), \
                           int(bBox.width * w), int(bBox.height * h)

                face = image[boundBox[1]:boundBox[1] + boundBox[3],
                       boundBox[0]:boundBox[0] + boundBox[2]]
                face = cv2.resize(face, (160, 160))
    return face


def plt_show(cv_img):
    img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.show()


def load_pickle(path):
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict


def save_pickle(path, obj):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)

def get_encode():
    pass

l2_normalizer = Normalizer('l2')

def normalize():
    pass
