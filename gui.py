from tkinter import *
import mediapipe as mp
import cv2
from PIL import Image, ImageTk
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
from utils import *

#=======Parameters===========#
encoder_model = 'Facenet\model/facenet_keras.h5'
people_dir = "E:\College-Project\Rice-ATM-Face-Verification\Facenet\database_face/"
encodings_path = 'E:\College-Project\Rice-ATM-Face-Verification\Facenet\encodings/encodings.pkl'
required_size = (160, 160)
start = 0
threshold = 6 #tuned threshold for l2 disabled euclidean distance
cap = cv2.VideoCapture(0)
start = 0
Count = 0
encoding_dict = load_pickle(encodings_path)
#================================#

#========LOAD=MODEL==============#
from tensorflow import keras
face_encoder = keras.models.load_model('E:\College-Project\Rice-ATM-Face-Verification\Facenet\model/facenet_keras.h5')
print("model built")
#================================#


class MyGUI:
    def __init__(self, master):
        self.master = master

        master.title('Cam Screen')
        master.configure(background='#CDCDCD')

        self.encoding_dict = load_pickle(encodings_path)
        self.mp_facedetector = mp.solutions.face_detection
        self.face_detection = self.mp_facedetector.FaceDetection(min_detection_confidence=0.7)

        # Create left and right frames
        self.left_frame = Frame(master, width=200, height=400, bg='grey')
        self.left_frame.grid(row=0, column=0, padx=10, pady=5)
        self.right_frame = Frame(master, width=800, height=600, bg='grey')
        self.right_frame.grid(row=0, column=1, padx=10, pady=5)
        self.encoding_dict = load_pickle(encodings_path)

        # Create tool box
        tool_bar = Frame(self.left_frame, width=180, height=185, bg="purple")
        tool_bar.grid(row=1, column=0, padx=5, pady=5)
        Label(self.left_frame, text="ToolBox").grid(row=0, column=0, padx=5, pady=5)

        # Extract face
        self.face= Frame(self.left_frame, width=180, height=185, bg="white")
        self.face.grid(row=2, column=0, padx=5, pady=5)


        #Video Screen
        self.lmain = Frame(self.right_frame, width=800, height=600, bg="white")
        self.lmain.grid(row=1, column=0, padx=5, pady=5)
        Label(self.right_frame,text="Video Screen").grid(row=0, column=0, padx=5, pady=5)


        self.Showcam_button = Button(tool_bar, text="Show Cam", command=self.show_frame).grid(row=0, column=0, padx=10, pady=10)
        self.Update_button = Button(tool_bar, text="Update", command=self.Update).grid(row=0, column=1, padx=10, pady=10)
        self.Prev_face_button = Button(self.left_frame, text="Show Previous Face", command=self.prevf).grid(row=3, column=0, padx=10, pady=10)
        self.Exit_Button = Button(tool_bar, text="Close", command=self.exit).grid(row=0, column=2, padx=10, pady=10)



        self.width, self.height = 800, 600
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)


    def preprocess_image(sefl,img):
        img = cv2.resize(img, (160, 160))
        img = np.asarray(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def exit(self):
        self.master.destroy()

    def prevf(self):
        image = cv2.imread(people_dir+"{}.jpg".format(str(self.person_name)))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)
        Label(self.face, image=image).grid(row=0, column=0, padx=5, pady=5)
        self.face.imgtk = image
        self.face.configure(background='#CDCDCD', foreground='#364156', image=image)

    def Update(self):
        if self.out == "Accept":
            id = len(encoding_dict) + 1
            face = self.image[self.boundBox[1]:self.boundBox[1] + self.boundBox[3], self.boundBox[0]:self.boundBox[0] + self.boundBox[2]]
            cv2.imwrite(people_dir+"{}.jpg".format(str(id)), face)
            face = self.preprocess_image(face)
            encode = face_encoder.predict(face)[0]
            encoding_dict[id] = encode
        else:
            pass

    def findEuclideanDistance(sefl,source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance


    def show_frame(self):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the image and find faces
        results = self.face_detection.process(image)
        # Convert the image color back so it can be displayed
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for id, detection in enumerate(results.detections):
                # mp_draw.draw_detection(image, detection)
                bBox = detection.location_data.relative_bounding_box
                h, w, c = self.image.shape
                self.boundBox = int(bBox.xmin * w*0.93), int(bBox.ymin * h * 0.8), \
                           int(bBox.width * w*1.1), int(bBox.height * h * 1.25)
                detected_face = self.image[self.boundBox[1]:self.boundBox[1] + self.boundBox[3], self.boundBox[0]:self.boundBox[0] + self.boundBox[2]]
                detected_face = cv2.resize(detected_face, (160, 160))  # resize to 224x224
                img_pixels = np.asarray(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                # employee dictionary is using preprocess_image and it normalizes in scale of [-1, +1]
                img_pixels = preprocess_input(img_pixels)
                captured_representation = face_encoder.predict(img_pixels)[0, :]
                distances = []
                for i in encoding_dict:  # O(n)
                    employee_name = i
                    source_representation = encoding_dict[i]
                    distance = self.findEuclideanDistance(captured_representation, source_representation)
                    print(employee_name, ": ", distance)
                    distances.append(distance)

                label_name = 'unknown'
                index = 0
                for i in encoding_dict:  # O(n)
                    self.person_name = i
                    if index == np.argmin(distances):
                        if distances[index] <= threshold:
                            print("detected: ", self.person_name)
                            label_name = "%s" % (self.person_name)
                            break
                    index = index + 1

                if label_name == 'unknown':
                    self.out = "Accept"
                    Color = (0, 255, 0)
                else:
                    self.out = "Reject"
                    Color = (0, 0, 255)

                cv2.rectangle(self.image, self.boundBox, Color, 2)
                cv2.putText(self.image, self.out, (self.boundBox[0], self.boundBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 2, Color, 2)

        cv2image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        image = ImageTk.PhotoImage(image=img)
        Label(self.lmain, image=image).grid(row=0, column=0, padx=5, pady=5)
        self.lmain.imgtk = image


root = Tk()
my_gui = MyGUI(root)
root.mainloop()