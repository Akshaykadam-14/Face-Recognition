import tkinter as tk
import tkinter.messagebox as messagebox
import cv2
import os
import time
import numpy as np

cv2_base_dir = os.path.dirname(os.path.abspath(cv2.__file__))
haar_model = os.path.join(cv2_base_dir, 'haarcascade_frontalface_default.xml')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def cut_faces(image, faces_coord):
    faces = []
    for (x, y, w, h) in faces_coord:
        w_rm = int(0.2 * w / 2)
        faces.append(image[y: y + h, x + w_rm: x + w - w_rm])
    return faces

def add_person():
    person_name = input_entry.get().lower()
    folder = 'C:\\people_folder' + '\\' + person_name
    
    # Check if the 'people_folder' directory exists, if not, create it
    if not os.path.exists('C:\\people_folder'):
        os.makedirs('C:\\people_folder')
        
    if not os.path.exists(folder):
        messagebox.showinfo("Instructions", "I will now take 20 pictures. Press OK when ready.")
        os.mkdir(folder)
        video = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        counter = 1
        timer = 0
        cv2.namedWindow('Video Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Video Feed', 800, 600)  # Set window size
        cv2.namedWindow('Saved Face', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Saved Face', 200, 200)  # Set window size
        while counter < 21:
            _, frame = video.read()
            if counter == 1:
                time.sleep(6)
            else:
                time.sleep(1)
            faces = detector.detectMultiScale(frame)
            if len(faces):
                cut_face = cut_faces(frame, faces)
                face_bw = cv2.cvtColor(cut_face[0], cv2.COLOR_BGR2GRAY)
                face_bw_eq = cv2.equalizeHist(face_bw)
                face_bw_eq = cv2.resize(face_bw_eq, (100, 100), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(folder + '\\' + str(counter) + '.jpg', face_bw_eq)
                print('Images Saved:' + str(counter))
                counter += 1
                cv2.imshow('Saved Face', face_bw_eq)
            cv2.imshow('Video Feed', frame)
            cv2.waitKey(50)
        messagebox.showinfo("Instructions", "Pictures captured successfully. Press OK to make any changes if needed.")
    else:
        messagebox.showerror("Error", "This name already exists. Please choose a different name.")

def live():
    cv2.namedWindow('Predicting for')
    images = []
    labels = []
    labels_dic = {}
    threshold = 105
    for person in os.listdir("C:\\people_folder"):
        labels_dic[len(labels_dic)] = person
        for image in os.listdir("C:\\people_folder/" + person):
            images.append(cv2.imread('C:\\people_folder/' + person + '/' + image, 0))
            labels.append(len(labels_dic) - 1)
    labels = np.array(labels)
    rec_lbhp = cv2.face.LBPHFaceRecognizer_create()
    rec_lbhp.train(images, labels)
    cv2.namedWindow('face')
    webcam = cv2.VideoCapture(0)
    while True:
        _, frame = webcam.read()
        # Apply Gaussian Blur to reduce noise
        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
        faces = face_cascade.detectMultiScale(frame_blur, 1.3, 5)
        num_people = len(faces)
        cv2.putText(frame, f'People count: {num_people}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        for (x, y, w, h) in faces:
            face = frame[y:y + h, x:x + w]
            gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(gray_face, (100, 100))
            face_resized = cv2.equalizeHist(face_resized)
            collector = cv2.face.StandardCollector_create()
            rec_lbhp.predict_collect(face_resized, collector)
            conf = collector.getMinDist()
            pred = collector.getMinLabel()
            if conf < threshold:
                person_name = labels_dic[pred].upper()
            else:
                person_name = 'Unknown'
            cv2.putText(frame, person_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2,
                        cv2.LINE_AA)
        cv2.imshow("Live", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    webcam.release()
    cv2.destroyAllWindows()


def on_add_person_click():
    add_person()

def on_live_click():
    live()

root = tk.Tk()
root.title("Face Recognition System")

label_instruction = tk.Label(root, text="Please enter the name of the new person:")
label_instruction.pack()

input_entry = tk.Entry(root)
input_entry.pack()

btn_add_person = tk.Button(root, text="Add a new face", command=on_add_person_click)
btn_add_person.pack()

btn_live_recognition = tk.Button(root, text="Live recognition", command=on_live_click)
btn_live_recognition.pack()

btn_exit = tk.Button(root, text="Exit", command=root.quit)
btn_exit.pack()

root.geometry("400x300")  # Set window size
root.mainloop()
