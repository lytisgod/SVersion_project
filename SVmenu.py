import tkinter as tk
import customtkinter as ck
import numpy as np
import mediapipe as mp
import cv2
import math as m
from PIL import Image, ImageTk
import time as tt
import pickle
import pandas as pd
landmark = ['class'] + [f"{coord}{i}" for i in range(1, 34) for coord in ('x', 'y', 'z', 'v')]
class ExerciseTracker:
    def __init__(self, title, exercise_type):
        self.window = tk.Toplevel()
        self.window.geometry("480x700")
        self.window.title(title)
        ck.set_appearance_mode("dark")
        self.exercise_type = exercise_type
        self.stage = ""
        self.counter = 0
        self.aligned = ""
        self.alarm = ""
        self.body_language_alignment = ''
        self.standing = ''
        self.setup_gui()
        self.setup_mediapipe()
        self.detect()
        self.window.mainloop()

    def setup_gui(self):
        labels = ["STAGE", "REPS", "ALARM", "OFFSET"]
        self.boxes = {}

        for i, label in enumerate(labels):
            lbl = ck.CTkLabel(self.window, height=40, width=120, font=("Arial", 20), text_color="black", text=label)
            lbl.place(x=10 + i*150, y=1)
            box = ck.CTkLabel(self.window, height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", text="0")
            box.place(x=10 + i*150, y=41)
            self.boxes[label] = box

        self.button = ck.CTkButton(self.window, text='RESET', command=self.reset_counter, height=40, width=120, font=("Arial", 20), fg_color="blue")
        self.button.place(x=10, y=600)

        self.frame = tk.Frame(self.window, height=480, width=480)
        self.frame.place(x=10, y=90)
        self.lmain = tk.Label(self.frame)
        self.lmain.place(x=0, y=0)

    def setup_mediapipe(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        return angle if angle <= 180 else 360 - angle

    def detect(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image)
        h, w = image.shape[:2]
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                                        self.mp_drawing.DrawingSpec(color=(106, 13, 173), thickness=4, circle_radius=5),
                                        self.mp_drawing.DrawingSpec(color=(255, 102, 0), thickness=5, circle_radius=10))

        try:
            landmarks = results.pose_landmarks.landmark
            shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
            elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
            wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
            knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
            knee2 = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x * w, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y * h - 100]
            hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w,landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
            hip2 = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x * w, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y * h - 100]
            shoulder_r = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x*w, landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y*h]
            elbow_r = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x*w, landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y*h]
            wrist_r = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x*w, landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y*h]
            ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x*w, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y*h]
            hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x*w, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y*h]
            offset = m.dist(shoulder, shoulder_r)
            angle = self.calculate_angle(shoulder, elbow, wrist)
            angle2 = self.calculate_angle(elbow, shoulder, hip)
            angle3 = self.calculate_angle(shoulder_r, elbow_r, wrist_r)
            angle4 = self.calculate_angle(shoulder, hip, hip2)
            angle5 = self.calculate_angle(hip, knee, knee2)
            angle6 = self.calculate_angle(shoulder, hip, ankle)
            angle7 = self.calculate_angle(hip, shoulder , wrist)
            if self.exercise_type == "barbell":
                self.aligned = "Aligned" if offset < 100 else "Not Aligned"
                print( angle,angle3)
                if offset < 100:
                    if angle > 150 :
                        self.stage = "down"

                    self.alarm = "Keep arms close to body" if  (angle2 > 20) else "GOOD"

                    if (angle < 55 ) and self.stage == "down":
                        self.stage = "up"
                        if self.alarm == "GOOD":
                            self.counter += 1
                else:
                    self.alarm = ""
                    self.stage = ""


            elif self.exercise_type == "squat":
                self.aligned = "Aligned" if offset < 100 else "Not Aligned"
                print( angle4,angle5)
                if offset < 100:
                    if angle5 < 10:
                        self.stage = "up"
                    if angle5 > 60 and self.stage == "up" and self.alarm == "GOOD" :
                        self.stage = "down"
                        self.counter += 1
                    if angle4 > 50:
                        self.alarm = "BEND BACKWARDS"
                    if 15 < angle5 < 70:
                        self.alarm = "LOWER YOUR HIPS"
                    if angle5 > 98:
                        self.alarm = "SQUAT TOO DEEP"
                    if 72 < angle5 < 97 and angle4 < 49:
                        self.alarm = "GOOD"
                else:
                    self.alarm = ""
                    self.stage = ""
            elif self.exercise_type == "deadlift":
                with open('DLlr3_22.pkl', 'rb') as f:
                    model_D = pickle.load(f)
                with open('Alignedlr3_22.pkl', 'rb') as f:
                    model = pickle.load(f)
                row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
                X = pd.DataFrame([row], columns=landmark[1:])
                D = pd.DataFrame([row], columns=landmark[1:])
                self.aligned = model.predict(X)[0]
                body_language_class = model_D.predict(D)[0]
                body_language_prob = model_D.predict_proba(D)[0]
                if self.aligned == 'Aligned' and body_language_class == 'DOWN' and body_language_prob[body_language_prob.argmax()] >= .6:
                    self.stage = 'DOWN'
                elif self.aligned == 'Aligned' and self.stage == 'DOWN' and body_language_class == 'UP' and body_language_prob[body_language_prob.argmax()]>= .6:
                    self.stage = "UP"
                    self.counter += 1
                if body_language_class == "UP" and 165 < angle6 < 172:
                    self.alarm = "normal"
                elif body_language_class == "UP" and 173 < angle6 < 180:
                    self.alarm = "narrow"
                elif body_language_class == "UP" and angle6 < 160:
                    self.alarm = "wide"

            elif self.exercise_type == "shoulderpress":
                with open('Alisholr905.pkl', 'rb') as f:
                    model = pickle.load(f)
                row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
                X = pd.DataFrame([row], columns=landmark[1:])
                self.aligned = model.predict(X)[0]
                if self.aligned == "Aligned":
                    if angle < 70:
                        self.stage = "down"
                        if angle2 < 40 and self.stage == "down":
                            self.alarm = "TOO CLOSE"
                        if angle < 50:
                            self.alarm = "ARM 90 degree"
                        elif angle2 > 40:
                            self.alarm = "GOOD"
                    if angle > 135 and self.stage =='down':
                        self.stage="up"
                        if angle7 > 160 :
                            self.alarm = "GOOD"
                            self.counter +=1
                            print(self.counter)
                        elif angle7 < 160 and  self.stage == "up":
                            self.alarm = "HANDS TOO LOW"
                    if angle > 170 and angle7 <45 :
                        if self.counter == 0 :
                            self.alarm = "START"
                            self.stage = "GO"
                        elif self.counter > 0:
                            self.alarm = "REST"
                            self.stage = "STOP"
                else:
                    self.alarm = " "
                    self.stage = " "
        except Exception as e:
            print(e)

        img = Image.fromarray(image[:, :460, :])
        self.lmain.imgtk = ImageTk.PhotoImage(img)
        self.lmain.configure(image=self.lmain.imgtk)
        self.lmain.after(10, self.detect)

        self.boxes["REPS"].configure(text=self.counter)
        self.boxes["ALARM"].configure(text=self.alarm)
        self.boxes["STAGE"].configure(text=self.stage)
        self.boxes["OFFSET"].configure(text=self.aligned)

    def reset_counter(self):
        self.counter = 0
def main():
    root = tk.Tk()
    root.geometry("480x700")
    root.title("Main Menu")
    ck.set_appearance_mode("dark")

    squat_button = ck.CTkButton(root, text="Squat", height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=lambda:ExerciseTracker("Squat", "squat"))
    squat_button.place(x=180, y=200)
    shoulderpress_button = ck.CTkButton(root, text="Sholderpress", height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=lambda:ExerciseTracker("ShoulderPress", "shoulderpress"))
    shoulderpress_button.place(x=180, y=600)

    barbell_button = ck.CTkButton(root, text="Barbell", height=40, width=120, font=("Arial", 20),
                              text_color="white", fg_color="blue", command=lambda: ExerciseTracker("Barbell", "barbell"))

    barbell_button.place(x=180, y=300)
    DL_button = ck.CTkButton(root, text="Deadlift", height=40, width=120, font=("Arial", 20), text_color="white", fg_color="blue", command=lambda:ExerciseTracker("DeadLift", "deadlift"))
    DL_button.place(x=180, y=500)
    root.mainloop()

if __name__ == "__main__":
    main()


