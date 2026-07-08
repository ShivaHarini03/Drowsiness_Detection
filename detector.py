import cv2
import mediapipe as mp
import numpy as np
import pygame
import time

class DrowsinessDetector:
    def __init__(self):
        
        self.mp_face_mesh = mp.solutions.face_mesh

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.mp_draw = mp.solutions.drawing_utils

        # FaceMesh eye landmarks
        # Left Eye
        # Left Eye
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]

        # Right Eye
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Thresholds
        self.EYE_OPEN_THRESHOLD = 0.39
        self.BLINK_FRAMES = 2
        self.DROWSY_FRAMES = 15
        self.MAR_THRESHOLD = 0.60
        self.YAWN_FRAMES = 15

        self.yawn_counter = 0
        self.yawns = 0
        self.counter = 0
        self.blinks = 0

        # Mouth landmarks
        self.MOUTH = {
            "top": 13,
            "bottom": 14,
            "left": 78,
            "right": 308
        }
        pygame.mixer.init()
        pygame.mixer.music.load("alarm.wav")

        self.last_alarm_time = 0
        self.ALARM_INTERVAL = 3   # Alarm every 3 seconds
        
    def process_frame(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)

        return results

    def get_landmarks(self, frame, results):

        if not results.multi_face_landmarks:
            return None

        face = results.multi_face_landmarks[0]

        h, w, _ = frame.shape

        landmarks = []

        for lm in face.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmarks.append((x, y))

        return landmarks
    
    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
    def calculate_eye_openness(self, eye, landmarks):

        p1 = landmarks[eye[0]]
        p2 = landmarks[eye[1]]
        p3 = landmarks[eye[2]]
        p4 = landmarks[eye[3]]
        p5 = landmarks[eye[4]]
        p6 = landmarks[eye[5]]

        vertical1 = self.distance(p2, p5)
        vertical2 = self.distance(p3, p6)

        horizontal = self.distance(p1, p4)

        ear = (vertical1 + vertical2) / (2.0 * horizontal)

        return ear
    def detect_drowsiness(self, frame, landmarks):
        
        if landmarks is None:
            return frame

        leftEye = self.calculate_eye_openness(self.LEFT_EYE, landmarks)
        rightEye = self.calculate_eye_openness(self.RIGHT_EYE, landmarks)
        print(f"Left Eye: {leftEye:.3f}")
        print(f"Right Eye: {rightEye:.3f}")
        eyeScore = (leftEye + rightEye) / 2
        mar = self.calculate_mar(landmarks)

        cv2.putText(
        frame,
        f"MAR : {mar:.2f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
        )
        print(f"Left Eye : {leftEye:.3f}")
        print(f"Right Eye: {rightEye:.3f}")
        print(f"Eye Score: {eyeScore:.3f}")
        print(f"Eye Score : {eyeScore:.2f} Counter={self.counter} Blinks={self.blinks}")
        cv2.putText(frame,
                f"Eye Score : {eyeScore:.2f}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,0),
                2)

        if eyeScore < self.EYE_OPEN_THRESHOLD:
            self.counter += 1
        else:
    # Count a blink
            if self.BLINK_FRAMES <= self.counter < self.DROWSY_FRAMES:
                self.blinks += 1

            self.counter = 0
        if mar > self.MAR_THRESHOLD:
            self.yawn_counter += 1
        else:
            if self.yawn_counter >= self.YAWN_FRAMES:
                self.yawns += 1

            self.yawn_counter = 0
# Drowsiness Warning
        current_time = time.time()
        if self.counter >= self.DROWSY_FRAMES:

            cv2.putText(
                frame,
                "DROWSINESS DETECTED!",
                (40,100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                3
            )

            current_time = time.time()

            if (current_time - self.last_alarm_time >= self.ALARM_INTERVAL
                and not pygame.mixer.music.get_busy()):

                pygame.mixer.music.play()
                self.last_alarm_time = current_time
        if self.yawn_counter >= self.YAWN_FRAMES:
            cv2.putText(
            frame,
            "YAWNING DETECTED!",
            (40, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 165, 255),
            3,
            )    
        cv2.putText(frame,
                f"Blinks : {self.blinks}",
                (20,140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2)
        cv2.putText(
            frame,
            f"Yawns : {self.yawns}",
            (20, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.putText(frame,
            f"Counter : {self.counter}",
            (20,180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2)

        
        return frame
    def draw_face_mesh(self, frame, results):

        if not results.multi_face_landmarks:
            return frame

        for face_landmarks in results.multi_face_landmarks:

            self.mp_draw.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_draw.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=1,
                    circle_radius=1
                )
            )

        return frame

    def draw_eye_points(self, frame, landmarks):

        if landmarks is None:
            return frame

        for idx in self.LEFT_EYE:
            cv2.circle(frame, landmarks[idx], 6, (255, 0, 0), -1)

        for idx in self.RIGHT_EYE:
            cv2.circle(frame, landmarks[idx], 6, (0, 0, 255), -1)

        return frame

    def draw_mouth_points(self, frame, landmarks):

        if landmarks is None:
            return frame

        for idx in self.MOUTH.values():
            cv2.circle(frame, landmarks[idx], 2, (0, 255, 255), -1)

        return frame
    def calculate_mar(self, landmarks):

        top = landmarks[self.MOUTH["top"]]
        bottom = landmarks[self.MOUTH["bottom"]]
        left = landmarks[self.MOUTH["left"]]
        right = landmarks[self.MOUTH["right"]]

        vertical = self.distance(top, bottom)
        horizontal = self.distance(left, right)

        mar = vertical / horizontal

        return mar
