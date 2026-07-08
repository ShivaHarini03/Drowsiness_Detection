import cv2
from detector import DrowsinessDetector

cap = cv2.VideoCapture(0)

detector = DrowsinessDetector()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = detector.process_frame(frame)

    detector.draw_face_mesh(frame, results)

    landmarks = detector.get_landmarks(frame, results)
    """if landmarks is None:
        print("No landmarks detected")
    else:
        print("Landmarks:", len(landmarks))"""
    
    frame = detector.detect_drowsiness(frame, landmarks)

    detector.draw_eye_points(frame, landmarks)

    detector.draw_mouth_points(frame, landmarks)

    cv2.imshow("Driver Monitoring", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
