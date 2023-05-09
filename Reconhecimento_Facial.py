import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)
face_recognition_solution = mp.solutions.face_detection
face_recognizer = face_recognition_solution.FaceDetection()
drawing = mp.solutions.drawing_utils

while True:
    cheker, frame = webcam.read()
    if not cheker:
        break
    face_list = face_recognizer.process(frame)
    if face_list.detections:
        for face in face_list.detections:
            drawing.draw_detection(frame, face)
        cv2.imshow("image", frame)

        if cv2.waitKey(5) == 27:
            break

webcam.release()
cv2.destroyAllWindows()