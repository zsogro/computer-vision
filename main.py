import cv2

from landmark_recognizer import LandmarkRecognizer

cap = cv2.VideoCapture(0)
recognizer = LandmarkRecognizer()

    
while True:
    success, img = cap.read()
    recognizer.recognize(img)
    recognizer.detect_wrists()
    recognizer.show_annotated_image()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()