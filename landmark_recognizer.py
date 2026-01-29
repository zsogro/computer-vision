import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from annotator_helpers import draw_hand_landmarks_tasks_only



class LandmarkRecognizer:
    def __init__(self,mirror=True):
        """Initializes the LandmarkRecognizer with the given model path."""
        self.base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        self.options = vision.HandLandmarkerOptions(base_options=self.base_options,
                                       num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(self.options)
        print("LandmarkRecognizer initialized.")
        self.mirror = mirror

    def recognize(self, img):
        """Recognizes landmarks in the given image."""
        self.frame = img
        if self.mirror:
            self.frame = cv2.flip(img, 1)
        self.annotated_image = self.frame.copy()
        self.height, self.width = self.annotated_image.shape[:2]

        frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        # image = mp.Image.create_from_file("image.jpg")

        detection_result = self.detector.detect(image)
        
        self.hand_landmarks_list = detection_result.hand_landmarks
        self.handedness_list = detection_result.handedness

        # for hand in self.handedness_list:
        #     # print(f'Handedness: {hand.category[0].label}, Score: {hand.category[0].score}')
        #     print(hand)
     
        # for hand in self.hand_landmarks_list:
        #     print('Hand landmarks:')
        #     for landmark in hand:
        #         print(f'  (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})')

    def detect_wrists(self):
        """Detects wrists in the recognized hand landmarks."""
        wrists = []
        for hand_landmarks in self.hand_landmarks_list:
            wrist = hand_landmarks[0]  # Wrist is the first landmark
            wrists.append((wrist.x, wrist.y, wrist.z))
        
        # print("Detected wrists (x, y, z):", wrists)
        if len(wrists) == 1:
            cv2.putText(self.annotated_image, "Single wrist detected", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(self.annotated_image, f"{wrist.x:.1f}  {wrist.y:.1f}  {wrist.z:.1f}", (10, 60), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        elif len(wrists) == 2:
            wrist_distance = ((wrists[0][0] - wrists[1][0]) ** 2 + (wrists[0][1] - wrists[1][1]) ** 2) ** 0.5
            cv2.putText(self.annotated_image, f"Distance: {wrist_distance:.2f}", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.line(self.annotated_image, (int(wrists[0][0]*self.width), int(wrists[0][1]*self.height)), 
                 (int(wrists[1][0]*self.width), int(wrists[1][1]*self.height)), (255, 0, 0), 2)
        else:
            cv2.putText(self.annotated_image, "No wrists detected", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

        self.annotate_landmarks()

    def annotate_landmarks(self):
        """Annotates the image with hand landmarks."""
        self.annotated_image = draw_hand_landmarks_tasks_only(self.annotated_image, self.hand_landmarks_list)
 

    def show_annotated_image(self):
        """Displays the image with annotated hand landmarks."""
        # cv2.putText(self.annotated_image, "No hands detected", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Annotated Image", self.annotated_image)