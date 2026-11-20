import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import poseLandmarker
import poses

def process_image(image):
    # Increase contrast: alpha > 1 increases contrast, beta balances brightness
    alpha = 2  # Contrast control (1.0-3.0)
    beta = .8  # Brightness control (0-1)

    image = cv2.convertScaleAbs(image, alpha, beta)

    return image#cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


# Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task',delegate=python.BaseOptions.Delegate.GPU)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    num_poses = 1)
detector = vision.PoseLandmarker.create_from_options(options)


camera = 1 
cap = cv2.VideoCapture(camera)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to connect to camera")
        continue
    image = process_image(image)
    mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

    if cv2.waitKey(1) & 0xFF == ord('q'): # Refreshes frame and checks if 'q' is pressed
        break
    # STEP 4: Detect hand landmarks from the input image.
    detection_result = detector.detect(mpImage)
    
    # STEP 5: Process the classification result. In this case, visualize it.
    annotated_image = poseLandmarker.draw_landmarks_on_image(image, detection_result)
    cv2.imshow("Image2",annotated_image)
    if len(detection_result.pose_landmarks) > 0:
        top_result = detection_result.pose_landmarks[0]
        print(f"TPose: {poses.t_pose(top_result)}\n"+\
                f"LHand: {poses.left_hand_up(top_result)}\n"+\
                f"RHand: {poses.right_hand_up(top_result)}\n"+\
                f"Hands: {poses.hands_up(top_result)}\n"+\
                f"OverLap: {poses.point_overlap(top_result,16,13)}")
