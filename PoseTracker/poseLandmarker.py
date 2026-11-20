import mediapipe as mp
import numpy as np
import cv2

# Uses numpy array for image and draws MP landmarks ontop.
def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)
  height, width = annotated_image.shape[:2]
  # Loop through the detected hands to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]
    for i in range(0,len(pose_landmarks_list[idx])):
        landmark = pose_landmarks_list[idx]
        x = int( landmark[i].x * width )
        y = int( landmark[i].y * height )
        cv2.circle(annotated_image, (x,y), 5, (0,255,0))
        cv2.putText(annotated_image, f"{i}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
  return annotated_image
