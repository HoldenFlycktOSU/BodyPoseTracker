import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import poseLandmarker
import poses

class PoseTracker:
    _detection_result:np.ndarray
    _image:np.ndarray
    _detector:vision.PoseLandmarker
    _human:bool
    _pos_updated:bool
    _x:float
    _y:float

    def __init__(self)->None:
        # Create an PoseLandmarker object.
        base_options = python.BaseOptions(model_asset_path='pose_landmarker.task',delegate=python.BaseOptions.Delegate.GPU)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            num_poses = 1)
        self._detector = vision.PoseLandmarker.create_from_options(options)
        
        self._detection_result = None
        self._image = None
        self._human = False
        self._pos_updated = False
   
    # Give a numpy image
    # Do this before calling anything else
    def update_image(self,image:np.ndarray)->None:
        self._image = image
        mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=image) # Go from numpy rgb to whatever MP uses
        # Detect pose landmarks from the input image.
        self._detection_result = self._detector.detect(mpImage)
        self.human_present()
        self._pos_updated = False
   
    def human_present(self)->bool:
         self._human = self._detection_result is not None and len(self._detection_result.pose_landmarks) > 0
         return self._human

    def get_annotated_image(self)->np.ndarray:
        # Draw pose on image
        annotated_image = poseLandmarker.draw_landmarks_on_image(self._image, self._detection_result)
        return annotated_image

    #returns confidence value for a t-pose
    #search for x > 0.95
    def t_pose(self)->float:
        if self._human:
            top_result = self._detection_result.pose_landmarks[0]
            return poses.t_pose( top_result )
    
    #returns bool for hand above head
    def left_hand_up(self)->bool:
        if self._human:
            top_result = self._detection_result.pose_landmarks[0]
            return poses.left_hand_up(top_result)
        return False

    #returns bool for hand above head
    def right_hand_up(self)->bool:
        if self._human:
            top_result = self._detection_result.pose_landmarks[0]
            return poses.right_hand_up(top_result)
        return False
    
    #returns bool for hand above head
    def hands_up(self)->bool:
        if self._human:
            top_result = self._detection_result.pose_landmarks[0]
            return poses.hands_up(top_result)
        return False
    
    #returns confidence value
    #Search mediapipe body pose id's or possibly an image will be in git
    def point_overlap(self,id1:int, id2:int)->float:
        if self._human:
            top_result = self._detection_result.pose_landmarks[0]
            return poses.point_overlap(top_result,id1, id2)
        return 0

    def get_detection_result(self)->np.ndarray:
        return self._detection_result

    def get_landmarks(self):
        return self._detection_result

    #internal
    def _update_pos(self)->None:
        if not self._pos_updated:
            self._x = poses.get_x(self._detection_result.pose_landmarks[0])
            self._y = poses.get_y(self._detection_result.pose_landmarks[0])

        self._pos_updated = True

    #returns 0-1 or -1 if no human
    def get_x(self)->float:
        if self._human:
            self._update_pos()
            return self._x
        return -1

    #returns 0-1 or -1 if no human
    def get_y(self)->float:
        if self._human:
            self._update_pos()
            return self._y
        return -1

