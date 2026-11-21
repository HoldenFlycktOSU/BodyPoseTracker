**Setup**\
First make an instance of the PoseTracker class (stored in poseTracker.py).\
After creating the instance call update_image() and pass it a numpy RGB image.(CV2 image format)\

tracker = PoseTracker()
tracker.update_image(image) #run whenever you get a new image

**Use**\
After setting the class up you now have access to a multitude of functions.It is recommended that\
you call these on a series of images before large motions as they occasionally misfire.

**t_pose()**\
Returns a scale from 0-1 on pose confidence.\
Consistently gives 0.95 or greater when pose is active.

**left_hand_up() and right_hand_up()**\
Returns a boolean if the hand is detected above the head visually from the camera's pov.

**hands_up()**\
Returns a boolean if both hands are above the head from the camera's pov.

**point_overlap(id1,id2)**\
Detects if points are overlapping. Returns a scale 0-1 for confidence.\
Recommended check for greater than 0.9. Has the most variance between frames.\
Id's can be found online by searching mediapipe pos id's or in an image on this repo.

**get_x()**\
Returns x from 0-1 based on averaged position of all points from left to right.\

**get_y()**\
Returns y from 0-1 based on averaged position of all points from top down.\
  



ID |Landmark |ID |Landmark
|---|-------------|---|-----------|
0 |nose |17 |left_pinky
1 |left_eye_inner |18 |right_pinky
2 |left_eye |19 |left_index
3 |left_eye_outer |20 |right_index
4 |right_eye_inner |21 |left_thumb
5 |right_eye |22 |right_thumb
6 |right_eye_outer |23 |left_hip
7 |left_ear |24 |right_hip
8 |right_ear |25 |left_knee
9 |mouth_left |26 |right_knee
10 |mouth_right |27 |left_ankle
11 |left_shoulder |28 |right_ankle
12 |right_shoulder |29 |left_heel
13 |left_elbow |30 |right_heel
14 |right_elbow |31 |left_foot_index
15 |left_wrist |32 |right_foot_index
16 |right_wrist
