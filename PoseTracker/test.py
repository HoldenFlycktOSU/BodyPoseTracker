from poseTracker import PoseTracker
import cv2
# Slightly increases contrast
def process_image(image):
    # Increase contrast: alpha > 1 increases contrast, beta balances brightness
    alpha = 2  # Contrast control (1.0-3.0)
    beta = .8  # Brightness control (0-1)

    image = cv2.convertScaleAbs(image, alpha, beta)

    return image#cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def main()->None:
    tracker = PoseTracker()

    camera = 1 
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Compress to same video feed on rover. 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Remove when moving to a ros node.

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to connect to camera")
            continue
        image = process_image(image)
        tracker.update_image(image)

        if cv2.waitKey(1) & 0xFF == ord('q'): # Refreshes frame and checks if 'q' is pressed
            break
    
        # Draw pose on image
        annotated_image = tracker.get_annotated_image()
        cv2.imshow("image",annotated_image)

        # Run through tests
        print(f"TPose: {tracker.t_pose()}\n"+\
            f"LHand: {tracker.left_hand_up()}\n"+\
            f"RHand: {tracker.right_hand_up()}\n"+\
            f"Hands: {tracker.hands_up()}\n"+\
            f"X: {tracker.get_x()}\n"+\
            f"Y: {tracker.get_y()}\n"+\
            f"OverLap: {tracker.point_overlap(16,13)}")

if __name__ == "__main__":
    main()
