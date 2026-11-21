import math
import mediapipe as mp
from mediapipe.tasks import python
import time

#pass a list of landmarks
#Check for .9 confidence or above
def t_pose(lms):
    if len(lms) == 0:
        return 0

    checkIndexes = [14,12,11,13]
    slope = (lms[15].y-lms[16].y)/(lms[15].x-lms[16].x)
    y_int = lms[15].y - (slope * lms[15].x)
    distance = math.sqrt( math.pow(lms[15].x - lms[16].x,2) + math.pow(lms[15].y-lms[16].y,2))
    
    pointDistance = 0
    for i in checkIndexes:
        estimatedValue = lms[i].x * slope + y_int
        pointDistance += math.fabs(lms[i].y - estimatedValue)

    pointDistance /= len(checkIndexes)
    pointDistance /= distance

    return 1 / (pointDistance + 1)

# Returns true if left hand is above head.
def left_hand_up(lms):
    if len(lms) == 0:
        return 0
    
    return lms[15].y < lms[0].y

# Returns true if right hand is above head.
def right_hand_up(lms):
    if len(lms) == 0:
        return 0
    
    return lms[16].y < lms[0].y

# Returns true if both hands are above head.
def hands_up(lms):
    return left_hand_up(lms) and right_hand_up(lms)

# Find points by searching MP pose landmarks, or it may be in the git if I remember to upload it.
def point_overlap(lms,id1,id2):
    # Get bounding box of human.
    minx = -1
    miny = -1
    maxx = -1
    maxy = -1
    for point in lms:
        if point.x < minx or minx == -1:
            minx = point.x
        elif point.x > maxx:
            maxx = point.x
        if point.y < miny or miny == -1:
            miny = point.y
        elif point.y > maxy:
            maxy = point.y
    # Use bounding box to understand scale (how far back the human is).
    size = math.sqrt( math.pow( maxx - minx, 2) + math.pow( maxy - miny, 2 ) )
    distance = math.sqrt( math.pow( lms[id1].x - lms[id2].x, 2) + math.pow( lms[id1].y - lms[id2].y, 2) )

    return 1 - (distance / size )

#give overlap points for gesture 1 and then gesture 2
def gesture_points(lms,p1,p2,p3,p4): # WIP: Probably not going to be implemented fully.
    if point_overlap(lms,p1,p2) > 0.95:
        if point_overlap(lms,p3,p4) > 0.95:
            return True
    return False

#scale of 0-1
def get_x(lms)->float:
    x_total = 0
    count = 0
    for point in lms:
        x_total += point.x
        count += 1

    return x_total / count

#scale of 0-1
def get_y(lms)->float:
    y_total = 0
    count = 0
    for point in lms:
        y_total += point.y
        count += 1

    return y_total / count
