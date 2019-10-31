import cv2
from yolo.detect import *

class TrackerList:
    """
    Class allowing to list the whole tracker's history.
    """
    def __init__(self):
        self.ListOftrackers = {}

    def add(self,frameNum, bbox, confidence):
        self.ListOftrackers.update({frameNum:[bbox,confidence]})

    def get_TrackerBox(self,frameNum):
        try:
            return True, self.ListOftrackers[frameNum][0]
        except:
            print("No tracker for frame"+str(frameNum))
            return False
    def get_TrackerConfidence(self,frameNum):
        try:
            return self.ListOftrackers[frameNum][1]
        except:
            print("No tracker for frame"+str(frameNum))
            return False

class DetecterList:
    """
    Class allowing to list the whole detection's history.
    """
    def __init__(self):
        self.ListOfDetections = {}

    def add(self, frameNum, bbox, confidence):
        self.ListOfDetections.update({frameNum:[bbox,confidence]})

    def get_TrackerBox(self,frameNum):
        try:
            return True, self.ListOfDetections[frameNum][0]
        except:
            print("No tracker for frame"+str(frameNum))
            return False
    def get_TrackerConfidence(self,frameNum):
        try:
            return self.ListOfDetections[frameNum][1]
        except:
            print("No tracker for frame"+str(frameNum))
            return False

#####
def plotTracking(frame, bbox, confidence, fps):
    """
    Draws Bounding box arround tracker "bbox" and writes on the frame the tracking information.
    :return: ImageDraw
    """
    ### Draw bounding box
    ImageDraw = frame.copy()
    ImageDraw = cv2.rectangle(ImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)),
                              (50, 255, 50), 2)
    # Display State and FPS on frame
    cv2.putText(ImageDraw, "GOTURN TRACKING", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 50, 50), 2)
    cv2.putText(ImageDraw, "FPS : " + str(int(fps)), (100, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 50, 50), 2)
    try:
        cv2.putText(ImageDraw, "Tracking accuracy confidence : " + str(round(confidence[0],2)), (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 50, 50), 2)
    except:
        cv2.putText(ImageDraw, "Tracking accuracy confidence : ", (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 50, 50), 2)
    return ImageDraw

def plotDetection(frame, DetectFlag, bbox=None,confidence=None):
    """
    Draws Bounding box arround detection "bbox" if there is a detection
    and writes on the frame the detection  information
    :return: ImageDraw
    """
    ImageDraw = frame.copy()
    ### Display State and FPS on frame
    cv2.putText(ImageDraw, "YOLO DETECTION", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 50, 50), 2)
    cv2.putText(ImageDraw, "Sliding Window Detection:  True", (100, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 50, 50), 2)
    if DetectFlag:
        ### Add bbox to the image with label
        label = '%s %.2f' % ('Drone', confidence)
        plot_one_box([bbox.x1, bbox.y1, bbox.x2, bbox.y2], ImageDraw, label=label, color = (223,82,134))
    return ImageDraw

def showSlidingWindow( frame, cropWidth, cropHeight, SlidingWinSize,fps):
    """
    Plots the search reagion on the frame as well as all relevant detection information.
    """
    ImageDraw = frame.copy()
    ### Display State and FPS on frame
    cv2.putText(ImageDraw, "YOLO DETECTION", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 50, 50), 2)
    cv2.putText(ImageDraw, "Sliding Window Detection:  True", (100, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 50, 50), 2)
    cv2.putText(ImageDraw, "FPS: "+str(round(fps)), (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 50, 50), 2)
    ### Display sliding window
    ImageDraw[cropHeight:cropHeight+5, cropWidth:cropWidth+SlidingWinSize] =(255,255,200)
    ImageDraw[cropHeight+SlidingWinSize:cropHeight+SlidingWinSize+5, cropWidth:cropWidth+SlidingWinSize]=(255,255,200)
    ImageDraw[cropHeight: cropHeight+SlidingWinSize,cropWidth:cropWidth+5]=(255,255,200)
    ImageDraw[cropHeight:cropHeight+SlidingWinSize,cropWidth+SlidingWinSize:cropWidth+SlidingWinSize+5]=(255,255,200)
    return ImageDraw
