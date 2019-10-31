import cv2
import time
import numpy as np
import torch

from imutils.video import FileVideoStream

class VideoSaver:
    """
    Class operating all the actions linked to the saving of the output
    """
    def __init__(self, output_video_name, fourcc, FPS, width, height):
        self.outVideo = cv2.VideoWriter(output_video_name, fourcc, FPS, (width, height))
        self.frameCountprev = 0
        self.FPS = FPS

    def write(self, frame, frameCount):
        missedFrames = frameCount - self.frameCountprev
        for i in range(0,  missedFrames):
            self.outVideo.write(frame)

        self.frameCountprev = frameCount

    def AddIntroFrame(self,inputName, height, width, conf_thres, nms_thres,
                      doubleCheckThres,trackingConfidenceThres, VerificationInterval, RealTimeSimu,
                      weights):
        Intro_image = np.zeros((height, width, 3), np.uint8)

        cv2.putText(Intro_image, "Author: Jonathan Boinet, Faculty of Electrical Engineering, Tel Aviv University",
                    (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(Intro_image, "Under the supervision of Mr. Yonatan Mandel",
                    (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(Intro_image, "Date:" + time.strftime(" %d/%m/%Y, %H:%M"),
                    (200, 280), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(Intro_image, "Tracking and detection on video: " + inputName,
                    (200, 330), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(Intro_image, "Detection confidence Threshold: " + str(conf_thres),
                    (200, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(Intro_image, "Detection Non-Maximal Suppression Threshold: " + str(nms_thres),
                    (200, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(Intro_image, "Detection double check Threshold: " + str(doubleCheckThres),
                    (200, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(Intro_image, "Tracking verification interval: " + str(VerificationInterval)+"sec",
                    (200, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(Intro_image, "Tracking Confidence Threshold: " + str(trackingConfidenceThres),
                    (200, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(Intro_image, "Real time simulation: " + str(RealTimeSimu),
                    (200, 620), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(Intro_image, "Detection weights: " + weights.split('/')[-1:][0],
                    (200, 660), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(Intro_image, "Device: " + str(torch.cuda.get_device_properties(0)),
                    (200, 860), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        for i in range(0, 180):
            self.outVideo.write(Intro_image)

    def release(self):
        self.outVideo.release()

class VideoReaderQueue:
    """
    Class based on PySearch Image code.
    Allows to buffer frames in a queue and load frames on another thread => Increased speed in frame reading.
    """
    def __init__(self, VideoPath):
        self.video = FileVideoStream(VideoPath).start()
        time.sleep(1.0)
        self.videoProp = cv2.VideoCapture(VideoPath)

    def read(self):
        frame = self.video.read()
        if type(frame) == np.ndarray:
            return True, frame
        else:
            return False, None

    def get(self, item):
        return self.videoProp.get(item)
    def release(self):
        self.videoProp.release()

class VideoReader:
    """
    Class allowing us to pre-load the frames so that navigating betweed frames goes faster.
    Pay attention that it is expensive in RAM memory.
    On Jetson TX2 pre-loading batches of 500 frames at resolution of 1080*1920 seems to be ideal.
    """
    def __init__(self, VideoPath, numOfFramesToLoad, JetsonCam = False):
        self.numOfFramesToLoad =numOfFramesToLoad
        self.JetsonCam = JetsonCam
        # Pre-load Video
        self.video = cv2.VideoCapture(VideoPath)
        self.VidFPS = self.video.get(cv2.CAP_PROP_FPS)
        self.numFrames = self.video.get(cv2.CAP_PROP_FRAME_COUNT)
        self.frameCount = 0
        self.LoadInterval = [0,0]       #loaded frame interval: [i,j] = from frame i (incl.) to j (non-incl.)
        self.load()

    def load(self):
        self.frames = []
        i = 0
        while i < self.numOfFramesToLoad:
            ok, frame = self.video.read()
            if not ok:
                print("Pre-loading finished")
                break
            self.frames.append(frame)
            print("Pre-loading video frame:"+str(i+self.LoadInterval[1])+'/'+str(self.numFrames))
            i+=1
        self.LoadInterval = [self.LoadInterval[1], self.LoadInterval[1]+i]

    def set(self,a, Numframe):
        if self.JetsonCam:
            return
        if Numframe >= self.LoadInterval[0] and Numframe < self.LoadInterval[1]:
            self.frameCount = Numframe
        else:
            self.LoadInterval[1] = Numframe
            self.frameCount = Numframe
            self.video.set(1,Numframe)
            self.load()

    def read(self, time):
        if time < 0:
            print('error negative time')
            return
        frame_num = round(time*self.VidFPS + 0.5)
        if frame_num >= self.LoadInterval[0] and frame_num < self.LoadInterval[1]-1:
            frame = self.frames[frame_num]
            return True, frame
        else:
            self.load()
            try:
                frame = self.frames[frame_num]
                return True, frame
            except:
                return False, None

    def read(self):
        if self.frameCount >= self.LoadInterval[0] and self.frameCount < self.LoadInterval[1]:
            # If frame to read already loaded => Return the frame
            frame = self.frames[self.frameCount - self.LoadInterval[0]]
            self.frameCount += 1
            return True, frame
        else:
            # Load the next batch of frames and then return frame
            self.load()
            try:
                frame = self.frames[self.frameCount - self.LoadInterval[0]]
                self.frameCount +=1
                return True, frame
            except:
                return False, None

    def get(self, item):
        return self.video.get(item)
    def release(self):
        self.video.release()
