import time
import cv2
from goturn.helper.BoundingBox import BoundingBox

opencv_version = cv2.__version__.split('.')[0]

class tracker_manager:

    """Docstring for tracker_manager. """

    def __init__(self, regressor, tracker, logger):
        """This is
        :videos: list of video frames and annotations
        :regressor: regressor object
        :tracker: tracker object
        :logger: logger object
        :returns: list of video sub directories
        """
        self.regressor = regressor
        self.tracker = tracker
        self.logger = logger

    def trackAll(self, inputVideo,save_tracking=False,show_tracking=True):
        """Track the objects in the video
        """
        objRegressor = self.regressor
        objTracker = self.tracker

        vid = cv2.VideoCapture(inputVideo)

        ok, frame_0 =vid.read()
        if not ok:
            print('Couldnt read first frame')
            exit()

        if save_tracking:
            movie_number =0
            tracker_type ='GoturnGPU'
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            FPS = vid.get(cv2.CAP_PROP_FPS)
            output_video_name = 'output_videos/movie_' + str(movie_number) + time.strftime(
                "_%d%m%Y_%H%M%S") + '_' + tracker_type + '.avi'
            outVideo = cv2.VideoWriter(output_video_name, fourcc, int(FPS), (width, height))

        box_0 = cv2.selectROI(frame_0)
        cv2.destroyAllWindows()
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        bbox_0 = BoundingBox(x1=box_0[0], y1=box_0[1], x2=box_0[0] +box_0[2], y2=box_0[1] +box_0[3])
        objTracker.init(frame_0, bbox_0, objRegressor)
        
        timerI = cv2.getTickCount()
        for i in range(1, num_frames-1):
            # vid.set(1,i)
            print('Tracking on frame:'+str(i)+'/'+str(num_frames))
            ok,frame = vid.read()
            if not ok:
                break
            ### Start timer
            timer = cv2.getTickCount()

            ### Update Tracker
            bbox = objTracker.track(frame, objRegressor)

            ### Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            ### Draw bounding box
            ImageDraw = frame.copy()
            ImageDraw = cv2.rectangle(ImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)), (255, 0, 0), 2)


            # Display FPS on frame
            cv2.putText(ImageDraw, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

            if show_tracking:
                cv2.imshow('Results', ImageDraw)
                cv2.waitKey(1)
            if save_tracking:
                outVideo.write(ImageDraw)
        timerF = cv2.getTickCount()
        Time_proc = (timerF-timerI)/cv2.getTickFrequency()
        print('Total processing time ='+ str(int(Time_proc))+'sec')
        print('Average FPS ='+ str(i / Time_proc))
        vid.release()
        if save_tracking:
                outVideo.release()

    def trackAllRealTimeSimu_OnVideo(self, inputVideoPath, save_tracking=False, show_tracking=True):
        """Track the objects in the video
        Simulates real time by taking the frame according to computation time of each itteration.
        """

        objRegressor = self.regressor
        objTracker = self.tracker

        vid = cv2.VideoCapture(inputVideoPath)
        FPS = vid.get(cv2.CAP_PROP_FPS)
        ok, frame_0 = vid.read()
        if not ok:
            print('Couldnt read first frame. Exit.')
            exit()

        if save_tracking:
            inputName = inputVideoPath.split('/')[-1:][0].split('.')[:-1][0]
            tracker_type = 'GoturnGPU'
            output_video_name = '../output/' + inputName +'_'+ tracker_type + time.strftime("_%-Y-%m-%d_%H-%M")+'.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

            outVideo = cv2.VideoWriter(output_video_name, fourcc, int(FPS), (width, height))

        box_0 = cv2.selectROI(frame_0)
        cv2.destroyAllWindows()
        num_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

        bbox = BoundingBox(x1=box_0[0], y1=box_0[1], x2=box_0[0] + box_0[2], y2=box_0[1] + box_0[3])
        # Init Goturn
        objTracker.init(frame_0, bbox, objRegressor)

        TimeSimulation = 0

        color = (50, 255, 50)
        for i in range(0,num_frames):
            # Simulate real time by taking the next frame
            TimeSimuFrame = TimeSimulation * FPS
            NumframeSimu = int(TimeSimuFrame)
            print('Tracking on frame:' + str(NumframeSimu) + '/' + str(num_frames))
            vid.set(1, NumframeSimu)
            ok, frame = vid.read()
            if not ok:
                break

            ### Start timer
            timeA = cv2.getTickCount()

            ### Update Tracker
            bbox = objTracker.track(frame, objRegressor)

            ### Calculate Frames per second (FPS)
            timeB = cv2.getTickCount()
            fps = cv2.getTickFrequency() /(timeB - timeA)
            TimeSimulation += 1/fps

            ### Draw bounding box
            ImageDraw = frame.copy()
            ImageDraw = cv2.rectangle(ImageDraw, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)),
                                      color, 2)

            # Display FPS on frame
            cv2.putText(ImageDraw, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

            if show_tracking:
                cv2.imshow('Results', ImageDraw)
                cv2.waitKey(1)
            if save_tracking:
                outVideo.write(ImageDraw)

        print('Total processing time =' + str(TimeSimulation))
        print('Average FPS =' + str(i/TimeSimulation))
        if save_tracking:
            outVideo.release()