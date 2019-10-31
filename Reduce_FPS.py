import cv2

FPS_ratio = 3
movie_number = 840
#### Pre processing
# Read video
print('Processing movie number ' + str(movie_number))

video = cv2.VideoCapture("Input/Trim884.mp4")
# Exit if video not opened.
if not video.isOpened():
    print("Could not open video")
    exit()


fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS_i = video.get(cv2.CAP_PROP_FPS)
output_video_name = 'Input/Expo884'+ str(movie_number) +'FPS20'+ '.mp4'
outVideo = cv2.VideoWriter(output_video_name, fourcc, 30, (width, height))
num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

for i in range(5400,num_frames,FPS_ratio):
    video.set(1,i)
    ok, frame = video.read()
    print('frame '+str(i)+'/'+str(num_frames))
    if not ok:
        break

    outVideo.write(frame)
    # cv2.imwrite('inputFrames/'+str(movie_number)+'frame_'+str(i)+'.png',frame)

video.release()
outVideo.release()