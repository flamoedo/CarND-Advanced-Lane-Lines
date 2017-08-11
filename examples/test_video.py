import cv2

def run_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    ret = True
    
    while(ret):
        # Capture frame-by-frame
        ret, img = cap.read()
        cv2.imshow('frame',img)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
        
    
        
if __name__ == "__main__":
    run_video('../project_video.mp4')
        
        
