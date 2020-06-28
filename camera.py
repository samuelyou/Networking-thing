import cv2
from realtime_demo import FaceCV
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        
        faceCV = FaceCV()
        image = faceCV.detect_face()
        
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
