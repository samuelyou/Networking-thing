import cv2
import sys
import numpy as np
import os

cap = cv2.VideoCapture(0)

cap.set(3, 480) #set width of the frame
cap.set(4, 640) #set height of the frame

#why what is this
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)


age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

def detect_face(img):
    
    #load LBP recognizer from OPENCV library
    cascPath = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + cascPath)
    faces = face_cascade.detectMultiScale(img, scaleFactor = 1.2, minNeighbors = 5)
    
    if (len(faces) == 0):
        return None, None
    
    (x,y,w,h) = faces[0]
    
    return img[y:y+w, x:x+h], faces[0]

def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def load_models():

    #replace with ur paths to the file

    age_NN = cv2.dnn.readNetFromCaffe('/Users/samuelyou/Desktop/networking thing/deploy_age.prototxt', '/Users/samuelyou/Desktop/networking thing/age_net.caffemodel')
    gender_NN = cv2.dnn.readNetFromCaffe('/Users/samuelyou/Desktop/networking thing/deploy_gender.prototxt', '/Users/samuelyou/Desktop/networking thing/gender_net.caffemodel')
    return(age_NN, gender_NN)

print("Predicting images...")

(age_NN, gender_NN) = load_models()

cap = cv2.VideoCapture(0)  
ret,frame = cap.read()

predicted_img = None
Age, Gender = None, None

while(True):
    
    ret,frame = cap.read() # reads one frame
    cv2.imshow('img1',frame) #display the frame
    
    face, rect = detect_face(frame)
    
    if face is not None:
        if cv2.imwrite('test.png',frame):
            
            #perform prediction
            blob = cv2.dnn.blobFromImage(face, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            #Predict Gender
            gender_NN.setInput(blob)
            gender_preds = gender_NN.forward()

            gender = gender_list[gender_preds[0].argmax()]
            print("Gender : " + gender)
    
            #Predict Age
            age_NN.setInput(blob)
            age_preds = age_NN.forward()
            age = age_list[age_preds[0].argmax()]
            print("Age Range: " + age) 
            
            #draw the green rectangle to make sure it's classifying the right thing
            (x, y, w, h) = rect
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            predicted_img = frame
            Age = age
            Gender = gender
            
            #breaks after identifying a correct frame

            break    


cap.release()

#display results
cv2.imshow((Age + ", " + Gender), predicted_img) 

cv2.waitKey(0)
cv2.destroyAllWindows()
