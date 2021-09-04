ML,Python,Terraform
# Authenticate and Launch
## _Using OpenCV and HaarCascade Model to detect face and authenticate ,once done Launch Infrastructure over AWS Cloud using Terraform_

I have used Haarcascade_frontalface weights to detect face as well as to train custom model.
👉 https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

## Step1
- We need to capture photo streams from video
Thus we have :

```sh
try:
    while True:
    #captures the frame from video stream
        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            #resizing the extracted frame
            face = cv2.resize(face_extractor(frame), (300, 300))
            #changing color of the frame so as to reduce features and computing
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Save file in specified directory with unique name
            file_name_path = './faces/user/' + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)

            # Put count on images and display live count
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Cropper', face)

        else:
            print("Face not found")
            pass

        if cv2.waitKey(3) == 13 or count == 100: #13 is the Enter Key
            break
    print("Collecting Samples Complete")
except Exception as e:
    print(f"Time taken : {e} ")
cap.release()
cv2.destroyAllWindows()
```

Meanwhile capturing the frames ,we are sending it to the function named as face_extractor:
```sh
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #we are using detectMulitScale function which helps to detect objects of different sizes in the input image.The detected objects then are returned as a list of rectangles
    #ARGUMNETS: 
    - image
    - scale factor - 	how much the image size is reduced at each image scale.
    - minNeighbours - neighbors each candidate rectangle should have to retain.
    faces = face_classifier.detectMultiScale(gray, 1.4, 5)
    
    #if none of faces found
    if faces == ():
        return None
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]

    return cropped_face
```
This function helps in getting relevant part of the image as in our case is face.

## Step2 
- Now once we have captured image stored in sepcified folder,we need to get those images and will use those images to train our model.
- Further we assign each a label
- In order to train model ,we need LBPHFaceRecognizer class consists of CV algorithm to create model for the purpose.
- Once we have created empty model ,we can now train it using the training data.

```sh
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Get the training data we previously made
data_path = './faces/user/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

# Create arrays for training data and labels
Training_Data, Labels = [], []

# Open training images in our datapath
# Create a numpy array for training data
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

# Create a numpy array for both training data and labels
Labels = np.asarray(Labels, dtype=np.int32)

# Initialize facial recognizer
# model = cv2.face.createLBPHFaceRecognizer()
# NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
# pip install opencv-contrib-python
# model = cv2.createLBPHFaceRecognizer()

ansh_model  = cv2.face_LBPHFaceRecognizer.create()
# Let's train our model 
ansh_model.train(np.asarray(Training_Data), np.asarray(Labels))
print("Model trained sucessefully")
```

Now your model is successfully trained.

> Now you can use this to detect the face through video streams.

### We have the train model which can be used further for detection
- We will load haarcascade classifier.
- Now we will have face_detector function.
- we will use pre-trained model earlier,to predict face.

#### We can use the results from prediction to validate the face we desired.

```sh
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
def face_detector(img, size=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(img, 1.3, 5)
    print(len(faces))
    if faces is ():
        return img, []
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi
# Open Webcam
cap = cv2.VideoCapture(0)
photos=[]
c = 0
while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    # Pass face to prediction model
    # "results" comprises of a tuple containing the label and the confidence value
    results = ansh_model.predict(face)
    #If confidence value is less than 500
    if results[1] < 500:
        #calculating confidence.
        confidence = int( 100 * (1 - (results[1])/400) )
        display_string = str(confidence) + '% Confident it is User'
    cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
    #If confidence is greater than 90
    #Desired user
    if confidence >= 90 :
        cv2.putText(image, "Hey Ansh", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Face Recognition', image )
        if c == 0:
            photos.append(image)
            print(c)
            print("In the count1")
            c = alert(photos,c)
            print(c)
        cv2.imshow('Face Recognition', image )
        cap.release()
        break
    else:
        #User is not recognized
        cv2.putText(image, "I dont know, who r u", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break
cap.release()
cv2.destroyAllWindows()
```
## Now once we have detected face ,next task is to launch Infrastructure on AWS using Terraform
 For this I have created function which will initiate terraform apply :
 
 ```sh
 import tf
 def alert(photos,c):
    t1 = time.time()
    print("Launching infra using terraform .....")
    #calling terraform function to launch the infrastructure
    tf.launch()
    t2=time.time()
    print(f"Infra launched \ntime taken : {t2-t1}s")
    return c
 ```
 
 ## Terraform py file
 [tf.py](https://github.com/Hubcodee/Detect-and-Launch/blob/main/tf.py)
 
 ## Terraform file
 [Terraform file](https://github.dev/Hubcodee/Detect-and-Launch/blob/main/first.tf)
 
 
 





