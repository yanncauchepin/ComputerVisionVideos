import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(f'{cv2.data.haarcascades}haarcascade_eye.xml')

class FaceDetector():

    def __init__(self):
        pass

    '''Hence, having some means of abstracting image detail is useful in producing 
    stable classification and tracking results. The abstractions are called features, 
    which are said to be extracted from the image data. There should be far fewer 
    features than pixels, though any pixel might influence multiple features. A set 
    of features is represented as a vector (conceptually, a set of coordinates in a 
    multidimensional space), and the level of similarity between two images can be 
    evaluated based on some measure of the distance between the images' corresponding 
    feature vectors.'''

    '''Haar-like features are one type of feature that is often applied to real-time 
    face detection.
    Each Haar-like feature describes the pattern of contrast among adjacent image regions. 
    For example, edges, vertices, and thin lines each generate a kind of feature. 
    Some features are distinctive in the sense that they typically occur in a certain 
    class of object (such as a face) but not in other objects. These distinctive features 
    can be organized into a hierarchy, called a cascade, in which the highest layers 
    contain features of greatest distinctiveness, enabling a classifier to quickly 
    reject subjects that lack these features. If a subject is a good match for the 
    higher-layer features, then the classifier considers the lower-layer features too 
    in order to weed out more false positives.'''

    '''For any given subject, the features may vary depending on the scale of the image 
    and the size of the neighborhood (the region of nearby pixels) in which contrast 
    is being evaluated. The neighborhood’s size is called the window size. To make a 
    Haar cascade classifier scale-invariant or, in other words, robust to changes in 
    scale, the window size is kept constant but images are rescaled a number of times; 
    hence, at some level of rescaling, the size of an object (such as a face) may match 
    the window size. Together, the original image and the rescaled images are called 
    an image pyramid, and each successive level in this pyramid is a smaller rescaled 
    image. OpenCV provides a scale-invariant classifier that can load a Haar cascade 
    from an XML file in a particular format. Internally, this classifier converts any 
    given image into an image pyramid.'''

    '''Haar cascades, as implemented in OpenCV, are not robust to changes in rotation 
    or perspective. For example, an upside-down face is not considered similar to an 
    upright face and a face viewed in profile is not considered similar to a face 
    viewed from the front. A more complex and resource-intensive implementation could 
    improve a Haar cascade’s robustness to rotation by considering multiple transformations 
    of images as well as multiple window sizes. However, we will confine ourselves to 
    the implementation in OpenCV.'''


    '''Your installation of OpenCV 5 should contain a subfolder called data. The path 
    to this folder is stored in an OpenCV variable called cv2.data.haarcascades.
    The data folder contains XML files that can be loaded by an OpenCV class called 
    cv2.CascadeClassifier. An instance of this class interprets a given XML file as 
    a Haar cascade, which provides a detection model for a type of object such as a 
    face. cv2.CascadeClassifier can detect this type of object in any image. As usual, 
    we could obtain a still image from a file, or we could obtain a series of frames 
    from a video file or a video camera.
    From the data folder, we will use the following cascade files:
    * haarcascade_frontalface_default.xml
    * haarcascade_eye.xml
    As their names suggest, these cascades are for detecting faces and eyes. They 
    require a frontal, upright view of the subject. We will use them later when building 
    a face detector.'''

    @staticmethod
    def face_detector_camera():
        camera = cv2.VideoCapture(0)
        while (cv2.waitKey(1) == -1):
            success, frame = camera.read()
            if success:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, 1.3, 5, minSize=(120, 120))
                # Assuming that our user is sitting close to the camera, it is safe to say that the user's face will be larger than 120x120 pixels.
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    roi_gray = gray[y:y+h, x:x+w]
                    '''The eye detector is a bit less accurate than the face detector. 
                    You might see shadows, parts of the frames of glasses, or other 
                    regions of the face falsely detected as eyes. To improve the results, 
                    you could try defining roi_gray as a smaller region of the face, 
                    since we can make a good guess about the eyes' location in an 
                    upright face. You could also try using a maxSize argument to avoid 
                    false positives that are too large to be eyes. Also, you could adjust 
                    minSize and maxSize so that the dimensions are proportional to w 
                    and h, the size of the detected face.'''
                    eyes = eye_cascade.detectMultiScale(
                        roi_gray, 1.11, 5, minSize=(40, 40))
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(frame, (x+ex, y+ey),
                                    (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
                cv2.imshow('Face Detection', frame)
        camera.release()