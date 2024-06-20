import os
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier(f'{cv2.data.haarcascades}haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(f'{cv2.data.haarcascades}haarcascade_eye.xml')

class FaceRecognition():

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

    '''If we trained a face recognizer on these samples, we would then have to run 
    face recognition on an image that contains the face of one of the sampled people. 
    This process might be educational, but perhaps not as satisfying as providing images 
    of our own.
    Let's go ahead and write a script that will generate those images for us. A few 
    images containing different expressions are all that we need, but it is preferable 
    that the training images are square and are all the same size. Our sample script 
    uses a size of 200x200, but most freely available datasets have smaller images 
    than this.'''

    @staticmethod
    def create_face_recognition_camera_database(path_to_dataset):
        camera = cv2.VideoCapture(0)
        folder_path = os.path.join(path_to_dataset)
        os.makedirs(folder_path, exist_ok=True)
        count = 0
        while (cv2.waitKey(1) == -1):
            success, frame = camera.read()
            if success:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, 1.3, 5, minSize=(120, 120))
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    face_img = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                    face_filename = '%s/%d.pgm' % (folder_path, count)
                    cv2.imwrite(face_filename, face_img)
                    count += 1
                cv2.imshow('Capturing Faces...', frame)
        camera.release()

    '''OpenCV 5 implements three different algorithms for recognizing faces: Eigenfaces, 
    Fisherfaces, and Local Binary Pattern Histograms (LBPHs). Eigenfaces and Fisherfaces 
    are derived from a more general-purpose algorithm called Principal Component 
    Analysis (PCA).
    they all follow a similar process; they take a set of classified observations 
    (our face database, containing numerous samples per individual), train a model 
    based on it, perform an analysis of face images (which may be face regions that 
    we detected in an image or video), and determine two things: the subject's 
    identity and a measure of confidence that this identification is correct. The 
    latter is commonly known as the confidence score.
    Eigenfaces performs PCA, which identifies principal components of a certain set 
    of observations (again, your face database), calculates the divergence of the 
    current observation (the face being detected in an image or frame) compared to 
    the dataset, and produces a value. The smaller the value, the smaller the difference 
    between the face database and the detected face; hence, a value of 0 is an exact 
    match.
    Fisherfaces also derives from PCA and evolves the concept, applying more complex 
    logic. While computationally more intensive, it tends to yield more accurate 
    results than Eigenfaces.
    LBPH instead divides a detected face into small cells and, for each cell, builds 
    a histogram that describes whether the brightness of the image is increasing when 
    comparing neighboring pixels in a given direction. This cell's histogram can be 
    compared to the histogram of the corresponding cell in the model, producing a 
    measure of similarity. Of the face recognizers in OpenCV, the implementation of 
    LBPH is the only one that allows the model sample faces and the detected faces 
    to be of a different shape and size. Hence, it is a convenient option, and the 
    authors of this book find that its accuracy compares favorably to the other two 
    options.'''


    from sample.recognition_camera_faces.preprocessing import load_dataframe

    '''Optionally, we could have passed two arguments to cv2.EigenFaceRecognizer_create:

    - num_components: This is the number of components to keep for the PCA.
    - threshold: This is a floating-point value specifying a confidence threshold. 
    Faces with a confidence score below the threshold will be discarded. By default, 
    the threshold is the maximum floating-point value so that no faces are discarded.

    To improve upon this approach and make it more robust, you could take further steps 
    such as correctly aligning and rotating detected faces so that the accuracy of the 
    recognition is maximized.
    '''
    @staticmethod
    def face_recognizer_eigenfaces(df_dataset):
        model = cv2.face.EigenFaceRecognizer_create()
        model.train(df_dataset['images'], df_dataset['labels'])
        camera = cv2.VideoCapture(0)
        while (cv2.waitKey(1) == -1):
            success, frame = camera.read()
            if success:
                faces = face_cascade.detectMultiScale(frame, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    roi_gray = gray[x:x+w, y:y+h]
                    if roi_gray.size == 0:
                        # The ROI is empty. Maybe the face is at the image edge.
                        # Skip it.
                        continue
                    roi_gray = cv2.resize(roi_gray, (200,200))
                    label, confidence = model.predict(roi_gray)
                    text = '%s, confidence=%.2f' % (df_dataset['label_map'][label], confidence)
                    cv2.putText(frame, text, (x, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('Face Recognition', frame)
                

    @staticmethod
    def face_recognizer_fisherfaces(df_dataset):
        model = cv2.face.FisherFaceRecognizer_create()
        model.train(df_dataset['images'], df_dataset['labels'])
        camera = cv2.VideoCapture(0)
        while (cv2.waitKey(1) == -1):
            success, frame = camera.read()
            if success:
                faces = face_cascade.detectMultiScale(frame, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    roi_gray = gray[x:x+w, y:y+h]
                    if roi_gray.size == 0:
                        # The ROI is empty. Maybe the face is at the image edge.
                        # Skip it.
                        continue
                    roi_gray = cv2.resize(roi_gray, (200,200))
                    label, confidence = model.predict(roi_gray)
                    text = '%s, confidence=%.2f' % (df_dataset['label_map'][label], confidence)
                    cv2.putText(frame, text, (x, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('Face Recognition', frame)


    '''
    For the LBPH algorithm, again, the process is similar. However, the algorithm 
    factory takes the following optional parameters (in order):

    - radius: The pixel distance between the neighbors that are used to calculate a 
    cell's histogram (by default, 1)
    - neighbors: The number of neighbors used to calculate a cell's histogram (by default, 8)
    - grid_x: The number of cells into which the face is divided horizontally (by default, 8)
    - grid_y: The number of cells into which the face is divided vertically (by default, 8)
    - confidence: The confidence threshold (by default, the highest possible 
    floating-point value so that no results are discarded)


    Note that, with LBPH, we do not need to resize images as the division into grids 
    allows a comparison of patterns identified in each cell.
    '''
    @staticmethod
    def face_recognizer_lbphffaces(df_dataset):
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(df_dataset['images'], df_dataset['labels'])
        camera = cv2.VideoCapture(0)
        while (cv2.waitKey(1) == -1):
            success, frame = camera.read()
            if success:
                faces = face_cascade.detectMultiScale(frame, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    roi_gray = gray[x:x+w, y:y+h]
                    if roi_gray.size == 0:
                        # The ROI is empty. Maybe the face is at the image edge.
                        # Skip it.
                        continue
                    roi_gray = cv2.resize(roi_gray, (200,200))
                    label, confidence = model.predict(roi_gray)
                    text = '%s, confidence=%.2f' % (df_dataset['label_map'][label], confidence)
                    cv2.putText(frame, text, (x, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow('Face Recognition', frame)

    '''
    The predict method returns a tuple, in which the first element is the label of 
    the recognized individual and the second is the confidence score. All algorithms 
    come with the option of setting a confidence score threshold, which measures the 
    distance of the recognized face from the original model; therefore, a score of 0 
    signifies an exact match.

    There may be cases in which you would rather retain all recognitions and then 
    apply further processing, so you can come up with your own algorithms to estimate 
    the confidence score of a recognition. For example, if you are trying to identify 
    people in a video, you may want to analyze the confidence score in subsequent 
    frames to establish whether the recognition was successful or not. In this case, 
    you can inspect the confidence score obtained by the algorithm and draw your own 
    conclusions.

    The typical range of the confidence score depends on the algorithm. Eigenfaces 
    and Fisherfaces produce values (roughly) in the range of 0 to 20,000, with any 
    score below 4,000-5,000 being a quite confident recognition. For LBPH, a good 
    recognition scores (roughly) below 50, and any value above 80 is considered a 
    poor confidence score.

    A normal custom approach would be to hold off drawing a rectangle around a recognized 
    face until we have a number of frames with a good confidence score (where “good” 
    is an arbitrary threshold we must choose, based on our algorithm and use case), 
    but you have total freedom to use OpenCV's face recognition module to tailor your 
    application to your needs. Next, let’s see how face detection and recognition work 
    in a specialized use case.
    '''


    