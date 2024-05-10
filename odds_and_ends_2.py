import os
import cv2
import numpy as np

"""Many motion detection techniques are based on the simple concept of background 
subtraction. For example, suppose that we have a stationary camera viewing a scene 
that is also mostly stationary. In addition to this, suppose that the camera's 
exposure and the lighting conditions in the scene are stable so that frames do 
not vary much in terms of brightness. Under these conditions, we can easily capture 
a reference image that represents the background or, in other words, the stationary 
components of the scene. Then, any time the camera captures a new frame, we can 
subtract the frame from the reference image and take the absolute value of this 
difference in order to obtain a measurement of motion at each pixel location in 
the frame. If any region of the frame is very different from the reference image, 
we conclude that the given region is a moving object.

Background subtraction techniques, in general, have the following limitations:
- Any camera motion, change in exposure, or change in lighting conditions can cause 
a change in pixel values throughout the entire scene all at once; therefore, the 
entire background model (or reference image) becomes outdated.
- A piece of the background model can become outdated if an object enters the scene 
and then just stays there for a long period of time. For example, suppose our scene 
is a hallway. Someone enters the hallway, puts a poster on the wall, and leaves 
the poster there. For all practical purposes, the poster is now just another part 
of the stationary background; however, it was not part of our reference image, 
so our background model has become partly outdated.

Another general limitation is that shadows and solid objects can affect a background 
subtractor in similar ways. For instance, we might get an inaccurate picture of a 
moving object's size and shape because we cannot differentiate the object from 
its shadow. However, advanced background subtraction techniques do attempt to 
distinguish between shadow regions and solid objects using various means.

Background subtractors generally have yet another limitation: they do not offer 
fine-grained control over the kind of motion that they detect. For example, if a 
scene shows a subway car that is continuously shaking as it travels on its track, 
this repetitive motion will affect the background subtractor. For practical purposes, 
we might consider the subway car's vibrations to be normal variations in a semi-stationary 
background. We might even know the frequency of these vibrations. However, a 
background subtractor does not embed any information about frequencies of motion, 
so it does not offer a convenient or precise way in which to filter out such predictable 
motions. To compensate for such shortcomings, we can apply preprocessing steps such 
as blurring the reference image and also blurring each new frame; in this way, certain 
frequencies are suppressed, albeit in a manner that is not very intuitive, efficient, 
or precise.
"""

def camera_basic_background_substractor():
    '''
    1. Start capturing frames from a camera.
    2. Discard the first nine frames so that the camera has time to properly adjust 
    its autoexposure to suit the lighting conditions in the scene.
    3. Take the 10th frame, convert it to grayscale, blur it, and use this blurred 
    image as the reference image of the background.
    4. For each subsequent frame, blur the frame, convert it to grayscale, and 
    compute the absolute difference between this blurred frame and the reference 
    image of the background. Perform thresholding, smoothing, and contour detection 
    on the differenced image. Draw and show the bounding boxes of the major contours.

    The use of a Gaussian blur should make our background subtractor less susceptible 
    to small vibrations, as well as digital noise. We will also use morphological 
    operations for the same purpose.
    '''
    BLUR_RADIUS = 21
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (9, 9))
    cap = cv2.VideoCapture(0)
    # Capture several frames to allow the camera's autoexposure to adjust.
    for i in range(10):
        success, frame = cap.read()
    if not success:
        exit(1)
    gray_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_background = cv2.GaussianBlur(gray_background,
                                       (BLUR_RADIUS, BLUR_RADIUS), 0)
    success, frame = cap.read()
    while success:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame,
                                      (BLUR_RADIUS, BLUR_RADIUS), 0)
        """Now, we can compare the blurred, grayscale version of the current frame 
        to the blurred, grayscale version of the background image. Specifically, 
        we will use OpenCV's cv2.absdiff function to find the absolute value 
        (or the magnitude) of the difference between these two images. Then, we 
        will apply a threshold to obtain a pure black-and-white image, and morphological 
        operations to smooth the thresholded image. Here is the relevant code:"""
        diff = cv2.absdiff(gray_background, gray_frame)
        _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        cv2.erode(thresh, erode_kernel, thresh, iterations=2)
        cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)
        """At this point, if our technique has worked well, our thresholded image 
        should contain white blobs wherever there is a moving object. Now, we want 
        to find the contours of the white blobs and draw bounding boxes around them. 
        As a further means of filtering out small changes that are probably not 
        real objects, we will apply a threshold based on the area of the contour. 
        If the contour is too small, we conclude that it is not a real moving object. 
        (Of course, the definition of too small may vary depending on your camera's 
        resolution and your application; in some circumstances, you might not wish 
        to apply this test at all.) Here is the code to detect contours and draw 
        bounding boxes:"""
        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 4000:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.imshow('diff', diff)
        cv2.imshow('thresh', thresh)
        cv2.imshow('detection', frame)
        k = cv2.waitKey(1)
        if k == 27: # Escape
            break
        success, frame = cap.read()


"""Like cv2.grabCut, the various subclass implementations of cv2.BackgroundSubtractor 
can produce a mask that assigns different values to different segments of the image. 
Specifically, a background subtractor can mark foreground segments as white (that is, 
an 8-bit grayscale value of 255), background segments as black (0), and (in some 
implementations) shadow segments as gray (127). Moreover, unlike GrabCut, the background 
subtractors update the foreground/background model over time, typically by applying 
machine learning to a series of frames. Many of the background subtractors are named 
after the statistical clustering technique on which they base their approach to machine 
learning."""
def mog_background_substractor(cap):
    """OpenCV has two implementations of a MOG background subtractor. Perhaps not 
    surprisingly, they are named cv2.bgsegm.BackgroundSubtractorMOG and 
    cv2.BackgroundSubtractorMOG2. The latter is a more recent and improved implementation, 
    which adds support for shadow detection.
    From our basic background subtraction script:
    1. Replace our basic background subtraction model with a MOG background subtractor.
    2. As input, use a video file instead of a camera.
    3. Remove the use of Gaussian blur.
    4. Adjust the parameters used in the thresholding, morphology, and contour analysis steps.
    5. Get and show the MOG subtractor’s model of the background."""
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    '''To eliminate the thresholding step, we could configure the background subtractor 
    to color the shadows black (instead of gray). To do this, we would use the 
    subtractor’s setShadowValue method:
    However, for our learning purposes, it is good to see a visualization of the 
    shadows in gray before we filter them out.
    '''
    # bg_subtractor.setShadowValue(0)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    success, frame = cap.read()
    while success:
        '''When we pass a frame to the background subtractor's apply method, the 
        subtractor updates its internal model of the background and then returns 
        a mask. As we previously discussed, the mask is white (255) for foreground 
        segments, gray (127) for shadow segments, and black (0) for background 
        segments. For our purposes, we treat shadows as the background, so we apply 
        a nearly white threshold (244) to the mask.'''
        fg_mask = bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        cv2.erode(thresh, erode_kernel, thresh, iterations=2)
        cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)
        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 1000:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.imshow('mog', fg_mask)
        cv2.imshow('thresh', thresh)
        cv2.imshow('background',
                   bg_subtractor.getBackgroundImage())
        cv2.imshow('detection', frame)
        k = cv2.waitKey(30)
        if k == 27:  # Escape
            break
        success, frame = cap.read()

    '''This scene contains not only shadows but also reflections due to the polished 
    floor and wall. When shadow detection is enabled (as in the preceding screenshots), 
    we are able to use a threshold to remove the shadows and reflections from our mask, 
    leaving us with an accurate detection rectangle around the man in the hall.'''

def knn_background_substractor(cap):
    bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    # bg_subtractor.setShadowValue(0)
    '''With the following changes, we can use morphology kernels that are slightly 
    better adapted to a horizontally elongated object (in this case, a car), 
    and we can use a video of traffic as input:'''
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 11))

    
    success, frame = cap.read()
    while success:
        fg_mask = bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        cv2.erode(thresh, erode_kernel, thresh, iterations=2)
        cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)
        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 1000:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.imshow('knn', fg_mask)
        cv2.imshow('thresh', thresh)
        cv2.imshow('background',
                   bg_subtractor.getBackgroundImage())
        cv2.imshow('detection', frame)
        k = cv2.waitKey(30)
        if k == 27:  # Escape
            break
        success, frame = cap.read()

'''You are free to experiment with your own modifications to our background subtraction 
script. If you have obtained OpenCV with the optional opencv_contrib modules, as 
described in Chapter 1, Setting Up OpenCV, then several more background subtractors 
are available to you in the cv2.bgsegm module. They can be created using the following 
functions:
    - cv2.bgsegm.createBackgroundSubtractorCNT
    - cv2.bgsegm.createBackgroundSubtractorGMG
    - cv2.bgsegm.createBackgroundSubtractorGSOC
    - cv2.bgsegm.createBackgroundSubtractorLSBP
    - cv2.bgsegm.createBackgroundSubtractorMOG
These functions do not support the detectShadows parameter, and they create background 
subtractors that do not support shadow detection. However, all the background subtractors 
support the apply method.
Moreover, some of them, including the GMG subtractor, do not support the 
getBackgroundImage method.
'''
def gmg_background_substractor(cap):
    '''OpenCV's implementation of GMG does not differentiate between shadows and 
    solid objects, so the detection rectangles are elongated in the direction of 
    the cars' shadows or reflections.'''
    bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGMG()
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 9))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 11))
    success, frame = cap.read()
    while success:
        fg_mask = bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        cv2.erode(thresh, erode_kernel, thresh, iterations=2)
        cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)
        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > 1000:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        cv2.imshow('gmg', fg_mask)
        cv2.imshow('thresh', thresh) 
        cv2.imshow('detection', frame)
        k = cv2.waitKey(30)
        if k == 27: # Escape
            break
        success, frame = cap.read()


"""
We have seen that background subtraction can be an effective technique for detecting 
moving objects; however, we know that it has some inherent limitations. Notably, 
it assumes that the current background can be predicted based on past frames. 
This assumption is fragile. For example, if the camera moves, the entire background 
model could suddenly become outdated. Thus, in a robust tracking system, it is 
important to build some kind of model of foreground objects rather than just the 
background.
If we were tracking cars, we would want a different model for each car in the scene 
so that a red car and a blue car would not get mixed up. We would want to track 
the motion of each car separately.

A color histogram may serve as a sufficiently unique description. Essentially, an 
object's color histogram is an estimate of the probability distribution of pixel 
colors in the object. For example, the histogram could indicate that each pixel 
in the object is 10% likely to be blue. The histogram is based on the actual colors 
observed in the object's region of a reference image. The histogram serves as a 
lookup table that directly maps pixel values to probabilities, so it enables us 
to use every pixel as a feature, at a low computational cost. In this way, we can 
afford to perform tracking with very fine spatial resolution in real time. To find 
the most likely location of an object that we are tracking, we just have to find 
the region of interest where the pixel values map to the maximum probability, 
according to the histogram.

Naturally, this approach is leveraged by an algorithm with a catchy name: MeanShift. 
For each frame in a video, the MeanShift algorithm performs tracking iteratively by 
computing a centroid based on probability values in the current tracking rectangle, 
shifting the rectangle's center to this centroid, recomputing the centroid based on 
values in the new rectangle, shifting the rectangle again, and so on. This process 
continues until convergence is achieved (meaning that the centroid ceases to move 
or nearly ceases to move) or until a maximum number of iterations is reached. 
"""
"""
For our first demonstration of MeanShift, we are not concerned with the approach 
to the initial detection of moving objects. We will use a naive approach that simply 
chooses the central part of the first video frame as our initial region of interest. 
(The user must ensure that an object of interest is initially located in the center 
of the video.) We will calculate a histogram of this initial region of interest. 
Then, in subsequent frames, we will use this histogram and the MeanShift algorithm 
to track the object.
"""
"""
To calculate a color histogram, OpenCV provides a function called cv2.calcHist. 
To apply a histogram as a lookup table, OpenCV provides another function called 
cv2.calcBackProject. The latter operation is known as histogram back-projection, 
and it transforms a given image into a probability map based on a given histogram.
A histogram can use any color model, such as blue-green-red (BGR), hue-saturation-value 
(HSV), or grayscale.

OpenCV represents H values with a range from 0 to 179, which, unfortunately, is a 
tad less precise than a typical HSV representation. Some other systems use a wider 
(more precise) range from 0 to 359 (like the degrees of a circle) or from 0 to 255 
(to match the range of one byte).

When we use cv2.calcHist to generate a hue histogram, it returns a 1D array.
Alternatively, depending on the parameters we provide, we could use cv2.calcHist 
to generate a histogram of a different channel or of two channels at once. In the 
latter case, cv2.calcHist would return a 2D array.

Once we have a histogram, we can back-project the histogram onto any image. 
cv2.calcBackProject produces a back-projection in the format of an 8-bit grayscale 
image, with pixel values that potentially range from 0 (indicating a low probability) 
to 255 (indicating a high probability), depending on how we scale the values.
"""
"""cv2.calcHist DOCUMENTATION"""
"""cv2.calcBackProject DOCUMENTATION"""
def centered_camera_meanshift_tracking():
    cap = cv2.VideoCapture(0)

    # Capture several frames to allow the camera's autoexposure to adjust.
    for i in range(20):
        success, frame = cap.read()
    if not success:
        exit(1)

    # Define an initial tracking window in the center of the frame.
    frame_h, frame_w = frame.shape[:2]
    w = frame_w//8
    h = frame_h//8
    x = frame_w//2 - w//2
    y = frame_h//2 - h//2
    track_window = (x, y, w, h)

    # Calculate the normalized HSV histogram of the initial window.
    roi = frame[y:y+h, x:x+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = None
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Define the termination criteria:
    # 10 iterations or convergence within 1-pixel radius.
    term_crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)

    success, frame = cap.read()
    while success:

        # Perform back-projection of the HSV histogram onto the frame.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # Perform tracking with MeanShift.
        num_iters, track_window = cv2.meanShift(
            back_proj, track_window, term_crit)

        # Draw the tracking window.
        x, y, w, h = track_window
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow('back-projection', back_proj)
        cv2.imshow('meanshift', frame)

        k = cv2.waitKey(1)
        if k == 27:  # Escape
            break

        success, frame = cap.read()

"""By now, you should have a good idea of how color histograms, back-projections, 
and MeanShift work. However, the preceding program (and MeanShift in general) has 
a limitation: the size of the window does not change with the size of the object 
in the frames being tracked.
Gary Bradski – one of the founders of the OpenCV project – published a paper in 
1998 to improve the accuracy of MeanShift. He described a new algorithm called 
Continuously Adaptive MeanShift (CAMShift or CamShift), which is very similar to 
MeanShift but also adapts the size of the tracking window when MeanShift reaches 
convergence.
Although CamShift is a more complex algorithm than MeanShift, OpenCV provides a 
very similar interface for the two algorithms. The main difference is that a call 
to cv2.CamShift returns a rectangle with a particular rotation that follows the 
rotation of the object being tracked.
"""
def centered_camera_camshift_tracking():
    cap = cv2.VideoCapture(0)

    # Capture several frames to allow the camera's autoexposure to adjust.
    for i in range(20):
        success, frame = cap.read()
    if not success:
        exit(1)

    # Define an initial tracking window in the center of the frame.
    frame_h, frame_w = frame.shape[:2]
    w = frame_w//8
    h = frame_h//8
    x = frame_w//2 - w//2
    y = frame_h//2 - h//2
    track_window = (x, y, w, h)

    # Calculate the normalized HSV histogram of the initial window.
    roi = frame[y:y+h, x:x+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = None
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    # Define the termination criteria:
    # 10 iterations or convergence within 1-pixel radius.
    term_crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)
    success, frame = cap.read()
    while success:
        # Perform back-projection of the HSV histogram onto the frame.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # Perform tracking with CamShift.
        rotated_rect, track_window = cv2.CamShift(
            back_proj, track_window, term_crit)
        # Draw the tracking window.
        box_points = cv2.boxPoints(rotated_rect)
        box_points = np.intp(box_points)
        cv2.polylines(frame, [box_points], True, (255, 0, 0), 2)
        cv2.imshow('back-projection', back_proj)
        cv2.imshow('camshift', frame)
        k = cv2.waitKey(1)
        if k == 27: # Escape
            break
        success, frame = cap.read()
        
    """We use the cv2.boxPoints function to find the vertices of the rotated tracking 
    rectangle. Then, we use the cv2.polylines function to draw the lines connecting 
    these vertices."""

"""The Kalman filter operates recursively on a stream of noisy input data to produce 
a statistically optimal estimate of the underlying system state. In the context of 
computer vision, the Kalman filter can smooth the estimate of a tracked object's position.
The Kalman filter itself is not gathering these tracking results, but it is updating 
its model of the object's motion based on the tracking results derived from another 
algorithm – in our case, a visual tracking algorithm such as MeanShift. The tracking 
algorithm produces results that are (probably) noisy but (hopefully) unbiased; that 
is to say, its errors do not tend toward any particular direction. Conversely, it 
is the Kalman filter’s role to smooth these results, to make them less noisy, at 
the risk of biasing the results based by seeking a trend to fit an idealized model 
of motion.
Naturally, the Kalman filter cannot foresee new forces acting; but it can update 
its model of the ball’s motion after the fact, when new tracking results deviate 
significantly from what the Kalman filter would have predicted.

From the preceding description, we gather that the Kalman filter's algorithm has 
two phases:
- Predict: In the first phase, the Kalman filter uses the covariance calculated 
up to the current point in time to estimate the object's new position.
- Update: In the second phase, the Kalman filter records the object's position 
and adjusts the covariance for the next cycle of calculations.
The update phase is – in OpenCV's terms – a correction. 

For the purpose of smoothly tracking objects, we will call the predict method to 
estimate the position of an object, and then use the correct method to instruct 
the Kalman filter to adjust its calculations based on a new tracking result from 
another algorithm such as MeanShift. However, before we combine the Kalman filter 
with a computer vision algorithm.
"""
"""
Our demo will implement the following sequence of operations:

1. Start by initializing a black image and a Kalman filter. Show the black image 
in a window.
2. Every time the windowed application processes input events, use the Kalman 
filter to predict the mouse's position. Then, correct the Kalman filter's model 
based on the actual mouse coordinates. On top of the black image, draw a red line 
from the old predicted position to the new predicted position, and then draw a 
green line from the old actual position to the new actual position. Show the 
drawing in the window.
3. When the user hits the Esc key, exit and save the drawing to a file.
"""
def demo_kalman_fitler():
    img = np.zeros((800, 800, 3), np.uint8)
    # Initialize the Kalman filter.
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]], np.float32) * 0.03
    """Based on the preceding initialization, our Kalman filter will track a 2D 
    object's position and velocity.
    For now, let's just take note of the two parameters in cv2.KalmanFilter(4, 2). 
    The first parameter is the number of variables tracked (or predicted) by the 
    Kalman filter, in this case, 4: the x position, the y position, the x velocity, 
    and the y velocity. The second parameter is the number of variables provided 
    to the Kalman filter as measurements, in this case, 2: the x position and the 
    y position. We also initialize several matrices that describe the relationships 
    among all these variables.
    
    Having initialized the image and the Kalman filter, we also have to declare 
    variables to hold the actual (measured) and predicted mouse coordinates. 
    Initially, we have no coordinates, so we will assign None to these variables:
    """
    
    """Then, we declare a callback function that handles mouse movement. This function 
    is going to update the state of the Kalman filter, and draw a visualization 
    of both the unfiltered mouse movement and the Kalman-filtered mouse movement. 
    The first time we receive mouse coordinates, we initialize the Kalman filter's 
    state so that its initial prediction is the same as the actual initial mouse 
    coordinates. (If we did not do this, the Kalman filter would assume that the 
    initial mouse position was (0, 0).) Subsequently, whenever we receive new 
    mouse coordinates, we correct the Kalman filter with the current measurement, 
    calculate the Kalman prediction, and, finally, draw two lines: a green line 
    from the last measurement to the current measurement and a red line from the 
    last prediction to the current prediction. Here is the callback function's 
    implementation:"""
    last_measurement = None
    last_prediction = None
    def on_mouse_moved(event, x, y, flags, param):
        nonlocal img, kalman, last_measurement, last_prediction
        measurement = np.array([[x], [y]], np.float32)
        if last_measurement is None:
            # This is the first measurement.
            # Update the Kalman filter's state to match the measurement.
            kalman.statePre = np.array(
                [[x], [y], [0], [0]], np.float32)
            kalman.statePost = np.array(
                [[x], [y], [0], [0]], np.float32)
            prediction = measurement
        else:
            kalman.correct(measurement)
            prediction = kalman.predict()  # Gets a reference, not a copy
            # Trace the path of the measurement in green.
            cv2.line(img, (int(last_measurement[0]), int(last_measurement[1])),
                     (int(measurement[0]), int(measurement[1])), (0, 255, 0))
            # Trace the path of the prediction in red.
            cv2.line(img, (int(last_prediction[0]), int(last_prediction[1])),
                     (int(prediction[0]), int(prediction[1])), (0, 0, 255))
        last_prediction = prediction.copy()
        last_measurement = measurement
    cv2.namedWindow('kalman_tracker')
    cv2.setMouseCallback('kalman_tracker', on_mouse_moved)
    while True:
        cv2.imshow('kalman_tracker', img)
        k = cv2.waitKey(1)
        if k == 27:  # Escape
            break

if __name__=='__main__':
    
    
    # camera_basic_background_substractor()
    
    # cap = cv2.VideoCapture('hallway.mpg')
    # mog_background_substractor(cap)
    
    # cap = cv2.VideoCapture('traffic.flv')
    # knn_background_substractor(cap)
    
    # cap = cv2.VideoCapture('traffic.flv')
    # gmg_background_substractor(cap)
    
    # centered_camera_meanshift_tracking()
    # centered_camera_camshift_tracking()
    
    # demo_kalman_fitler()