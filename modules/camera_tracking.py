import os
import cv2
import numpy as np

class CameraTracking():

    def __init__(self):
        pass

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
    @staticmethod
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
    @staticmethod
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