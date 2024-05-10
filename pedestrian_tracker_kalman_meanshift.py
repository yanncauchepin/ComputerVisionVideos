import os
import cv2
import numpy as np

"""The application will adhere to the following logic:
1. Capture frames from a video file.
2. Use the first 20 frames to populate the history of a background subtractor.
3. Based on background subtraction, use the 21st frame to identify moving 
foreground objects. We will treat these as pedestrians. For each pedestrian, 
assign an ID and an initial tracking window, and then calculate a histogram.
4. For each subsequent frame, track each pedestrian using a Kalman filter 
and MeanShift."""

OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])

class Pedestrian():
    """A tracked pedestrian with a state including an ID, tracking
    window, histogram, and Kalman filter.
    """

    def __init__(self, id, hsv_frame, track_window):

        self.id = id

        self.track_window = track_window
        self.term_crit = \
            (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 1)

        # Initialize the histogram.
        x, y, w, h = track_window
        roi = hsv_frame[y:y+h, x:x+w]
        roi_hist = cv2.calcHist([roi], [0, 2], None, [15, 16],
                                [0, 180, 0, 256])
        self.roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255,
                                      cv2.NORM_MINMAX)

        # Initialize the Kalman filter.
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0],
             [0, 1, 0, 1],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1]], np.float32) * 0.03
        cx = x+w/2
        cy = y+h/2
        self.kalman.statePre = np.array(
            [[cx], [cy], [0], [0]], np.float32)
        self.kalman.statePost = np.array(
            [[cx], [cy], [0], [0]], np.float32)

    def update(self, frame, hsv_frame):

        back_proj = cv2.calcBackProject(
            [hsv_frame], [0, 2], self.roi_hist, [0, 180, 0, 256], 1)

        ret, self.track_window = cv2.meanShift(
            back_proj, self.track_window, self.term_crit)
        x, y, w, h = self.track_window
        center = np.array([x+w/2, y+h/2], np.float32)

        prediction = self.kalman.predict()
        estimate = self.kalman.correct(center)
        center_offset = estimate[:,0][:2] - center
        self.track_window = (x + int(center_offset[0]),
                             y + int(center_offset[1]), w, h)
        x, y, w, h = self.track_window

        # Draw the predicted center position as a blue circle.
        cv2.circle(frame, (int(prediction[0]), int(prediction[1])),
                   4, (255, 0, 0), -1)

        # Draw the corrected tracking window as a cyan rectangle.
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 255, 0), 2)

        # Draw the ID above the rectangle in blue text.
        cv2.putText(frame, 'ID: %d' % self.id, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0),
                    1, cv2.LINE_AA)

def main():

    cap = cv2.VideoCapture('/home/yanncauchepin/Git/PublicProjects/ComputerVisionVideos/pedestrians.avi')

    # Create the KNN background subtractor.
    bg_subtractor = cv2.createBackgroundSubtractorKNN()
    history_length = 20
    bg_subtractor.setHistory(history_length)

    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3))
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 7))

    pedestrians = []
    num_history_frames_populated = 0
    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break

        # Apply the KNN background subtractor.
        fg_mask = bg_subtractor.apply(frame)

        # Let the background subtractor build up a history.
        if num_history_frames_populated < history_length:
            num_history_frames_populated += 1
            continue

        # Create the thresholded image.
        _, thresh = cv2.threshold(fg_mask, 127, 255,
                                  cv2.THRESH_BINARY)
        cv2.erode(thresh, erode_kernel, thresh, iterations=2)
        cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

        # Detect contours in the thresholded image.
        if OPENCV_MAJOR_VERSION >= 4:
            # OpenCV 4 or a later version is being used.
            contours, hier = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            # OpenCV 3 or an earlier version is being used.
            # cv2.findContours has an extra return value.
            # The extra return value is the thresholded image, which
            # is unchanged, so we can ignore it.
            _, contours, hier = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Draw green rectangles around large contours.
        # Also, if no pedestrians are being tracked yet, create some.
        should_initialize_pedestrians = len(pedestrians) == 0
        id = 0
        for c in contours:
            if cv2.contourArea(c) > 500:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x+w, y+h),
                              (0, 255, 0), 1)
                if should_initialize_pedestrians:
                    pedestrians.append(
                        Pedestrian(id, hsv_frame,
                                   (x, y, w, h)))
            id += 1

        # Update the tracking of each pedestrian.
        for pedestrian in pedestrians:
            pedestrian.update(frame, hsv_frame)

        cv2.imshow('Pedestrians Tracked', frame)

        k = cv2.waitKey(110)
        if k == 27:  # Escape
            break

if __name__ == "__main__":
    main()
    
"""The preceding program can be expanded and improved in various ways, depending 
on the requirements of a particular application. Consider the following examples:

- You could remove a Pedestrian object from the pedestrians list (and thereby 
destroy the Pedestrian object) if the Kalman filter predicts the pedestrian's 
position to be outside the frame.
- You could check whether each detected moving object corresponds to an existing 
Pedestrian instance in the pedestrians list, and, if not, add a new object to the 
list so that it will be tracked in subsequent frames.
- You could train a support vector machine (SVM) and use it to classify each moving 
object. By these means, you could establish whether or not the moving object is 
something you intend to track. For instance, a dog might enter the scene but your 
application might require the tracking of humans only. For more information on 
training an SVM, refer to Chapter 7, Building Custom Object Detectors.
- You could adopt a tracking technique that is based on machine learning, instead 
of using MeanShift (or CamShift) and a Kalman filter. OpenCV provides easy-to-use 
implementations of several machine learning trackers, with a common interface 
called cv2.Tracker."""

