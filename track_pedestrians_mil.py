import cv2
import numpy as np

OPENCV_MAJOR_VERSION = int(cv2.__version__.split('.')[0])

"""By its nature, a video of a moving object shows us many variations of that object. 
For example, in some frames, we might see a person’s face and hair, yet in other 
frames, we might see only the person’s hair because she has turned her head away. 
Later, we might see the person’s upper face but not her mouth or chin because she 
is drinking from a coffee mug. To track this person’s head, it would be helpful 
to build a model of various segments rather than just the whole. Over a series of 
several frames, the tracker could learn that the features of the eyes, nose, mouth, 
and hair really are part of this woman’s head, yet the coffee mug is not part of 
her head; it moves separately. This type of learning is, essentially, what MILTrack 
tries to do. MILTrack breaks up the contents of the tracking window and nearby 
regions into several patches or samples subsections, it detects features in each 
patch, and it learns which of these local clusters of features do (or do not) move 
together over time. The optional parameters argument is an instance of 
cv2.TrackerMIL_Params, which has the following properties:
- featureSetNumFeatures: The maximum number of Haar features to detect in each patch. 
The default is 250.
- samplerInitInRadius: The pixel radius, beyond the tracking window, to search 
for positive patches during initialization. The default is 3.0.
- samplerInitMaxNegNum: The maximum number of negative patches, at the end of 
initialization. The default is 65.
- samplerSearchWinSize: The pixel radius, beyond the tracking window, to search 
for negative patches. The same value is used during initialization and updates. 
The default is 25.0.
- samplerTrackInRadius: The pixel radius, beyond the tracking window, to search 
for positive patches during updates. The default is 4.0.
- samplerTrackMaxNegNum: The maximum number of negative patches, at the end of 
each update. The default is 65.
- samplerTrackMaxPosNum: The maximum number of positive patches, at the end of 
each update. The default is 100,000.

After creating an instance of TrackerMIL, we can treat it as a black box; it does 
not expose any further parameters that are specific to MIL. To use TrackerMIL, we 
simply pass an initial frame and tracking window to its init method, and subsequently 
we pass each new frame to its update method, which returns an updated tracking window. 
The init and update methods are part of the cv2.Tracker interface, which TrackerMIL 
implements.

Let’s put MIL into action by modifying our pedestrian tracker. First, make a copy 
of the script (the one containing the Pedestrian class and main method) so that 
later, you can do a before-and-after comparison of the tracking results. Then, 
modify the constructor of our Pedestrian class to remove everything relating to 
histogram back-projection, CamShift, and the Kalman filter; instead, just create 
and initialize a TrackerMIL. The changes are highlighted in the following block 
of code:
"""

class Pedestrian():
    """A tracked pedestrian with a state including an ID and a
    MIL tracker.
    """

    def __init__(self, id, frame, track_window):

        self.id = id

        # Initialize the MIL tracker.
        tracker_params = cv2.TrackerMIL_Params()
        tracker_params.samplerInitMaxNegNum = 100
        tracker_params.samplerSearchWinSize = 30.0
        tracker_params.samplerTrackMaxNegNum = 100
        self.tracker = cv2.TrackerMIL_create(tracker_params)
        self.tracker.init(frame, track_window)

    def update(self, frame):

        # Update the MIL tracker.
        ret, (x, y, w, h) = self.tracker.update(frame)

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

        contours, hier = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                        Pedestrian(id, frame, (x, y, w, h)))
            id += 1

        # Update the tracking of each pedestrian.
        for pedestrian in pedestrians:
            pedestrian.update(frame)

        cv2.imshow('Pedestrians Tracked', frame)

        k = cv2.waitKey(110)
        if k == 27:  # Escape
            break

if __name__ == "__main__":
    main()
