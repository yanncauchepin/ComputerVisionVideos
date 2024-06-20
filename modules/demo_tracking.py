import cv2
import numpy as np

class DemoTracking():

    def __init__(self):
        pass

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
    @staticmethod
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