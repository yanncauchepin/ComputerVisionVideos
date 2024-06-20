import cv2
import numpy as np

class BackgroundSubtractor():

    def init(self):
        pass

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

    @static_method
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
    @static_method
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

    @static_method
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
    @static_method
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