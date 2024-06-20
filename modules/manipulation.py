import os
import cv2
import numpy as np

class Manipulation()

    def __init__(self):
        pass

    @staticmethod
    def read_video(path_to_video, path_to_save, codec=0):
        videoCapture = cv2.VideoCapture(path_to_video)
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        map_ = {
            0: 0
            }
        '''
            0: This option is an uncompressed raw video file. The file extension should be .avi.
            cv2.VideoWriter_fourcc('I','4','2','0'): This option is an uncompressed YUV encoding, 4:2:0 chroma subsampled. This encoding is widely compatible but produces large files. The file extension should be .avi.
            cv2.VideoWriter_fourcc('P','I','M','1'): This option is MPEG-1. The file extension should be .avi.
            cv2.VideoWriter_fourcc('X','V','I','D'): This option is a relatively old MPEG-4 encoding. It is a good option if you want to limit the size of the resulting video. The file extension should be .avi.
            cv2.VideoWriter_fourcc('M','P','4','V'): This option is another relatively old MPEG-4 encoding. It is a good option if you want to limit the size of the resulting video. The file extension should be .mp4.
            cv2.VideoWriter_fourcc('X','2','6','4'): This option is a relatively new MPEG-4 encoding. It may be the best option if you want to limit the size of the resulting video. The file extension should be .mp4.
            cv2.VideoWriter_fourcc('T','H','E','O'): This option is Ogg Vorbis. The file extension should be .ogv.

        '''
        if codec not in map_.keys():
            raise Exception(f'Codec {codec} not recognized.')
        videoWriter = cv2.VideoWriter(
            path_to_save, map_[codec], fps, size)
        success, frame = videoCapture.read()
        while success:  # Loop until there are no more frames.
            videoWriter.write(frame)
            success, frame = videoCapture.read()

    @staticmethod
    def read_gpu_video(path_to_video, path_to_save, codec=0):
        videoCapture = cv2.VideoCapture(
            path_to_video, cv2.CAP_ANY,
            [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        map_ = {
            0: 0
            }
        '''
            0: This option is an uncompressed raw video file. The file extension should be .avi.
            cv2.VideoWriter_fourcc('I','4','2','0'): This option is an uncompressed YUV encoding, 4:2:0 chroma subsampled. This encoding is widely compatible but produces large files. The file extension should be .avi.
            cv2.VideoWriter_fourcc('P','I','M','1'): This option is MPEG-1. The file extension should be .avi.
            cv2.VideoWriter_fourcc('X','V','I','D'): This option is a relatively old MPEG-4 encoding. It is a good option if you want to limit the size of the resulting video. The file extension should be .avi.
            cv2.VideoWriter_fourcc('M','P','4','V'): This option is another relatively old MPEG-4 encoding. It is a good option if you want to limit the size of the resulting video. The file extension should be .mp4.
            cv2.VideoWriter_fourcc('X','2','6','4'): This option is a relatively new MPEG-4 encoding. It may be the best option if you want to limit the size of the resulting video. The file extension should be .mp4.
            cv2.VideoWriter_fourcc('T','H','E','O'): This option is Ogg Vorbis. The file extension should be .ogv.

        '''
        if codec not in map_.keys():
            raise Exception(f'Codec {codec} not recognized.')
        videoWriter = cv2.VideoWriter(
            path_to_save, map_[codec], fps, size,
            [cv2.VIDEOWRITER_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
        success, frame = videoCapture.read()
        while success: # Loop until there are no more frames.
            videoWriter.write(frame)
            success, frame = videoCapture.read()


    @staticmethod
    def read_camera_video(path_to_save, codec=0):
        cameraCapture = cv2.VideoCapture(0)
        fps = 30  # An assumption
        size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        '''For some cameras on certain systems, cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH) 
        and cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT) may return inaccurate results. 
        To be more certain of the actual image dimensions, you can first capture a 
        frame and then get its height and width with code such as h, w = frame.shape[:2].'''
        map_ = {
            0: 0,
            1: cv2.VideoWriter_fourcc('I','4','2','0')
            }
        '''
            0: This option is an uncompressed raw video file. The file extension should be .avi.
            cv2.VideoWriter_fourcc('I','4','2','0'): This option is an uncompressed YUV encoding, 4:2:0 chroma subsampled. This encoding is widely compatible but produces large files. The file extension should be .avi.
            cv2.VideoWriter_fourcc('P','I','M','1'): This option is MPEG-1. The file extension should be .avi.
            cv2.VideoWriter_fourcc('X','V','I','D'): This option is a relatively old MPEG-4 encoding. It is a good option if you want to limit the size of the resulting video. The file extension should be .avi.
            cv2.VideoWriter_fourcc('M','P','4','V'): This option is another relatively old MPEG-4 encoding. It is a good option if you want to limit the size of the resulting video. The file extension should be .mp4.
            cv2.VideoWriter_fourcc('X','2','6','4'): This option is a relatively new MPEG-4 encoding. It may be the best option if you want to limit the size of the resulting video. The file extension should be .mp4.
            cv2.VideoWriter_fourcc('T','H','E','O'): This option is Ogg Vorbis. The file extension should be .ogv.

        '''
        if codec not in map_.keys():
            raise Exception(f'Codec {codec} not recognized.')
        videoWriter = cv2.VideoWriter(
            path_to_save, map_[codec], fps, size)
        success, frame = cameraCapture.read()
        numFramesRemaining = 10 * fps - 1 # 10 seconds of frames
        while success and numFramesRemaining > 0:
            videoWriter.write(frame)
            success, frame = cameraCapture.read()
            numFramesRemaining -= 1
        cameraCapture.release()
        videoWriter.release()