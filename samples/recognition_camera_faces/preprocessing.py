import os
import cv2
import numpy as np

root_path = "/media/yanncauchepin/ExternalDisk/Datasets/ComputerVisionFaces/recognition_camera_faces/"    
    
def load_dataframe() :
    
    label_map = {
        0 : 'Yann'
        }
    
    images = []
    labels = []
    
    class_dir = os.path.join(root_path)
    for image_file in os.listdir(class_dir):
        image = cv2.imread(os.path.join(class_dir, image_file),
                         cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image = cv2.resize(image, (200,200))
            # cv2.imshow('image', image)
            # cv2.waitKey(3)
            images.append(image)
            labels.append(0)
    
    images = np.asarray(images, np.uint8)
    labels = np.asarray(labels, np.int32)
            
    return {"images" : images, "labels" : labels, 'label_map': label_map}


if __name__ == '__main__' :

    """EXAMPLE"""

    df_recognition_camera = load_dataframe()