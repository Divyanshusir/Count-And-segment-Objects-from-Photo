import pixellib
from pixellib.instance import instance_segmentation
import cv2
import numpy as np
# np.bool = np.bool_
np.bool = bool

segmentation_model = instance_segmentation()
segmentation_model.load_model('C:/models/mask_rcnn_coco.h5')

segmask , output = segmentation_model.segmentImage('C:/Users/nikhi/Desktop/COUNT_OBJECTS/temp5.jpg' , extract_segmented_objects=True ,save_extracted_objects=True , show_bboxes = True , output_image_name = 'C:/Users/nikhi//Desktop/COUNT_OBJECTS/tempOut.jpg')

cv2.imshow('img' , output)
cv2.waitKey(0)

cv2.destroyAllWindows()


# import cv2
# import pixellib
# from pixellib.instance import instance_segmentation

# import numpy as np

# Define a function to replace np.bool with bool
# def my_bool(*args, **kwargs):
#     return bool(*args, **kwargs)

# Monkey patch np.bool with my_bool
# np.bool = bool


# segmentation_model = instance_segmentation()
# segmentation_model.load_model('C:/models/mask_rcnn_coco.h5')  # Load pre-trained model

# segmented_image, _ = segmentation_model.segmentImage('C:/Users/nikhi/Desktop/COUNT_OBJECTS/temp5.jpg', show_bboxes=True)

# if segmented_image is not None and isinstance(segmented_image, np.ndarray):
#     # Display segmented image
#     cv2.imshow('Segmented Image', segmented_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("Error: Segmented image is invalid or empty.")