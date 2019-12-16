# -*- coding: utf-8 -*-

import cv2


def rectify(rectified_shape, img, roi):
    """Return a rectified perspective of the image

    PARAMETERS:
    rectified_shape: np.float shape(_,2)
        Points matrix (x,y) that define the rectified shape
    imgs: opencv image (BGR color)
        Actual image
    rois:  np.float shape(_,2)
        Points matrix (x,y) that define the Region Of Interest of the image.
        shape.sshape = rectified shape.shape (n x 2)
        Polygon thay will be rectified

    RETURN:
    img_rect: opencv image
        Rectified img with rectified_shape
    hom_m: np matrix
        Transformation matrix
    """
    retval = []
    def get_range(values):
        return max(values)-min(values)
    height = get_range([h for _,h in rectified_shape])
    width = get_range([w for w,_ in rectified_shape])
    hom_m = cv2.getPerspectiveTransform(src=roi, dst=rectified_shape)
    img_rect = cv2.warpPerspective(img, hom_m, (width, height))
    return(img_rect, hom_m)
