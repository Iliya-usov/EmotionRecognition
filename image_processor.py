import cv2


def get_gray_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def histogram_equalization(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = cv2.split(image)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, image)
    return cv2.cvtColor(image, cv2.COLOR_YCrCb2BGR)


def binarization(image, thresh):
    dummy, image = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    return image
