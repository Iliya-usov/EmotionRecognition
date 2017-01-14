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


def get_rotation_image(image, rotation_matrix):
    height, width = image.shape[:2]
    return cv2.warpAffine(image, rotation_matrix, (width, height))


def get_rectangle(image, points, edging=10):
    height, width = image.shape[:2]
    min_x = max(min(list(map(lambda x: x[0], points))) - edging, 0)
    min_y = max(min(list(map(lambda x: x[1], points))) - edging, 0)
    max_x = min(max(list(map(lambda x: x[0], points))) + edging, width)
    max_y = min(max(list(map(lambda x: x[1], points))) + edging, height)

    return image[min_y:max_y, min_x:max_x]