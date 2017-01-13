import openface
from geometric_helper import *
from image_processor import *

predictor_model = "shape_predictor_68_face_landmarks.dat"
face_aligner = openface.AlignDlib(predictor_model)


def get_roi_of_faces(image):
    roi_of_faces = []
    detected_faces = face_aligner.getAllFaceBoundingBoxes(image)
    for i, face_rect in enumerate(detected_faces):
        landmarks_pos = face_aligner.findLandmarks(image, face_rect)
        center = get_center_between_eyes(landmarks_pos)
        rotation_matrix = get_rotation_matrix(center, landmarks_pos)
        rotation_image = get_rotation_image(image, rotation_matrix)
        rotation_landmarks_pos = get_rotation_points(landmarks_pos, center, rotation_matrix)
        roi = get_roi(rotation_image, rotation_landmarks_pos)
        roi_of_faces.append((roi, face_rect))
    return roi_of_faces


def get_roi(image, landmarks_pos):
    left_eye = get_rectangle(image, landmarks_pos[17:22] + landmarks_pos[36:42])
    right_eye = get_rectangle(image, landmarks_pos[22:27] + landmarks_pos[42:48])
    lips = get_rectangle(image, landmarks_pos[48:60] + landmarks_pos[60:68])

    return left_eye, right_eye, lips


def get_rectangle(image, points, edging=10):
    min_x = min(list(map(lambda x: x[0], points))) - edging
    min_y = min(list(map(lambda x: x[1], points))) - edging
    max_x = max(list(map(lambda x: x[0], points))) + edging
    max_y = max(list(map(lambda x: x[1], points))) + edging

    return image[min_y:max_y, min_x:max_x]


def get_rotation_matrix(center, landmarks_pos):
    angle = get_angle(landmarks_pos[36], landmarks_pos[45])
    return cv2.getRotationMatrix2D(center, angle, 1)


def get_center_between_eyes(landmarks_pos):
    return get_center(landmarks_pos[36], landmarks_pos[45])
