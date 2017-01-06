import openface
import cv2
import numpy as np

predictor_model = "shape_predictor_68_face_landmarks.dat"
face_aligner = openface.AlignDlib(predictor_model)


def get_emotion_templates(image):
    templates = []
    detected_faces = face_aligner.getAllFaceBoundingBoxes(image)
    for i, face_rect in enumerate(detected_faces):
        pose_landmarks = face_aligner.findLandmarks(image, face_rect)
        height, width, channels = image.shape
        emotion_template = draw_emotion_template(pose_landmarks, height, width)
        aligned_template = face_aligner.align(90,
                                              emotion_template,
                                              face_rect,
                                              pose_landmarks,
                                              openface.AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
        templates.append((aligned_template, face_rect))
    return templates


def draw_emotion_template(pose_landmarks, h, w):
    template = np.zeros((h, w), "uint8")

    cv2.polylines(template, np.array(
        [pose_landmarks[17:22]]), False, 255, 2, cv2.LINE_AA)
    cv2.polylines(template, np.array(
        [pose_landmarks[22:27]]), False, 255, 2, cv2.LINE_AA)

    cv2.polylines(template, np.array(
        [pose_landmarks[36:42]]), True, 255, 2, cv2.LINE_AA)
    cv2.polylines(template, np.array(
        [pose_landmarks[42:48]]), True, 255, 2, cv2.LINE_AA)

    cv2.polylines(template, np.array(
        [pose_landmarks[48:60]]), True, 255, 2, cv2.LINE_AA)
    cv2.polylines(template, np.array(
        [pose_landmarks[60:68]]), True, 255, 2, cv2.LINE_AA)
    return template


def parts_of_face(pose_landmarks, image):
    rows, cols = image.shape
    left_eye = pose_landmarks[36:42]
    right_eye = pose_landmarks[42:48]
    lips = pose_landmarks[48:68]

    def bound_lines(lines, x_coef = 1, y_coef = 1):
        min_x = min(list(map(lambda x: x[1], lines)))
        min_y = min(list(map(lambda x: x[0], lines)))
        max_x = max(list(map(lambda x: x[1], lines)))
        max_y = max(list(map(lambda x: x[0], lines)))
    
        edging_x = int((max_x - min_x)*x_coef)
        edging_y = int((max_y - min_y)*y_coef)

        max_x += edging_x
        if max_x > rows: max_x = rows
        max_y = max_y + edging_y
        if max_y > cols: max_y = cols
        min_x = min_x - edging_x
        if min_x < 0: min_x = 0
        min_y = min_y - edging_y
        if min_y < 0: min_y = 0
    
        return [max_x, max_y, min_x, min_y]

    def rotateImage(center, left_point, right_point):
        vector = (right_point[0] - left_point[0],
                  right_point[1] - left_point[1])
        ungle = np.arccos((vector[1]) / (cv2.norm(np.array(vector))))
        ungle = (ungle * 180) / np.pi
        if vector[0] > 0:
            ungle = -ungle + 90
        matrix = cv2.getRotationMatrix2D(center, ungle, 1)
        cv2.getr
        return cv2.warpAffine(image, matrix, (cols, rows))

    max_x, max_y, min_x, min_y = bound_lines(left_eye, 2.6, 0.7)
    center = ((max_y + min_y) / 2, (max_x + min_x) / 2)
    left_eye_image = rotateImage(center, left_eye[0], left_eye[3])[
        min_x: max_x, min_y: max_y]

    max_x, max_y, min_x, min_y = bound_lines(right_eye, 2.6, 0.7)
    center = ((max_y + min_y) / 2, (max_x + min_x) / 2)
    right_eye_image = rotateImage(center, right_eye[0], right_eye[3])[
        min_x: max_x, min_y: max_y]

    max_x, max_y, min_x, min_y = bound_lines(lips, 0.5, 0.5)
    center = ((max_y + min_y) / 2, (max_x + min_x) / 2)
    lips_image = rotateImage(center, lips[0], lips[6])[
        min_x: max_x, min_y: max_y]

    return [left_eye_image, right_eye_image, lips_image]



