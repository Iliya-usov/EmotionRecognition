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

    cv2.polylines(template, np.array([pose_landmarks[17:22]]), False, 255, 2, cv2.LINE_AA)
    cv2.polylines(template, np.array([pose_landmarks[22:27]]), False, 255, 2, cv2.LINE_AA)

    cv2.polylines(template, np.array([pose_landmarks[36:42]]), True, 255, 2, cv2.LINE_AA)
    cv2.polylines(template, np.array([pose_landmarks[42:48]]), True, 255, 2, cv2.LINE_AA)

    cv2.polylines(template, np.array([pose_landmarks[48:60]]), True, 255, 2, cv2.LINE_AA)
    cv2.polylines(template, np.array([pose_landmarks[60:68]]), True, 255, 2, cv2.LINE_AA)
    return template
