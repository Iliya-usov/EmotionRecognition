import cv2
import numpy as np
from math import sqrt, pow


def get_center(left_point, right_point):
    return (left_point[0] + (right_point[0] - left_point[0]) / 2,
            left_point[1] + (right_point[1] - left_point[1]) / 2)


def get_angle(left_point, right_point):
    vector = (right_point[0] - left_point[0], right_point[1] - left_point[1])
    angle = np.arccos((vector[1]) / (cv2.norm(np.array(vector))))
    angle = (angle * 180) / np.pi
    return angle if vector[0] <= 0 else 90 - angle


def get_rotation_points(points, center, rotation_matrix):
    new_points = list(map(
        lambda x:
        (x[0] - center[0], x[1] - center[1], 0),
        points))

    matrix = np.matrix(rotation_matrix)

    rotation_points = list(map(
        lambda x:
        ((matrix * (np.matrix(x)).T).T + np.matrix(center)).tolist(),
        new_points))
    return list(map(lambda x: (int(x[0][0]), int(x[0][1])), rotation_points))


def get_rotation_matrix(center, angle, scale):
    return cv2.getRotationMatrix2D(center, angle, scale)


def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))


def distance_between_numbers(x1, x2):
    return abs(x1 - x2)


def eccentricity(a, b):
    return sqrt(abs(pow(a, 2) - pow(b, 2))) / a
