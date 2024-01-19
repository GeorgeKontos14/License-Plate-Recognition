import numpy as np
import cv2

def find_rotation(image: np.ndarray) -> float:
    """ Find rotation angle of license plate. This function takes an image 
        after Canny edge detection and identifies the angle of rotation of 
        a license plate within the image.

        Parameters:
        image (np.ndarray): Input image after Canny edge detection.

        Returns:
        float: Angle of rotation of the license plate.
    """
    rotation: float = 0.0
    lines = cv2.HoughLinesP(image, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

    angles: list = [np.arctan2(line[0][3] - line[0][1], line[0][2] - line[0][0]) for line in lines]

    if angles:
        rotation = -np.degrees(np.median(angles))


    return rotation

def rotate(image: np.ndarray, theta: float) -> np.ndarray:
    """ Rotate image to show axis-aligned license plate. This function 
        takes an input image and a rotation angle in degrees (theta). 
        The angle is used to rotate the image, making the license 
        plate axis-aligned.

        Parameters:
        image (np.ndarray): Input image.
        theta (float): Rotation angle in degrees.

        Returns:
        np.ndarray: Rotated image with axis-aligned license plate.
    """
    height, width= image.shape[:2]

    # Convert angle to radians
    theta_rad: float = np.radians(theta)

    # Calculate rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -theta, 1)  # Adjusted here

    # Apply rotation to the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    return rotated_image

def rotation_pipeline(image: np.ndarray) -> np.ndarray:
    """ Rotation pipeline for license plate alignment.

        Parameters:
        - image (np.ndarray): Input image containing a license plate.

        Returns:
        - np.ndarray: Resulting image with an axis-aligned license plate.
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    rotation_angle: float = find_rotation(edges)
    
    return rotate(image, rotation_angle)