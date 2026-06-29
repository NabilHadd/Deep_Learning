"""Automatic face detection, rotation alignment and cropping using OpenCV Haar cascades.

Used by E6 to preprocess raw UTKFace images with the same pipeline that the
Streamlit application would apply to a photo uploaded by a user.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class FaceAligner:
    """Detect the largest frontal face, rotate to a horizontal eye line and crop.

    The alignment step estimates the rotation angle from the eye positions so
    that the processed crop matches what an aligned dataset provides.  When no
    face is found the original image is returned unchanged and ``face_found``
    is ``False``.

    Pipeline:
        1. Convert to greyscale.
        2. Run Haar frontal-face cascade → select largest bounding box.
        3. Run Haar eye cascade inside the face crop.
        4. Compute the angle between the two eye centres.
        5. Rotate the face crop around the midpoint of the eyes.
        6. Return the aligned crop as a PIL image.
    """

    def __init__(self) -> None:
        cascades = Path(cv2.data.haarcascades)
        self._face_cascade = cv2.CascadeClassifier(
            str(cascades / "haarcascade_frontalface_default.xml")
        )
        self._eye_cascade = cv2.CascadeClassifier(
            str(cascades / "haarcascade_eye.xml")
        )
        if self._face_cascade.empty():
            raise RuntimeError(
                "No se pudo cargar haarcascade_frontalface_default.xml. "
                "Verifica la instalación de opencv-python."
            )

    def align_and_crop(self, image: Image.Image) -> tuple[Image.Image, bool]:
        """Return ``(processed_image, face_found)``.

        When a face is detected the returned image is the rotated and cropped
        face region. When no face is found the original image is returned as-is
        and ``face_found`` is ``False``.
        """
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        if len(faces) == 0:
            return image, False

        x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
        face_rgb = rgb[y : y + h, x : x + w]
        face_gray = gray[y : y + h, x : x + w]

        aligned = self._rotate_by_eyes(face_rgb, face_gray)
        return Image.fromarray(aligned), True

    def _rotate_by_eyes(
        self, face_rgb: np.ndarray, face_gray: np.ndarray
    ) -> np.ndarray:
        """Rotate the face crop so the line connecting both eyes is horizontal."""
        eyes = self._eye_cascade.detectMultiScale(
            face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15)
        )
        if len(eyes) < 2:
            return face_rgb

        # Keep only the two largest eye detections sorted left→right.
        eyes = sorted(eyes, key=lambda e: int(e[2]) * int(e[3]), reverse=True)[:2]
        eyes = sorted(eyes, key=lambda e: e[0])
        (x1, y1, w1, h1), (x2, y2, w2, h2) = eyes

        left_cx, left_cy = x1 + w1 // 2, y1 + h1 // 2
        right_cx, right_cy = x2 + w2 // 2, y2 + h2 // 2

        angle = np.degrees(np.arctan2(right_cy - left_cy, right_cx - left_cx))
        pivot_x = (left_cx + right_cx) // 2
        pivot_y = (left_cy + right_cy) // 2

        h, w = face_rgb.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((float(pivot_x), float(pivot_y)), angle, 1.0)
        return cv2.warpAffine(face_rgb, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
