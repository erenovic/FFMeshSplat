import cv2
import numpy as np


def resize_video(video: np.ndarray, new_width: int, new_height: int, multiple_of: int) -> np.ndarray:
    resized_video = []
    for frame in video:
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        if multiple_of is not None:
            pad_width = (multiple_of - (new_width % multiple_of)) // 2
            pad_height = (multiple_of - (new_height % multiple_of)) // 2
            resized_frame = pad_video(resized_frame, pad_width, pad_height)

        resized_video.append(resized_frame)
    return np.array(resized_video)


def pad_video(frame: np.ndarray, width_padding: int, height_padding: int) -> np.ndarray:
    return cv2.copyMakeBorder(
        frame,
        top=height_padding,
        bottom=height_padding,
        left=width_padding,
        right=width_padding,
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0],
    )
