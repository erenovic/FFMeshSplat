import numpy as np
import cv2


def resize_video(video: np.ndarray, new_width: int, new_height: int) -> np.ndarray:
    resized_video = []
    for frame in video:
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        resized_video.append(resized_frame)
    return np.array(resized_video)
