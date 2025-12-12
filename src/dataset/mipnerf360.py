import random
from pathlib import Path

import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from omegaconf import DictConfig
from src.dataset.resizing import resize_video
from torch.utils.data import Dataset


class MipNerf360Dataset(Dataset):
    def __init__(self, cfg: DictConfig, target_part: str | None = None):
        self.data_dir = Path(cfg.data_dir)
        self.target_part = target_part

        self.num_frames = cfg.num_frames
        self.resize_width = cfg.resize_width
        self.resize_height = cfg.resize_height
        self.multiple_of = cfg.multiple_of

        print(f"target_part: {self.target_part}")
        print(f"num_frames: {self.num_frames}")

        if self.target_part is not None:
            self.data_dir = self.data_dir / f"{self.target_part}_mp4"
            videos = list(self.data_dir.glob("*.mp4"))
        else:
            # Walk through all subfolders recursively and find all the mp4 files
            videos = list(self.data_dir.rglob("*.mp4"))

        print(f"Found {len(videos)} videos in {self.data_dir}")
        self.videos = videos
        # self.videos = self._filter_short_videos(videos)
        # print(f"  Filtered out {len(videos) - len(self.videos)} short videos")
        # print(f"  Keeping {len(self.videos)} videos")

    def _filter_short_videos(self, videos):
        valid = []
        for vp in videos:
            vp = str(vp)
            try:
                with VideoFileClip(vp) as clip:
                    duration = clip.duration or 0.0
                if duration >= self.seg_dur:
                    valid.append(vp)
            except Exception:
                # optionally: log the broken video path here
                pass
        return valid

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index: int) -> dict:
        video_path = self.videos[index]

        with VideoFileClip(video_path) as clip:
            duration = clip.duration
            fps = clip.fps
            seg_dur = self.num_frames / clip.fps
            max_start = max(duration - seg_dur, 0.0)

            start_t = random.uniform(0.0, max_start)
            end_t = start_t + seg_dur

            sub = clip.subclipped(start_t, end_t)

            frames = []
            for frame in sub.iter_frames(fps=fps):
                frames.append(frame)
                if len(frames) >= self.num_frames:
                    break

        video = np.stack(frames)
        video = resize_video(
            video, new_width=self.resize_width, new_height=self.resize_height, multiple_of=self.multiple_of
        )
        video = (video.transpose(0, 3, 1, 2).astype(np.float32) / 255.0) * 2.0 - 1.0

        return {
            "frames": video,
            "video_path": str(video_path),
            "start_time": start_t,
            "end_time": end_t,
            "fps": fps,
        }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    cfg = DictConfig(
        {
            "data_dir": "data/raw/OpenVidHD",
            "target_part": "part_1",
            "num_frames": 25,
        }
    )

    dataset = OpenVidDataset(cfg, target_part=cfg.target_part)
    print(len(dataset))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for batch in dataloader:
        print(batch.keys())
        breakpoint()
