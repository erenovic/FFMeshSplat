from pathlib import Path

import accelerate
import hydra
import torch
from accelerate.utils import tqdm
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from src.dataset.openvid import OpenVidDataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image


def visualize_features_pca(feats: torch.Tensor, n_components: int = 3) -> torch.Tensor:
    """
    Reduce feature channels using PCA for visualization.

    Args:
        feats: Feature tensor of shape (N, C, H, W) - should be on CPU
        n_components: Number of PCA components (3 for RGB, 1 for grayscale)

    Returns:
        Tensor of shape (N, n_components, H, W) normalized to [0, 1]
    """
    n, c, h, w = feats.shape

    # Reshape to (N*H*W, C) for PCA
    feats_flat = feats.permute(0, 2, 3, 1).reshape(-1, c).numpy()

    # Apply PCA
    pca = PCA(n_components=n_components)
    feats_pca = pca.fit_transform(feats_flat)

    # Reshape back to (N, H, W, n_components) -> (N, n_components, H, W)
    feats_pca = feats_pca.reshape(n, h, w, n_components)
    feats_pca = torch.from_numpy(feats_pca).permute(0, 3, 1, 2)

    # Normalize to [0, 1] for visualization
    feats_pca = (feats_pca - feats_pca.min()) / (feats_pca.max() - feats_pca.min() + 1e-8)

    return feats_pca


@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def main(cfg: DictConfig):
    output_dir = Path("feat_vis")
    output_dir.mkdir(parents=True, exist_ok=True)
    accelerator = accelerate.Accelerator()

    model = torch.hub.load(
        "tum-vision/flowfeat",
        "flowfeat",
        name="dinov2_vits14_yt",
        pretrained=True,
    )

    model = accelerator.prepare(model)
    model.eval()

    dataset = OpenVidDataset(cfg.dataset)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    dataloader = accelerator.prepare(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        # 'frames', 'video_path', 'start_time', 'end_time', 'fps'
        x = batch["frames"]

        feats = []
        frames = []

        pbar = tqdm(range(x.shape[1]), desc="Processing frames")
        for frame_idx in pbar:
            with torch.no_grad():
                y_enc, y_dec = model(x[:, frame_idx])
                feats.append(y_dec[0].cpu())
                frames.append(x[0, frame_idx].cpu())

            pbar.set_postfix(frame_idx=frame_idx, shape=y_dec.shape)

        # Stack on CPU (PCA runs on CPU anyway)
        feats = torch.stack(feats)
        frames = torch.stack(frames)

        # Reduce 128 channels to 3 (RGB) using PCA
        feats_vis = visualize_features_pca(feats, n_components=3)

        crop_height = (frames.shape[-2] - cfg.dataset.resize_height) // 2
        crop_width = (frames.shape[-1] - cfg.dataset.resize_width) // 2
        frames = frames[:, :, crop_height:-crop_height, crop_width:-crop_width]
        feats_vis = feats_vis[:, :, crop_height:-crop_height, crop_width:-crop_width]

        combined = torch.stack([frames, feats_vis], dim=1)
        combined = combined.view(-1, 3, cfg.dataset.resize_height, cfg.dataset.resize_width)

        grid = make_grid(combined, nrow=2, normalize=True, padding=2)
        save_image(grid, output_dir / f"feature_vis_{batch_idx:05d}.png")
        print(f"Saved visualization to {output_dir / f'feature_vis_{batch_idx:05d}.png'}")

        torch.cuda.empty_cache()

        breakpoint()


if __name__ == "__main__":
    main()
