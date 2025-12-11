
import accelerate
import hydra
import torch
from omegaconf import DictConfig


@hydra.main(config_path="../config", config_name="main.yaml", version_base=None)
def main(cfg: DictConfig):
    # Load a pretrained FlowFeat model from PyTorch Hub
    accelerator = accelerate.Accelerator()

    model = torch.hub.load(
        "tum-vision/flowfeat",
        "flowfeat",
        name="dinov2_vits14_yt",   # model variant
        pretrained=True
    )

    model = accelerator.prepare(model)
    model.eval()

    x = torch.randn(1, 3, 224, 224).to(accelerator.device)  # example input
    with torch.no_grad():
        y_enc, y_dec = model(x)

    print(y_enc.shape) # encoder features, e.g. (1,384,16,16)
    print(y_dec.shape) # decoder features, e.g. (1,128,224,224)

    breakpoint()



if __name__ == "__main__":
    main()
