from __future__ import annotations

import torch


def main() -> None:
    print(f"torch_version={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device={torch.cuda.get_device_name(0)}")
        print(f"cuda_device_count={torch.cuda.device_count()}")
    else:
        print("cuda_device=no cuda")


if __name__ == "__main__":
    main()
