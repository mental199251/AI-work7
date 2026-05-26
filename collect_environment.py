import json
import platform
import sys
from pathlib import Path

import torch
import torchvision


def main() -> None:
    output_path = Path("outputs/metrics/environment.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    information = {
        "operating_system": platform.platform(),
        "python_version": sys.version.replace("\n", " "),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_devices": [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memory_gigabytes": round(torch.cuda.get_device_properties(index).total_memory / (1024**3), 2),
            }
            for index in range(torch.cuda.device_count())
        ],
    }
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(information, output_file, indent=2, ensure_ascii=False)
    print(json.dumps(information, indent=2, ensure_ascii=False))
    print(f"Environment saved to: {output_path}")


if __name__ == "__main__":
    main()
