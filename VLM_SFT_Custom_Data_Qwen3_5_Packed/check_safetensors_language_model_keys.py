import sys
from pathlib import Path

from safetensors import safe_open


def main() -> None:
    if len(sys.argv) != 2:
        print(f"Usage: python {Path(__file__).name} <model_dir>")
        raise SystemExit(1)

    model_dir = Path(sys.argv[1])
    safetensors_path = model_dir / "model.safetensors"

    with safe_open(str(safetensors_path), framework="pt") as f:
        for key in f.keys():
            if "language_model" in key:
                print(key)


if __name__ == "__main__":
    main()
