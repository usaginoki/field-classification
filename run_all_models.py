#!/usr/bin/env python3
"""Run classification pipeline for multiple feature extraction models."""

import subprocess
import sys
import yaml
from pathlib import Path


# Model configurations to run
MODELS = [
    {
        "name": "clip-vit-large-patch14",
        "extraction_mode": "clip",
        "clip_model": "openai/clip-vit-large-patch14",
        "output_dir": "output/clip-vit-large-patch14",
    },
    {
        "name": "remoteclip-vit-l-14",
        "extraction_mode": "remote_clip",
        "remote_clip_variant": "ViT-L-14",
        "output_dir": "output/remoteclip-vit-l-14",
    },
    {
        "name": "siglip2-base-patch16-224",
        "extraction_mode": "siglip2",
        "siglip2_model": "google/siglip2-base-patch16-224",
        "output_dir": "output/siglip2-base-patch16-224",
    },
    {
        "name": "qwen3-vision-encoder",
        "extraction_mode": "vision_encoder",
        "output_dir": "output/qwen3-vision-encoder",
    },
]


def load_base_config() -> dict:
    """Load the base configuration from config.yaml."""
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def run_pipeline(model_config: dict, base_config: dict) -> bool:
    """Run the pipeline for a specific model configuration."""
    print(f"\n{'='*60}")
    print(f"Running pipeline: {model_config['name']}")
    print(f"{'='*60}\n")

    # Create a temporary config for this run
    config = base_config.copy()
    config["model"] = base_config["model"].copy()
    config["data"] = base_config["data"].copy()

    # Update model settings
    config["model"]["extraction_mode"] = model_config["extraction_mode"]

    if "clip_model" in model_config:
        config["model"]["clip_model"] = model_config["clip_model"]
    if "remote_clip_variant" in model_config:
        config["model"]["remote_clip_variant"] = model_config["remote_clip_variant"]
    if "siglip2_model" in model_config:
        config["model"]["siglip2_model"] = model_config["siglip2_model"]

    # Set output directory and ensure full dataset
    config["data"]["output_dir"] = model_config["output_dir"]
    config["data"]["use_sample"] = False

    # Write temporary config
    temp_config_path = Path(f"config_temp_{model_config['name']}.yaml")
    with open(temp_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Run the pipeline
    try:
        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--config", str(temp_config_path),
                "--holdout",
            ],
            check=True,
        )
        success = result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"Error running pipeline for {model_config['name']}: {e}")
        success = False
    finally:
        # Clean up temporary config
        if temp_config_path.exists():
            temp_config_path.unlink()

    return success


def main():
    """Run all model pipelines sequentially."""
    print("=" * 60)
    print("Multi-Model Classification Pipeline")
    print("=" * 60)
    print(f"\nModels to run: {len(MODELS)}")
    for i, model in enumerate(MODELS, 1):
        print(f"  {i}. {model['name']} -> {model['output_dir']}")
    print()

    # Load base configuration
    base_config = load_base_config()

    # Track results
    results = []

    # Run each model
    for model_config in MODELS:
        success = run_pipeline(model_config, base_config)
        results.append({
            "name": model_config["name"],
            "output_dir": model_config["output_dir"],
            "success": success,
        })

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for result in results:
        status = "SUCCESS" if result["success"] else "FAILED"
        print(f"  [{status}] {result['name']} -> {result['output_dir']}")

    # Return exit code based on results
    all_success = all(r["success"] for r in results)
    sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
