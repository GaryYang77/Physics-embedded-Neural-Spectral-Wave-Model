import json
from pathlib import Path
from typing import Dict, Optional, Union, List

import torch


def load_input_norm(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Load input normalization stats from JSON.
    
    Supports two schema formats:
    - Legacy (scale-only): {"scale": [...]}
    - New (mode-aware): {"mode": "minmax"|"scale", "min": [...], "scale": [...], "features": [...]}
    
    Returns dict with keys: mode, min (optional), scale, features (optional).
    """
    with open(path, "r") as f:
        data = json.load(f)
    
    # Determine mode: default to "scale" for backward compatibility if not specified
    mode = data.get("mode", "scale")
    
    if "scale" not in data:
        raise ValueError("input norm json missing 'scale' field")
    
    scale = torch.tensor(data["scale"], dtype=torch.float32, device=device)
    result = {"mode": mode, "scale": scale}
    
    # Load min if present (required for minmax mode)
    if "min" in data:
        result["min"] = torch.tensor(data["min"], dtype=torch.float32, device=device)
    elif mode == "minmax":
        raise ValueError("input norm json with mode='minmax' missing 'min' field")
    
    # Load optional feature names for UNet normalization
    if "features" in data:
        result["features"] = data["features"]
    
    return result


def save_input_norm(
    path: str, 
    scale: torch.Tensor, 
    min_vals: Optional[torch.Tensor] = None,
    mode: str = "minmax",
    features: Optional[List[str]] = None,
) -> None:
    """
    Persist input normalization stats to JSON, creating parent directories if needed.
    
    Args:
        path: Output JSON path.
        scale: Per-channel scale tensor (max - min for minmax, or raw scale for scale mode).
        min_vals: Per-channel min tensor (required for minmax mode).
        mode: Normalization mode, one of "minmax" or "scale".
        features: Optional list of feature names (for UNet channel-wise norm).
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "mode": mode,
        "scale": scale.cpu().tolist(),
    }
    
    if min_vals is not None:
        payload["min"] = min_vals.cpu().tolist()
    elif mode == "minmax":
        raise ValueError("save_input_norm: mode='minmax' requires min_vals")
    
    if features is not None:
        payload["features"] = features
    
    with open(target, "w") as f:
        json.dump(payload, f, indent=2)
