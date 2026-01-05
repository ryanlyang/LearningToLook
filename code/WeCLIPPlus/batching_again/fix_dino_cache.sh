#!/bin/bash
# Script to fix the broken DINOv1 cache on the cluster

echo "Fixing DINOv1 cache issue..."

# Remove the corrupted cache
CACHE_DIR="$HOME/.cache/torch/hub/facebookresearch_dino_main"
if [ -d "$CACHE_DIR" ]; then
    echo "Removing corrupted cache at $CACHE_DIR"
    rm -rf "$CACHE_DIR"
    echo "Cache removed successfully"
else
    echo "Cache directory not found (already clean)"
fi

# Install timm if not present
echo "Checking for timm installation..."
python -c "import timm" 2>/dev/null && {
    echo "timm is already installed"
} || {
    echo "Installing timm..."
    pip install -q timm
    echo "timm installed successfully"
}

echo "Done! You can now run your training script."
