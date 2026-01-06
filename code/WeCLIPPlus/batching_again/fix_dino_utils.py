#!/usr/bin/env python3
"""
Fix the broken torch.hub DINO cache by patching the utils.py file
"""
import os
import sys

# Find the DINO cache directory
torch_hub_dir = os.path.expanduser("~/.cache/torch/hub")
dino_dir = os.path.join(torch_hub_dir, "facebookresearch_dino_main")

if not os.path.exists(dino_dir):
    print(f"DINO cache not found at {dino_dir}")
    sys.exit(1)

utils_file = os.path.join(dino_dir, "utils.py")

# Check if utils.py exists
if not os.path.exists(utils_file):
    print(f"utils.py not found at {utils_file}")
    sys.exit(1)

# Read the file
with open(utils_file, 'r') as f:
    content = f.read()

# Check if trunc_normal_ exists
if 'def trunc_normal_' in content:
    print("trunc_normal_ already exists in utils.py - no fix needed")
    sys.exit(0)

# Add the trunc_normal_ function at the top of the file after imports
trunc_normal_code = '''
# Added to fix import error
import math
import warnings

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Fills the input Tensor with values drawn from a truncated normal distribution."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor
'''

# Find the first import statement and add our code after all imports
lines = content.split('\n')
import_end_idx = 0
for i, line in enumerate(lines):
    if line.strip() and not line.strip().startswith('import') and not line.strip().startswith('from') and not line.strip().startswith('#'):
        import_end_idx = i
        break

# Insert the function
lines.insert(import_end_idx, trunc_normal_code)
new_content = '\n'.join(lines)

# Write back
with open(utils_file, 'w') as f:
    f.write(new_content)

print(f"Successfully patched {utils_file}")
print("Added trunc_normal_ function to utils.py")
