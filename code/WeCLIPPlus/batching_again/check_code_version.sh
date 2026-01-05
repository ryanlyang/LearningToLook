#!/bin/bash
# Check what version of dinov1_loader.py is on the cluster

echo "Checking dinov1_loader.py on cluster..."
echo ""
echo "Looking for 'timm' references:"
grep -n "import timm" /home/ryreu/guided_cnn/code/SwitchDINO/LearningToLook/code/WeCLIPPlus/pretrained/dinov1_loader.py || echo "NOT FOUND"
echo ""
echo "Looking for debug messages:"
grep -n "Using timm to load DINO" /home/ryreu/guided_cnn/code/SwitchDINO/LearningToLook/code/WeCLIPPlus/pretrained/dinov1_loader.py || echo "NOT FOUND"
echo ""
echo "Line 60-65:"
sed -n '60,65p' /home/ryreu/guided_cnn/code/SwitchDINO/LearningToLook/code/WeCLIPPlus/pretrained/dinov1_loader.py
echo ""
echo "Line 107-112:"
sed -n '107,112p' /home/ryreu/guided_cnn/code/SwitchDINO/LearningToLook/code/WeCLIPPlus/pretrained/dinov1_loader.py
