import os
import re

#  EDIT THESE PATHS 
source_config = '/home/ryan/ComputerScience/LearnToLook/Learning2/LearningToLook/code/WeCLIPPlus/configs/voc_attn_reg.yaml'
output_configs_dir = '/home/ryan/ComputerScience/LearnToLook/Learning2/LearningToLook/code/WeCLIPPlus/configs/NICO_configs'
# 

def main(source_config, output_configs_dir, class_name):
    """
    Duplicate a config file and modify dataset paths for a specific class.

    Args:
        source_config: Path to the source config file (e.g., voc_attn_reg.yaml)
        output_configs_dir: Directory where modified configs will be saved
        class_name: The class name (e.g., 'bear', 'car', 'bicycle')
    """
    # Strip whitespace from class_name
    class_name = class_name.strip()

    # Create output directory if it doesn't exist
    os.makedirs(output_configs_dir, exist_ok=True)

    # Read the source config file
    with open(source_config, 'r') as f:
        content = f.read()

    # Modify root_dir and name_list_dir to include the class name
    # root_dir: '/home/ryreu/guided_cnn/code/LearningToLook/code/WeCLIPPlus/VOCdevkit/VOC2012'
    # becomes: '/home/ryreu/guided_cnn/code/LearningToLook/code/WeCLIPPlus/VOCdevkit/VOC2012/bear'
    content = re.sub(
        r"(root_dir:\s*')([^']*VOC2012)(')",
        rf"\1\2/{class_name}\3",
        content
    )

    # name_list_dir: '/home/ryreu/guided_cnn/code/LearningToLook/code/WeCLIPPlus/VOCdevkit/VOC2012/ImageSets/Main'
    # becomes: '/home/ryreu/guided_cnn/code/LearningToLook/code/WeCLIPPlus/VOCdevkit/VOC2012/bear/ImageSets/Main'
    content = re.sub(
        r"(name_list_dir:\s*')([^']*VOC2012)(/ImageSets/Main)(')",
        rf"\1\2/{class_name}\3\4",
        content
    )

    # Write the modified config to the output directory
    output_file = os.path.join(output_configs_dir, f'voc_attn_reg_{class_name}.yaml')
    with open(output_file, 'w') as f:
        f.write(content)

    print(f"Created config: {output_file}")
    return output_file

if __name__ == "__main__":
    # Example usage: create configs for bear, car, bicycle
    classes = ['bear', 'car', 'bicycle']
    for cls in classes:
        main(source_config, output_configs_dir, cls)
