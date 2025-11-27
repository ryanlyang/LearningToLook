import argparse
import os
import sys
sys.path.append(".")
from utils.dcrf import DenseCRF
from utils.imutils import encode_cmap
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import multiprocessing
from tqdm import tqdm
import joblib
from datasets import voc
from utils import evaluate
from WeCLIP_Plus.model_attn_aff_voc import WeCLIP_Plus
import imageio.v2 as imageio
from torch.utils.data import Subset
from clip.clip_text import new_class_names
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='configs/voc_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--work_dir", default="results", type=str, help="work_dir")
parser.add_argument("--bkg_score", default=0.45, type=float, help="bkg_score")
parser.add_argument("--eval_set", default="val", type=str, help="eval_set") #val
parser.add_argument("--model_path", default="/data1/zbf_data/FinalCodeGithub/WeCLIP_final_version/WeCLIP+/scripts/work_dir_voc/checkpoints/2025-03-11-10-47/wetr_iter_30000.pth", type=str, help="model_path")
parser.add_argument(
    "--chunk_size",
    default=None,
    type=int,
    help="If set, run inference in chunks of this many images"
)
parser.add_argument(
    "--image_set_dir",
    default="/home/ryan/ComputerScience/LearnToLook/Learning2/LearningToLook/code/WeCLIPPlus/VOCdevkit/VOC2012/ImageSets/2025-11-19_15-38-26",
    type=str,
    help="Path to the directory containing image label files"
)
args = parser.parse_args([])
my_image_set_dir = r"/home/ryreu/guided_cnn/code/LearningToLook/code/WeCLIPPlus/VOCdevkit/VOC2012/ImageSets/Main"

def load_image_class_mapping(image_set_dir, eval_set='val'):
    """
    Load mapping of image names to their class label from the image set directory.

    The image set directory contains files like 'airplane_val.txt', 'cat_val.txt', etc.
    Each file contains image names that belong to that class (marked with 1).

    Returns:
        dict: mapping of image_name -> class_name
    """
    image_to_class = {}

    if not os.path.exists(image_set_dir):
        print(f"Warning: Image set directory not found: {image_set_dir}")
        return image_to_class

    # Iterate through all class files
    for filename in os.listdir(image_set_dir):
        if filename.endswith(f'_{eval_set}.txt'):
            # Extract class name from filename (e.g., 'airplane_val.txt' -> 'airplane')
            class_name = filename.replace(f'_{eval_set}.txt', '')

            filepath = os.path.join(image_set_dir, filename)
            with open(filepath, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        image_name = parts[0]
                        label = int(parts[1])
                        # If label is 1, this image belongs to this class
                        if label == 1:
                            image_to_class[image_name] = class_name

    return image_to_class

def validate(model, dataset, cfg, test_scales=None, images_path=None):

    # Load image to class mapping from ImageSets directory
    image_to_class = load_image_class_mapping(my_image_set_dir, args.eval_set)

    _preds, _gts, _msc_preds, cams = [], [], [], []

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)
    model.cuda()
    model.eval()

    num = 0

    _preds_hist = np.zeros((21, 21))
    _msc_preds_hist = np.zeros((21, 21))
    _cams_hist = np.zeros((21, 21))

    # Initialize CRF post-processor
    post_processor = DenseCRF(
        iter_max=10,
        pos_xy_std=3,
        pos_w=3,
        bi_xy_std=64,
        bi_rgb_std=5,
        bi_w=4,
    )

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
        num+=1

        name, inputs, labels, cls_labels = data
        names = name+name

        inputs = inputs.cuda()
        labels = labels.cuda()

        _, _, h, w = inputs.shape
        ratio = cfg.clip_init.resize_long / max(h,w)
        _h, _w = int(h*ratio), int(w*ratio)
        inputs = F.interpolate(inputs, size=(_h, _w), mode='bilinear', align_corners=False)

        segs_list = []
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        segs_clip_cat, segs_dino_cat, cam, attn_loss = model(inputs_cat, names, mode = 'val')
        torch.cuda.empty_cache()

        segs_cat = 0.5 * segs_dino_cat + 0.5*segs_clip_cat

        cam = cam[0].unsqueeze(0)
        segs = segs_cat[0].unsqueeze(0)

        _segs = (segs_cat[0,...] + segs_cat[1,...].flip(-1)) / 2
        segs_list.append(_segs)

        _, _, s_h, s_w = segs_cat.shape

        for s in test_scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, scale_factor=s, mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                segs_clip_cat, segs_dino_cat, cam_cat, attn_loss = model(inputs_cat, names, mode='val')
                torch.cuda.empty_cache()

                segs_cat = 0.5* segs_dino_cat + 0.5*segs_clip_cat

                _segs_cat = F.interpolate(segs_cat, size=(s_h, s_w), mode='bilinear', align_corners=False)
                _segs = (_segs_cat[0,...] + _segs_cat[1,...].flip(-1)) / 2
                segs_list.append(_segs)

        msc_segs = torch.mean(torch.stack(segs_list, dim=0), dim=0).unsqueeze(0)

        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        seg_preds = torch.argmax(resized_segs, dim=1)
        print('seg_shape', seg_preds.shape, 'labels', labels.shape, 'cam', cam.shape)

        resized_msc_segs = F.interpolate(msc_segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        msc_seg_preds = torch.argmax(resized_msc_segs, dim=1)

        cams += list(cam.cpu().numpy().astype(np.int16))
        _preds += list(seg_preds.cpu().numpy().astype(np.int16))
        _msc_preds += list(msc_seg_preds.cpu().numpy().astype(np.int16))
        _gts += list(labels.cpu().numpy().astype(np.int16))


        if num % 1000 == 0:
            _preds_hist, seg_score = evaluate.scores(_gts, _preds, _preds_hist)
            _msc_preds_hist, msc_seg_score = evaluate.scores(_gts, _msc_preds, _msc_preds_hist)
            _cams_hist, cam_score = evaluate.scores(_gts, cams, _cams_hist)
            _preds, _gts, _msc_preds, cams = [], [], [], []

        # Save logits
        np.save(args.work_dir+ '/logit/' + name[0] + '.npy', {"segs":segs.detach().cpu().numpy(), "msc_segs":msc_segs.detach().cpu().numpy()})

        # Generate and save predictions + cmaps on the fly
        logit = msc_segs.detach().cpu().numpy()
        logit = torch.FloatTensor(logit)

        # Load original image for CRF
        if images_path:
            image_name = os.path.join(images_path, name[0] + ".jpg")
            if os.path.exists(image_name):
                image = imageio.imread(image_name).astype(np.float32)

                if image.ndim == 2:
                    image = np.stack([image, image, image], axis=-1)

                H, W, _ = image.shape
                logit_resized = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
                prob = F.softmax(logit_resized, dim=1)[0].numpy()

                image = image.astype(np.uint8)
                prob = post_processor(image, prob)
                pred = np.argmax(prob, axis=0)
            else:
                # Fallback if image not found
                prob = F.softmax(logit, dim=1)[0].numpy()
                pred = np.argmax(prob, axis=0)
        else:
            # No CRF, just softmax + argmax
            prob = F.softmax(logit, dim=1)[0].numpy()
            pred = np.argmax(prob, axis=0)

        # Filter predictions to only show the class that exists in the ground truth
        # All other pixels are set to black (0)
        image_name = name[0]
        filtered_pred = np.zeros_like(pred)

        if image_name in image_to_class:
            target_class = image_to_class[image_name]
            print(f"Image {image_name} belongs to class: {target_class}")

            # Create a mapping of class names to indices based on VOC dataset
            # This is a standard mapping for VOC12
            voc_class_names = new_class_names
            # Try to find the class index
            class_idx = None
            for idx, class_name in enumerate(voc_class_names):
                if class_name == target_class or class_name.replace('_', ' ').lower() == target_class.replace('_', ' ').lower():
                    class_idx = idx
                    break

            if class_idx is not None:
                # Keep pixels where target class value is higher than background class value
                # prob has shape (num_classes, H, W)
                target_class_prob = prob[class_idx]
                background_prob = prob[0]
                mask = target_class_prob > background_prob
                filtered_pred[mask] = class_idx
                print(f"  Using class index: {class_idx}")
            else:
                print(f"  Warning: Could not find class index for {target_class}")
        else:
            print(f"Image {image_name} not found in class mapping")

        # Save prediction and cmap
        imageio.imsave(os.path.join(args.work_dir, "prediction", name[0] + ".png"), np.squeeze(filtered_pred).astype(np.uint8))
        imageio.imsave(os.path.join(args.work_dir, "prediction_cmap", name[0] + ".png"), encode_cmap(np.squeeze(filtered_pred)).astype(np.uint8))

    return _gts, _preds, _msc_preds, cams, _preds_hist, _msc_preds_hist, _cams_hist


def crf_proc(config):
    print("crf post-processing...")

    # Load image to class mapping from ImageSets directory
    image_to_class = load_image_class_mapping(my_image_set_dir, args.eval_set)

    txt_name = os.path.join(config.dataset.name_list_dir, args.eval_set) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    images_path = os.path.join(config.dataset.root_dir, 'JPEGImages',)
    labels_path = os.path.join(config.dataset.root_dir, 'SegmentationClassAug')

    post_processor = DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=3,   # 3
        pos_w=3,        # 3
        bi_xy_std=64,  # 64
        bi_rgb_std=5,   # 5
        bi_w=4,         # 4
    )

    def _job(i):

        name = name_list[i]
        logit_name = os.path.join(args.work_dir, "logit", name + ".npy")

        logit = np.load(logit_name, allow_pickle=True).item()
        logit = logit['msc_segs']

        image_name = os.path.join(images_path, name + ".jpg")
        image = imageio.imread(image_name).astype(np.float32)

        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)#[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        # fg_conf = prob[1]                              # confidence for your "object" class
        # pred    = np.where(fg_conf > args.bkg_score, 1, 0)
        pred = np.argmax(prob, axis=0)

        return name, pred

    # n_jobs = int(multiprocessing.cpu_count() * 0.8)
    n_jobs = 1
    batch_size = 100

    # Process and save in batches of 100
    for batch_start in range(0, len(name_list), batch_size):
        batch_end = min(batch_start + batch_size, len(name_list))
        batch_indices = list(range(batch_start, batch_end))

        print(f"\nProcessing batch: images {batch_start} to {batch_end-1}")

        results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")(
            [joblib.delayed(_job)(i) for i in batch_indices]
        )

        # Save all results from this batch
        print(f"Saving batch: images {batch_start} to {batch_end-1}")

        # VOC class names mapping
        voc_class_names = [
            'background', 'airplane', 'bear', 'bicycle', 'bird', 'butterfly', 'cactus', 'car',
            'chair', 'clock', 'cloud', 'crab', 'crocodile', 'cup', 'deer', 'dog', 'dolphin',
            'elephant', 'fish', 'flower', 'fox', 'frog', 'giraffe', 'glass', 'glove', 'gorilla',
            'guitar', 'hat', 'hippopotamus', 'hot_air_balloon', 'jellyfish', 'kangaroo',
            'key', 'knife', 'ladder', 'lamp', 'laptop', 'leopard', 'lion', 'lizard', 'lobster',
            'motorcycle', 'mushroom', 'musical_instrument', 'orange', 'otter', 'owl', 'paintbrush',
            'palm_tree', 'panda', 'penguin', 'piano', 'pineapple', 'pink_flamingo', 'pizza',
            'plastic_bag', 'plate', 'player_piano', 'polar_bear', 'pool', 'porcupine'
        ]

        for idx, (name, pred) in enumerate(results):
            # Filter predictions to only show the class that exists in the ground truth
            # All other pixels are set to black (0)
            filtered_pred = np.zeros_like(pred)

            if name in image_to_class:
                target_class = image_to_class[name]
                print(f"Image {name} belongs to class: {target_class}")

                # Try to find the class index
                class_idx = None
                for idx_num, class_name in enumerate(voc_class_names):
                    if class_name == target_class or class_name.replace('_', ' ').lower() == target_class.replace('_', ' ').lower():
                        class_idx = idx_num
                        break

                if class_idx is not None:
                    # Only keep predictions that match the target class
                    filtered_pred[pred == class_idx] = class_idx
                    print(f"  Using class index: {class_idx}")
                else:
                    print(f"  Warning: Could not find class index for {target_class}")
            else:
                print(f"Image {name} not found in class mapping")

            imageio.imsave(os.path.join(args.work_dir, "prediction", name + ".png"), np.squeeze(filtered_pred).astype(np.uint8))
            imageio.imsave(os.path.join(args.work_dir, "prediction_cmap", name + ".png"), encode_cmap(np.squeeze(filtered_pred)).astype(np.uint8))

        print(f"Batch saved successfully!")

    # preds, gts = zip(*results)
    # hist = np.zeros((21, 21))
    # hist, score = evaluate.scores(gts, preds, hist, 21)

    # print(score)

    return True

def main(cfg, model_path):

    val_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=args.eval_set,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    model = WeCLIP_Plus(num_classes=cfg.dataset.num_classes,
                     clip_model=cfg.clip_init.clip_pretrain_path,
                     dino_model=cfg.dino_init.dino_model,
                     dino_fts_dim=cfg.dino_init.dino_fts_fuse_dim,
                     decoder_layers=cfg.dino_init.decoder_layer,
                     embedding_dim=cfg.clip_init.embedding_dim,
                     in_channels=cfg.clip_init.in_channels,
                     dataset_root_path=cfg.dataset.root_dir,
                     clip_flag=cfg.clip_init.clip_flag,
                     device='cuda')

    trained_state_dict = torch.load(model_path, map_location="cpu")

    model.load_state_dict(state_dict=trained_state_dict, strict=False)
    model.eval()

    # Get images path for CRF processing
    images_path = os.path.join(cfg.dataset.root_dir, 'JPEGImages')

    gts, preds, msc_preds, cams, preds_hist, msc_preds_hist, cams_hist = validate(model=model, dataset=val_dataset, cfg=cfg, test_scales=[0.5, 1], images_path=images_path) #[1, 0.75] [1, 1.5]
    #[0.75, 1.0, 1.25, 1.5]
    torch.cuda.empty_cache()

    preds_hist, seg_score = evaluate.scores(gts, preds, preds_hist)
    msc_preds_hist, msc_seg_score = evaluate.scores(gts, msc_preds, msc_preds_hist)
    cams_hist, cam_score = evaluate.scores(gts, cams, cams_hist)

    print("cams score:")
    print(cam_score)
    print("segs score:")
    print(seg_score)
    print("msc segs score:")
    print(msc_seg_score)

    return True


def outer_main(model_path, config_path=None):

    if config_path is None:
        config_path = args.config  # default from parser

    cfg = OmegaConf.load(config_path)


    args.work_dir = os.path.join(args.work_dir, args.eval_set)

    os.makedirs(args.work_dir + "/logit", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction_cmap", exist_ok=True)

    main(cfg, model_path)


if __name__ == '__main__':
    outer_main()