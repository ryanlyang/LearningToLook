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
args = parser.parse_args([])

def validate(model, dataset, cfg, test_scales=None):
    """
    Process validation dataset.
    """

    _preds, _gts, _msc_preds, cams = [], [], [], []

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=False)
    model.cuda()
    model.eval()

    num = 0

    _preds_hist = np.zeros((21, 21))
    _msc_preds_hist = np.zeros((21, 21))
    _cams_hist = np.zeros((21, 21))

    total_images = len(data_loader)

    for idx, data in tqdm(enumerate(data_loader), total=total_images, ncols=100, ascii=" >="):
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

        segs_cat = 0.5 * segs_dino_cat + 0.5*segs_clip_cat
        # Free memory from CLIP and DINO outputs
        del segs_clip_cat, segs_dino_cat, attn_loss

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

                segs_cat = 0.5* segs_dino_cat + 0.5*segs_clip_cat
                # Free memory
                del segs_clip_cat, segs_dino_cat, cam_cat, attn_loss, _inputs

                _segs_cat = F.interpolate(segs_cat, size=(s_h, s_w), mode='bilinear', align_corners=False)
                _segs = (_segs_cat[0,...] + _segs_cat[1,...].flip(-1)) / 2
                segs_list.append(_segs)
                # Free intermediate tensors
                del segs_cat, _segs_cat

        msc_segs = torch.mean(torch.stack(segs_list, dim=0), dim=0).unsqueeze(0)
        del segs_list

        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        seg_preds = torch.argmax(resized_segs, dim=1)

        resized_msc_segs = F.interpolate(msc_segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        msc_seg_preds = torch.argmax(resized_msc_segs, dim=1)

        cams += list(cam.cpu().numpy().astype(np.int16))
        _preds += list(seg_preds.cpu().numpy().astype(np.int16))
        _msc_preds += list(msc_seg_preds.cpu().numpy().astype(np.int16))
        _gts += list(labels.cpu().numpy().astype(np.int16))

        # Save logits for this image
        np.save(args.work_dir+ '/logit/' + name[0] + '.npy', {"segs":segs.detach().cpu().numpy(), "msc_segs":msc_segs.detach().cpu().numpy()})

        # Free memory for this iteration
        del inputs, inputs_cat, labels, segs, msc_segs, cam, resized_segs, resized_msc_segs, seg_preds, msc_seg_preds

        if num % 1000 == 0:
            _preds_hist, seg_score = evaluate.scores(_gts, _preds, _preds_hist)
            _msc_preds_hist, msc_seg_score = evaluate.scores(_gts, _msc_preds, _msc_preds_hist)
            _cams_hist, cam_score = evaluate.scores(_gts, cams, _cams_hist)
            _preds, _gts, _msc_preds, cams = [], [], [], []

    return _gts, _preds, _msc_preds, cams, _preds_hist, _msc_preds_hist, _cams_hist


def crf_proc(config):
    print("crf post-processing...")

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
        # label_name = os.path.join(labels_path, name + ".png")
        # if "test" in args.eval_set:
        #     label = image[:,:,0]
        # else:
        #     label = imageio.imread(label_name)

        label = None

        if image.ndim == 2:                             
            image = np.stack([image, image, image], axis=-1)

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)#[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        # fg_conf = prob[1]                              # confidence for your “object” class
        # pred    = np.where(fg_conf > args.bkg_score, 1, 0)
        pred = np.argmax(prob, axis=0)

        imageio.imsave(os.path.join(args.work_dir, "prediction", name + ".png"), np.squeeze(pred).astype(np.uint8))
        imageio.imsave(os.path.join(args.work_dir, "prediction_cmap", name + ".png"), encode_cmap(np.squeeze(pred)).astype(np.uint8))
        return pred, label

    # n_jobs = int(multiprocessing.cpu_count() * 0.8)
    n_jobs = 1
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])

    # preds, gts = zip(*results)
    # hist = np.zeros((21, 21))
    # hist, score = evaluate.scores(gts, preds, hist, 21)

    # print(score)
    
    return True

def main(cfg, model_path, chunk_size=10):
    """
    Main validation function with chunked processing.
    Processes dataset in isolated chunks - each chunk goes through complete pipeline
    (validation + CRF) before moving to next chunk.

    Args:
        cfg: Configuration object
        model_path: Path to model checkpoint
        chunk_size: Number of images to process in each isolated batch (default: 10)
    """
    import gc

    # Load full dataset to get total count and indices
    full_dataset = voc.VOC12SegDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=args.eval_set,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    total_images = len(full_dataset)
    num_chunks = (total_images + chunk_size - 1) // chunk_size  # Ceiling division

    # Use args.chunk_size if provided, otherwise use the parameter
    effective_chunk_size = args.chunk_size if args.chunk_size is not None else chunk_size

    print(f"\n{'='*70}")
    print(f"CHUNKED PROCESSING MODE")
    print(f"Total images: {total_images}")
    print(f"Chunk size: {effective_chunk_size}")
    print(f"Number of chunks: {num_chunks}")
    print(f"{'='*70}\n")

    # Aggregate results across all chunks
    all_preds_hist = np.zeros((21, 21))
    all_msc_preds_hist = np.zeros((21, 21))
    all_cams_hist = np.zeros((21, 21))

    # Process each chunk independently
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * effective_chunk_size
        end_idx = min(start_idx + effective_chunk_size, total_images)
        chunk_indices = list(range(start_idx, end_idx))

        print(f"\n{'='*70}")
        print(f"Processing Chunk {chunk_idx + 1}/{num_chunks}")
        print(f"Images {start_idx + 1} to {end_idx} (total: {len(chunk_indices)} images)")
        print(f"{'='*70}\n")

        # Create subset dataset for this chunk
        chunk_dataset = Subset(full_dataset, chunk_indices)

        # Load model fresh for each chunk
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

        # Run validation on this chunk
        gts, preds, msc_preds, cams, preds_hist, msc_preds_hist, cams_hist = validate(
            model=model,
            dataset=chunk_dataset,
            cfg=cfg,
            test_scales=[1, 1.25]  #[1, 0.75] [1, 1.5] [0.75, 1.0, 1.25, 1.5]
        )

        # Accumulate histograms
        all_preds_hist += preds_hist
        all_msc_preds_hist += msc_preds_hist
        all_cams_hist += cams_hist

        # Calculate scores for this chunk
        # _, seg_score = evaluate.scores(gts, preds, np.zeros((21, 21)))
        # _, msc_seg_score = evaluate.scores(gts, msc_preds, np.zeros((21, 21)))
        # _, cam_score = evaluate.scores(gts, cams, np.zeros((21, 21)))

        # print(f"\n--- Chunk {chunk_idx + 1} Results ---")
        # print(f"Segs mIoU: {seg_score['Mean IoU']:.4f}")
        # print(f"MSC Segs mIoU: {msc_seg_score['Mean IoU']:.4f}")
        # print(f"CAMs mIoU: {cam_score['Mean IoU']:.4f}")

        # Completely clean up this chunk
        del model, gts, preds, msc_preds, cams, trained_state_dict
        torch.cuda.empty_cache()
        gc.collect()

        print(f"Chunk {chunk_idx + 1} complete. Memory cleared.\n")

    # Final aggregate scores
    print(f"\n{'='*70}")
    print("FINAL AGGREGATE RESULTS")
    print(f"{'='*70}")

    # Note: We can't recalculate exact scores from histograms, but we have them accumulated
    print("All validation chunks processed successfully!")
    print(f"Total images processed: {total_images}")

    # Run CRF post-processing on all saved logits
    print(f"\n{'='*70}")
    print("Running CRF post-processing on all predictions...")
    print(f"{'='*70}\n")
    crf_proc(config=cfg)

    return True


def outer_main(model_path, config_path=None, chunk_size=10):
    """
    Outer wrapper for validation with memory-efficient chunked processing.

    Args:
        model_path: Path to model checkpoint
        config_path: Path to config file (optional)
        chunk_size: Number of images to process before memory cleanup (default: 10)
    """

    if config_path is None:
        config_path = args.config  # default from parser

    cfg = OmegaConf.load(config_path)

    args.work_dir = os.path.join(args.work_dir, args.eval_set)

    os.makedirs(args.work_dir + "/logit", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction", exist_ok=True)
    os.makedirs(args.work_dir + "/prediction_cmap", exist_ok=True)

    main(cfg, model_path, chunk_size=chunk_size)


if __name__ == '__main__':
    outer_main(r"/workspace/LearningToLook/code/WeCLIPPlus/work_dir_voc/checkpoints/2025-10-19-05-29/wetr_iter_2000.pth")