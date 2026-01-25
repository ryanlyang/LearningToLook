import argparse
import datetime
import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import sys
sys.path.append(".")
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import voc
from utils.losses import get_aff_loss
from utils import evaluate
from utils.AverageMeter import AverageMeter
from utils.camutils import cams_to_affinity_label
from utils.optimizer import PolyWarmupAdamW
from WeCLIP_Plus.model_attn_aff_voc import WeCLIP_Plus
from WeCLIP_Plus.dice_loss import DiceLoss

from functools import partial
import math


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    default='/data1/zbf_data/Project2024/FCLIP_DINO/configs/voc_attn_reg.yaml',
                    type=str,
                    help="config")
parser.add_argument("--seg_detach", action="store_true", help="detach seg")
parser.add_argument("--work_dir", default=None, type=str, help="work_dir")
parser.add_argument("--radius", default=8, type=int, help="radius")

args = parser.parse_args([])




def smart_collate(batch, clip_flag=16, max_h=512, max_w=512):
    names, images_np, cls_labels, img_boxes = zip(*batch)

    # figure out how big to pad up to (multiples of clip_flag, capped by max_h/max_w)
    Hs = [min(im.shape[1], max_h) for im in images_np]
    Ws = [min(im.shape[2], max_w) for im in images_np]
    H_pad = math.ceil(max(Hs) / clip_flag) * clip_flag
    W_pad = math.ceil(max(Ws) / clip_flag) * clip_flag

    batch_size = len(images_np)
    # allocate a CPU tensor of zeros
    padded = torch.zeros((batch_size, 3, H_pad, W_pad), dtype=torch.float32)

    for i, im_np in enumerate(images_np):
        im = torch.as_tensor(im_np, dtype=torch.float32)  # still CPU
        C, H, W = im.shape

        # if this sample still exceeds the padded size, downsample it
        if H > H_pad or W > W_pad:
            im = F.interpolate(
                im.unsqueeze(0),
                size=(min(H, H_pad), min(W, W_pad)),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
            C, H, W = im.shape

        padded[i, :, :H, :W] = im

    # stack labels on CPU
    cls_tensor = torch.stack(
        [torch.as_tensor(lbl, dtype=torch.float32) for lbl in cls_labels],
        dim=0
    )

    return names, padded, cls_tensor, img_boxes



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_logger(filename='test.log'):
    ## setup logger
    logFormatter = logging.Formatter('%(asctime)s - %(filename)s - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fHandler = logging.FileHandler(filename, mode='w')
    fHandler.setFormatter(logFormatter)
    logger.addHandler(fHandler)

    cHandler = logging.StreamHandler()
    cHandler.setFormatter(logFormatter)
    logger.addHandler(cHandler)


def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now()
    time_now = time_now.replace(microsecond=0)

    scale = (total_iter-cur_iter) / float(cur_iter)
    delta = (time_now - time0)
    eta = (delta*scale)
    time_fin = time_now + eta
    eta = time_fin.replace(microsecond=0) - time_now
    return str(delta), str(eta)


def validate(model=None, data_loader=None, cfg=None):

    preds, gts, cams, aff_gts = [], [], [], []
    num = 1
    seg_hist = np.zeros((21, 21))
    cam_hist = np.zeros((21, 21))
    for _, data in tqdm(enumerate(data_loader),
                        total=len(data_loader), ncols=100, ascii=" >="):
        name, inputs, labels, cls_label = data

        inputs = inputs.cuda()
        labels = labels.cuda()
        
        b,c,h,w = inputs.shape
        if (h//cfg.clip_init.clip_flag==0) and (w//cfg.clip_init.clip_flag)==0:
            inputs = inputs
        else:
            new_h = round(h/cfg.clip_init.clip_flag)*cfg.clip_init.clip_flag
            new_w = round(w/cfg.clip_init.clip_flag)*cfg.clip_init.clip_flag
            inputs = F.interpolate(inputs, size=(new_h, new_w), mode='bilinear', align_corners=False)

        segs_clip,segs_dino, cam, attn_loss = model(inputs, name, 'val')
        segs = 0.5*(segs_dino+segs_clip)

        resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

        preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
        cams += list(cam.cpu().numpy().astype(np.int16))
        gts += list(labels.cpu().numpy().astype(np.int16))

        num+=1

        if num % 1000 ==0:
            seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist)
            cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist)
            preds, gts, cams, aff_gts = [], [], [], []

    seg_hist, seg_score = evaluate.scores(gts, preds, seg_hist)
    cam_hist, cam_score = evaluate.scores(gts, cams, cam_hist)
    model.train()
    return seg_score, cam_score


def get_seg_loss(pred, label, ignore_index=255):
    bg_label = label.clone()
    bg_label[label!=0] = ignore_index
    bg_loss = F.cross_entropy(pred, bg_label.type(torch.long), ignore_index=ignore_index)
    fg_label = label.clone()
    fg_label[label==0] = ignore_index
    fg_loss = F.cross_entropy(pred, fg_label.type(torch.long), ignore_index=ignore_index)

    return (bg_loss + fg_loss) * 0.5


def get_mask_by_radius(h=20, w=20, radius=8):
    hw = h * w
    mask  = np.zeros((hw, hw))
    for i in range(hw):
        _h = i // w
        _w = i % w

        _h0 = max(0, _h - radius)
        _h1 = min(h, _h + radius+1)
        _w0 = max(0, _w - radius)
        _w1 = min(w, _w + radius+1)
        for i1 in range(_h0, _h1):
            for i2 in range(_w0, _w1):
                _i2 = i1 * w + i2
                mask[i, _i2] = 1
                mask[_i2, i] = 1

    return mask





def train(cfg):
    last_path = None

    num_workers = 10
    
    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)
    
    train_dataset = voc.VOC12ClsDataset(
        root_dir=cfg.dataset.root_dir,
        name_list_dir=cfg.dataset.name_list_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        img_fliplr=True,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )
    print("Found classes:", train_dataset.num_classes)
    print(f"Number of training samples: {len(train_dataset)}")
    
    # val_dataset = voc.VOC12SegDataset(
    #     root_dir=cfg.dataset.root_dir,
    #     name_list_dir=cfg.dataset.name_list_dir,
    #     split=cfg.val.split,
    #     stage='train',
    #     aug=False,
    #     ignore_index=cfg.dataset.ignore_index,
    #     num_classes=cfg.dataset.num_classes,
    # )

    # collate = partial(
    #     smart_collate,
    #     clip_flag=cfg.clip_init.clip_flag,
    #     max_h=cfg.dataset.crop_size,
    #     max_w=cfg.dataset.crop_size
    # )

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.train.samples_per_gpu,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=False,
                              drop_last=True,
                              prefetch_factor=4,
                            #   collate_fn=smart_collate
                              )

    # val_loader = DataLoader(val_dataset,
    #                         batch_size=1,
    #                         shuffle=False,
    #                         num_workers=num_workers,
    #                         pin_memory=False,
    #                         drop_last=False)


    clip_pretrained = cfg.clip_init.get("clip_pretrained", None)
    model = WeCLIP_Plus(
        num_classes=cfg.dataset.num_classes,
        clip_model=cfg.clip_init.clip_pretrain_path,
        clip_pretrained=clip_pretrained,
        dino_model=cfg.dino_init.dino_model,
        dino_fts_dim = cfg.dino_init.dino_fts_fuse_dim,
        decoder_layers = cfg.dino_init.decoder_layer,
        embedding_dim=cfg.clip_init.embedding_dim,
        in_channels=cfg.clip_init.in_channels,
        dataset_root_path=cfg.dataset.root_dir,
        clip_flag=cfg.clip_init.clip_flag,
        device='cuda'
    )
    logging.info('\nNetwork config: \n%s'%(model))
    param_groups = model.get_param_groups()
    model.cuda()

    mask_size = int(cfg.dataset.crop_size // cfg.clip_init.clip_flag)
    attn_mask = get_mask_by_radius(h=mask_size, w=mask_size, radius=args.radius)
    writer = SummaryWriter(cfg.work_dir.tb_logger_dir)

    optimizer = PolyWarmupAdamW(
        params=[
            {
                "params": param_groups[0],
                "lr": cfg.optimizer.learning_rate,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[1],
                "lr": 0.0,
                "weight_decay": 0.0,
            },
            {
                "params": param_groups[2],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
            {
                "params": param_groups[3],
                "lr": cfg.optimizer.learning_rate*10,
                "weight_decay": cfg.optimizer.weight_decay,
            },
        ],
        lr = cfg.optimizer.learning_rate,
        weight_decay = cfg.optimizer.weight_decay,
        betas = cfg.optimizer.betas,
        warmup_iter = cfg.scheduler.warmup_iter,
        max_iter = cfg.train.max_iters,
        warmup_ratio = cfg.scheduler.warmup_ratio,
        power = cfg.scheduler.power
    )
    logging.info('\nOptimizer: \n%s' % optimizer)

    train_loader_iter = iter(train_loader)

    avg_meter = AverageMeter()

    criterion_dice = DiceLoss().cuda()


    for n_iter in range(cfg.train.max_iters):

        try:
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_labels, img_box = next(train_loader_iter)

        segs_clip, segs_dino, cam, attn_pred = model(inputs.cuda(), img_name)

        # pseudo_label = cam
        # generate our pseudo-labels from the CAM and make sure they’re valid ints
        pseudo_label = cam.long()
        # any value <0 or ≥ num_classes, mark as ignore_index so cross_entropy will skip it
        invalid = (pseudo_label < 0) | (pseudo_label >= cfg.dataset.num_classes)
        pseudo_label[invalid] = cfg.dataset.ignore_index

        segs= 0.5*segs_clip+0.5*segs_dino
        segs = F.interpolate(segs, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        segs_clip = F.interpolate(segs_clip, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        segs_dino = F.interpolate(segs_dino, size=pseudo_label.shape[1:], mode='bilinear', align_corners=False)

        pred_clip_max, pred_label_clip = torch.max(F.softmax(segs_clip, dim=1), dim=1)
        pred_dino_max, pred_label_dino = torch.max(F.softmax(segs_dino, dim=1), dim=1)
        pred_max, pred_label_seg = torch.max(F.softmax(segs, dim=1), dim=1)

        fts_cam = cam.clone()

        aff_label = cams_to_affinity_label(fts_cam, mask=attn_mask, ignore_index=cfg.dataset.ignore_index, clip_flag=cfg.clip_init.clip_flag)
        attn_loss, pos_count, neg_count = get_aff_loss(attn_pred, aff_label)

        if n_iter > 18000:
            pseudo_label[pred_max>0.75] = pred_label_seg[pred_max>0.75]
        seg_loss = get_seg_loss(segs, pseudo_label.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        seg_loss2 = criterion_dice(segs, pseudo_label.type(torch.long))

        seg_clip_loss2 = get_seg_loss(segs_clip, pred_label_dino.type(torch.long), ignore_index=cfg.dataset.ignore_index)
        seg_dino_loss2 = get_seg_loss(segs_dino, pred_label_clip.type(torch.long), ignore_index=cfg.dataset.ignore_index)


        loss = 1 * seg_loss + 0.1 * attn_loss + 0.1 * (seg_clip_loss2 + seg_dino_loss2) + 1*seg_loss2

        avg_meter.add({'seg_loss': seg_loss.item(), 'attn_loss': attn_loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (n_iter + 1) % cfg.train.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter+1, cfg.train.max_iters)
            cur_lr = optimizer.param_groups[0]['lr']

            preds = torch.argmax(segs,dim=1).cpu().numpy().astype(np.int16)
            gts = pseudo_label.cpu().numpy().astype(np.int16)

            seg_mAcc = (preds==gts).sum()/preds.size

            logging.info("Iter: %d; Elasped: %s; ETA: %s; LR: %.3e;, pseudo_seg_loss: %.4f, attn_loss: %.4f, pseudo_seg_mAcc: %.4f"%(n_iter+1, delta, eta, cur_lr, avg_meter.pop('seg_loss'), avg_meter.pop('attn_loss'),  seg_mAcc))

            writer.add_scalars('train/loss',  {"seg_loss": seg_loss.item(), "attn_loss": attn_loss.item()}, global_step=n_iter)


        # if (n_iter +1) % cfg.train.eval_iters == 0:
        #     ckpt_name = os.path.join(cfg.work_dir.ckpt_dir, "wetr_iter_%d.pth"%(n_iter+1))
        #     logging.info('Validating...')
        #     if (n_iter + 1) > 26000:
        #         torch.save(model.state_dict(), ckpt_name)
        #     seg_score, cam_score = validate(model=model, data_loader=val_loader, cfg=cfg)
        #     logging.info("cams score:")
        #     logging.info(cam_score)
        #     logging.info("segs score:")
        #     logging.info(seg_score)

        #Validation Commented Out
        if (n_iter + 1) % 10000 == 0 or (n_iter+1 == cfg.train.max_iters):
            ckpt_path = os.path.join(
                cfg.work_dir.ckpt_dir,
                f"wetr_iter_{n_iter+1}.pth"
            )
            if n_iter + 1 == cfg.train.max_iters:
                last_path = ckpt_path
            torch.save(model.state_dict(), ckpt_path)

    return last_path


def main(config):
    # args = parser.parse_args()
    cfg = OmegaConf.load(config)

    if args.work_dir is not None:
        cfg.work_dir.dir = args.work_dir

    timestamp = "{0:%Y-%m-%d-%H-%M}".format(datetime.datetime.now())

    # Use CLIP_TEXT_VERSION env var to create unique subdirectory if available
    clip_text_version = os.environ.get('CLIP_TEXT_VERSION', 'default')
    timestamp_with_version = f"{timestamp}_{clip_text_version}"

    cfg.work_dir.ckpt_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.ckpt_dir, timestamp_with_version)
    cfg.work_dir.pred_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.pred_dir)
    cfg.work_dir.tb_logger_dir = os.path.join(cfg.work_dir.dir, cfg.work_dir.tb_logger_dir, timestamp_with_version)

    os.makedirs(cfg.work_dir.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.pred_dir, exist_ok=True)
    os.makedirs(cfg.work_dir.tb_logger_dir, exist_ok=True)

    setup_logger(filename=os.path.join(cfg.work_dir.dir, timestamp+'.log'))
    logging.info('\nargs: %s' % args)
    logging.info('\nconfigs: %s' % cfg)

    setup_seed(1)
    final_path = train(cfg=cfg)
    return final_path



if __name__ == "__main__":
    # args = parser.parse_args()

    # args = parser.parse_args()
    config = args.config
    main(config)
