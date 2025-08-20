import numpy as np
from torch.utils.data import Dataset
import os
import imageio.v2 as imageio
from . import transforms
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torch
import cv2




def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str)
    return img_name_list

def load_cls_label_list(name_list_dir):
    """
    Build a dict: { basename -> one-hot np.array of shape (C,) }
    by reading:
      - train.txt / val.txt         (one basename per line)
      - <class>_train.txt           (basename 1/-1)
      - <class>_val.txt             (basename 1/-1)
    """
    # 1) discover your classes by looking at *_train.txt filenames
    class_files = [f for f in os.listdir(name_list_dir) if f.endswith('_train.txt')]
    classes = sorted({fname.split('_')[0] for fname in class_files})

    # 2) initialize the label map
    label_map = {}

    # 3) for each split, read the bare list + per-class labels
    for split in ['train', 'val']:
        split_path = os.path.join(name_list_dir, f'{split}.txt')
        if not os.path.isfile(split_path):
            continue
        with open(split_path, 'r') as f:
            basenames = [ln.strip() for ln in f if ln.strip()]

        # ensure an entry for every image
        for b in basenames:
            label_map[b] = np.zeros(len(classes), dtype=np.uint8)

        # read each class’s file for this split
        for ci, cls in enumerate(classes):
            cls_path = os.path.join(name_list_dir, f'{cls}_{split}.txt')
            if not os.path.isfile(cls_path):
                continue
            with open(cls_path, 'r') as f:
                for ln in f:
                    b, lab = ln.strip().split()
                    if lab == '1':
                        label_map[b][ci] = 1

    return label_map


class VOC12Dataset(Dataset):
    def __init__(
        self,
        root_dir=None,
        name_list_dir=None,
        split='train',
        stage='train',
    ):
        super().__init__()

        self.root_dir = root_dir
        self.stage = stage
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.label_dir = os.path.join(root_dir, 'SegmentationClassAug')
        self.name_list_dir = os.path.join(name_list_dir, split + '.txt')
        self.name_list = load_img_name_list(self.name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        _img_name = self.name_list[idx]
        img_name = os.path.join(self.img_dir, _img_name+'.jpg')
        image = np.asarray(imageio.imread(img_name))
        # image = Image.open(img_name)

        # if self.stage == "train":

        #     label_dir = os.path.join(self.label_dir, _img_name+'.JPEG')
        #     label = np.asarray(imageio.imread(label_dir))

        # elif self.stage == "val":

        #     label_dir = os.path.join(self.label_dir, _img_name+'.JPEG')
        #     label = np.asarray(imageio.imread(label_dir))

        # elif self.stage == "test":
        #     label = image[:,:,0]

        # return _img_name, image, label
        return _img_name, image, None

def _transform_resize():
    return Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class VOC12ClsDataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 num_classes=21,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        self.color_jittor = transforms.PhotoMetricDistortion()

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)
        self.normalize = _transform_resize()
        self.scale = 1
        self.patch_size = 16

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image):
        # img_box = None
        # if self.aug:
        #     image = np.array(image)
        #     # print('image', image.shape)
        #     '''
        #     if self.resize_range: 
        #         image, label = transforms.random_resize(
        #             image, label, size_range=self.resize_range)
        #     '''
        #     # if self.rescale_range:
        #     #     image = transforms.random_scaling(
        #     #         image,
        #     #         scale_range=self.rescale_range)
        #     #Reenable if you can
            
        #     if self.img_fliplr:
        #         image = transforms.random_fliplr(image)
        #     #image = self.color_jittor(image)
        #     if self.crop_size:
        #         image, img_box = transforms.random_crop(
        #             image,
        #             crop_size=self.crop_size,
        #             mean_rgb=[0,0,0],#[123.675, 116.28, 103.53], 
        #             ignore_index=self.ignore_index)
        # '''
        # if self.stage != "train":
        #     image = transforms.img_resize_short(image, min_size=min(self.resize_range))
        # '''
        # image = transforms.normalize_img(image)
        # # image = self.normalize(image)
        # # image = image.numpy()
        
        # ## to chw
        # image = np.transpose(image, (2, 0, 1))

        # return image, img_box

        # img_box = None
        # image = np.array(image)           # H x W x C
        # image = transforms.normalize_img(image)
        # image = np.transpose(image, (2,0,1))  # C x H x W
        # return image, img_box


        # img_box = None

        # # 1) force to ndarray H×W×C (or H×W if grayscale)
        # image = np.array(image)

        # # 2) your augmentations (uncomment / re-enable as needed)
        # if self.aug:
        #     # example: random scaling
        #     # image = transforms.random_scaling(image, scale_range=self.rescale_range)
        #     if self.img_fliplr:
        #         image = transforms.random_fliplr(image)
        #     # if self.crop_size:
        #     #     image, img_box = transforms.random_crop(
        #     #         image,
        #     #         crop_size=self.crop_size,
        #     #         mean_rgb=[0,0,0],
        #     #         ignore_index=self.ignore_index
        #     #     )
        #     # …etc…

        # # 3) if it’s now H×W (no channel dim), stack to make H×W×3
        # if image.ndim == 2:
        #     image = np.stack([image, image, image], axis=-1)

        # # 4) normalize to float32 in [0..1], then to your clip means/stds
        # image = transforms.normalize_img(image)

        # # 5) H×W×3 → 3×H×W
        # image = np.transpose(image, (2, 0, 1))

        # return image, img_box
        # 1) force to numpy H×W×C
        img_np = np.asarray(image)

        # 2) if grayscale or single-channel, make 3 channels
        if img_np.ndim == 2:
            img_np = np.stack([img_np]*3, axis=-1)
        elif img_np.ndim == 3 and img_np.shape[2] == 1:
            img_np = np.concatenate([img_np]*3, axis=2)

        # 3) optional flip
        if self.aug and self.img_fliplr:
            img_np = transforms.random_fliplr(img_np)

        # 4) resize to exactly (crop_size x crop_size)
        img_resized = cv2.resize(
            img_np,
            (self.crop_size, self.crop_size),
            interpolation=cv2.INTER_LINEAR
        )

        # 5) normalize pixel values then CLIP mean/std
        img_norm = transforms.normalize_img(img_resized)

        # 6) HWC → CHW and convert to float32
        img_chw = img_norm.transpose(2, 0, 1).astype(np.float32)

        # 7) wrap in tensor and return with empty img_box
        return torch.from_numpy(img_chw), []
    

    @staticmethod
    def _to_onehot(label_mask, num_classes, ignore_index):
        #label_onehot = F.one_hot(label, num_classes)
        
        _label = np.unique(label_mask).astype(np.int16)
        # exclude ignore index
        _label = _label[_label != ignore_index]
        # exclude background
        _label = _label[_label != 0]

        label_onehot = np.zeros(shape=(num_classes), dtype=np.uint8)
        label_onehot[_label] = 1
        return label_onehot

    def __getitem__(self, idx):

#         img_name, image, _ = super().__getitem__(idx)
        
# #         ori_height = image.size[1]
# #         ori_width = image.size[0]
        
# #         new_height = int(np.ceil(self.scale * int(ori_height) / self.patch_size) * self.patch_size)
# #         new_width = int(np.ceil(self.scale * int(ori_width) / self.patch_size) * self.patch_size)
        
# #         image = Resize((new_height, new_width), interpolation=BICUBIC)(image)
# #         image = image.convert("RGB")

#         image, img_box = self.__transforms(image=image)

#         cls_label = self.label_list[img_name]

#         if self.aug:
#             return img_name, image, cls_label, img_box
#         else:
#             return img_name, image, cls_label

     
        # img_name, image_np, _ = super().__getitem__(idx)
        # image_np, img_box = self.__transforms(image=image_np)
        # cls_label_np = self.label_list[img_name]

        # # --- HERE: convert *all* numpy arrays into fresh, resizable Tensors ---
        # image = torch.tensor(image_np, dtype=torch.float32)         # C×H×W
        # cls_label = torch.tensor(cls_label_np, dtype=torch.uint8)   # e.g. (num_classes,)
        # if img_box is not None:
        #     img_box = torch.tensor(img_box, dtype=torch.int64)      # whatever shape your box uses

        # # return exactly the same four slots
        # return img_name, image, cls_label, img_box
        name, img_np, _ = super().__getitem__(idx)
        img_t, _     = self.__transforms(img_np)
        cls_lbl_np   = self.label_list[name]
        cls_lbl_t    = torch.tensor(cls_lbl_np, dtype=torch.uint8)
        return name, img_t, cls_lbl_t, []


class VOC12SegDataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.color_jittor = transforms.PhotoMetricDistortion()
        self.normalize = _transform_resize()

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)
        self.scale = 1
        self.patch_size = 16
        

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label):
        # if self.aug:
        #     image = np.array(image)
        #     '''
        #     if self.resize_range: 
        #         image, label = transforms.random_resize(
        #             image, label, size_range=self.resize_range)
            
        #     if self.rescale_range:
        #         image, label = transforms.random_scaling(
        #             image,
        #             label,
        #             scale_range=self.rescale_range)
        #     '''
        #     if self.img_fliplr:
        #         image, label = transforms.random_fliplr(image, label)
        #     image = self.color_jittor(image)
        #     if self.crop_size:
        #         image, label, img_box = transforms.random_crop(
        #             image,
        #             label,
        #             crop_size=self.crop_size,
        #             # mean_rgb=[123.675, 116.28, 103.53], 
        #             ignore_index=self.ignore_index)
        # '''
        # if self.stage != "train":
        #     image = transforms.img_resize_short(image, min_size=min(self.resize_range))
        # '''
        # # image = self.normalize(image)
        # # image = image.numpy()
        
        # image = transforms.normalize_img(image)
        # ## to chw
        # image = np.transpose(image, (2, 0, 1))

        # return image, label
         
        img_box = None
        image = np.asarray(image)   # ensure numpy


        if image.ndim == 2:               # ADDED STUFF
            image = np.stack([image]*3, axis=-1)

        # resize to exactly (crop_size × crop_size)
        image = cv2.resize(
            image,
            (self.crop_size, self.crop_size),
            interpolation=cv2.INTER_LINEAR
        )
        # normalize & to C×H×W
        image = transforms.normalize_img(image)

        

        image = np.transpose(image, (2, 0, 1))
        return image, img_box

    def __getitem__(self, idx):
        img_name, image, label = super().__getitem__(idx)
#         ori_height = image.size[1]
#         ori_width = image.size[0]
        
#         new_height = int(np.ceil(self.scale * int(ori_height) / self.patch_size) * self.patch_size)
#         new_width = int(np.ceil(self.scale * int(ori_width) / self.patch_size) * self.patch_size)
        
#         image = Resize((new_height, new_width), interpolation=BICUBIC)(image)
#         image = image.convert("RGB")

        image, label = self.__transforms(image=image, label=label)

        if self.stage=='test':
            cls_label = 0
        else:
            cls_label = self.label_list[img_name]
        
        dummy_label = np.zeros((self.crop_size, self.crop_size), dtype=np.int16)
        return img_name, image, torch.from_numpy(dummy_label), cls_label

       