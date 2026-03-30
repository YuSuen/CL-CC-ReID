import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset#, ImageDataset_CC, ImageDataset_CC_ID
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler#, RandomIdentitySamplerCC
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP#, RandomIdentitySamplerCC_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
from .ltcc import LTCC
from .prcc import PRCC
from .vcclothes import VCClothes, VCClothesSameClothes, VCClothesClothesChanging
from .last import LaST
from .deepchange import DeepChange

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
    'ltcc': LTCC,
    'prcc': PRCC,
    'vcclothes': VCClothes,
    'vcclothes_sc': VCClothesSameClothes,
    'vcclothes_cc': VCClothesClothesChanging,
    'deepchange': DeepChange,
    'last': LaST

}

def train_collate_fn(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, viewids , _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, viewids,

def val_collate_fn(batch):
    imgs, pids, camids, viewids, img_paths = zip(*batch)
    viewids = torch.tensor(viewids, dtype=torch.int64)
    camids_batch = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, camids_batch, viewids, img_paths

def train_collate_fn_cc(batch):
    """
    # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
    """
    imgs, pids, camids, clothes_ids, img_path = zip(*batch)  # [img_path, pid, camid, clothes_id]
    pids = torch.tensor(pids, dtype=torch.int64)
    clothes_ids = torch.tensor(clothes_ids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, clothes_ids, img_path

def val_collate_fn_cc(batch):
    imgs, pids, camids, clothes_ids, img_path = zip(*batch) # [img_path, pid, camid, clothes_id]
    pids = torch.tensor(pids, dtype=torch.int64)
    clothes_ids = torch.tensor(clothes_ids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, clothes_ids, img_path
    
class IterLoader:
    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self):
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)

def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    if cfg.DATASETS.NAMES in ['ltcc', 'prcc', 'vcclothes', 'vcclothes_sc', 'vcclothes_cc', 'deepchange', 'last']:
        train_set = ImageDataset(dataset.train, train_transforms)
        train_set_normal = ImageDataset(dataset.train, val_transforms)

    if cfg.DATASETS.NAMES in ['ltcc', 'prcc', 'vcclothes', 'vcclothes_sc', 'vcclothes_cc', 'deepchange', 'last']:
        num_classes = dataset.num_train_pids
        clt_num = dataset.num_train_clothes
        cam_num = dataset.num_train_cams
        view_num = dataset.num_train_clothes
        pid2clothes = dataset.pid2clothes
    else:
        clt_num = 0
        cam_num = dataset.num_train_cams
        view_num = dataset.num_train_vids
        num_classes = dataset.num_train_pids

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            if cfg.DATASETS.NAMES in ['ltcc', 'prcc', 'vcclothes', 'vcclothes_sc', 'vcclothes_cc', 'deepchange', 'last']:
                pass
            else:
                mini_batch_size = cfg.SOLVER.STAGE2.IMS_PER_BATCH // dist.get_world_size()
                data_sampler = RandomIdentitySampler_DDP(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                                         cfg.DATALOADER.NUM_INSTANCE)
                batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
                train_loader_stage2 = torch.utils.data.DataLoader(
                    train_set,
                    num_workers=num_workers,
                    batch_sampler=batch_sampler,
                    collate_fn=train_collate_fn,
                    pin_memory=True,
                )
        else:
            if cfg.DATASETS.NAMES in ['ltcc', 'prcc', 'vcclothes', 'vcclothes_sc', 'vcclothes_cc', 'deepchange', 'last']:
                train_loader_stage2 = DataLoader(
                    train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                    sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                                  cfg.DATALOADER.NUM_INSTANCE),
                    num_workers=num_workers, collate_fn=train_collate_fn_cc
                )
            else:
                train_loader_stage2 = DataLoader(
                    train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                    sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                    num_workers=num_workers, collate_fn=train_collate_fn
                )
    elif cfg.DATALOADER.SAMPLER == 'softmax':
        print('using softmax sampler')
        train_loader_stage2 = DataLoader(
            train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn)
    else:
        print('unsupported sampler! expected softmax or triplet but got {}'.format(cfg.SAMPLER))


    if cfg.DATASETS.NAMES in ['ltcc', 'prcc', 'vcclothes', 'vcclothes_sc', 'vcclothes_cc', 'deepchange', 'last']:

        train_loader_stage1 = DataLoader(train_set_normal, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=True,
                                         num_workers=num_workers,
                                         collate_fn=val_collate_fn_cc) # NO sample strategy apply

        gallery_set = ImageDataset(dataset.gallery, val_transforms)
        galleryloader = DataLoader(gallery_set,  batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
                collate_fn=val_collate_fn_cc)

        if cfg.DATASETS.NAMES in ['ltcc', 'vcclothes', 'vcclothes_sc', 'vcclothes_cc', 'deepchange', 'last']:
            query_set = ImageDataset(dataset.query, val_transforms)
            queryloader = DataLoader(query_set,  batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
                                            collate_fn=val_collate_fn_cc)

            return train_loader_stage2, train_loader_stage1, galleryloader, queryloader, len(dataset.query), \
                   num_classes, cam_num, view_num, clt_num, pid2clothes, dataset

        else:
            queryloader_same_set = ImageDataset(dataset.query_same, val_transforms)
            queryloader_diff_set = ImageDataset(dataset.query_diff, val_transforms)
            queryloader_same = DataLoader(queryloader_same_set,  batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
                                            collate_fn=val_collate_fn_cc)
            queryloader_diff = DataLoader(queryloader_diff_set,  batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
                                            collate_fn=val_collate_fn_cc)

            return train_loader_stage2, train_loader_stage1, galleryloader, queryloader_same, queryloader_diff, \
                   len(dataset.query_same)+len(dataset.query_diff), num_classes, cam_num, view_num, clt_num, pid2clothes, dataset

    else:
        train_loader_stage1 = DataLoader(train_set_normal, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True,
                                         num_workers=num_workers,
                                         collate_fn=train_collate_fn)

        val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

        val_loader = DataLoader(val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
            collate_fn=val_collate_fn)

        return train_loader_stage2, train_loader_stage1, val_loader, len(dataset.query), num_classes, cam_num, view_num

def make_dataloader2(cfg, train_data, val=False):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS
    if val:
        train_set_normal = ImageDataset(train_data, val_transforms)
    else:
        if cfg.DATASETS.NAMES in ['ltcc', 'prcc', 'vcclothes', 'vcclothes_sc', 'vcclothes_cc', 'deepchange', 'last']:
            train_set = ImageDataset(train_data, train_transforms)
        else:
            train_set = ImageDataset(train_data, train_transforms)

    if 'triplet' in cfg.DATALOADER.SAMPLER:
        if cfg.MODEL.DIST_TRAIN:
            print('DIST_TRAIN START')
            if cfg.DATASETS.NAMES in ['ltcc', 'prcc', 'vcclothes', 'vcclothes_sc', 'vcclothes_cc', 'deepchange', 'last']:
                pass
            else:
                mini_batch_size = cfg.SOLVER.STAGE2.IMS_PER_BATCH // dist.get_world_size()
                data_sampler = RandomIdentitySampler_DDP(train_data, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                                         cfg.DATALOADER.NUM_INSTANCE)
                batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)
                train_loader_stage2 = torch.utils.data.DataLoader(
                    train_set,
                    num_workers=num_workers,
                    batch_sampler=batch_sampler,
                    collate_fn=train_collate_fn,
                    pin_memory=True,
                )
        else:
            if val:
                train_loader_stage2 = DataLoader(train_set_normal, batch_size=cfg.SOLVER.STAGE1.IMS_PER_BATCH, shuffle=True,
                                                         num_workers=num_workers,
                                                         collate_fn=train_collate_fn_cc) # NO sample strategy apply
            else:
                if cfg.DATASETS.NAMES in ['ltcc', 'prcc', 'vcclothes', 'vcclothes_sc', 'vcclothes_cc', 'deepchange', 'last']:
                    train_loader_stage2 = DataLoader(
                        train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                        sampler=RandomIdentitySampler(train_data, cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                                                      cfg.DATALOADER.NUM_INSTANCE),
                        num_workers=num_workers, collate_fn=train_collate_fn_cc
                    )
                else:
                    train_loader_stage2 = DataLoader(
                        train_set, batch_size=cfg.SOLVER.STAGE2.IMS_PER_BATCH,
                        sampler=RandomIdentitySampler(train_data, cfg.SOLVER.STAGE2.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
                        num_workers=num_workers, collate_fn=train_collate_fn
                    )

    return train_loader_stage2



