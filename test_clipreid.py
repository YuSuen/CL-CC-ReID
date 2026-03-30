import os
from config import cfg
import argparse
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from processor.processor_clipreid_stage2 import do_inference
from utils.logger import setup_logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    if cfg.DATASETS.NAMES in ['ltcc', 'vcclothes', 'vcclothes_cc', 'vcclothes_sc', 'last', 'deepchange']:
        train_loader_stage2, train_loader_stage1, galleryloader, queryloader, num_query, \
                   num_classes, camera_num, view_num, clt_num, pid2clothes, dataset = make_dataloader(cfg)
    elif cfg.DATASETS.NAMES=='prcc':
        train_loader_stage2, train_loader_stage1, galleryloader, queryloader_same, queryloader_diff, \
                   num_query, num_classes, camera_num, view_num, clt_num, pid2clothes, dataset = make_dataloader(cfg)
        queryloader = [queryloader_same, queryloader_diff]
    else:
        train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    if cfg.DATASETS.NAMES in ['ltcc', 'PRCC'，'vcclothes', 'vcclothes_cc', 'vcclothes_sc', 'last', 'deepchange']:
        model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num, clothes_num = clt_num)
    else:
        model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
        
    model.load_param(cfg.TEST.WEIGHT)

    if cfg.DATASETS.NAMES in ['ltcc', 'prcc', 'vcclothes', 'vcclothes_sc', 'vcclothes_cc', 'deepchange', 'last']:
        rank1 = test(model, queryloader, galleryloader)
    elif cfg.DATASETS.NAMES == 'prcc':     
        queryloader_same, queryloader_diff = queryloader[0], queryloader[1]
        rank1 = test_prcc(model, queryloader_same, queryloader_diff, galleryloader)
    elif cfg.DATASETS.NAMES == 'VehicleID':
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            rank_1, rank5, mAP = do_inference(cfg,
                 model,
                 val_loader,
                 num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
                all_mAP = mAP
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5
                all_mAP = all_mAP + mAP

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, mAP, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}, sum_mAP {:.1%}".format(all_rank_1.sum()/10.0, all_rank_5.sum()/10.0, all_mAP.sum()/10.0))
    else:
       do_inference(cfg,
                 model,
                 val_loader,
                 num_query)


