import logging
import os
import time
import json
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
from evaluation import test, test_prcc
from datasets.make_dataloader_clipreid import make_dataloader2
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, silhouette_samples

def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    # train
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes-batch* (num_classes//batch)
    if left != 0 :
        i_ter = i_ter+1
    text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i+1 != i_ter:
                l_list = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list = torch.arange(i*batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label = l_list, get_text = True)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        scheduler.step()

        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            with amp.autocast(enabled=True):
                score, feat, image_features = model(x = img, label = target, cam_label=target_cam, view_label=target_view)
                logits = image_features @ text_features.t()
                loss = loss_fn(score, feat, target, target_cam, logits)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else: 
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else: 
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else: 
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else: 
                            target_view = None
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)


    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]

import numpy as np
import collections
import random
def do_train_stage2_cc(cfg,
                    model,
                    center_criterion,
                    dataset,
                    train_loader_stage2,
                    train_loader_stage1,
                    galleryloader,
                    queryloader,
                    pid2clothes,
                    optimizer,
                    optimizer_center,
                    scheduler,
                    loss_fn,
                    num_query, local_rank):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    device = "cuda"

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
            num_classes = model.module.num_classes
            clt_classes = model.module.clothes_num
        else:
            clt_classes = model.clothes_num
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()

    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    p2c_dict = {i: np.nonzero(row)[0].tolist() for i, row in enumerate(pid2clothes)}

    from loss.triplet_loss import TripletLoss_weight, euclidean_dist
    from loss.softmax_loss import CrossEntropyLabelSmooth_weight

    triplet = TripletLoss_weight(cfg.SOLVER.MARGIN).to(device)
    ce_criterion = CrossEntropyLabelSmooth_weight(num_classes=num_classes).to(device)

    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes - batch * (num_classes // batch)
    if left != 0:
        i_ter = i_ter + 1
    text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i + 1 != i_ter:
                l_list = torch.arange(i * batch, (i + 1) * batch)
            else:
                l_list = torch.arange(i * batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label=l_list, get_text=True)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()

    use_clothes_label = cfg.MODEL.USE_CLOTHES_LABEL
    if use_clothes_label:
        logger.info('Using clothes label!')
        train_data = train_loader_stage1.dataset.dataset
        p2c_dict = {i: np.nonzero(row)[0].tolist() for i, row in enumerate(pid2clothes)}
    else:
        logger.info('Do not use clothes label!')

        feats_dir = os.path.join(os.getcwd(), "feats")
        dataset_name = cfg.DATASETS.NAMES
        feat_json_path = os.path.join(feats_dir, f"{dataset_name}_feats.json")

        if not os.path.isdir(feats_dir):
            os.makedirs(feats_dir, exist_ok=True)

        if os.path.isfile(feat_json_path):
            logger.info("Loading cached features from %s", feat_json_path)
            with open(feat_json_path, "r", encoding="utf-8") as file:
                dataset_feats = json.load(file)
        else:
            logger.info("Extracting features for %s and saving to %s", dataset_name, feat_json_path)
            dataset_feats = {}
            with torch.no_grad():
                for n_iter, (img, vid, target_cam, clothes_id, img_paths) in enumerate(train_loader_stage1):
                    img = img.to(device)
                    with amp.autocast(enabled=False):
                        image_feature = model(img, get_image=True)
                    for img_feat, img_path in zip(image_feature, img_paths):
                        dataset_feats[img_path] = img_feat.detach().cpu().float().tolist()

            with open(feat_json_path, "w", encoding="utf-8") as file:
                json.dump(dataset_feats, file)

        id_data = {pid: [] for pid in range(num_classes)}
        id_features = collections.defaultdict(list)
        with torch.no_grad():
            for n_iter, (img, vid, target_cam, clothes_id, img_paths) in enumerate(train_loader_stage1):  # if cc [img, pid, camid, clothes_id]
                img = img.to(device)
                vid = vid.data.cpu().numpy()
                cams = target_cam.data.cpu().numpy()
                clts = clothes_id.data.cpu().numpy()

                for id, cam, clt, img_path in zip(vid, cams, clts, img_paths):
                    id_features[id].append(torch.tensor(np.array(dataset_feats[img_path])).unsqueeze(0).float())
                    id_data[id].append((img_path, id, cam, clt))

        from utils.rerank import compute_jaccard_dist
        from sklearn.cluster import DBSCAN
        new_dataset = []
        index_offset = 0
        p2c_dict = {pid: [] for pid in range(num_classes)}
        for id in id_data.keys():
            features = []
            for data in id_data[id]:
                features = id_features[data[1]]

            features = torch.cat(features, dim=0)
            features = F.normalize(features, dim=1).contiguous()
            logger.info('==> Create clothes pseudo labels for ID {}'.format(id))
            rerank_dist = compute_jaccard_dist(features, k1=20, k2=6)
            eps = 0.6
            # cluster
            cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=-1)
            pseudo_labels = cluster.fit_predict(rerank_dist)
            num_ids = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            clothes_num = num_ids
            logger.info('==> Create {} clothes pseudo labels for ID {}'.format(clothes_num, id))
            # relabel clothes
            for i, (item, label) in enumerate(zip(id_data[id], pseudo_labels)):
                if label == -1: continue
                new_dataset.append((item[0], item[1], item[2], label + index_offset))
                if label + index_offset in p2c_dict[item[1]]:
                    pass
                else:
                    p2c_dict[item[1]].append(label + index_offset)
            index_offset += clothes_num
        del id_features
        logger.info(index_offset)
        train_data = new_dataset
        train_loader = make_dataloader2(cfg, train_data, val=True)
        train_loader_stage1 = train_loader
        clt_classes = index_offset
        class_accuracies_pid = {i: [] for i in range(num_classes)}
        class_accuracies_cid = {i: [] for i in range(clt_classes)}


    print(num_classes, clt_classes)
    class_accuracies_pid = {i: [] for i in range(num_classes)}
    class_accuracies_cid = {i: [] for i in range(clt_classes)}
    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_clt, _) in enumerate(train_loader_stage1):
            img = img.to(device)
            vid = vid.to(device)
            target_clt = target_clt.to(device)
            if cfg.MODEL.SIE_CAMERA:
                cam_label = target_cam.to(device)
            else:
                cam_label = None
            with amp.autocast(enabled=True):
                pred, _, _ = model(img, cam_label=cam_label)
                for pid, clt, prediction in zip(vid, target_clt, pred[1].max(1)[1]):
                    class_accuracies_pid[pid.item()].append(1 if pid == prediction else 0)
                    class_accuracies_cid[clt.item()].append(1 if pid == prediction else 0)

    logger.info('clothes sampling')
    data = []
    clt_count = collections.defaultdict(set)
    nested_dict = {}
    for data_info in train_data:
        pid = data_info[1]
        cid = data_info[3]
        clt_count[pid].add(cid)
        if pid not in nested_dict:
            nested_dict[pid] = {}
        if cid not in nested_dict[pid]:
            nested_dict[pid][cid] = []
        nested_dict[pid][cid].append(data_info)
    max_clothes = 0
    for key, items_set in clt_count.items():
        max_clothes = max(max_clothes, len(items_set))

    selected_clothes = []
    sc = []
    for id in list(nested_dict.keys()):
        if nested_dict[id]:
            id_infos = nested_dict[id]
            select_clt = random.choice(list(id_infos.keys()))
            data.extend(id_infos[select_clt])
            if len(id_infos.keys())==1:
                sc.append(id)
            id_infos.pop(select_clt)
            selected_clothes.append(select_clt)
        else:
            continue

    logger.info('>>>>>>>>>>>>>>>>>>start training')
    best_rank1 = -1.0
    best_model_path = os.path.join(cfg.OUTPUT_DIR, f"{cfg.MODEL.NAME}_best.pth")
    for epoch in range(1, epochs + 1):
        # start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step()
        model.train()

        for i in range(len(class_accuracies_pid)):
            if class_accuracies_pid[i] == []:
                class_accuracies_pid[i] = torch.tensor(0).float()
            else:
                class_accuracies_pid[i] = torch.tensor(class_accuracies_pid[i]).float().mean()

        for i in range(len(class_accuracies_cid)):
            if class_accuracies_cid[i] == []:
                class_accuracies_cid[i] = torch.tensor(0).float()
            else:
                class_accuracies_cid[i] = torch.tensor(class_accuracies_cid[i]).float().mean()

        weight_pid = torch.stack(list(class_accuracies_pid.values())).cuda()
        weight_cid = torch.stack(list(class_accuracies_cid.values())).cuda()

        class_accuracies_pid = {i: [] for i in range(num_classes)}
        class_accuracies_cid = {i: [] for i in range(clt_classes)}

        threshold = 0.1
        logger.info(f"threshold:{threshold}")
        if epoch > 1 and (len(data) < len(train_data)):
            logger.info('clothes sampling')
            logger.info('Extracting features for measuring, this may take several minutes.')
            clothes_centers = collections.defaultdict(list)
            feature = []
            label = []
            with torch.no_grad():
                for n_iter, (img, vid, target_cam, clothes_id, _) in enumerate(train_loader_stage1):  # if cc [img, pid, camid, clothes_id]
                    img = img.to(device)
                    colthes = clothes_id.data.cpu().numpy()
                    if cfg.MODEL.SIE_CAMERA:
                        cam_label = target_cam.to(device)
                    else:
                        cam_label = None
                    # if cfg.DATASETS.NAMES == 'prcc':
                    #     cam_label = clothes_id.to(device)
                    with amp.autocast(enabled=True):
                        _, _, image_feature = model(img, cam_label=cam_label)
                        for clt, pid, img_feat in zip(colthes, vid, image_feature):
                            clothes_centers[clt].append(img_feat.unsqueeze(0).cpu())
                            if clt in selected_clothes:
                                feature.append(img_feat.unsqueeze(0).cpu())
                                label.append(pid)
                clothes_centers = [torch.cat(clothes_centers[i], 0).mean(0) for i in sorted(clothes_centers.keys())]

                feature = torch.cat(feature, 0).data.cpu().numpy()
                label = np.array(label)
                silhouette_score = silhouette_samples(feature, label)

                selected_features = collections.defaultdict(list)
                unselected_features = collections.defaultdict(list)
                for id in p2c_dict:
                    for cid in p2c_dict[id]:
                        if cid in selected_clothes:
                            selected_features[id].append(clothes_centers[cid].unsqueeze(0))
                        else:
                            unselected_features[id].append(clothes_centers[cid].unsqueeze(0))
                selected_features = [torch.cat(selected_features[i], 0).mean(0) for i in sorted(selected_features.keys())]
                for id in list(nested_dict.keys()):
                    if nested_dict[id]:
                        id_infos = nested_dict[id]
                        if len(list(id_infos.keys())) == 1 and silhouette_score[id] > threshold:
                            select_clt = list(id_infos.keys())[0]
                        else:
                            a = selected_features[id].unsqueeze(0)
                            b = torch.cat(unselected_features[id], 0)
                            dist = euclidean_dist(a.float(), b.float()).half()
                            value, index = torch.sort(dist, dim=1, descending=True)
                            index = index.squeeze(0).cpu().numpy()[0]
                            select_clt = list(id_infos.keys())[index]

                        data.extend(id_infos[select_clt])
                        id_infos.pop(select_clt)
                        selected_clothes.append(select_clt)
                    else:
                        continue

            del clothes_centers
            del selected_features
            del unselected_features

        logger.info(len(data))

        train_loader = make_dataloader2(cfg, data)

        if cfg.DATASETS.NAMES in ['ltcc', 'prcc', 'vcclothes', 'vcclothes_sc', 'vcclothes_cc', 'deepchange', 'last']:
            for n_iter, (img, vid, target_cam, target_clt, _) in enumerate(train_loader):
                optimizer.zero_grad()
                optimizer_center.zero_grad()

                img = img.to(device)

                target = vid.to(device)
                target_clt = target_clt.to(device)

                if cfg.MODEL.SIE_CAMERA:
                    target_cam = target_cam.to(device)
                else:
                    target_cam = None
                if cfg.MODEL.SIE_VIEW:
                    target_view = target_view.to(device)
                else:
                    target_view = None

                weights = 1 - weight_pid[target] * weight_cid[target_clt]

                with amp.autocast(enabled=True):

                    score, feat, img_features = model(x=img, label=target, cam_label=target_cam,
                                                            view_label=target_view)

                    ID_LOSS = [ce_criterion(scor, target, weights) for scor in score[0:]]
                    ID_LOSS = sum(ID_LOSS) * cfg.MODEL.ID_LOSS_WEIGHT

                    TRI_LOSS = [triplet(f, target, weights)[0] for f in feat[0:]]
                    TRI_LOSS = sum(TRI_LOSS)

                    logits = img_features @ text_features.t()

                    I2TLOSS = ce_criterion(logits, target, weights)
                    loss = ID_LOSS + TRI_LOSS + I2TLOSS

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()

                if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                    for param in center_criterion.parameters():
                        param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                    scaler.step(optimizer_center)
                    scaler.update()

                for pid, clt, prediction in zip(target, target_clt, score[1].max(1)[1]):
                    class_accuracies_pid[pid.item()].append(1 if pid == prediction else 0)
                    class_accuracies_cid[clt.item()].append(1 if pid == prediction else 0)

                acc = (logits.max(1)[1] == target).float().mean()
                acc1 = (score[1].max(1)[1] == target).float().mean()

                loss_meter.update(loss.item(), img.shape[0])
                acc_meter.update(acc, 1)
                torch.cuda.synchronize()
                if (n_iter + 1) % log_period == 0:
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                                    .format(epoch, (n_iter + 1), len(train_loader),
                                            loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))
        else:
            pass

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank()==0:
                    model.eval()
                    if cfg.DATASETS.NAMES == 'prcc':
                        queryloader_same, queryloader_diff = queryloader[0], queryloader[1]
                        rank1 = test_prcc(model, queryloader_same, queryloader_diff, galleryloader)
                    else:
                        rank1 = test(model, queryloader, galleryloader)

                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("Rank-1: {:.1%}".format(rank1))
                    if rank1 > best_rank1:
                        best_rank1 = rank1
                        torch.save(model.state_dict(), best_model_path)
                        logger.info("Saved best Rank-1 model to {} ({:.1%})".format(best_model_path, best_rank1))
                    torch.cuda.empty_cache()
                else:
                    pass
            else:
                model.eval()
                if cfg.DATASETS.NAMES == 'prcc':
                    queryloader_same, queryloader_diff = queryloader[0], queryloader[1]
                    rank1 = test_prcc(model, queryloader_same, queryloader_diff, galleryloader)
                else:
                    rank1 = test(model, queryloader, galleryloader)

                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("Rank-1: {:.1%}".format(rank1))
                if rank1 > best_rank1:
                    best_rank1 = rank1
                    torch.save(model.state_dict(), best_model_path)
                    logger.info("Saved best Rank-1 model to {} ({:.1%})".format(best_model_path, best_rank1))
                torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    logger.info("Best Rank-1: {:.1%}".format(best_rank1))
    logger.info("Best model saved to: {}".format(best_model_path))
    print(cfg.OUTPUT_DIR)
