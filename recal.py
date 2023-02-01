from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from args import argument_parser, dataset_kwargs, optimizer_kwargs, lr_scheduler_kwargs
from vehiclereid.data_manager import ImageDataManager
from vehiclereid import models
from vehiclereid.losses import CrossEntropyLoss, TripletLoss, DeepSupervision
from vehiclereid.utils.iotools import check_isfile
from vehiclereid.utils.avgmeter import AverageMeter
from vehiclereid.utils.loggers import Logger, RankLogger
from vehiclereid.utils.torchtools import count_num_param, accuracy, \
    load_pretrained_weights, save_checkpoint, resume_from_checkpoint
from vehiclereid.utils.visualtools import visualize_ranked_results
from vehiclereid.utils.generaltools import set_random_seed
from vehiclereid.eval_metrics import evaluate
from vehiclereid.optimizers import init_optimizer
from vehiclereid.lr_schedulers import init_lr_scheduler
import xml.etree.ElementTree as ET

# global variables
parser = argument_parser()
args = parser.parse_args()


def compute_distance(query_features, test_features):
    m, n = query_features.size(0), test_features.size(0)
    distmat = torch.pow(query_features, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(test_features, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, query_features, test_features.t())
    distmat = distmat.cpu().numpy()
    return distmat


def compute_weight(args, epoch):
    alpha = 0.0
    if epoch >= args.t1:
        alpha = (epoch - args.t1) / (args.t2 - args.t1) * args.af
        if epoch > args.t2:
            alpha = args.af
    return alpha


def load_labels_from_veri(path='datasets/VeRi/test_label.xml'):
    encoding = 'gb2312'
    # xmlp = ET.XMLParser(encoding=encoding)
    # f = ET.parse('a.xml',parser=xmlp)
    ids = {}
    print(path)
    root = ET.parse(path)
    prefix = 333 if 'Sim' in path else 0
    print(path)
    print('prefix: ', prefix)
    for item in root.find('Items').findall('Item'):
        attr = item.attrib
        ids[attr['imageName']] = {
            'type': int(attr['typeID'])-1,
            'color': int(attr['colorID'])-1
        }
    return ids



def compute_alpha(i, epoch, len_dataloader, n_epoch):
    p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
    alpha = 2. / (1. + np.exp(-10 * p)) - 1

    return alpha


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):

    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    original_dist = np.concatenate(
      [np.concatenate([q_q_dist, q_g_dist], axis=1),
       np.concatenate([q_g_dist.T, g_g_dist], axis=1)],
      axis=0)
    original_dist = np.power(original_dist, 2).astype(np.float32)
    original_dist = np.transpose(1. * original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[0] + q_g_dist.shape[1]
    all_num = gallery_num

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = 1.*weight/np.sum(weight)
    original_dist = original_dist[:query_num,]
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)


    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2.-temp_min)

    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist


def main():
    global args

    set_random_seed(args.seed)
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False
    log_name = 'log_test.txt' if args.evaluate else 'log_train.txt'
    sys.stdout = Logger(osp.join(args.save_dir, log_name))
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    print('==========\nArgs:{}\n=========='.format(args))

    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        warnings.warn('Currently using CPU, however, GPU is highly recommended')

    print('Initializing image data manager')
    dm = ImageDataManager(use_gpu, **dataset_kwargs(args))
    trainloader, testloader_dict = dm.return_dataloaders()

    print('Initializing model: {}'.format(args.arch))
    model = models.init_model(name=args.arch, num_classes=dm.num_train_pids, loss={'xent', 'htri'},
                              pretrained=not args.no_pretrained, use_gpu=use_gpu, include_sim=args.include_sim, recal=True)
    print('Model size: {:.3f} M'.format(count_num_param(model)))

    if args.load_weights and check_isfile(args.load_weights):
        load_pretrained_weights(model, args.load_weights)


    model = nn.DataParallel(model).cuda() if use_gpu else model

    criterion_xent = CrossEntropyLoss(num_classes=dm.num_train_pids, use_gpu=use_gpu, label_smooth=args.label_smooth)
    criterion_color = CrossEntropyLoss(num_classes=12, use_gpu=use_gpu, label_smooth=args.label_smooth)
    criterion_orientation = CrossEntropyLoss(num_classes=6, use_gpu=use_gpu, label_smooth=args.label_smooth)
    criterion_type = CrossEntropyLoss(num_classes=11, use_gpu=use_gpu, label_smooth=args.label_smooth)

    criterion_htri = TripletLoss(margin=args.margin)
    optimizer = init_optimizer(model, **optimizer_kwargs(args))
    scheduler = init_lr_scheduler(optimizer, **lr_scheduler_kwargs(args))

    if args.resume and check_isfile(args.resume):
        args.start_epoch = resume_from_checkpoint(args.resume, model, optimizer=optimizer)

    if args.evaluate:
        print('Evaluate only')

        for name in args.target_names:
            print('Evaluating {} ...'.format(name))
            queryloader = testloader_dict[name]['query']
            galleryloader = testloader_dict[name]['gallery']
            distmat = test(model, queryloader, galleryloader, use_gpu, name, return_distmat=True)

            if args.visualize_ranks:
                visualize_ranked_results(
                    distmat, dm.return_testdataset_by_name(name),
                    save_dir=osp.join(args.save_dir, 'ranked_results', name),
                    topk=20
                )
        return

    time_start = time.time()
    ranklogger = RankLogger(args.source_names, args.target_names)
    print('=> Start training')
    '''
    if args.fixbase_epoch > 0:
        print('Train {} for {} epochs while keeping other layers frozen'.format(args.open_layers, args.fixbase_epoch))
        initial_optim_state = optimizer.state_dict()

        for epoch in range(args.fixbase_epoch):
            train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu, fixbase=True)

        print('Done. All layers are open to train for {} epochs'.format(args.max_epoch))
        optimizer.load_state_dict(initial_optim_state)
    '''
    # one_set = next(iter(trainloader))
    # trainloader = [one_set]
    for epoch in range(args.start_epoch, args.max_epoch):
        train(epoch, model, [criterion_xent, criterion_color, criterion_orientation, criterion_type], criterion_htri, optimizer, trainloader, use_gpu)

        scheduler.step()

        if (epoch + 1) > args.start_eval and args.eval_freq > 0 and (epoch + 1) % args.eval_freq == 0 or (
                epoch + 1) == args.max_epoch:
            print('=> Test')

            ##### skip eval
            # for name in args.target_names:
            #     print('Evaluating {} ...'.format(name))
            #     queryloader = testloader_dict[name]['query']
            #     galleryloader = testloader_dict[name]['gallery']
            #     rank1 = test(model, queryloader, galleryloader, use_gpu)
            #     ranklogger.write(name, epoch + 1, rank1)

        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'arch': args.arch,
            'optimizer': optimizer.state_dict(),
        }, args.save_dir)

    elapsed = round(time.time() - time_start)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print('Elapsed {}'.format(elapsed))
    ranklogger.show_summary()


def train(epoch, model, criterion_xent, criterion_htri, optimizer, trainloader, use_gpu):
    xent_losses = AverageMeter()
    htri_losses = AverageMeter()
    color_losses = AverageMeter()
    orientation_losses = AverageMeter()
    type_losses = AverageMeter()
    accs = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # for name, param in model.named_parameters():
    #     if not 'rclb' in name:
    #         param.requires_grad = False

    criterion_xent, criterion_color, criterion_orientation, criterion_type = criterion_xent

    model.train()
    # if args.include_sim:
    #     assert(any([args.include_color, args.include_ori, args.include_type]))
    # if args.ssl:
    #     assert(all([args.include_sim, len(args.ssl_weights) > 0]))
    # for p in model.parameters():
    #     p.requires_grad = True    # open all layers

    end = time.time()
    for batch_idx, (imgs, pids, camid, _) in enumerate(trainloader):
        data_time.update(time.time() - end)
        camid = camid[1:]
        if use_gpu:
            imgs, pids = imgs.cuda(), pids.cuda()
            if args.include_sim:
                camid = [f.cuda() for f in camid]

        outputs, features, color, orientation, typeid, dis = model(imgs, compute_alpha(batch_idx, epoch, len(trainloader), args.max_epoch))
        if isinstance(outputs, (tuple, list)):
            xent_loss = DeepSupervision(criterion_xent, outputs, pids)
        else:
            if args.include_sim:
                # only select real data
                mask = camid[0] == -1
                mask = mask.float().unsqueeze(-1)
                xent_loss = criterion_xent(outputs, pids.clone().detach(), mask)

                with torch.no_grad():
                    mask = mask.squeeze(1).detach()
                    outputs = outputs[mask.bool()].reshape(-1, outputs.shape[1])
                    accu_pid = pids[mask.bool()]
            else:
                xent_loss = criterion_xent(outputs, pids.clone().detach())

        if isinstance(features, (tuple, list)):
            htri_loss = DeepSupervision(criterion_htri, features, pids)
        else:
            htri_loss = criterion_htri(features, pids.clone().detach())
            if args.include_sim:
                mask = camid[0] == -1
                htri_loss = htri_loss + F.cross_entropy(dis, mask.long())

        if args.include_sim:
            # only select simulation data
            mask = camid[0] > -1
            mask = mask.float().unsqueeze(-1)

            if args.include_color:
                color_loss = criterion_color(color, camid[0], mask)

                if args.ssl and mask_not.sum() > 0:
                    color_pseudo = torch.max(color, dim=1)[1].detach()
                    color_loss = color_loss + compute_weight(args, epoch) * criterion_color(color, color_pseudo, mask_not)
            else:
                color_loss = 0.
            
            if args.include_ori:
                orientation_loss = criterion_orientation(orientation, camid[1], mask) # (F.mse_loss(orientation.squeeze(1), camid[1], reduction='none') * mask).sum() / mask.sum()
                if args.ssl and mask_not.sum() > 0:
                    orientation_pseudo = torch.max(orientation, dim=1)[1].detach()
                    orientation_loss = orientation_loss + compute_weight(args, epoch) * criterion_color(orientation, orientation_pseudo, mask_not)
            else:
                orientation_loss = 0.
            
            if args.include_type:
                type_loss = criterion_type(typeid, camid[2], mask)
                if args.ssl and mask_not.sum() > 0:
                    typeid_pseudo = torch.max(typeid, dim=1)[1].detach()
                    type_loss = type_loss + compute_weight(args, epoch) * criterion_color(typeid, typeid_pseudo, mask_not)
            else:
                type_loss = 0.
        else:
            color_loss, orientation_loss, type_loss = 0., 0., 0.
        
        loss = args.lambda_xent * xent_loss + args.lambda_htri * htri_loss + color_loss + orientation_loss + type_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        if args.include_sim:
            real_count = (1 - mask).sum().item()
            sim_count = mask.sum().item()
            real_count = 1 if real_count == 0 else real_count
            sim_count = 1 if sim_count == 0 else sim_count

        htri_losses.update(htri_loss.item(), pids.size(0))
        # if (args.include_sim and args.ssl):
        #     xent_losses.update(xent_loss.item(), pids.size(0))
        #     color_losses.update(color_loss.item() if torch.is_tensor(color_loss) else 0., pids.size(0))
        #     orientation_losses.update(orientation_loss.item() if torch.is_tensor(orientation_loss) else 0., pids.size(0))
        #     type_losses.update(type_loss.item() if torch.is_tensor(type_loss) else 0., pids.size(0))
        #     accs.update(accuracy(outputs, accu_pid)[0])

        if args.include_sim: # include sim
            xent_losses.update(xent_loss.item(), real_count)
            color_losses.update(color_loss.item() if torch.is_tensor(color_loss) else 0., sim_count)
            orientation_losses.update(orientation_loss.item() if torch.is_tensor(orientation_loss) else 0., sim_count)
            type_losses.update(type_loss.item() if torch.is_tensor(type_loss) else 0., sim_count)
            accs.update(accuracy(outputs, accu_pid)[0])

        else: # not include sim
            xent_losses.update(xent_loss.item(), pids.size(0))
            color_losses.update(0, pids.size(0))
            orientation_losses.update(0, pids.size(0))
            type_losses.update(0, pids.size(0))
            accs.update(accuracy(outputs, pids)[0])
        
        # if args.include_sim:
        #     xent_losses.update(xent_loss.item(), real_count)
        # else:
        #     xent_losses.update(xent_loss.item(), pids.size(0))
        # htri_losses.update(htri_loss.item(), pids.size(0))
        # if args.include_sim:
        #     color_losses.update(color_loss.item() if torch.is_tensor(color_loss) else 0., sim_count)
        #     orientation_losses.update(orientation_loss.item() if torch.is_tensor(orientation_loss) else 0., sim_count)
        #     type_losses.update(type_loss.item() if torch.is_tensor(type_loss) else 0., sim_count)
        # else:
        #     color_losses.update(0, pids.size(0))
        #     orientation_losses.update(0, pids.size(0))
        #     type_losses.update(0, pids.size(0))

        # if (args.include_sim and accu_pid.shape[0] > 0) or not args.include_sim:
        #     accs.update(accuracy(outputs, pids if not args.include_sim else accu_pid)[0])

        if (batch_idx + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.4f} ({data_time.avg:.4f})\t'
                    'Xent {xent.val:.4f} ({xent.avg:.4f})\t'
                    'Htri {htri.val:.4f} ({htri.avg:.4f})\t'
                    'color {color.val:.4f} ({color.avg:.4f})\t'
                    'orientation {orientation.val:.4f} ({orientation.avg:.4f})\t'
                    'type {type.val:.4f} ({type.avg:.4f})\t'
                    'Acc {acc.val:.2f} ({acc.avg:.2f})\t'.format(
                epoch + 1, batch_idx + 1, len(trainloader),
                batch_time=batch_time,
                data_time=data_time,
                xent=xent_losses,
                htri=htri_losses,
                color=color_losses,
                orientation=orientation_losses,
                type=type_losses,
                acc=accs
            ))

        end = time.time()


def test(model, queryloader, galleryloader, use_gpu, target_name, ranks=[1, 5, 10, 20], return_distmat=False):
    batch_time = AverageMeter()
    color_map = [
        'yellow',
        'orange',
        'green',
        'gray',
        'red',
        'blue',
        'white',
        'golden',
        'brown',
        'black',
        'purple',
        'pink'
    ]
    type_map = [
        'sedan',
        'suv',
        'van',
        'hatchback',
        'mpv',
        'pickup', 
        'bus', 
        'truck', 
        'estate', 'sportscar', 'RV'
    ]

    orientation_map = [
        '0 - 60',
        '60 - 120',
        '120 - 180',
        '180 - 240',
        '240 - 300',
        '300 - 360']

    veri_color = [
        'yellow', 'orange', 'green', 'gray', 'red', 'blue',
        'white',
        'golden',
        'brown',
        'black'
    ]

    veri_type = [
        'sedan',
        'suv',
        'van',
        'hatchback',
        'mpv',
        'pickup',
        'bus',
        'truck',
        'estate'
    ]
    veri_data = load_labels_from_veri()

    model.eval()
    color_q, type_q, orientation_q = [], [], []
    with torch.no_grad():
        qf, q_pids, q_camids = [], [], []
        for batch_idx, (imgs, pids, camids, _) in enumerate(queryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features, color_f, type_f, orientation_f = model(imgs, 0)
            # print(features.shape)
            batch_time.update(time.time() - end)
            # color_f = F.softmax(color_f, dim=1)
            # type_f = F.softmax(type_f, dim=1)
            # orientation_f = F.softmax(orientation_f, dim=1)
            # features = torch.cat([features, color_f, type_f], dim=1)

            # features = features.data.cpu()
            qf.append(features)
            q_pids.extend(pids)
            q_camids.extend(camids)
            # color_q.extend(color_f)
            # type_q.extend(type_f)
            # orientation_q.extend(orientation_f)
            # print(torch.max(color_f, dim=1))
            # color_q.extend([color_map[t] for t in torch.argmax(color_f, dim=1)])
            # color_q.extend([[color_map[i], v]  for v, i in zip(torch.max(color_f, dim=1)[0], torch.max(color_f, dim=1)[1])])
            # type_q.extend([type_map[t] for t in torch.argmax(type_f, dim=1)])
            # orientation_q.extend([orientation_map[t] for t in torch.argmax(orientation_f, dim=1)])
        qf = torch.cat(qf, 0)
        q_pids = np.asarray(q_pids)
        q_camids = np.asarray(q_camids)
        # for d, (i, j, k) in enumerate(zip(color_q, type_q, orientation_q)):
        #     print(f'imag {d}: color: {i}, type: {j}, orientation: {k}')
#        return

        print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        gf, g_pids, g_camids = [], [], []
        color_g, type_g = [], []
        total = 0
        correct_type = 0
        correct_color = 0
        for batch_idx, (imgs, pids, camids, files) in enumerate(galleryloader):
            if use_gpu:
                imgs = imgs.cuda()

            end = time.time()
            features, color, typeid, _ = model(imgs, 0)
            batch_time.update(time.time() - end)
            # color_f = F.softmax(color_f, dim=1)
            # type_f = F.softmax(type_f, dim=1)
            # print(files)
            # color = [color_map[t] for t in torch.argmax(color, dim=1)]
            # typeid = [type_map[t] for t in torch.argmax(typeid, dim=1)]
            # for c, t, gt in zip(color, typeid, files):
            #     gt = os.path.basename(gt)
            #     gt = veri_data[gt]
            #     gt_color, gt_type = gt['color'], gt['type']
            #     # print(c)
            #     # print(gt_color)
            #     if veri_color[gt_color] == c:
            #         correct_color += 1
            #     if veri_type[gt_type] == t:
            #         correct_type += 1
            # total += len(files)
            # print(f'acc type: {correct_type/total}, acc color: {correct_color/total}')
                # print(f'{c}, {t}, gt: {veri_color[gt_color]}, {veri_type[gt_type]}')
            
            # print(len(files), ' ', len(color))
            # features = torch.cat([features, color_f, type_f], dim=1)

            # features = features.data.cpu()
            gf.append(features)
            g_pids.extend(pids)
            g_camids.extend(camids)
            # color_g.extend(color_f)
            # type_g.extend(type_f)
        gf = torch.cat(gf, 0)
        test_image_id = np.array([str(g.item()) for g in g_pids])
        g_pids = np.asarray(g_pids)
        g_camids = np.asarray(g_camids)

        print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    print('=> BatchTime(s)/BatchSize(img): {:.3f}/{}'.format(batch_time.avg, args.test_batch_size))

    distmat = compute_distance(qf, gf)
    # distmat = re_ranking(
    #     compute_distance(qf, gf),
    #     compute_distance(qf, qf),
    #     compute_distance(gf, gf)
    # )
    if target_name == 'aicity':
        distmat_argsort = distmat.argsort(axis=1, kind='mergesort')[:, :100]
        if not os.path.exists(os.path.join(args.save_dir, 'viz')):
            os.mkdir(os.path.join(args.save_dir, 'viz'))
        f = open(os.path.join(args.save_dir, 'track2.txt'), 'w')
        for index, (dm, distance) in enumerate(zip(distmat, distmat_argsort)):
            top_n = test_image_id[distance.tolist()]
            f.write(' '.join(top_n) + '\n')
            with open(os.path.join(args.save_dir, 'viz', str(index+1).zfill(6) + '.txt'), 'w') as vizf:
                for t in top_n[:50]:
                    vizf.write(t + '\n')
        f.close()
    if target_name != 'aicity':
        print('Computing CMC and mAP')
        # cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids, args.target_names)
        cmc, mAP = evaluate(distmat, q_pids, g_pids, q_camids, g_camids)

        print('Results ----------')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}: {:.1%}'.format(r, cmc[r - 1]))
        print('------------------')

        if return_distmat:
            return distmat
        return cmc[0]


if __name__ == '__main__':
    main()
