import argparse
import random
import copy
import tqdm
import sys
import json
import os
import time
import contextlib
from pathlib import Path
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel

# from engine import *
from models.criterion import post_process, get_pseudo_labels
from utils.box_utils import box_cxcywh_to_xyxy, convert_to_xywh
from build_modules import *
from datasets.augmentations import train_trans, val_trans, strong_trans
from utils import get_rank, init_distributed_mode, resume_and_load, save_ckpt, selective_reinitialize

import numpy as np
import skimage.io
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from detectron2.structures import BoxMode

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Intersections', 'scripts'))
from evaluation import evaluate_cocovalid, evaluate_masked

video_id_list = ['001', '003', '005', '006', '007', '008', '009', '011', '012', '013', '014', '015', '016', '017', '019', '020', '023', '025', '027', '034', '036', '039', '040', '043', '044', '046', '048', '049', '050', '051', '053', '054', '055', '056', '058', '059', '060', '066', '067', '068', '069', '070', '071', '073', '074', '075', '076', '077', '080', '085', '086', '087', '088', '090', '091', '092', '093', '094', '095', '098', '099', '105', '108', '110', '112', '114', '115', '116', '117', '118', '125', '127', '128', '129', '130', '131', '132', '135', '136', '141', '146', '148', '149', '150', '152', '154', '156', '158', '159', '160', '161', '164', '167', '169', '170', '171', '172', '175', '178', '179']
thing_classes = ['person', 'vehicle']
bbox_rgbs = ['#FF0000', '#0000FF']


def get_args_parser(parser):
    # Model Settings
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--pos_encoding', default='sine', type=str)
    parser.add_argument('--num_classes', default=9, type=int)
    parser.add_argument('--num_queries', default=300, type=int)
    parser.add_argument('--num_feature_levels', default=4, type=int)
    parser.add_argument('--with_box_refine', default=False, type=bool)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--num_encoder_layers', default=6, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--feedforward_dim', default=1024, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    # Optimization hyperparameters
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--eval_batch_size', default=1, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj', default=2e-5, type=float)
    parser.add_argument('--sgd', default=False, type=bool)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--clip_max_norm', default=0.5, type=float, help='gradient clipping max norm')
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--epoch_lr_drop', default=40, type=int)
    # Loss coefficients
    parser.add_argument('--teach_box_loss', default=False, type=bool)
    parser.add_argument('--coef_class', default=2.0, type=float)
    parser.add_argument('--coef_boxes', default=5.0, type=float)
    parser.add_argument('--coef_giou', default=2.0, type=float)
    parser.add_argument('--coef_target', default=1.0, type=float)
    parser.add_argument('--coef_domain', default=1.0, type=float)
    parser.add_argument('--coef_domain_bac', default=0.3, type=float)
    parser.add_argument('--coef_mae', default=1.0, type=float)
    parser.add_argument('--alpha_focal', default=0.25, type=float)
    parser.add_argument('--alpha_ema', default=0.9996, type=float)

    # Dataset parameters
    parser.add_argument('--dataset', type=str)

    # Retraining parameters
    parser.add_argument('--epoch_retrain', default=40, type=int)
    parser.add_argument('--keep_modules', default=["decoder"], type=str, nargs="+")
    # MAE parameters
    parser.add_argument('--mae_layers', default=[2], type=int, nargs="+")
    parser.add_argument('--mask_ratio', default=0.8, type=float)
    parser.add_argument('--epoch_mae_decay', default=10, type=float)
    # Dynamic threshold (DT) parameters
    parser.add_argument('--threshold', default=0.3, type=float)
    parser.add_argument('--alpha_dt', default=0.5, type=float)
    parser.add_argument('--gamma_dt', default=0.9, type=float)
    parser.add_argument('--max_dt', default=0.45, type=float)
    # mode settings
    parser.add_argument("--mode", default="single_domain", type=str,
                        help="'single_domain' for single domain training, "
                             "'cross_domain_mae' for cross domain training with mae, "
                             "'teaching' for teaching process, 'eval' for evaluation only.")
    # Other settings
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', default='./output', type=str)
    parser.add_argument('--random_seed', default=8008, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--flush', default=True, type=bool)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--distributed", default=False, type=bool)

    parser.add_argument("--opt", type=str)


@torch.no_grad()
def evaluate(model, data_loader_val, device):
    start_time = time.time()
    model.eval()
    results_all = {}
    for i, (images, masks, annotations) in tqdm.tqdm(enumerate(data_loader_val), ascii=True, total=len(data_loader_val)):
        # To CUDA
        images = images.to(device)
        masks = masks.to(device)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        # Forward
        out = model(images, masks)
        logits_all, boxes_all = out['logits_all'], out['boxes_all']
        # Get pseudo labels
        results = get_pseudo_labels(logits_all[-1], boxes_all[-1], [0.4 for _ in range(9)])
        for anno, res in zip(annotations, results):
            image_id = anno['image_id'].item()
            orig_image_size = anno['orig_size']
            img_h, img_w = orig_image_size.unbind(0)
            scale_fct = torch.stack([img_w, img_h, img_w, img_h])
            converted_boxes = convert_to_xywh(box_cxcywh_to_xyxy(res['boxes'] * scale_fct))
            converted_boxes = converted_boxes.detach().cpu().numpy().tolist()
            for label, box in zip(res['labels'].detach().cpu().numpy().tolist(), converted_boxes):
                pseudo_anno = {
                    'id': 0,
                    'image_id': image_id,
                    'category_id': label,
                    'iscrowd': 0,
                    'area': box[-2] * box[-1],
                    'bbox': box
                }
        orig_image_sizes = torch.stack([anno['orig_size'] for anno in annotations], dim=0)
        results = post_process(logits_all[-1], boxes_all[-1], orig_image_sizes, 100)
        results = {anno['image_id'].item(): res for anno, res in zip(annotations, results)}
        results_all.update({
            i: [{'bbox': list(map(float, b)), 'bbox_mode': BoxMode.XYXY_ABS, 'segmentation': [], 'category_id': int(l) - 1, 'score': float(s)} for s, l, b in zip(results[i]['scores'], results[i]['labels'], results[i]['boxes'])]
            for i in results
        })
    return results_all


def eval_single(args):
    # for key, value in args.__dict__.items():
    #     print(key, value, flush=args.flush)
    # Build model
    device = torch.device(args.device)
    model = build_model(args, device)
    model = resume_and_load(model, args.resume, device)
    # print(model, flush=True)

    if args.dataset == 'mscoco':
        with open(os.path.join('../MSCOCO2017', 'annotations', 'instances_val2017.json'), 'r') as fp:
            dataset = json.load(fp)
        images = [{'file_name': im['file_name'], 'image_id': im['id'], 'height': im['height'], 'width': im['width'], 'annotations': []} for im in dataset['images']]
        dataset = CocoStyleDataset(root_dir='../MSCOCO2017', dataset_name='mscoco', domain='target', split='val', transforms=val_trans)
        batch_sampler = build_sampler(args, dataset, 'val')
        data_loader = DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=CocoStyleDataset.collate_fn, num_workers=args.num_workers)
        detections = evaluate(model, data_loader, device)
        images_detections = []
        for im in images:
            if im['image_id'] in detections:
                im['annotations'] = detections[im['image_id']]
                images_detections.append(im)
        # for im in images_detections:
        #     im_arr = skimage.io.imread(os.path.join('../MSCOCO2017/images/val2017', im['file_name']))
        #     im_arr = Image.fromarray(im_arr, 'RGB')
        #     draw = ImageDraw.Draw(im_arr)
        #     for ann in im['annotations']:
        #         if ann['score'] < 0.5:
        #             continue
        #         x1, y1, x2, y2 = ann['bbox']
        #         draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill=bbox_rgbs[ann['category_id']], width=3)
        #     plt.figure()
        #     plt.imshow(np.array(im_arr))
        #     plt.show()
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            resuls_AP = evaluate_cocovalid('../MSCOCO2017', images_detections)
    elif args.dataset in video_id_list:
        dataset = CocoStyleDatasetScenes100(args.dataset, 'val', val_trans)
        batch_sampler = build_sampler(args, dataset, 'val')
        data_loader = DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=CocoStyleDatasetScenes100.collate_fn, num_workers=args.num_workers)
        detections = evaluate(model, data_loader, device)
        images_detections = copy.deepcopy(dataset.annotations_d2)
        assert len(images_detections) == len(detections)
        for im in images_detections:
            im['annotations'] = detections[im['image_id']]
        # for im in images_detections:
        #     im_arr = skimage.io.imread(os.path.join(dataset.root_dir, 'unmasked', im['file_name']))
        #     im_arr = Image.fromarray(im_arr, 'RGB')
        #     draw = ImageDraw.Draw(im_arr)
        #     for ann in im['annotations']:
        #         if ann['score'] < 0.5:
        #             continue
        #         x1, y1, x2, y2 = ann['bbox']
        #         draw.line(((x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)), fill=bbox_rgbs[ann['category_id']], width=3)
        #     plt.figure()
        #     plt.imshow(np.array(im_arr))
        #     plt.show()
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            resuls_AP = evaluate_masked(args.dataset, images_detections)

    print(   '             %s' % '/'.join(resuls_AP['metrics']))
    for c in sorted(resuls_AP['results'].keys()):
        print('%10s  ' % c, end='')
        print('/'.join(map(lambda x: '%05.2f' % (x * 100), resuls_AP['results'][c])))
    return resuls_AP


def eval_base_scenes100(args):
    device = torch.device(args.device)
    model = build_model(args, device)
    model = resume_and_load(model, args.resume, device)
    results_all = {}

    for video_id in video_id_list:
        dataset = CocoStyleDatasetScenes100(video_id, 'val', val_trans)
        batch_sampler = build_sampler(args, dataset, 'val')
        data_loader = DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=CocoStyleDatasetScenes100.collate_fn, num_workers=args.num_workers)
        detections = evaluate(model, data_loader, device)
        images_detections = copy.deepcopy(dataset.annotations_d2)
        assert len(images_detections) == len(detections)
        for im in images_detections:
            im['annotations'] = detections[im['image_id']]
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            resuls_AP = evaluate_masked(video_id, images_detections)

        print(video_id)
        print(   '             %s' % '/'.join(resuls_AP['metrics']))
        for c in sorted(resuls_AP['results'].keys()):
            print('%10s  ' % c, end='')
            print('/'.join(map(lambda x: '%05.2f' % (x * 100), resuls_AP['results'][c])))
        results_all[video_id] = resuls_AP
    with open('results_AP_base.json', 'w') as fp:
        json.dump(results_all, fp)


def apg_scenes100(args):
    import glob
    import matplotlib.pyplot as plt

    ckpts = sorted(glob.glob(os.path.join(args.resume, '*.pth')))
    ckpts = [(os.path.basename(f).split('.')[0], f) for f in ckpts]
    print('%d presented video checkpoints:' % len(ckpts))
    print(' '.join([v for (v, _) in ckpts]))
    print('missing:')
    print(' '.join(sorted(list(set(video_id_list) - set([v for (v, _) in ckpts])))))

    device = torch.device(args.device)
    model = build_model(args, device)
    with open('results_AP_base.json', 'r') as fp:
        base_AP = json.load(fp)
    results_file = os.path.join(args.resume, 'results_AP')

    results = {}
    for video_id, f in ckpts:
        model = resume_and_load(model, f, device)
        dataset = CocoStyleDatasetScenes100(video_id, 'val', val_trans)
        batch_sampler = build_sampler(args, dataset, 'val')
        data_loader = DataLoader(dataset=dataset, batch_sampler=batch_sampler, collate_fn=CocoStyleDatasetScenes100.collate_fn, num_workers=args.num_workers)
        detections = evaluate(model, data_loader, device)
        images_detections = copy.deepcopy(dataset.annotations_d2)
        assert len(images_detections) == len(detections)
        for im in images_detections:
            im['annotations'] = detections[im['image_id']]
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            results[video_id] = evaluate_masked(video_id, images_detections)

    videos = sorted(list(results.keys()))
    categories = ['person', 'vehicle', 'overall', 'weighted']
    improvements = {c: [] for c in categories}
    for video_id in videos:
        AP1, AP2 = base_AP[video_id]['results'], results[video_id]['results']
        for cat in categories:
            improvements[cat].append([AP2[cat][0] - AP1[cat][0], AP2[cat][1] - AP1[cat][1]])
    for cat in categories:
        improvements[cat] = np.array(improvements[cat]) * 100.0
    xs = np.arange(0, len(videos), 1)
    fig, axes = plt.subplots(2, 2, figsize=(28, 16))
    axes = axes.reshape(-1)
    for i in range(0, len(categories)):
        axes[i].plot([-1, xs.max() + 1], [0, 0], 'k-')
        axes[i].plot(xs, improvements[categories[i]][:, 0], 'r.-')
        axes[i].plot(xs, improvements[categories[i]][:, 1], 'b.-')
        axes[i].legend(['0', 'mAP %.4f' % improvements[categories[i]][:, 0].mean(), 'AP50 %.4f' % improvements[categories[i]][:, 1].mean()])
        axes[i].set_xticks(xs)
        axes[i].set_xticklabels(videos, rotation='vertical', fontsize=10)
        axes[i].set_xlim(0, xs.max())
        # axes[i].set_ylim(-3, 3)
        axes[i].set_ylabel('AP improvement (0-100)')
        axes[i].grid(True)
        axes[i].set_title('<%s>' % (categories[i]))
    # plt.tight_layout()
    plt.suptitle(results_file)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    plt.savefig(results_file + '.pdf')
    plt.close()
    print('saved to:', results_file)


if __name__ == '__main__':
    # Parse arguments
    parser_main = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    get_args_parser(parser_main)
    args = parser_main.parse_args()
    if args.opt == 'eval':
        eval_single(args)
    elif args.opt == 'base':
        eval_base_scenes100(args)
    elif args.opt == 'compare':
        apg_scenes100(args)

'''
python eval.py --opt eval --num_classes 3 --resume r50_model_best.pth --dataset 001
python eval.py --opt base --num_classes 3 --resume r50_model_best.pth

python eval.py --opt compare --num_classes 3 --resume /mnt/f/intersections_results/cvpr24/mrt/cross
'''