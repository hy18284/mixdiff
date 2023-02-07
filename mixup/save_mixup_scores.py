import random
import copy
from collections import defaultdict
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import numpy as np
import wandb
from jsonargparse import (
    ArgumentParser,
    ActionConfigFile,
)
from pandas import DataFrame

from .ood_score_calculators.ood_score_calculator import OODScoreCalculator
from .ood_datamodules.base_ood_datamodule import BaseOODDataModule


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--r', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb_name', type=str, default='mixup_v1')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--abs', action='store_true')
    parser.add_argument('--top_1', action='store_true')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--max_samples', type=int, default=None, required=False)
    parser.add_argument('--truncate_by_max_samples', type=bool, default=False)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_subclass_arguments(OODScoreCalculator, 'score_calculator')
    parser.add_subclass_arguments(BaseOODDataModule, 'datamodule')
    parser.add_argument('--config', action=ActionConfigFile)

    args = parser.parse_args()
    classes = parser.instantiate_classes(args)
    return args, classes


if __name__ == '__main__':
    args, classes = parse_args()
    score_calculator: OODScoreCalculator = classes['score_calculator']
    datamodule: BaseOODDataModule = classes['datamodule']

    torch.set_grad_enabled(False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    N, M, R = args.n, args.m, args.r
    batch_size = N

    device = torch.device(args.device)
    score_calculator.load_model('ViT-B/32', device)

    wandb.init(
        config=args,
        name=f'{args.wandb_name}_{score_calculator}_{datamodule}',
        project='ZOC',
    )
    wandb.config.method = str(score_calculator)
    wandb.config.dataset = str(datamodule)
     
    aurocs = []
    for j, (seen_labels, seen_idx, given_images) in enumerate(
        datamodule.get_splits(n_samples_per_class=M, seed=args.seed)
    ):
        seen_idx = seen_idx.to(device) 
        given_images = given_images.to(device)

        wandb.log({
            'seen_classes': seen_labels,
            'seen_idx': seen_idx.tolist(),
        })

        r_start = 1.0 / (R + 1)
        r_end = 1 - r_start
        rates = torch.linspace(r_start, r_end, R, device=device)
        print(rates)
        targets = []
        scores = []
        table = defaultdict(list)

        loader = datamodule.construct_loader(
            batch_size=batch_size, 
            shuffle=args.shuffle,
        )
        score_calculator.on_eval_start(copy.deepcopy(seen_labels))
        cur_num_samples = 0
        
        for i, (images, labels) in enumerate(tqdm(loader)):
            N, C, H, W = images.size()
            NC = len(seen_labels)
            # (N, C, H, W)
            images = images.to(device)
            labels = labels.to(device)
            image_kwargs = score_calculator.process_images(images)

            if score_calculator.utilize_mixup:
                chosen_images = score_calculator.select_given_images(
                    given_images, 
                    **image_kwargs
                )

                def calculate_known_logits(chosen_images, images):
                    # (N, M, C, H, W) -> (N, N, M, C, H, W)
                    chosen_images = chosen_images.expand(N, -1, -1, -1, -1, -1)
                    # (N, N, M, C, H, W) -> (M, N, N, C, H, W)
                    chosen_images = chosen_images.permute(2, 1, 0, 3, 4, 5)

                    # (N, C, H, W) -> (M, N, N, C, H, W)
                    images_m = images.expand(M, N, -1, -1, -1, -1)

                    # (M, N, N, C, H, W, 1) * (R) -> (M, N, N, C, H, W, R)
                    chosen_images = chosen_images.unsqueeze(-1) * rates 
                    images_m = images_m.unsqueeze(-1) * (1 - rates)
                    known_mixup = chosen_images + images_m
                    del chosen_images
                    del images_m
                    known_mixup = known_mixup.permute(0, 1, 2, 6, 3, 4, 5)
                    known_mixup = known_mixup.reshape(M * N * N * R, C, H, W)

                    known_logits = score_calculator.process_mixup_images(known_mixup)

                    NS = known_logits.size(-1)
                    # (M * N * N * R, NS) -> (M, N, N, R, NS)
                    known_logits = known_logits.view(M, N, N, R, -1)
                    # (M, N, N, R, NS) -> (N, N, R, NS)
                    known_logits = torch.mean(known_logits, dim=0)
                    known_logits = known_logits.view(-1, NS)
                    return known_logits
                
                known_logits = calculate_known_logits(chosen_images, images)

                def calculate_unknown_logits(images):
                    # (N, C, H, W) -> (N, N, C, H, W) -> (N, N, C, H, W, R)
                    images_n_1 = images.expand(N, -1, -1, -1, -1)
                    images_n_1 = images_n_1.permute(1, 0, 2, 3, 4).unsqueeze(-1) * rates
                    images_n_2 = images.expand(N, -1, -1, -1, -1).unsqueeze(-1) * (1 - rates)
                    # (N, C, H, W) -> (M, N, C, H, W)
                    unknown_mixup = images_n_1 + images_n_2
                    unknown_mixup = unknown_mixup.permute(0, 1, 5, 2, 3, 4)
                    unknown_mixup = unknown_mixup.reshape(N * N * R, C, H, W)
                    unknown_logits = score_calculator.process_mixup_images(unknown_mixup)
                    return unknown_logits

                unknown_logits = calculate_unknown_logits(images)
                
                dists = score_calculator.calculate_diff(known_logits, unknown_logits)

                if args.abs:
                    dists = torch.abs(dists)

                dists = dists.view(N, N, R)
                dists = torch.mean(dists, dim=-1)
                mask = torch.ones_like(dists)
                mask.fill_diagonal_(0.0)
                dists = dists * mask

                if args.top_1:
                    abs_dists = torch.abs(dists)
                    abs_max_idx = torch.argmax(abs_dists, dim=-1)
                    dists = dists[torch.arange(dists.size(0)), abs_max_idx]
                else:
                    dists = torch.sum(dists, dim=-1) / torch.sum(mask > 0.5, dim=-1)
            
            base_scores = score_calculator.calculate_base_scores(**image_kwargs)
            if score_calculator.utilize_mixup:
                table['mixdiff'] += dists.tolist()
                dists = base_scores + args.gamma * dists
            else:
                dists = base_scores
            
            targets += [int(label not in seen_idx) for label in labels]
            scores += dists.tolist()

            table['base_score'] += base_scores.tolist()
            table['label'] += labels.tolist()
            table['is_ood'] += [label not in seen_idx for label in labels]
            table['batch_idx'] += [i] * labels.size(0)
            max_idx = torch.argmax(image_kwargs['logits'], dim=-1)
            chosen_img_labels = seen_idx[max_idx]
            table['chosen_img_label'] += chosen_img_labels.tolist()
            table['base+gamma*mixdiff'] += dists.tolist()

            cur_num_samples += N
            if args.max_samples is not None and cur_num_samples >= args.max_samples:
                break
            
        score_calculator.on_eval_end()

        if args.truncate_by_max_samples:
            for key in table.keys():
                table[key] = table[key][:args.max_samples]
            targets = targets[:args.max_samples]
            scores = scores[:args.max_samples]

        ood_scores = [score for score, target in zip(scores, targets) if target == 1]
        id_scores = [score for score, target in zip(scores, targets) if target == 0]
        ood_mean = np.mean(ood_scores)
        id_mean = np.mean(id_scores)
        print(f'ood_mean: {ood_mean}')
        print(f'id_mean: {id_mean}')
        print(f'ood - id mean: {ood_mean - id_mean}')
            
        auroc = roc_auc_score(targets, scores)
        print(f'auroc: {auroc}')
        wandb.log({'auroc': auroc})
        aurocs.append(auroc)
        
        table['sample_idx'] = list(range(len(table['base_score'])))

        df = DataFrame(table)

        path = Path(f'mixup/logs/{datamodule}_{score_calculator}_gamma_{args.gamma}_m_{args.m}_n_{args.n}_r_{args.r}.xlsx')
        path.parent.mkdir(exist_ok=True, parents=True)
        df.to_excel(
            path,
            sheet_name=f'gamma_{args.gamma}_m_{args.m}_n_{args.n}_r_{args.r}_auroc_{auroc:.2f}',
            index=False,
        )
        break

    
    print('all auc scores:', aurocs)
    avg_auroc = np.mean(aurocs)
    auroc_std = np.std(aurocs)
    print('avg', avg_auroc, 'std', auroc_std)
    wandb.log({'avg_auroc': avg_auroc, 'auroc_std': auroc_std})
