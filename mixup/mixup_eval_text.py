import random
import copy

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

from .ood_score_calculators.ood_score_calculator import OODScoreCalculator
from .ood_datamodules.base_ood_datamodule import BaseOODDataModule
from .mixup_operators.base_mixup_operator import BaseMixupOperator


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
    parser.add_argument('--model_path', type=str)
    parser.add_subclass_arguments(OODScoreCalculator, 'score_calculator')
    parser.add_subclass_arguments(BaseOODDataModule, 'datamodule')
    parser.add_subclass_arguments(BaseMixupOperator, 'mixup_operator')
    parser.add_argument('--config', action=ActionConfigFile)

    args = parser.parse_args()
    classes = parser.instantiate_classes(args)
    return args, classes


if __name__ == '__main__':
    args, classes = parse_args()
    score_calculator: OODScoreCalculator = classes['score_calculator']
    datamodule: BaseOODDataModule = classes['datamodule']
    mixup_fn: BaseMixupOperator = classes['mixup_operator']

    torch.set_grad_enabled(False)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    N, M, R = args.n, args.m, args.r
    batch_size = N

    device = torch.device(args.device)
    score_calculator.load_model(args.model_path, device)
    
    if score_calculator.utilize_mixup: 
        name=f'{args.wandb_name}_{score_calculator}_{datamodule}_{mixup_fn}'
    else:
        name=f'{args.wandb_name}_{score_calculator}_{datamodule}'
    wandb.init(
        config=args,
        name=name,
        project='ZOC',
    )
    wandb.config.method = str(score_calculator)
    wandb.config.dataset = str(datamodule)
     
    aurocs = []
    for j, (seen_labels, seen_idx, given_images) in enumerate(
        datamodule.get_splits(n_samples_per_class=M, seed=args.seed)
    ):
        seen_idx = seen_idx.to(device) 

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

        loader = datamodule.construct_loader(batch_size=batch_size)
        score_calculator.on_eval_start(copy.deepcopy(seen_labels))
        cur_num_samples = 0

        for i, (images, labels) in enumerate(tqdm(loader)):
            orig_n_samples = len(images)
            if len(images) != N:
                images += prev_images[len(images):]

            NC = len(seen_labels)
            labels = labels.to(device)
            image_kwargs = score_calculator.process_images(images)

            if score_calculator.utilize_mixup:
                chosen_images = score_calculator.select_given_images(
                    given_images, 
                    **image_kwargs
                )
                
                known_mixup, unknown_mixup = mixup_fn(chosen_images, images, rates) 
                # (N * M * N * R * NC) -> (N, M, N, R, NC) -> (N, N, R, NC) -> (N * N * R * NC)
                known_logits = score_calculator.process_mixup_images(known_mixup)
                known_logits = known_logits.view(N, M, N, R, -1)
                known_logits = torch.mean(known_logits, dim=1).view(-1, NC)

                # (N * N * R)
                unknown_logits = score_calculator.process_mixup_images(unknown_mixup)

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
                dists = base_scores + args.gamma * dists
            else:
                dists = base_scores
            
            targets += [int(label not in seen_idx) for label in labels]
            scores += dists.tolist()[:orig_n_samples]

            cur_num_samples += N
            prev_images = images
            if args.max_samples is not None and cur_num_samples >= args.max_samples:
                break
            
        score_calculator.on_eval_end()

        ood_scores = [score for score, target in zip(scores, targets) if target == 1]
        id_scores = [score for score, target in zip(scores, targets) if target == 0]
        ood_mean = np.mean(ood_scores)
        id_mean = np.mean(id_scores)
        print(f'ood_mean: {ood_mean}')
        print(f'id_mean: {id_mean}')
        print(f'ood - id mean: {ood_mean - id_mean}')
        
        auroc = roc_auc_score(targets[:args.max_samples], scores[:args.max_samples])
        print(f'auroc: {auroc}')
        wandb.log({'auroc': auroc})
        aurocs.append(auroc)
    
    print('all auc scores:', aurocs)
    avg_auroc = np.mean(aurocs)
    auroc_std = np.std(aurocs)
    print('avg', avg_auroc, 'std', auroc_std)
    wandb.log({'avg_auroc': avg_auroc, 'auroc_std': auroc_std})
