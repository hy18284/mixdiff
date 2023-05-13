import random
import copy
import itertools

import torch
from sklearn.metrics import (
    roc_auc_score,
)
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
from .utils import (
    log_mixup_samples,
    calculate_fnr_at,
    calculate_fpr_at,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--r', type=int, default=3)
    parser.add_argument('--p', type=int, default=10)
    parser.add_argument('--r_ref', type=float, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--wandb_name', type=str, default='mixup_v1')
    parser.add_argument('--wandb_project', type=str, default='ZOC')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=[])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--abs', action='store_true')
    parser.add_argument('--top_1', action='store_true')
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--max_samples', type=int, default=None, required=False)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--ref_mode', type=str, default='in_batch')
    parser.add_argument('--fnr_at', type=float, default=0.95)
    parser.add_argument('--fpr_at', type=float, default=0.95)
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

    datamodule.ref_mode = args.ref_mode
    score_calculator.ref_mode = args.ref_mode

    torch.set_grad_enabled(False)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    N, M, R, P = args.n, args.m, args.r, args.p
    batch_size = N
    if args.ref_mode == 'oracle':
        P = M 
    elif args.ref_mode == 'in_batch':
        P = N

    device = torch.device(args.device)
    # TODO: Remove this. Use model_path from get_splits
    score_calculator.load_model(args.model_path, device)
    
    if score_calculator.utilize_mixup: 
        if args.ref_mode == 'in_batch':
            ref = 't'
        elif args.ref_mode == 'oracle':
            ref = 'o'
        elif args.ref_mode == 'rand_id':
            ref = 'i'

        name=f'{args.wandb_name}_{score_calculator}_{ref}_{datamodule}_{mixup_fn}'
    else:
        name=f'{args.wandb_name}_{score_calculator}_{datamodule}'

    wandb.init(
        config=args,
        name=name,
        project=args.wandb_project,
        tags=args.wandb_tags,
    )
    wandb.config.method = str(score_calculator)
    wandb.config.dataset = str(datamodule)
     
    aurocs = []
    fprs = []
    fnrs = []
    mixdiff_scores = []
    known_mixup_table = wandb.Table(['x', 'y', 'mixup', 'rate', 'split'])
    unknown_mixup_table = wandb.Table(['x', 'y', 'mixup', 'rate', 'split'])
    wandb.Table.MAX_ROWS = 200000

    for j, (
        seen_labels, 
        seen_idx, 
        given_images, 
        ref_images, 
        model_path,
    ) in enumerate(
        datamodule.get_splits(
            n_samples_per_class=M, 
            seed=args.seed, 
            n_ref_samples=P,
        )
    ):
        seen_idx = seen_idx.to(device) 

        wandb.log({
            'seen_classes': seen_labels,
            'seen_idx': seen_idx.tolist(),
        })
        
        start = args.r_ref
        end = 1.0 - start
        
        r_start = (end - start) / (R + 1)
        r_end = end - start - r_start

        rates = torch.linspace(r_start, r_end, R, device=device)
        rates += start

        print(rates)
        targets = []
        scores = []

        loader = datamodule.construct_loader(batch_size=batch_size)
        score_calculator.on_eval_start(
            seen_labels=copy.deepcopy(seen_labels),
            given_images=copy.deepcopy(given_images),
            mixup_fn=mixup_fn,
            ref_images=copy.deepcopy(ref_images),
            rates=rates,
            seed=args.seed,
            iter_idx=j,
            model_path=model_path,
        )
        cur_num_samples = 0

        for i, (images, labels) in enumerate(tqdm(loader)):
            orig_n_samples = len(images)
            if len(images) != batch_size and score_calculator.utilize_mixup:
                dummy = [images[0]] * (batch_size - len(images))
                images += dummy

            NC = len(seen_labels)

            labels = labels.to(device)
            image_kwargs = score_calculator.process_images(images)

            if score_calculator.utilize_mixup:
                chosen_images = score_calculator.select_given_images(
                    given_images, 
                    **image_kwargs
                )
                
                if args.ref_mode == 'in_batch':
                    # O: (N, M), R: (N), T: (N)
                    # TM: (N, N, R)
                    known_mixup, unknown_mixup = mixup_fn(
                        oracle=chosen_images, 
                        references=images,
                        targets=images,
                        rates=rates,
                    ) 
                elif args.ref_mode == 'oracle':
                    # (N), (N, P) -> (N * P * R)
                    unknown_mixup = []
                    for image, chosen in zip(images, chosen_images):
                        # (1 * P * R)
                        unknown = mixup_fn(
                            references=chosen,
                            targets=[image],
                            rates=rates,
                            seed=args.seed,
                        )
                        unknown_mixup += unknown
                    known_mixup = None
                elif args.ref_mode == 'rand_id':
                    # (N), (P) -> (N, P, R)
                    unknown_mixup = []
                    for image in images:
                        # (1 * P * R)
                        unknown = mixup_fn(
                            references=ref_images,
                            targets=[image],
                            rates=rates,
                            seed=args.seed,
                        )
                        unknown_mixup += unknown
                    known_mixup = None

                if args.ref_mode == 'oracle':
                    ref_images_log = chosen_images
                elif args.ref_mode == 'in_batch':
                    ref_images_log = list(itertools.repeat(images, N))
                elif args.ref_mode == 'rand_id':
                    ref_images_log = list(itertools.repeat(ref_images, P))

                log_mixup_samples(
                    ref_images=ref_images_log,
                    known_mixup_table=known_mixup_table, 
                    unknown_mixup_table=unknown_mixup_table, 
                    known_mixup=known_mixup,
                    unknown_mixup=unknown_mixup, 
                    images=images, 
                    chosen_images=chosen_images, 
                    rates=rates, 
                    j=j,
                    N=N,
                    P=P,
                    R=R,
                    M=M,
                )
                
                # (N * M * P * R * NC) -> (N, M, P, R, NC) -> (N, P, R, NC) -> (N * P * R * NC)
                known_logits = score_calculator.process_mixed_oracle(
                    known_mixup, 
                    **image_kwargs
                )
                if args.ref_mode == 'in_batch' or args.ref_mode == 'rand_id':
                    known_logits = known_logits.view(N, M, P, R, -1)
                    known_logits = torch.mean(known_logits, dim=1).view(-1, NC)
                elif args.ref_mode == 'oracle':
                    known_logits = known_logits.view(N, M, M, R, -1)
                    mask = torch.ones(
                        (M, M), 
                        device=known_logits.device
                    )
                    mask.fill_diagonal_(0)
                    mask = mask[None, :, :, None, None]
                    known_logits = known_logits * mask
                    known_logits = torch.sum(known_logits, dim=1)
                    knwon_logits = known_logits / (M - 1)
                    known_logits = knwon_logits.view(-1, NC)
                    
                # (N * P * R) -> (N * P * R * NC)
                unknown_logits = score_calculator.process_mixed_target(
                    unknown_mixup,
                    **image_kwargs
                )
                # (N * P * R * NC) -> (N * P * R)
                dists = score_calculator.calculate_diff(known_logits, unknown_logits)

                if args.abs:
                    dists = torch.abs(dists)

                if args.ref_mode == 'in_batch':
                    dists = dists.view(N, N, R)
                    dists = torch.mean(dists, dim=-1)
                    mask = torch.ones_like(dists)
                    mask.fill_diagonal_(0.0)
                    dists = dists * mask
                elif args.ref_mode == 'oracle':
                    dists = dists.view(N, M, R)
                    dists = torch.mean(dists, dim=-1)
                elif args.ref_mode == 'rand_id':
                    dists = dists.view(N, P, R)
                    dists = torch.mean(dists, dim=-1)

                if args.top_1:
                    abs_dists = torch.abs(dists)
                    abs_max_idx = torch.argmax(abs_dists, dim=-1)
                    dists = dists[torch.arange(dists.size(0)), abs_max_idx]
                else:
                    if args.ref_mode == 'in_batch':
                        dists = torch.sum(dists, dim=-1) / torch.sum(mask > 0.5, dim=-1)
                    elif args.ref_mode == 'oracle' or args.ref_mode == 'rand_id':
                        dists = torch.mean(dists, dim=-1)
            
            base_scores = score_calculator.calculate_base_scores(**image_kwargs)

            if score_calculator.utilize_mixup:
                mixdiff_scores += (args.gamma * dists).tolist()[:orig_n_samples]
                dists = base_scores + args.gamma * dists[:orig_n_samples]
            else:
                dists = base_scores[:orig_n_samples]
            
            targets += [int(label not in seen_idx) for label in labels][:orig_n_samples]
            scores += dists.tolist()[:orig_n_samples]

            cur_num_samples += N
            if args.max_samples is not None and cur_num_samples >= args.max_samples:
                break
            
        score_calculator.on_eval_end(iter_idx=j)

        if not score_calculator.utilize_mixup:
            mixdiff_scores = itertools.repeat(0.0, len(scores))
        table = wandb.Table(
            columns=['id', 'ood_score','base_score', 'mixdiff_score', 'is_ood'],
            data=[
                (i, score, score - mixdiff_score, mixdiff_score, target)
                for i, (score, mixdiff_score, target) 
                in enumerate(zip(scores, mixdiff_scores, targets))
            ]
        )
        wandb.log({f'ood_scores_{j}': table})

        ood_scores = [score for score, target in zip(scores, targets) if target == 1]
        id_scores = [score for score, target in zip(scores, targets) if target == 0]
        ood_mean = np.mean(ood_scores)
        id_mean = np.mean(id_scores)
        print(f'ood_mean: {ood_mean}')
        print(f'id_mean: {id_mean}')
        print(f'ood - id mean: {ood_mean - id_mean}')
        
        auroc = roc_auc_score(targets[:args.max_samples], scores[:args.max_samples])
        print(f'auroc: {auroc}')
        aurocs.append(auroc)

        fpr_at = calculate_fpr_at(scores, targets, args.fpr_at)
        fprs.append(fpr_at)

        fnr_at = calculate_fnr_at(scores, targets, args.fnr_at)
        fnrs.append(fnr_at)

        wandb.log({
            'auroc': auroc,
            f'fpr{args.fpr_at}': fpr_at,
            f'fnr{args.fnr_at}': fnr_at,
        })
    
    if args.ref_mode == 'in_batch':
        wandb.log({
            'oracle_mixup': known_mixup_table,
            'target_mixup': unknown_mixup_table,
        })
    else:
        wandb.log({
            'target_mixup': unknown_mixup_table,
        })

    print('all auc scores:', aurocs)
    avg_auroc = np.mean(aurocs)
    auroc_std = np.std(aurocs)
    print('avg', avg_auroc, 'std', auroc_std)

    wandb.log({
        'avg_auroc': avg_auroc, 
        'auroc_std': auroc_std,
        f'avg_fpr{round(args.fpr_at * 100)}': np.mean(fprs), 
        f'fpr{round(args.fpr_at * 100)}_std': np.std(fprs),
        f'avg_fnr{round(args.fnr_at * 100)}': np.mean(fnrs), 
        f'fnr{round(args.fnr_at * 100)}_std': np.std(fnrs),
    })
