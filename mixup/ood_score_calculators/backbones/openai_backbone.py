from typing import (
    Optional,
)
import random
import os
import time

import torch
import torch.nn.functional as F
import openai
import numpy as np
from litellm import completion

from .base_backbone import BaseBackbone
from ...mixup_operators.base_mixup_operator import BaseMixupOperator

TEMPLATE = '''
Select one most appropriate intent for the user query below. Provide the answer only without any additional texts.

Below are some examples:
{}

Here is the query:
{}

These are the intents candidates: 
{}
'''


class OpenAIBackbone(BaseBackbone):
    def __init__(
        self, 
        n_examples: int, 
        n_calls: int,
        api_key_path: str = '.openai_api_key', 
        ood_name: str = 'None of the above',
    ):
        # openai.api_key_path = api_key_path
        with open(api_key_path) as f:
            key = f.readline()
        os.environ['OPENAI_API_KEY'] = key
        self.n_examples = n_examples
        self.n_calls = n_calls
        self.ood_name = ood_name
    
    def load_model(self, backbone_name, device):
        self.backbone_name = backbone_name
        self.device = device

    def on_eval_start(
        self, 
        seen_labels,
        given_images, 
        mixup_fn: Optional[BaseMixupOperator],
        ref_images,
        rates,
        seed,
        iter_idx,
        model_path,
        few_shot_samples,
    ):
        self.few_shot_samples = few_shot_samples
        self.seen_labels = seen_labels
        self.all_labels = self.seen_labels + [self.ood_name]

        self.label_orders = []
        self.few_shot_samples_list = []
        print(few_shot_samples)
        print('###########')

        for i in range(self.n_examples):
            order = list(range(len(self.seen_labels)))
            np.random.default_rng(seed + i).shuffle(order)
            self.label_orders.append(order)
            partial_samples = np.random.default_rng(seed + i).choice(
                self.few_shot_samples,
                size=self.n_examples,
                shuffle=False,
            )
            print(partial_samples)
            print('###########')
            self.few_shot_samples_list.append(partial_samples)
        print(self.few_shot_samples_list)
        
    def process_images(self, images):
        mean_preds = []
        for image in images:
            predictions = []
            for examples, order in zip(self.few_shot_samples_list, self.label_orders):
                candidates = [self.seen_labels[idx] for idx in order]
                candidates.append(self.ood_name)
                candidates = '\n'.join(candidates)

                print(examples)
                examples = [
                    f'query: {query}, intent: {intent}' 
                    for query, intent in examples
                ]
                examples = '\n'.join(examples)

                sample = TEMPLATE.format(
                    examples,
                    image,
                    candidates,
                )
                # res = openai.ChatCompletion.create(
                n_attempts = 0
                successful = False
                while not successful:
                    try:
                        if n_attempts > 5:
                            raise ValueError('Too many attempts')
                        res = completion(
                            model=self.backbone_name,
                            request_timeout=10,
                            messages=[
                                {'role': 'user', 'content': sample}
                            ],
                            n=1,
                        )
                        successful = True
                    except Exception as e:
                        print(e)
                        n_attempts += 1
                        print(f'# of attempts: {n_attempts}')
                        time.sleep(5)
                    
                prediction = (res.choices[0]['message']['content'])
                print(prediction)
                print(sample)
                if prediction in self.all_labels:
                    prediction = self.all_labels.index(prediction)
                    prediction = torch.tensor(prediction)
                    prediction = F.one_hot(prediction, num_classes=len(self.all_labels))
                    prediction = prediction.to(self.device)
                    prediction = prediction.to(torch.float)
                else:
                    prediction = torch.ones(
                        len(self.all_labels), 
                        device=self.device, 
                        dtype=torch.float,
                    )
                    prediction /= len(self.all_labels)
                print(prediction)
                predictions.append(prediction)
            mean_pred = torch.stack(predictions).mean(0)
            mean_preds.append(mean_pred)

        mean_preds = torch.stack(mean_preds)
        print(mean_preds)
        return mean_preds