from typing import (
    Optional,
)

import torch
from torchvision.transforms import (
    Compose, 
    Resize, 
    CenterCrop, 
    ToTensor, 
    Normalize, 
)
from PIL import Image

from .base_backbone import BaseBackbone
import clip
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
from ...mixup_operators.base_mixup_operator import BaseMixupOperator
from .ash import apply_ash


class ClipBackboneASH(BaseBackbone):
    def __init__(self, ash_method: str) -> None:
        super().__init__()
        self.transform_fn = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
        ])
        self.post_transform_fn = Normalize((0.4913, 0.4821, 0.4465), (0.2470, 0.2434, 0.2615))
        self.ash_method = ash_method

    def load_model(self, backbone_name, device):
        self.clip_model, _ = clip.load(
            backbone_name, 
            device=device, 
            download_root='trained_models'
        )
        self.clip_model.eval()
        self.cliptokenizer = clip_tokenizer()
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
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]
        prompts_ids = [clip.tokenize(prompt) for prompt in seen_descriptions]
        prompts_ids = torch.cat(prompts_ids, dim=0).to(self.device)
        prompts_ids = prompts_ids.to(self.device)
        self.prompts_embeds = self.clip_model.encode_text(prompts_ids)
        self.prompts_embeds /= torch.norm(self.prompts_embeds, dim=-1, keepdim=True)
        self.seen_labels = seen_labels

    def on_eval_end(self, iter_idx: int):
        del self.prompts_embeds
        del self.seen_labels
    
    def process_images(
        self, 
        images, 
        return_embeds: bool = False,
    ):
        image_embeds = self.clip_model.encode_image(images)
        image_embeds = apply_ash(image_embeds[:, :, None, None], method=getattr(self, 'ash_method'))
        image_embeds = image_embeds.squeeze(-1).squeeze(-1)
        image_embeds = image_embeds / torch.norm(image_embeds, dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_embeds @ self.prompts_embeds.t()
        if return_embeds:
            return logits.float(), image_embeds, self.prompts_embeds.detach().clone()
        else:
            return logits.float()

    def transform(self, images):
        return self.transform_fn(images)
   
    def post_transform(self, images):
        orig_size = images.size()
        C, H, W = orig_size[-3:]
        images = images.view(-1, C, H, W)
        images = self.post_transform_fn(images)
        images = images.view(orig_size)
        return images