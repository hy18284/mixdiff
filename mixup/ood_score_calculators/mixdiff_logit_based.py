import torch
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
import clip

from .ood_score_calculator import OODScoreCalculator


class MixDiffLogitBasedMixin:
    def __init__(
        self,
        batch_size: int,
        utilize_mixup: bool = True,
        add_base_score: bool = True,
    ):
        self.batch_size = batch_size
        self.utilize_mixup = utilize_mixup
        self.add_base_score = add_base_score
    
    def load_model(self, backbone_name, device):
        self.clip_model, _ = clip.load(
            backbone_name, 
            device=device, 
            download_root='trained_models'
        )
        self.clip_model.eval()
        self.cliptokenizer = clip_tokenizer()
        self.device = device

    def on_eval_start(self, seen_labels):
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]
        prompts_ids = [clip.tokenize(prompt) for prompt in seen_descriptions]
        prompts_ids = torch.cat(prompts_ids, dim=0).to(self.device)
        prompts_ids = prompts_ids.to(self.device)
        self.prompts_embeds = self.clip_model.encode_text(prompts_ids)
        self.prompts_embeds /= torch.norm(self.prompts_embeds, dim=-1, keepdim=True)
        self.seen_labels = seen_labels
    
    def on_eval_end(self):
        del self.prompts_embeds
        del self.seen_labels

    @torch.no_grad()
    def process_images(
        self,
        images,
    ):
        image_embeds = self.clip_model.encode_image(images)
        image_embeds /= torch.norm(image_embeds, dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_embeds @ self.prompts_embeds.t()
        kwargs = {
            'logits': logits,
        }
        return kwargs
    
    @torch.no_grad()
    def select_given_images(
        self,
        given_images,
        logits,
        **kwargs,
    ):
        # (N)
        max_indices = torch.argmax(logits, dim=-1)
        # (NC, M, C, H, W) -> (N, M, C, H, W)
        chosen_images = given_images[max_indices, ...]
        return chosen_images

    @torch.no_grad()
    def process_mixup_images(
        self,
        images,
    ):
        images_list = torch.split(images, self.batch_size, dim=0)
        logits_list = []
        for split_images in images_list:
            image_embeds = self.clip_model.encode_image(split_images)
            image_embeds /= torch.norm(image_embeds, dim=-1, keepdim=True)
            logit_scale = self.clip_model.logit_scale.exp()
            split_logits = logit_scale * image_embeds @ self.prompts_embeds.t()
            logits_list.append(split_logits) 
        return torch.cat(logits_list, dim=0)