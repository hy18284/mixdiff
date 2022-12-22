import torch
from clip.simple_tokenizer import SimpleTokenizer as clip_tokenizer
import clip
from transformers import (
    BertGenerationTokenizer, 
    BertGenerationDecoder, 
    BertGenerationConfig,
)

from .ood_score_calculator import OODScoreCalculator


class MixDiffZOC(OODScoreCalculator):

    utilize_mixup = True

    def __init__(
        self,
        zoc_checkpoint_path: str,
        batch_size: int,
        utilize_mixup: bool = True,
        add_base_scores: bool = True,
        follow_zoc: bool = True,
    ):
        self.batch_size = batch_size
        self.zoc_checkpoint_path = zoc_checkpoint_path
        self.utilize_mixup = utilize_mixup
        self.add_base_scores = add_base_scores
        self.follow_zoc = follow_zoc
    
    def load_model(self, backbone_name, device):
        self.device = device

        self.clip_model, _ = clip.load(
            backbone_name, 
            device=self.device, 
            download_root='trained_models'
        )
        self.clip_model.eval()
        self.cliptokenizer = clip_tokenizer()

        self.berttokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
        bert_config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
        bert_config.is_decoder=True
        bert_config.add_cross_attention=True
        self.bert_model = BertGenerationDecoder.from_pretrained(
            'google/bert_for_seq_generation_L-24_bbc_encoder',
            config=bert_config,
        )
        self.bert_model.load_state_dict(
            torch.load(
                self.zoc_checkpoint_path,
                map_location=self.device
            )['net']
        )
        self.bert_model.eval()
        self.bert_model.to(self.device)
    
    def on_eval_start(self, seen_labels):
        seen_descriptions = [f"This is a photo of a {label}" for label in seen_labels]
        prompts_ids = [clip.tokenize(prompt) for prompt in seen_descriptions]
        prompts_ids = torch.cat(prompts_ids, dim=0).to(self.device)
        prompts_ids = prompts_ids.to(self.device)
        self.prompts_embeds = self.clip_model.encode_text(prompts_ids).float()
        self.seen_labels = seen_labels
    
    def on_eval_end(self):
        del self.prompts_embeds
        del self.seen_labels

    def process_images(
        self,
        images,
    ):
        image_embeds = self.clip_model.encode_image(images).float()
        # Follow ZOC's logit scaling.
        logits = 100.0 * image_embeds @ self.prompts_embeds.t()
        return {
            'logits': logits,
            'images': images,
        }
    
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

    def process_mixup_images(
        self,
        images,
    ):
        images_list = torch.split(images, self.batch_size, dim=0)
        scores_list = []
        for split_images in images_list:
            split_scores = self._compute_zoc_scores(split_images) 
            scores_list.append(split_scores) 
        return torch.cat(scores_list, dim=0)

    def calculate_diff(
        self,
        known_logits,
        unknown_logits,
    ):
        return unknown_logits - known_logits        

    def calculate_base_scores(
        self,
        images,
        **kwargs,
    ):
        images_list = torch.split(images, self.batch_size, dim=0)
        scores_list = []
        for split_images in images_list:
            split_scores = self._compute_zoc_scores(split_images) 
            scores_list.append(split_scores) 
        return torch.cat(scores_list, dim=0)

    def _compute_zoc_scores(self, images):
        B = images.size(0)
        clip_out = self.clip_model.encode_image(images).float()
        clip_extended_embeds = clip_out.repeat(1, 2).type(torch.FloatTensor)
        clip_extended_embeds = clip_extended_embeds.unsqueeze(1)

        #greedy generation
        _, top_k_idx_list = self.greedysearch_generation_topk(clip_extended_embeds)
        
        unseen_ids_list = []
        top_k_tokens_list = []
        for top_k_idx in top_k_idx_list:
            top_k_tokens = [
                self.berttokenizer.decode(int(pred_idx.cpu().numpy())) 
                for pred_idx in top_k_idx
            ]
            top_k_tokens_list.append(top_k_tokens)
            unseen_prompts = [f"This is a photo of a {label}" for label in top_k_tokens]
            unseen_ids = self.tokenize_for_clip(unseen_prompts).to(self.device)
            unseen_ids_list.append(unseen_ids)

        # (B * NU, H)
        unseen_ids = torch.cat(unseen_ids_list, dim=0)
        unseen_text_embeds = self.clip_model.encode_text(unseen_ids)
        # (B * NU, H) -> (B, NU, H)
        unseen_text_embeds = unseen_text_embeds.view(B, -1, unseen_text_embeds.size(-1))

        seen_text_embeds = self.prompts_embeds.expand(B, -1, -1)
        text_features = torch.cat([seen_text_embeds, unseen_text_embeds], dim=1).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

        image_features = self.clip_model.encode_image(images).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # (B, H) * (B, NS + NU, H)
        # TODO: Find a better way.
        logits_list = []
        for image_embed, text_embed in zip(image_features, text_features):
            # (H) * (H, NS + NU) -> (NS + NU)
            logits = 100.0 * image_embed @ text_embed.t()
            logits_list.append(logits)
        zeroshot_logits = torch.stack(logits_list, dim=0)

        unseen_mask = []
        for top_k_tokens in top_k_tokens_list:
            row = [token in self.seen_labels for token in top_k_tokens]
            unseen_mask.append(row)
        unseen_mask = torch.tensor(unseen_mask, device=self.device)

        zeroshot_logits[:, self.prompts_embeds.size(0):].masked_fill_(
            unseen_mask,
            float('-inf'),
        )
        zeroshot_probs = zeroshot_logits.softmax(dim=-1)
        ood_probs = torch.sum(zeroshot_probs[:, self.prompts_embeds.size(0):], dim=-1)
        return ood_probs.unsqueeze(-1)
    
    def tokenize_for_clip(self, batch_sentences):
        default_length = 77  # CLIP default
        sot_token = self.cliptokenizer.encoder['<|startoftext|>']
        eot_token = self.cliptokenizer.encoder['<|endoftext|>']
        tokenized_list = []
        for sentence in batch_sentences:
            text_tokens = [sot_token] + self.cliptokenizer.encode(sentence) + [eot_token]
            tokenized = torch.zeros((default_length), dtype=torch.long)
            tokenized[:len(text_tokens)] = torch.tensor(text_tokens)
            tokenized_list.append(tokenized)
        tokenized_list = torch.stack(tokenized_list)
        return tokenized_list

    def greedysearch_generation_topk(self, clip_embeds):
        B = clip_embeds.size(0)
        max_len = 77
        target = torch.tensor(self.berttokenizer.bos_token_id, device=self.device)
        target = target.expand(B, 1)
        top_k_indices = torch.Tensor(B, 0).to(self.device)
        self.bert_model.eval()
        for i in range(max_len):
            if self.follow_zoc:
                # Follow ZOC's code, even if it does not make sense at all.
                L = 1
            else:
                L = target.size(1)
            position_ids = torch.arange(0, L).expand(B, L).to(self.device)
            with torch.no_grad():
                out = self.bert_model(
                    input_ids=target.to(self.device),
                    position_ids=position_ids,
                    attention_mask=torch.ones(B, L).to(self.device),
                    encoder_hidden_states=clip_embeds.to(self.device),
            )
            # (B, L, V) -> (B, L, 1) -> (B, 1)
            pred_idx = out.logits.argmax(dim=2, keepdim=True)[:, -1, :]
            target = torch.cat([target, pred_idx], dim=-1)

            # (B, L, V) -> (B, L, K)
            _, top_k = torch.topk(out.logits, dim=2, k=35)
            # (B, L, K) -> (B, K)
            top_k = top_k[:, -1, :]
            top_k_indices = torch.cat([top_k_indices, top_k], dim=1)
            #if pred_idx == berttokenizer.eos_token_id or len(target_list)==10: #the entitiy word is in at most first 10 words
            if i == 9:  # the entitiy word is in at most first 10 words
                break
        return target, top_k_indices

    def __str__(self) -> str:
        if not self.utilize_mixup:
            return 'zoc'
        if self.add_base_scores:
            return 'mixdiff_zoc+'
        else:
            return 'mixdiff_zoc'