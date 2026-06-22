import io
from PIL import Image
from typing import Dict, Any

from .base import BaseTaskHandler

from flagscale.train.megatron.bagel_energon.data_utils import pil_img2rgb


class T2IHandler(BaseTaskHandler):

    def encode(self, sample, **kwargs):
        """Encode text-to-image sample: caption + VAE image."""
        caption = sample.get('caption', '')
        image_bytes = sample.get('image_bytes')

        transform = kwargs.get("transform")
        transform_stride = self.transform.stride

        image_tensor_list = []
        text_ids_list = []
        sequence_plan = []
        num_tokens = 0

        # Load and transform image for VAE
        if image_bytes is not None:
            img = pil_img2rgb(Image.open(io.BytesIO(image_bytes)))
            image_tensor = transform(img)
            image_tensor_list.append(image_tensor)
            _, h, w = image_tensor.shape
            latent_h = min(h // self.data_config.vae_image_downsample, self.data_config.max_latent_size)
            latent_w = min(w // self.data_config.vae_image_downsample, self.data_config.max_latent_size)
            num_tokens += latent_h * latent_w

        # Tokenize caption
        text_ids = self.tokenizer.encode(caption)
        if len(text_ids) > 0:
            text_ids_list.append(text_ids)
            num_tokens += len(text_ids)
            sequence_plan.append({
                'type': 'text',
                'enable_cfg': 1,
                'loss': 0,
                'special_token_loss': 0,
                'special_token_label': None,
            })

        # VAE image plan
        if image_tensor_list:
            sequence_plan.append({
                'type': 'vae_image',
                'enable_cfg': 0,
                'loss': 1,
                'special_token_loss': 0,
                'special_token_label': None,
            })

        return dict({
            "image_tensor_list": image_tensor_list,
            "text_ids_list": text_ids_list,
            "sequence_plan": sequence_plan,
            "num_tokens": num_tokens,
            "is_mandatory": sample.get('__subflavors__', {}).get('is_mandatory', False),
            "task": sample.get('__subflavors__', {}).get("task", "t2i"),
        })
