# define a processor

from typing import List

import torch
from PIL import Image

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_image_token


class LlaVaProcessor:
    def __init__(self, tokenizer, image_processor, mm_use_im_start_end):
        self.mm_use_im_start_end = mm_use_im_start_end
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_mode = "llava_v1"

    def format_text(self, text: str):
        if self.mm_use_im_start_end:
            text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
        else:
            text = DEFAULT_IMAGE_TOKEN + "\n" + text

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()

        return text

    def load_image(self, image_path: str):
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
        """Pad a sequence to the desired max length."""
        if len(sequence) >= max_length:
            return sequence
        return torch.cat([torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype), sequence])

    def get_processed_tokens(self, text: str, image_path: str):
        prompt = self.format_text(text)
        image = self.load_image(image_path)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        return image_tensor, input_ids

    def get_processed_tokens_batch(self, batch_text: List[str], image_paths: List[str]):
        prompt = [self.format_text(text) for text in batch_text]
        images = [self.load_image(image_path) for image_path in image_paths]

        batch_input_ids = [
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in prompt
        ]

        # Determine the maximum length of input_ids in the batch
        max_len = max([len(seq) for seq in batch_input_ids])
        # Pad each sequence in input_ids to the max_len
        padded_input_ids = [self.pad_sequence_to_max_length(seq.squeeze(), max_len) for seq in batch_input_ids]
        batch_input_ids = torch.stack(padded_input_ids).to("cuda")

        batch_image_tensor = self.image_processor(images, return_tensors="pt")["pixel_values"].to("cuda")

        return batch_image_tensor, batch_input_ids