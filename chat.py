import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)


def parse_args():
    parser = argparse.ArgumentParser(description="LISA chat")
    parser.add_argument("--version", default="xinlai/LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args()


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

# >>>>>>>>>>>>>>>>>>>>>>>>>>>> Additions >>>>>>>>>>>>>
import torch.nn as nn
class Lisa(nn.Module):

    def __init__(self,):
        super(Lisa, self).__init__()

        self.args = parse_args()
        os.makedirs(self.args.vis_save_path, exist_ok=True)

        # Create model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.version,
            cache_dir=None,
            model_max_length=self.args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.args.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]


        torch_dtype = torch.float32
        if self.args.precision == "bf16":
            torch_dtype = torch.bfloat16
        elif self.args.precision == "fp16":
            torch_dtype = torch.half

        kwargs = {"torch_dtype": torch_dtype}
        if self.args.load_in_4bit:
            kwargs.update(
                {
                    "torch_dtype": torch.half,
                    "load_in_4bit": True,
                    "quantization_config": BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        llm_int8_skip_modules=["visual_model"],
                    ),
                }
            )
        elif self.args.load_in_8bit:
            kwargs.update(
                {
                    "torch_dtype": torch.half,
                    "quantization_config": BitsAndBytesConfig(
                        llm_int8_skip_modules=["visual_model"],
                        load_in_8bit=True,
                    ),
                }
            )

        self.model = LISAForCausalLM.from_pretrained(
            self.args.version, low_cpu_mem_usage=True, vision_tower=self.args.vision_tower, seg_token_idx=self.args.seg_token_idx, **kwargs
        )

        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model.get_model().initialize_vision_modules(self.model.get_model().config)
        vision_tower = self.model.get_model().get_vision_tower()
        vision_tower.to(dtype=torch_dtype)

        if self.args.precision == "bf16":
            self.model = self.model.bfloat16().cuda()
        elif (
            self.args.precision == "fp16" and (not self.args.load_in_4bit) and (not self.args.load_in_8bit)
        ):
            vision_tower = self.model.get_model().get_vision_tower()
            self.model.model.vision_tower = None
            import deepspeed

            model_engine = deepspeed.init_inference(
                model=self.model,
                dtype=torch.half,
                replace_with_kernel_inject=True,
                replace_method="auto",
            )
            self.model = model_engine.module
            self.model.model.vision_tower = vision_tower.half().cuda()
        elif self.args.precision == "fp32":
            self.model = self.model.float().cuda()

        vision_tower = self.model.get_model().get_vision_tower()
        vision_tower.to(device=self.args.local_rank)

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(self.model.config.vision_tower)
        self.transform = ResizeLongestSide(self.args.image_size)
        
        self.model.eval()

    def forward(self, imgs, qtns):
        final_pred = []
        final_text = []

        for _i,_q in zip(imgs,qtns):
            image_path,prompt = _i,_q
            image_path = f"/home/gauravs/data/clevrmath_data/images/{int(_i.item())}.png"

            conv = conversation_lib.conv_templates[self.args.conv_type].copy()
            conv.messages = []

            prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
            if self.args.use_mm_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
                )
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()

            if not os.path.exists(image_path):
                print("File not found in {}".format(image_path))
                continue

            image_np = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]

            image_clip = (
                self.clip_image_processor.preprocess(image_np, return_tensors="pt")[
                    "pixel_values"
                ][0]
                .unsqueeze(0)
                .cuda()
            )
            if self.args.precision == "bf16":
                image_clip = image_clip.bfloat16()
            elif self.args.precision == "fp16":
                image_clip = image_clip.half()
            else:
                image_clip = image_clip.float()

            image = self.tokenizer.apply_image(image_np)
            resize_list = [image.shape[:2]]

            image = (
                preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
                .unsqueeze(0)
                .cuda()
            )
            if self.args.precision == "bf16":
                image = image.bfloat16()
            elif self.args.precision == "fp16":
                image = image.half()
            else:
                image = image.float()

            input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
            input_ids = input_ids.unsqueeze(0).cuda()

            output_ids, pred_masks = self.model.evaluate(
                image_clip,
                image,
                input_ids,
                resize_list,
                original_size_list,
                max_new_tokens=512,
                tokenizer=self.tokenizer,
            )

            final_pred.append(pred_masks[0])
            final_text.append(output_ids)

            return torch.stack(final_pred), torch.stack(final_text)