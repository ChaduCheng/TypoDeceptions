import sys

import argparse
import os
import random
from collections import defaultdict

import cv2
import re
import glob
from tqdm import tqdm

import numpy as np
from PIL import Image
import torch
import html
import gradio as gr

import torchvision.transforms as T
import torch.backends.cudnn as cudnn

from minigpt4.common.config import Config

from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Conversation, SeparatorStyle, Chat

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def remove_image_extensions(text):
    text = text.replace(".jpg", "")
    text = text.replace(".png", "")
    return text


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", default='eval_configs/minigptv2_eval.yaml',
                        help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
             "in xxx=yyy format will be merged into config file (deprecate), "
             "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

cudnn.benchmark = False
cudnn.deterministic = True

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

device = 'cuda:{}'.format(args.gpu_id)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to(device)
bounding_box_size = 100

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

model = model.eval()

CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

chat = Chat(model, vis_processor, device=device)
print('Initialization Finished')

 
log_root = './'
data_root = '../LLaVA/dataset/classification'
model_name = 'minigpt4v2'
total_datasets = [['color-r' + str(r) for r in range(2)],]

 
prefixs =  ["Answer with the option's letter from the given choices directly. ",
            "Focus on the visual aspects of the image, including colors, shapes, composition, and any notable visual themes. Answer with the option's letter from the given choices directly. ",]
questions = ['{} (a) {} (b) {}']

for q in questions:
    for p in prefixs:
        prompt = '[vqa] ' + p + q
        for datasets in total_datasets:
            for dataset in datasets:
                with open(os.path.join(log_root, dataset + '-' + model_name + '-' + p), 'w') as f:
                    sys.stdout = f
                    
                    images = []
                    for extension in ["*.jpg", "*.jpeg", "*.png"]:
                        images.extend(glob.glob(os.path.join(data_root, dataset, extension)))

                    for img in tqdm(images):
                        chat_state = CONV_VISION.copy()
                        img_list= []
                        
                        label = remove_image_extensions(img).split('-')[-2]
                        mislabel = remove_image_extensions(img).split('-')[-1]
                        challenge = remove_image_extensions(img).split('-')[-3]
                        inp = prompt.format(challenge, label, mislabel)
                        
                        chat.upload_img(img, chat_state, img_list)
                        chat.ask(inp, chat_state)
                        chat.encode_img(img_list)
                        llm_message = chat.answer(conv=chat_state, img_list=img_list, temperature=0.001, max_new_tokens=500, max_length=2000)[0]
                        print(img)
                        print("USER:", inp)
                        print("ASSISTANT:", llm_message)
                        print()
                    
                    sys.stdout = sys.__stdout__