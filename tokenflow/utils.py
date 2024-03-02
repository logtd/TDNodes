from pathlib import Path
from PIL import Image
import torch
import yaml

import torchvision.transforms as T
from torchvision.io import read_video, write_video
import os
import random
import numpy as np
from torchvision.io import write_video


def isinstance_str(x: object, cls_name: str):
  """
  Checks whether x has any class *named* cls_name in its ancestry.
  Doesn't require access to the class's implementation.

  Useful for patching!
  """

  for _cls in x.__class__.__mro__:
    if _cls.__name__ == cls_name:
      return True

  return False


def print_module(m):
  for _, module in m.named_modules():
    print(module.__class__.__name__)


def batch_cosine_sim(x, y):
  if type(x) is list:
    x = torch.cat(x, dim=0)
  if type(y) is list:
    y = torch.cat(y, dim=0)
  x = x / x.norm(dim=-1, keepdim=True)
  y = y / y.norm(dim=-1, keepdim=True)
  similarity = x @ y.T
  return similarity


def load_imgs(data_path, n_frames, device='cuda', pil=False):
  imgs = []
  pils = []
  for i in range(n_frames):
    img_path = os.path.join(data_path, "%05d.jpg" % i)
    if not os.path.exists(img_path):
      img_path = os.path.join(data_path, "%05d.png" % i)
    img_pil = Image.open(img_path)
    pils.append(img_pil)
    img = T.ToTensor()(img_pil).unsqueeze(0)
    imgs.append(img)
  if pil:
    return torch.cat(imgs).to(device), pils
  return torch.cat(imgs).to(device)


def save_video(raw_frames, save_path, fps=10):
  video_codec = "libx264"
  video_options = {
      # Constant Rate Factor (lower value = higher quality, 18 is a good balance)
      "crf": "18",
      # Encoding preset (e.g., ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
      "preset": "slow",
  }

  frames = (raw_frames * 255).to(torch.uint8).cpu().permute(0, 2, 3, 1)
  write_video(save_path, frames, fps=fps,
              video_codec=video_codec, options=video_options)


def seed_everything(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  random.seed(seed)
  np.random.seed(seed)
