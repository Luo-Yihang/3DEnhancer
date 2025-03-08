import warnings
warnings.filterwarnings('ignore')

import os
import argparse
from PIL import Image
import torch
from torchvision import transforms as T
import kiui

import sys
sys.path.insert(0, "src")
from src.utils.camera import get_c2ws
from src.enhancer import Enhancer


def load_img_file_to_tensor(path):
    """Load an image from a file path and convert it to a tensor."""
    img = Image.open(path).convert("RGB")
    img = img.resize((512, 512), resample=Image.Resampling.LANCZOS)
    img_tensor = T.ToTensor()(img)
    return img_tensor


def save_tensor_to_img_file(img_tensor, path):
    """Save a tensor as an image file."""
    img = T.ToPILImage()(img_tensor)
    img.save(path)


def main():
    parser = argparse.ArgumentParser(description="Inference script for multi-view image enhancement.")
    parser.add_argument('-m', '--model_path', type=str, default="pretrained_models/3DEnhancer/model.safetensors", help="Path to the model checkpoint.")
    parser.add_argument('-c', '--config_path', type=str, default="src/configs/config.py", help="Path to the configuration file.")
    parser.add_argument('-i', '--input_folder', type=str, required=True, help="Folder containing four input images, with filenames sorted in lexicographical order.")
    parser.add_argument('-o', '--output_folder', type=str, default="results", help="Folder to save the enhanced images.")
    parser.add_argument('-p', '--prompt', type=str, required=True, help="Prompt for the object.")
    parser.add_argument('-e', '--elevations', type=int, nargs='+', default=[0,0,0,0], help="Elevations for each image.")
    parser.add_argument('-a', '--amuziths', type=int, nargs='+', default=[0,90,180,270], help="Amuziths for each image.")
    parser.add_argument('-n', '--noise_level', type=int, default=0, choices=range(0, 301), help="Noise level for the inference.")
    parser.add_argument('--steps', type=int, default=20, help="Sample steps for the inference.")
    parser.add_argument('--cfg_scale', type=float, default=4.5, help="CFG scale for the inference.")
    parser.add_argument('--seed', type=int, default=0, help="Seed for the inference.")
    parser.add_argument('--color_shift', type=str, default=None, choices=["adain", "wavelet", None], help="Color shift for the results.")

    args = parser.parse_args()

    enhancer = Enhancer(
        model_path=args.model_path,
        config_path=args.config_path
    )

    img_tensor_list = []
    file_name_list = []
    for file in sorted(os.listdir(args.input_folder)):
        path = os.path.join(args.input_folder, file)
        file_name_list.append(os.path.basename(path))
        img_tensor = load_img_file_to_tensor(path)
        img_tensor_list.append(img_tensor)

    img_tensors = torch.stack(img_tensor_list)

    kiui.seed_everything(args.seed)
    output_img_tensors = enhancer.inference(
        mv_imgs=img_tensors, 
        c2ws=get_c2ws(args.elevations, args.amuziths), 
        prompt=args.prompt, 
        noise_level=args.noise_level,
        cfg_scale=args.cfg_scale,
        sample_steps=args.steps,
        color_shift=args.color_shift
    )

    os.makedirs(args.output_folder, exist_ok=True)
    for i, output_img_tensor in enumerate(output_img_tensors):
        save_tensor_to_img_file(
            output_img_tensor,
            os.path.join(args.output_folder, file_name_list[i])
        )


if __name__ == "__main__":
    main()