import warnings
warnings.filterwarnings('ignore')

import os
import tyro
import imageio
import numpy as np
import tqdm
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms as T
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera
import rembg
import gradio as gr
from gradio_imageslider import ImageSlider

import sys
sys.path.insert(0, "src")
from src.enhancer import Enhancer
from src.utils.camera import get_c2ws

# import LGM
sys.path.insert(0, "extern/LGM")
from core.options import AllConfigs
from core.models import LGM
from mvdream.pipeline_mvdream import MVDreamPipeline


### Title and Description ###
#### Description ####
title = r"""<h1 align="center">3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement</h1>"""

important_link = r"""
<div align='center'>
<a href='https://arxiv.org/abs/2412.18565'>[arxiv]</a>
&ensp; <a href='https://Luo-Yihang.github.io/projects/3DEnhancer'>[Project Page]</a>
&ensp; <a href='https://github.com/Luo-Yihang/3DEnhancer'>[Code]</a>
</div>
"""

authors = r"""
<div align='center'>
 <a href='https://github.com/Luo-Yihang'>Yihang Luo</a>
 &ensp; <a href='https://shangchenzhou.com/'>Shangchen Zhou</a>
&ensp; <a href='https://nirvanalan.github.io/'>Yushi Lan</a>
&ensp; <a href='https://xingangpan.github.io/'>Xingang Pan</a>
&ensp; <a href='https://www.mmlab-ntu.com/person/ccloy/index.html'>Chen Change Loy</a>
</div>
"""

affiliation = r"""
<div align='center'>
 <a href='https://www.mmlab-ntu.com/'>S-Lab, NTU Singapore</a>
</div>
"""

description = r"""
<b>Official Gradio demo</b> for <a href='https://yihangluo.com/projects/3DEnhancer' target='_blank'><b>3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement</b></a>.<br>
🔥 3DEnhancer employs a multi-view diffusion model to enhance multi-view images, thus improving 3D models. Our contributions include a robust data augmentation pipeline, and the view-consistent blocks that integrate multi-view row attention and near-view epipolar aggregation modules to promote view consistency. <br>
"""

article = r"""
<br>If 3DEnhancer is helpful, please help to ⭐ the <a href='https://github.com/Luo-Yihang/3DEnhancer' target='_blank'>Github Repo</a>. Thanks! 
[![GitHub Stars](https://img.shields.io/github/stars/Luo-Yihang/3DEnhancer)](https://github.com/Luo-Yihang/3DEnhancer)
---
---
📝 **License**
<br>
This project is licensed under <a href="https://github.com/Luo-Yihang/3DEnhancer/blob/main/LICENSE">S-Lab License 1.0</a>, 
Redistribution and use for non-commercial purposes should follow this license.
<br>
📝 **Citation**
<br>
If our work is useful for your research, please consider citing:
```bibtex
@article{luo20243denhancer,
    title={3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement}, 
    author={Yihang Luo and Shangchen Zhou and Yushi Lan and Xingang Pan and Chen Change Loy},
    booktitle={arXiv preprint arXiv:2412.18565}
    year={2024},
}
```
📧 **Contact**
<br>
If you have any questions, please feel free to reach me out at <b>luo_yihang@outlook.com</b>.
"""


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
BASE_SAVE_PATH = 'gradio_results'
GRADIO_VIDEO_PATH = f'{BASE_SAVE_PATH}/gradio_output.mp4'
GRADIO_PLY_PATH = f'{BASE_SAVE_PATH}/gradio_output.ply'
GRADIO_ENHANCED_VIDEO_PATH = f'{BASE_SAVE_PATH}/gradio_enhanced_output.mp4'
GRADIO_ENHANCED_PLY_PATH = f'{BASE_SAVE_PATH}/gradio_enhanced_output.ply'
DEFAULT_NEG_PROMPT = "ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality, artifacts, duplicate"
DEFAULT_SEED = 0
os.makedirs(BASE_SAVE_PATH, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load dreams
pipe_text = MVDreamPipeline.from_pretrained(
    'ashawkey/mvdream-sd2.1-diffusers', # remote weights
    torch_dtype=torch.float16,
	trust_remote_code=True
)
pipe_text = pipe_text.to(device)

pipe_image = MVDreamPipeline.from_pretrained(
    "ashawkey/imagedream-ipmv-diffusers", # remote weights
    torch_dtype=torch.float16,
	trust_remote_code=True
)
pipe_image = pipe_image.to(device)

# load lgm
lgm_opt = tyro.cli(AllConfigs, args=["big"])

tan_half_fov = np.tan(0.5 * np.deg2rad(lgm_opt.fovy))
proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
proj_matrix[0, 0] = 1 / tan_half_fov
proj_matrix[1, 1] = 1 / tan_half_fov
proj_matrix[2, 2] = (lgm_opt.zfar + lgm_opt.znear) / (lgm_opt.zfar - lgm_opt.znear)
proj_matrix[3, 2] = - (lgm_opt.zfar * lgm_opt.znear) / (lgm_opt.zfar - lgm_opt.znear)
proj_matrix[2, 3] = 1

lgm_model = LGM(lgm_opt)
lgm_model = lgm_model.half().to(device)
ckpt = load_file("pretrained_models/LGM/model_fp16_fixrot.safetensors", device='cpu')
lgm_model.load_state_dict(ckpt, strict=False)
lgm_model.eval()

# load 3denhancer
enhancer = Enhancer(
	model_path = "pretrained_models/3DEnhancer/model.safetensors",
	config_path = "src/configs/config.py",
)

# load rembg
bg_remover = rembg.new_session()

@torch.no_grad()
def gen_mv(ref_image, ref_text):
	kiui.seed_everything(DEFAULT_SEED)

	# text-conditioned
	if ref_image is None:
		mv_image_uint8 = pipe_text(ref_text, negative_prompt=DEFAULT_NEG_PROMPT, num_inference_steps=30, guidance_scale=7.5, elevation=0)
		mv_image_uint8 = (mv_image_uint8 * 255).astype(np.uint8)
		# bg removal
		mv_image = []
		for i in range(4):
			image = rembg.remove(mv_image_uint8[i], session=bg_remover) # [H, W, 4]
			# to white bg
			image = image.astype(np.float32) / 255
			image = recenter(image, image[..., 0] > 0, border_ratio=0.2)
			image = image[..., :3] * image[..., -1:] + (1 - image[..., -1:])
			mv_image.append(image)
	# image-conditioned (may also input text, but no text usually works too)
	else:
		ref_image = np.array(ref_image) # uint8
		# bg removal
		carved_image = rembg.remove(ref_image, session=bg_remover) # [H, W, 4]
		mask = carved_image[..., -1] > 0
		image = recenter(carved_image, mask, border_ratio=0.2)
		image = image.astype(np.float32) / 255.0
		image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
		mv_image = pipe_image(ref_text, image, negative_prompt=DEFAULT_NEG_PROMPT, num_inference_steps=30, guidance_scale=5.0, elevation=0)

	# mv_image, a list of 4 np_arrays in shape (256, 256, 3) in range (0.0, 1.0)
	mv_image_512 = []
	for i in range(len(mv_image)):
		mv_image_512.append(cv2.resize(mv_image[i], (512, 512), interpolation=cv2.INTER_LINEAR))

	return mv_image_512[0], mv_image_512[1], mv_image_512[2], mv_image_512[3], ref_text, 120


@torch.no_grad()
def gen_3d(image_0, image_1, image_2, image_3, elevation, output_video_path, output_ply_path):
	kiui.seed_everything(DEFAULT_SEED)

	mv_image = [image_0, image_1, image_2, image_3]
	for i in range(len(mv_image)):
		if type(mv_image[i]) is tuple:
			mv_image[i] = mv_image[i][1]
		mv_image[i] = np.array(mv_image[i]).astype(np.float32) / 255.0
		mv_image[i] = cv2.resize(mv_image[i], (256, 256), interpolation=cv2.INTER_AREA)

	# generate gaussians
	input_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32
	input_image = torch.from_numpy(input_image).permute(0, 3, 1, 2).float().to(device) # [4, 3, 256, 256]
	input_image = F.interpolate(input_image, size=(lgm_opt.input_size, lgm_opt.input_size), mode='bilinear', align_corners=False)
	input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

	rays_embeddings = lgm_model.prepare_default_rays(device, elevation=elevation)
	input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]

	with torch.no_grad():
		with torch.autocast(device_type='cuda', dtype=torch.float16):
				# generate gaussians
				gaussians = lgm_model.forward_gaussians(input_image)
		lgm_model.gs.save_ply(gaussians, output_ply_path)
		
		# render 360 video 
		images = []
		elevation = 0
		if lgm_opt.fancy_video:
			azimuth = np.arange(0, 720, 4, dtype=np.int32)
			for azi in tqdm.tqdm(azimuth):
				cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=lgm_opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
				cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
				
				# cameras needed by gaussian rasterizer
				cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
				cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
				cam_pos = - cam_poses[:, :3, 3] # [V, 3]

				scale = min(azi / 360, 1)

				image = lgm_model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=scale)['image']
				images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))
		else:
			azimuth = np.arange(0, 360, 2, dtype=np.int32)
			for azi in tqdm.tqdm(azimuth):
				cam_poses = torch.from_numpy(orbit_camera(elevation, azi, radius=lgm_opt.cam_radius, opengl=True)).unsqueeze(0).to(device)
				cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
				
				# cameras needed by gaussian rasterizer
				cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
				cam_view_proj = cam_view @ proj_matrix # [V, 4, 4]
				cam_pos = - cam_poses[:, :3, 3] # [V, 3]

				image = lgm_model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
				images.append((image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8))

		images = np.concatenate(images, axis=0)
		imageio.mimwrite(output_video_path, images, fps=30)

	return output_video_path, output_ply_path


@torch.no_grad()
def enhance(image_0, image_1, image_2, image_3, prompt, elevation, noise_level, cfg_scale, steps, seed, color_shift):
	kiui.seed_everything(seed)

	mv_image = [image_0, image_1, image_2, image_3]
	img_tensor_list = []
	for image in mv_image:
		img_tensor_list.append(T.ToTensor()(image))

	img_tensors = torch.stack(img_tensor_list)

	color_shift = None if color_shift=="disabled" else color_shift
	output_img_tensors = enhancer.inference(
		mv_imgs=img_tensors, 
		c2ws=get_c2ws(elevations=[elevation]*4, amuziths=[0,90,180,270]), 
		prompt=prompt, 
		noise_level=noise_level,
		cfg_scale=cfg_scale,
		sample_steps=steps,
		color_shift=color_shift,
	)

	mv_image_512 = output_img_tensors.permute(0,2,3,1).cpu().numpy()

	# return to the image slider component
	return (image_0, mv_image_512[0]), (image_1, mv_image_512[1]), (image_2, mv_image_512[2]), (image_3, mv_image_512[3])


def check_video(input_video):
    if input_video:
        return gr.update(interactive=True)
    return gr.update(interactive=False)


i2mv_examples = [
	["assets/examples/i2mv/cake.png", "cake"], 
	["assets/examples/i2mv/skull.png", "skull"],
	["assets/examples/i2mv/sea_turtle.png", "sea turtle"],
	["assets/examples/i2mv/house2.png", "house"],
	["assets/examples/i2mv/cup.png", "cup"],
	["assets/examples/i2mv/mannequin.png", "mannequin"],
	["assets/examples/i2mv/boy.jpg", "boy"],
	["assets/examples/i2mv/dragontoy.jpg", "dragon toy"],
	["assets/examples/i2mv/gso_rabbit.jpg", "rabbit car"],
	["assets/examples/i2mv/Mario_New_Super_Mario_Bros_U_Deluxe.png", "standing Mario"],
]

t2mv_examples = [
	"teddy bear",
	"hamburger",
	"oldman's head sculpture",
	"headphone",
	"mech suit",
	"wooden barrel",
	"scary zombie"
]

mv_examples = [
	[
		"assets/examples/mv_lq_prerendered/vase.mp4",
		"assets/examples/mv_lq/vase/00.png",
		"assets/examples/mv_lq/vase/01.png",
		"assets/examples/mv_lq/vase/02.png",
		"assets/examples/mv_lq/vase/03.png",
		"vase",
		0
	],
	[
		"assets/examples/mv_lq_prerendered/tower.mp4",
		"assets/examples/mv_lq/tower/00.png",
		"assets/examples/mv_lq/tower/01.png",
		"assets/examples/mv_lq/tower/02.png",
		"assets/examples/mv_lq/tower/03.png",
		"brick tower",
		0
	],
	[
		"assets/examples/mv_lq_prerendered/truck.mp4",
		"assets/examples/mv_lq/truck/00.png", 
		"assets/examples/mv_lq/truck/01.png", 
		"assets/examples/mv_lq/truck/02.png",
		"assets/examples/mv_lq/truck/03.png",
		"truck",
		0
	],
	[
		"assets/examples/mv_lq_prerendered/gascan.mp4",
		"assets/examples/mv_lq/gascan/00.png",
		"assets/examples/mv_lq/gascan/01.png",
		"assets/examples/mv_lq/gascan/02.png",
		"assets/examples/mv_lq/gascan/03.png",
		"gas can",
		0
	],
	[
		"assets/examples/mv_lq_prerendered/fish.mp4",
		"assets/examples/mv_lq/fish/00.png",
		"assets/examples/mv_lq/fish/01.png",
		"assets/examples/mv_lq/fish/02.png",
		"assets/examples/mv_lq/fish/03.png", 
		"sea fish with eyes",
		0
	],
	[
		"assets/examples/mv_lq_prerendered/tshirt.mp4",
		"assets/examples/mv_lq/tshirt/00.png",
		"assets/examples/mv_lq/tshirt/01.png",
		"assets/examples/mv_lq/tshirt/02.png",
		"assets/examples/mv_lq/tshirt/03.png",
		"t-shirt",
		0
	],
	[
		"assets/examples/mv_lq_prerendered/turtle.mp4",
		"assets/examples/mv_lq/turtle/00.png",
		"assets/examples/mv_lq/turtle/01.png",
		"assets/examples/mv_lq/turtle/02.png",
		"assets/examples/mv_lq/turtle/03.png",
		"sea turtle",
		200
	],
	[
		"assets/examples/mv_lq_prerendered/cake.mp4",
		"assets/examples/mv_lq/cake/00.png",
		"assets/examples/mv_lq/cake/01.png",
		"assets/examples/mv_lq/cake/02.png",
		"assets/examples/mv_lq/cake/03.png",
		"cake",
		120
	],
	[
		"assets/examples/mv_lq_prerendered/lamp.mp4",
		"assets/examples/mv_lq/lamp/00.png",
		"assets/examples/mv_lq/lamp/01.png",
		"assets/examples/mv_lq/lamp/02.png",
		"assets/examples/mv_lq/lamp/03.png",
		"lamp",
		0
	],
	[
		"assets/examples/mv_lq_prerendered/oldman.mp4",
		"assets/examples/mv_lq/oldman/00.png",
		"assets/examples/mv_lq/oldman/00.png",
		"assets/examples/mv_lq/oldman/00.png",
		"assets/examples/mv_lq/oldman/00.png",
		"old man sculpture",
		120
	],
	[
		"assets/examples/mv_lq_prerendered/mario.mp4",
		"assets/examples/mv_lq/mario/00.png",
		"assets/examples/mv_lq/mario/01.png",
		"assets/examples/mv_lq/mario/02.png",
		"assets/examples/mv_lq/mario/03.png",
		"standing mario",
		120
	],
	[
		"assets/examples/mv_lq_prerendered/house.mp4",
		"assets/examples/mv_lq/house/00.png",
		"assets/examples/mv_lq/house/01.png",
		"assets/examples/mv_lq/house/02.png",
		"assets/examples/mv_lq/house/03.png",
		"house",
		120
	],
]


# gradio UI
demo = gr.Blocks().queue()
with demo:
	gr.Markdown(title)
	gr.Markdown(authors)
	gr.Markdown(affiliation)
	gr.Markdown(important_link)
	gr.Markdown(description)
	
	original_video_path = gr.State(GRADIO_VIDEO_PATH)
	original_ply_path = gr.State(GRADIO_PLY_PATH)
	enhanced_video_path = gr.State(GRADIO_ENHANCED_VIDEO_PATH)
	enhanced_ply_path = gr.State(GRADIO_ENHANCED_PLY_PATH)

	with gr.Column(variant='panel'):
		with gr.Accordion("Generate Multi Views (LGM)", open=False, elem_id="mv-gen"):
			gr.Markdown("*Don't have multi-view images on hand? Generate them here using a single image, text, or a combination of both.*")
			with gr.Row():
				with gr.Column():
					ref_image = gr.Image(label="Reference Image", type='pil', height=400, interactive=True)
					ref_text = gr.Textbox(label="Prompt", value="", interactive=True)
				with gr.Column():
					gr.Examples(
						examples=i2mv_examples,
						inputs=[ref_image, ref_text],
						examples_per_page=3,
						label='Image-to-Multiviews Examples',
					)

					gr.Examples(
						examples=t2mv_examples,
						inputs=[ref_text], 
						outputs=[ref_image, ref_text],
						cache_examples=False,
						run_on_click=True,
						fn=lambda x: (None, x),
						label='Text-to-Multiviews Examples',
					)
					
			with gr.Row():
				gr.Column()  # Empty column for spacing
				button_gen_mv = gr.Button("Generate Multi Views", scale=1)
				gr.Column()  # Empty column for spacing

		with gr.Column():
			gr.Markdown("Let's enhance!")
			with gr.Row():
				with gr.Column(scale=2):
					with gr.Tab("Multi Views"):
						gr.Markdown("*Upload your multi-view images and enhance them with 3DEnhancer. You can also generate 3D model using LGM.*")
						with gr.Row():
							input_image_0 = gr.Image(label="[Input] view-0", type='pil', height=320)
							input_image_1 = gr.Image(label="[Input] view-1", type='pil', height=320)
							input_image_2 = gr.Image(label="[Input] view-2", type='pil', height=320)
							input_image_3 = gr.Image(label="[Input] view-3", type='pil', height=320)
						gr.Markdown("---")
						gr.Markdown("Enhanced Output")
						with gr.Row():
							enhanced_image_0 = ImageSlider(label="[Enhanced] view-0", type='pil', height=350, interactive=False)
							enhanced_image_1 = ImageSlider(label="[Enhanced] view-1", type='pil', height=350, interactive=False)
							enhanced_image_2 = ImageSlider(label="[Enhanced] view-2", type='pil', height=350, interactive=False)
							enhanced_image_3 = ImageSlider(label="[Enhanced] view-3", type='pil', height=350, interactive=False)
					with gr.Tab("Generated 3D"):
						gr.Markdown("Coarse Input")
						with gr.Column():
							with gr.Row():
								gr.Column()  # Empty column for spacing
								with gr.Column():
									input_3d_video = gr.Video(label="[Input] Rendered Video", height=300, scale=1, interactive=False)
									with gr.Row():
										button_gen_3d = gr.Button("Render 3D")
										button_download_3d = gr.DownloadButton("Download Ply", interactive=False)
										# button_download_3d = gr.File(label="Download Ply", interactive=False, height=50)
								gr.Column()  # Empty column for spacing
							gr.Markdown("---")
							gr.Markdown("Enhanced Output")
							with gr.Row():
								gr.Column()  # Empty column for spacing
								with gr.Column():
									enhanced_3d_video = gr.Video(label="[Enhanced] Rendered Video", height=300, scale=1, interactive=False)
									with gr.Row():
										enhanced_button_gen_3d = gr.Button("Render 3D")
										enhanced_button_download_3d = gr.DownloadButton("Download Ply", interactive=False)
								gr.Column()  # Empty column for spacing
			
			with gr.Column():
				with gr.Row():
					enhancer_text = gr.Textbox(label="Prompt", value="", scale=1)
					enhancer_noise_level = gr.Slider(label="enhancer noise level", minimum=0, maximum=300, step=1, value=0, interactive=True)
				with gr.Accordion("Addvanced Setting", open=False):
					with gr.Column():
						with gr.Row():
							with gr.Column():
								elevation = gr.Slider(label="elevation", minimum=-90, maximum=90, step=1, value=0)
								cfg_scale = gr.Slider(label="cfg scale", minimum=0, maximum=10, step=0.1, value=4.5)
							with gr.Column():
								seed = gr.Slider(label="random seed", minimum=0, maximum=100000, step=1, value=0)
								steps = gr.Slider(label="inference steps", minimum=1, maximum=100, step=1, value=20)
						with gr.Row():
							color_shift = gr.Radio(label="color shift", value="disabled", choices=["disabled", "adain", "wavelet"])
				with gr.Row():
					gr.Column()  # Empty column for spacing
					button_enhance = gr.Button("Enhance", scale=1, variant="primary")
					gr.Column()  # Empty column for spacing

			gr.Examples(
				examples=mv_examples,
				inputs=[input_3d_video, input_image_0, input_image_1, input_image_2, input_image_3, enhancer_text, enhancer_noise_level],
				examples_per_page=3,
				label='Multiviews Examples',
			)

			gr.Markdown("*Don't have multi-view images on hand but want to generate your own multi-viwes? Generate them [Here](#mv-gen).*")

	gr.Markdown(article)

	button_gen_mv.click(
		gen_mv, 
		inputs=[ref_image, ref_text], 
		outputs=[input_image_0, input_image_1, input_image_2, input_image_3, enhancer_text, enhancer_noise_level]
	)

	button_gen_3d.click(
		gen_3d,
		inputs=[input_image_0, input_image_1, input_image_2, input_image_3, elevation, original_video_path, original_ply_path],
		outputs=[input_3d_video, button_download_3d]
	).success(
        lambda: gr.Button(interactive=True),
        outputs=[button_download_3d],
    )

	enhanced_button_gen_3d.click(
		gen_3d,
		inputs=[enhanced_image_0, enhanced_image_1, enhanced_image_2, enhanced_image_3, elevation, original_video_path, original_ply_path],
		outputs=[enhanced_3d_video, enhanced_button_download_3d]
	).success(
        lambda: gr.Button(interactive=True),
        outputs=[enhanced_button_download_3d],
    )

	button_enhance.click(
		enhance,
		inputs=[input_image_0, input_image_1, input_image_2, input_image_3, enhancer_text, elevation, enhancer_noise_level, cfg_scale, steps, seed, color_shift],
		outputs=[enhanced_image_0, enhanced_image_1, enhanced_image_2, enhanced_image_3]
	).success(
		gen_3d,
		inputs=[input_image_0, input_image_1, input_image_2, input_image_3, elevation, original_video_path, original_ply_path],
		outputs=[input_3d_video, button_download_3d]
	).success(
        lambda: gr.Button(interactive=True),
        outputs=[button_download_3d],
    ).success(
		gen_3d,
		inputs=[enhanced_image_0, enhanced_image_1, enhanced_image_2, enhanced_image_3, elevation, enhanced_video_path, enhanced_ply_path],
		outputs=[enhanced_3d_video, enhanced_button_download_3d]
	).success(
        lambda: gr.Button(interactive=True),
        outputs=[enhanced_button_download_3d],
    )

demo.launch()