import math
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer
from safetensors.torch import load_model

from diffusion import IDDPM, DPMS
from diffusion.utils.misc import read_config
from diffusion.model.nets import PixArtMS_XL_2, ControlPixArtMSMVHalfWithEncoder
from diffusion.utils.data import ASPECT_RATIO_512_TEST
from utils.camera import get_camera_poses
from utils.postprocess import adaptive_instance_normalization, wavelet_reconstruction


class Enhancer:
    def __init__(self, model_path, config_path):
        self.config = read_config(config_path)

        self.image_size = self.config.image_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weight_dtype = torch.float16

        self._load_model(model_path, self.config.pipeline_load_from)

    def _load_model(self, model_path, pipeline_load_from):
        self.tokenizer = T5Tokenizer.from_pretrained(pipeline_load_from, subfolder="tokenizer")
        self.text_encoder = T5EncoderModel.from_pretrained(pipeline_load_from, subfolder="text_encoder", torch_dtype=self.weight_dtype).to(self.device)

        self.vae = AutoencoderKL.from_pretrained(pipeline_load_from, subfolder="vae", torch_dtype=self.weight_dtype).to(self.device)
        del self.vae.encoder  # we do not use vae encoder

        # only support fixed latent size currently
        latent_size = self.image_size // 8
        lewei_scale = {512: 1, 1024: 2}
        model_kwargs = {
            "model_max_length": self.config.model_max_length, 
            "qk_norm": self.config.qk_norm, 
            "kv_compress_config": self.config.kv_compress_config if self.config.kv_compress else None, 
            "micro_condition": self.config.micro_condition,
            "use_crossview_module": getattr(self.config, 'use_crossview_module', False),
        }
        model = PixArtMS_XL_2(input_size=latent_size, pe_interpolation=lewei_scale[self.image_size], **model_kwargs).to(self.device)
        model = ControlPixArtMSMVHalfWithEncoder(model).to(self.weight_dtype).to(self.device)
        load_model(model, model_path)
        model.eval()
        self.model = model

        self.noise_maker = IDDPM(str(self.config.train_sampling_steps))
        
    @torch.no_grad()
    def _encode_prompt(self, text_prompt, n_views):
        txt_tokens = self.tokenizer(
            text_prompt, 
            max_length=self.config.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        caption_embs = self.text_encoder(
            txt_tokens.input_ids, 
            attention_mask=txt_tokens.attention_mask)[0][:, None]
        emb_masks = txt_tokens.attention_mask

        caption_embs = caption_embs.repeat_interleave(n_views, dim=0).to(self.weight_dtype)
        emb_masks = emb_masks.repeat_interleave(n_views, dim=0).to(self.weight_dtype)

        return caption_embs, emb_masks
    
    @torch.no_grad()
    def inference(self, mv_imgs, c2ws, prompt="", fov=math.radians(49.1), noise_level=120, cfg_scale=4.5, sample_steps=20, color_shift=None):
        mv_imgs = F.interpolate(mv_imgs, size=(512, 512), mode='bilinear', align_corners=False)

        n_views = mv_imgs.shape[0]
        # pixle-sigma input tensor range is [-1, 1]
        mv_imgs = 2.*mv_imgs - 1.

        originial_mv_imgs = mv_imgs.clone().to(self.device)
        if noise_level == 0:
            noise_level = torch.zeros((n_views,)).long().to(self.device)
        else:
            noise_level = noise_level * torch.ones((n_views,)).long().to(self.device)
            mv_imgs = self.noise_maker.q_sample(mv_imgs.to(self.device), noise_level-1)

        cur_camera_pose, epipolar_constrains, cam_distances = get_camera_poses(c2ws=c2ws, fov=fov, h=mv_imgs.size(-2), w=mv_imgs.size(-1))
        epipolar_constrains = epipolar_constrains.to(self.device)
        cam_distances = cam_distances.to(self.weight_dtype).to(self.device)
        
        caption_embs, emb_masks = self._encode_prompt(prompt, n_views)
        null_y = self.model.y_embedder.y_embedding[None].repeat(n_views, 1, 1)[:, None]

        latent_size_h, latent_size_w = mv_imgs.size(-2) // 8, mv_imgs.size(-1) // 8
        z = torch.randn(n_views, 4, latent_size_h, latent_size_w, device=self.device)
        z_lq = self.model.encode(
            mv_imgs.to(self.weight_dtype).to(self.device), 
            cur_camera_pose.to(self.weight_dtype).to(self.device),
            n_views=n_views,
        )

        model_kwargs = dict(
            c=torch.cat([z_lq] * 2), 
            data_info={}, 
            mask=emb_masks, 
            noise_level=torch.cat([noise_level] * 2), 
            epipolar_constrains=torch.cat([epipolar_constrains] * 2), 
            cam_distances=torch.cat([cam_distances] * 2),
            n_views=n_views,
        )
        dpm_solver = DPMS(
            self.model.forward_with_dpmsolver,
            condition=caption_embs,
            uncondition=null_y,
            cfg_scale=cfg_scale,
            model_kwargs=model_kwargs
        )
        samples = dpm_solver.sample(
            z,
            steps=sample_steps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
            disable_progress_ui=False,
        )

        samples = samples.to(self.weight_dtype)

        output_mv_imgs = self.vae.decode(samples / self.vae.config.scaling_factor).sample

        if color_shift == "adain":
            for i, output_mv_img in enumerate(output_mv_imgs):
                output_mv_imgs[i] = adaptive_instance_normalization(output_mv_img.unsqueeze(0), originial_mv_imgs[i:i+1]).squeeze(0)
        elif color_shift == "wavelet":
            for i, output_mv_img in enumerate(output_mv_imgs):
                output_mv_imgs[i] = wavelet_reconstruction(output_mv_img.unsqueeze(0), originial_mv_imgs[i:i+1]).squeeze(0)

        output_mv_imgs = torch.clamp((output_mv_imgs + 1.) / 2., 0, 1)

        torch.cuda.empty_cache()
        return output_mv_imgs