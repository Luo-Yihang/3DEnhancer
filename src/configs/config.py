image_size = 512

# model setting
model = 'PixArtMS_XL_2'
use_crossview_module = True

mixed_precision = 'bf16'  # ['fp16', 'fp32', 'bf16']
fp32_attention = True
pipeline_load_from = "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
aspect_ratio_type = 'ASPECT_RATIO_512'
pe_interpolation = 1.0

# pixart-sigma
scale_factor = 0.13025
model_max_length = 300
kv_compress = False
kv_compress_config = {
    'sampling': 'conv',  # ['conv', 'uniform', 'ave']
    'scale_factor': 2,
    'kv_compress_layer': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
}
qk_norm = False
micro_condition = False

# controlnet
copy_blocks_num = 13

# diffusion sampling
train_sampling_steps = 1000