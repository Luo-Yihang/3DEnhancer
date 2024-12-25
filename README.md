<div align="center">
    <h1>
    3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement
    </h1>
</div>

<div>
    <h4 align="center">
        <a href="https://yihangluo.com/projects/3DEnhancer" target='_blank'>
        <img src="https://img.shields.io/badge/🐳-Project%20Page-blue">
        </a>
        <a href="https://arxiv.org/abs/2412.18565" target='_blank'>
        <img src="https://img.shields.io/badge/arXiv-2312.06640-b31b1b.svg">
        </a>
        <a href="https://youtu.be/N7bfyd7B4D8" target='_blank'>
        <img src="https://img.shields.io/badge/Demo%20Video-%23FF0000.svg?logo=YouTube&logoColor=white">
        </a>
        <img src="https://api.infinitescript.com/badgen/count?name=sczhou/3DEnhancer&ltext=Visitors&color=3977dd">
    </h4>
</div>

<div align="center">
    <p>
        <span style="font-variant: small-caps;"><strong>3DEnhancer</strong></span> employs a multi-view 
        diffusion model to enhance multi-view images, thus improving 3D models.
    </p>
    <img width="820" alt="pipeline" src="assets/teaser.gif">
    :open_book: For more visual results, go checkout our <a href="https://yihangluo.com/projects/3DEnhancer" target="_blank">project page</a>
</div>

<br>

<details>
<summary><b>Introducing 3DEnhancer</b></summary>
    <br>
    <div align="center">
        <img width="820" alt="pipeline" src="assets/method_overview.png">
        <p align="justify">
            Despite advances in neural rendering, due to the scarcity of high-quality 3D datasets 
            and the inherent limitations of multi-view diffusion models, view synthesis and 3D model 
            generation are restricted to low resolutions with suboptimal multi-view consistency. 
            In this study, we present a novel 3D enhancement pipeline, dubbed <span style="font-variant: small-caps;"><strong>3DEnhancer</strong></span>, which employs 
            a multi-view latent diffusion model to enhance coarse 3D inputs while preserving multi-view consistency. 
            Our method includes a <strong>pose-aware encoder</strong> and a <strong>diffusion-based denoiser</strong> to refine low-quality 
            multi-view images, along with <strong>data augmentation</strong> and a <strong>multi-view attention module with epipolar 
            aggregation</strong> to maintain consistent, high-quality 3D outputs across views. Unlike existing video-based 
            approaches, our model supports seamless multi-view enhancement with improved coherence across diverse 
            viewing angles. Extensive evaluations show that <span style="font-variant: small-caps;">3DEnhancer</span> significantly outperforms existing methods, 
            boosting both multi-view enhancement and per-instance 3D optimization tasks.
        </p>
    </div>
</details>


## :fire: News

- [2024/12/25] Our paper and project page are now live. Merry Christmas!


### :calendar: TODO

- [x] Release paper and project page.
- [ ] Release code (coming soon!).
- [ ] Release Gradio demo.


## :pencil: Citation

If you find our code or paper helps, please consider citing:

```bibtex
@article{luo20243denhancer,
    title={3DEnhancer: Consistent Multi-View Diffusion for 3D Enhancement}, 
    author={Yihang Luo and Shangchen Zhou and Yushi Lan and Xingang Pan and Chen Change Loy},
    booktitle={arXiv preprint arXiv:2412.18565}
    year={2024},
}
```

## :mailbox: Contact
If you have any questions, please feel free to reach us at `luo_yihang@outlook.com`. 