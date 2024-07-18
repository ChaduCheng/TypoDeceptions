<br />
<p align="center">
  <h1 align="center">Unveiling Typographic Deceptions: Insights of the Typographic Vulnerability in Large Vision-Language Model</h1>
  <p align="center">
    Hao Cheng*,
    Erjia Xiao*,
    Jindong Gu,
    Le Yang,
    Jinhao Duan,
    Jize Zhang,
    Jiahang Cao,
    Kaidi Xu,
    Renjing Xu<sup>†</sup>
  </p>
  <p align="center" >
    <em>HKUST & University of Oxford & Drexel University & Xi’an Jiaotong University</em> 
  </p>
  <p align="center">
    <a href='https://arxiv.org/abs/2402.19150'>
      <img src='https://img.shields.io/badge/Paper-Arxiv-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://huggingface.co/datasets/erjiaxiao/Typographic-Dataset' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Dataset-Hugging%20Face-blue?style=flat&logo=HuggingFace&logoColor=white' alt='Dataset'>
    </a>
  </p>
  <p align="center">
    <img src="figs/main.png" alt="Logo" width="100%">
  </p>
</p>



## Installation

1. Please follow the instructions in [LLaVA](https://github.com/haotian-liu/LLaVA), [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) and [MiniGPT4](https://github.com/Vision-CAIR/MiniGPT-4) to set up the codebase, model weights and conda environment for further experiments.

2. Download the [Typographic Dataset](https://huggingface.co/datasets/erjiaxiao/Typographic-Dataset).

3. Clone this repository into the codebase mentioned above. For instance, after installing [LLaVA](https://github.com/haotian-liu/LLaVA),
```bash
cd LLaVA
git clone https://github.com/ChaduCheng/TypoDeceptions.git
```

## Acknowledgement

* [LLaVA](https://github.com/haotian-liu/LLaVA): Large Language and Vision Assistant
* [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4): Enhancing Vision-language Understanding with Advanced Large Language Models
* [InstructBLIP](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip): Towards General-purpose Vision-Language Models with Instruction Tuning
* [CLIP](https://github.com/openai/CLIP): Learning Transferable Visual Models From Natural Language Supervision

If you find our work useful for your research and applications, please cite using this BibTeX:

```
@article{cheng2024unveiling,
  title={Unveiling Typographic Deceptions: Insights of the Typographic Vulnerability in Large Vision-Language Model},
  author={Cheng, Hao and Xiao, Erjia and Gu, Jindong and Yang, Le and Duan, Jinhao and Zhang, Jize and Cao, Jiahang and Xu, Kaidi and Xu, Renjing},
  journal={ECCV},
  year={2024}
}
```
