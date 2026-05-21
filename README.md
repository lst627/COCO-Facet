# COCO-Facet

[![arXiv](https://img.shields.io/badge/arXiv-2505.15877-b31b1b.svg)](https://arxiv.org/abs/2505.15877) [![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/lst627/COCO-Facet)

This repository contains code for the COCO-Facet benchmark for attribute-focused text-to-image retrieval ("Facets" of the images). The benchmark can be downloaded from [Hugging Face](https://huggingface.co/datasets/lst627/COCO-Facet) or [Dropbox](https://www.dropbox.com/scl/fo/hbkknl14pj5wwgpphbt6l/AC15YovOLv65Ek3hE4kib1o?rlkey=fhphyfml0uc6ctnb70v95id1n&st=qrhpzs3o&dl=0). Please place the downloaded json files in the "benchmark" folder for evaluation.

## Downloading the Images

The annotations are from MSCOCO 2017, COCO-Stuff, Visual7W, and VisDial about COCO images. Since they reindexed the images, we recommend downloading the images at [MSCOCO_val2017](http://images.cocodataset.org/zips/val2017.zip), [VisDial_val2018](https://www.dropbox.com/s/twmtutniktom7tu/VisualDialog_val2018.zip), [Visual7W](http://vision.stanford.edu/yukezhu/), or jointly from [Hugging Face](https://huggingface.co/datasets/lst627/COCO-Facet)

## Environment

```bash
conda create -n facet python=3.10
pip install -r VLM2Vec/requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

## Evaluation

Please first modify the dataset path and huggingface model path in the scripts. Then you can start evaluation inside the "VLM2Vec" folder.

For CLIP-ViT-L/14-336px:
```bash
sh eval_b.sh
```

For VLM2Vec without any attribute-specific prompt:
```bash
sh eval_d.sh

```
For VLM2Vec with GPT prompts:
```bash
sh eval_f.sh
```
We also attached the human-written prompts in eval_f.py.

For the text-based retrieval:
```bash
sh eval_t_detailed.sh
```

For VLM2Vec with GPT-chosen prompts at test time:
```bash
sh eval_e.sh
```
We have attached the GPT responses under output/outputs_e, which can be reused.

For VLM2Vec with linear approximated promptable embeddings:
```bash
sh eval_a.sh
```
Note that we need the embeddings given by "eval_f.sh" and "eval_d.sh" to derive the matrix W.

We include the collators for other MLLM-based universal multimodal embedders in VLM2Vec/src/collator.py.

## Dataset Construction

We attach the dataset construction process in the .ipynb files in the "construction" folder.

## Acknowledgment

This code is mainly based on the [VLM2Vec repository](https://github.com/TIGER-AI-Lab/VLM2Vec).


## Citation

If you find our code, data, or the paper useful, please cite the paper:

```
@inproceedings{NEURIPS2025_a6e3dac8,
 author = {Li, Siting and Gao, Xiang and Du, Simon},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {D. Belgrave and C. Zhang and H. Lin and R. Pascanu and P. Koniusz and M. Ghassemi and N. Chen},
 pages = {115077--115110},
 publisher = {Curran Associates, Inc.},
 title = {Highlighting What Matters: Promptable Embeddings for Attribute-Focused Image Retrieval},
 url = {https://proceedings.neurips.cc/paper_files/paper/2025/file/a6e3dac86d06ce2a815a946ef008c1de-Paper-Conference.pdf},
 volume = {38},
 year = {2025}
}
```
