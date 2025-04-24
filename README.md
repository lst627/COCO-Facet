# COCO-Facet

This repository contains code for the COCO-Facet benchmark for text-to-image retrieval based on selective semantic elements ("Facets" of the images). The benchmark can be downloaded [here](https://www.dropbox.com/scl/fo/hbkknl14pj5wwgpphbt6l/AC15YovOLv65Ek3hE4kib1o?rlkey=fhphyfml0uc6ctnb70v95id1n&st=p1zui6ni&dl=0). Please put the downloaded json files into the "benchmark" folder.

The images are from COCO val2014 and val2017. 

```bash
git clone https://github.com/lst627/COCO-Facet.git
cd COCO-Facet
git submodule init
git submodule update
```

## Acknowledgment

This code is based on the [VLM2Vec repository](https://github.com/TIGER-AI-Lab/VLM2Vec).