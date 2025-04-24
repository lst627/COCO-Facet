export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/your/path/hf_cache/"
# Used for side results
# python eval_d.py --model_name TIGER-Lab/VLM2Vec-Full \
#   --model_backbone phi3_v \
#   --encode_output_path outputs_d/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name Country211_retrieval \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir /your/path/eval_images/

# Used for Original COCO
# python eval_d.py --model_name TIGER-Lab/VLM2Vec-Full \
#   --model_backbone phi3_v \
#   --encode_output_path outputs_d/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name Original_COCO_retrieval \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir /your/path/


# Used for time and scene and weather
# python eval_d.py --model_name TIGER-Lab/VLM2Vec-Full \
#   --model_backbone phi3_v \
#   --encode_output_path outputs_d/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name Visual7W_time_retrieval \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir /your/path/visual7w/images/

# python eval_d.py --model_name TIGER-Lab/VLM2Vec-Full \
#   --model_backbone phi3_v \
#   --encode_output_path outputs_d/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name Visual7W_scene_retrieval \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir /your/path/visual7w/images/

python eval_d.py --model_name TIGER-Lab/VLM2Vec-Full \
  --model_backbone phi3_v \
  --encode_output_path outputs_d/ \
  --num_crops 4 --max_len 256 \
  --pooling last --normalize True \
  --dataset_name TIGER-Lab/MMEB-eval \
  --subset_name Visual7W_people_num_retrieval \
  --dataset_split test --per_device_eval_batch_size 64 \
  --image_dir /your/path/visual7w/images/

# python eval_d.py --model_name TIGER-Lab/VLM2Vec-Full \
#   --model_backbone phi3_v \
#   --encode_output_path outputs_d/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name mix_weather_retrieval \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir /your/path/

# Used for objects
# python eval_d.py --model_name TIGER-Lab/VLM2Vec-Full \
#   --model_backbone phi3_v \
#   --encode_output_path outputs_d/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name COCO_object_retrieval \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir /your/path/
  
# Used for gestures and material
# python eval_d.py --model_name TIGER-Lab/VLM2Vec-Full \
#   --model_backbone phi3_v \
#   --encode_output_path outputs_d/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name COCO_gesture_retrieval \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir /your/path/val2017/

# python eval_d.py --model_name TIGER-Lab/VLM2Vec-Full \
#   --model_backbone phi3_v \
#   --encode_output_path outputs_d/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name COCOStuff_material_retrieval \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir /your/path/val2017/











# python eval.py --lora --model_name microsoft/Phi-3.5-vision-instruct --checkpoint_path TIGER-Lab/VLM2Vec-LoRA \
#   --model_backbone phi3_v \
#   --encode_output_path outputs/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name N24News CIFAR-100 HatefulMemes VOC2007 SUN397 ImageNet-A ImageNet-R ObjectNet Country211 \
#   --dataset_split test --per_device_eval_batch_size 16 \
#   --image_dir eval_images/