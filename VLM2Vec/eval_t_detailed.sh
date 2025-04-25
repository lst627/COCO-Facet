export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/your-path/hf_cache/"
DATASET_PATH="/your-path"

# Used for Original COCO
# python eval_t_detailed.py --model_name llava-hf/llava-1.5-7b-hf \
#   --model_backbone llava-hf/llava-1.5-7b-hf \
#   --encode_output_path output/outputs_t_detailed/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name Original_COCO_retrieval \
#   --embedding_type unimodal \
#   --image_resolution low \
#   --dataset_split test --per_device_eval_batch_size 32 \
#   --image_dir $DATASET_PATH/

# Used for time and scene and weather

# python eval_t_detailed.py --model_name llava-hf/llava-1.5-7b-hf \
#   --model_backbone llava-hf/llava-1.5-7b-hf \
#   --encode_output_path output/outputs_t_detailed/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name Visual7W_time_retrieval \
#   --embedding_type unimodal \
#   --image_resolution low \
#   --dataset_split test --per_device_eval_batch_size 32 \
#   --image_dir $DATASET_PATH/visual7w/images/


python eval_t_detailed.py --model_name llava-hf/llava-1.5-7b-hf \
  --model_backbone llava-hf/llava-1.5-7b-hf \
  --encode_output_path output/outputs_t_detailed/ \
  --num_crops 4 --max_len 256 \
  --pooling last --normalize True \
  --dataset_name TIGER-Lab/MMEB-eval \
  --subset_name Visual7W_scene_retrieval \
  --embedding_type unimodal \
  --image_resolution low \
  --dataset_split test --per_device_eval_batch_size 32 \
  --image_dir $DATASET_PATH/visual7w/images/

# python eval_t_detailed.py --model_name llava-hf/llava-1.5-7b-hf \
#   --model_backbone llava-hf/llava-1.5-7b-hf \
#   --encode_output_path output/outputs_t_detailed/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name Visual7W_people_num_retrieval \
#   --embedding_type unimodal \
#   --image_resolution low \
#   --dataset_split test --per_device_eval_batch_size 32 \
#   --image_dir $DATASET_PATH/visual7w/images/

# python eval_t_detailed.py --model_name llava-hf/llava-1.5-7b-hf \
#   --model_backbone llava-hf/llava-1.5-7b-hf \
#   --encode_output_path output/outputs_t_detailed/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name mix_weather_retrieval \
#   --embedding_type unimodal \
#   --image_resolution low \
#   --dataset_split test --per_device_eval_batch_size 32 \
#   --image_dir $DATASET_PATH/

# Used for objects
# python eval_t_detailed.py --model_name llava-hf/llava-1.5-7b-hf \
#   --model_backbone llava-hf/llava-1.5-7b-hf \
#   --encode_output_path output/outputs_t_detailed/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name COCO_object_retrieval \
#   --embedding_type unimodal \
#   --image_resolution low \
#   --dataset_split test --per_device_eval_batch_size 32 \
#   --image_dir $DATASET_PATH/

# Used for gestures and material
# python eval_t_detailed.py --model_name llava-hf/llava-1.5-7b-hf \
#   --model_backbone llava-hf/llava-1.5-7b-hf \
#   --encode_output_path output/outputs_t_detailed/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name COCO_gesture_retrieval \
#   --embedding_type unimodal \
#   --image_resolution low \
#   --dataset_split test --per_device_eval_batch_size 32 \
#   --image_dir $DATASET_PATH/val2017


# python eval_t_detailed.py --model_name llava-hf/llava-1.5-7b-hf \
#   --model_backbone llava-hf/llava-1.5-7b-hf \
#   --encode_output_path output/outputs_t_detailed/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name COCOStuff_material_retrieval \
#   --embedding_type unimodal \
#   --image_resolution low \
#   --dataset_split test --per_device_eval_batch_size 32 \
#   --image_dir $DATASET_PATH/val2017
