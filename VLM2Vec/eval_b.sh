export CUDA_VISIBLE_DEVICES=1
export HF_HOME="/your-path/hf_cache/"

DATASET_PATH="/your-path"

# Used for Original COCO
# python eval_b.py --model_name openai/clip-vit-large-patch14-336 \
#   --model_backbone openai/clip-vit-large-patch14-336 \
#   --encode_output_path output/outputs_b/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name Original_COCO_retrieval \
#   --embedding_type unimodal \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir $DATASET_PATH/

# Used for time and scene and weather and people num
# python eval_b.py --model_name openai/clip-vit-large-patch14-336 \
#   --model_backbone openai/clip-vit-large-patch14-336 \
#   --encode_output_path output/outputs_b/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name Visual7W_time_retrieval \
#   --embedding_type unimodal \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir $DATASET_PATH/visual7w/images/

python eval_b.py --model_name openai/clip-vit-large-patch14-336 \
  --model_backbone openai/clip-vit-large-patch14-336 \
  --encode_output_path output/outputs_b/ \
  --num_crops 4 --max_len 256 \
  --pooling last --normalize True \
  --dataset_name TIGER-Lab/MMEB-eval \
  --subset_name Visual7W_scene_retrieval \
  --embedding_type unimodal \
  --dataset_split test --per_device_eval_batch_size 64 \
  --image_dir $DATASET_PATH/visual7w/images/

# python eval_b.py --model_name openai/clip-vit-large-patch14-336 \
#   --model_backbone openai/clip-vit-large-patch14-336 \
#   --encode_output_path output/outputs_b/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name Visual7W_people_num_retrieval \
#   --embedding_type unimodal \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir $DATASET_PATH/visual7w/images/

# python eval_b.py --model_name openai/clip-vit-large-patch14-336 \
#   --model_backbone openai/clip-vit-large-patch14-336 \
#   --encode_output_path output/outputs_b/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name mix_weather_retrieval \
#   --embedding_type unimodal \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir $DATASET_PATH/



# Used for objects
# python eval_b.py --model_name openai/clip-vit-large-patch14-336 \
#   --model_backbone openai/clip-vit-large-patch14-336 \
#   --encode_output_path output/outputs_b/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name COCO_object_retrieval \
#   --embedding_type unimodal \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir $DATASET_PATH/

# python eval_b.py --model_name openai/clip-vit-large-patch14-336 \
#   --model_backbone openai/clip-vit-large-patch14-336 \
#   --encode_output_path output/outputs_b/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name COCO_animal_retrieval \
#   --embedding_type unimodal \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir $DATASET_PATH/

# Used for gestures and material
# python eval_b.py --model_name openai/clip-vit-large-patch14-336 \
#   --model_backbone openai/clip-vit-large-patch14-336 \
#   --encode_output_path output/outputs_b/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name COCO_gesture_retrieval \
#   --embedding_type unimodal \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir $DATASET_PATH/val2017/

# python eval_b.py --model_name openai/clip-vit-large-patch14-336 \
#   --model_backbone openai/clip-vit-large-patch14-336 \
#   --encode_output_path output/outputs_b/ \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name COCOStuff_material_retrieval \
#   --embedding_type unimodal \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir $DATASET_PATH/val2017/

