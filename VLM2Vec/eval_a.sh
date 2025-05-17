export CUDA_VISIBLE_DEVICES=0
export HF_HOME="/your-path/hf_cache/"
DATASET_PATH="/your-path"

# Used for Original COCO
# python eval_a_la.py --model_name TIGER-Lab/VLM2Vec-Full \
#   --model_backbone phi3_v \
#   --encode_output_path output/outputs_a \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name Original_COCO_retrieval \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir $DATASET_PATH/

# # Used for time and scene and weather
# python eval_a_la.py --model_name TIGER-Lab/VLM2Vec-Full \
#   --model_backbone phi3_v \
#   --encode_output_path output/outputs_a \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name Visual7W_time_retrieval \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir $DATASET_PATH/visual7w/images/

# python eval_a_la.py --model_name TIGER-Lab/VLM2Vec-Full \
#   --model_backbone phi3_v \
#   --encode_output_path output/outputs_a \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name Visual7W_scene_retrieval \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir $DATASET_PATH/visual7w/images/

# python eval_a_la.py --model_name TIGER-Lab/VLM2Vec-Full \
#   --model_backbone phi3_v \
#   --encode_output_path output/outputs_a \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name Visual7W_people_num_retrieval \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir $DATASET_PATH/visual7w/images/

# python eval_a_la.py --model_name TIGER-Lab/VLM2Vec-Full \
#   --model_backbone phi3_v \
#   --encode_output_path output/outputs_a \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name mix_weather_retrieval \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir $DATASET_PATH/

# # # Used for objects
# python eval_a_la.py --model_name TIGER-Lab/VLM2Vec-Full \
#   --model_backbone phi3_v \
#   --encode_output_path output/outputs_a \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name COCO_object_retrieval \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir $DATASET_PATH/

# python eval_a_la.py --model_name TIGER-Lab/VLM2Vec-Full \
#   --model_backbone phi3_v \
#   --encode_output_path output/outputs_a \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name COCO_animal_retrieval \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir $DATASET_PATH/

# # # Used for gestures and material
# python eval_a_la.py --model_name TIGER-Lab/VLM2Vec-Full \
#   --model_backbone phi3_v \
#   --encode_output_path output/outputs_a \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name COCO_gesture_retrieval \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir $DATASET_PATH/val2017/

# python eval_a_la.py --model_name TIGER-Lab/VLM2Vec-Full \
#   --model_backbone phi3_v \
#   --encode_output_path output/outputs_a \
#   --num_crops 4 --max_len 256 \
#   --pooling last --normalize True \
#   --dataset_name TIGER-Lab/MMEB-eval \
#   --subset_name COCOStuff_material_retrieval \
#   --dataset_split test --per_device_eval_batch_size 64 \
#   --image_dir $DATASET_PATH/val2017/




