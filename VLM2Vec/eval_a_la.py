import json
import sys

from src.arguments import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoProcessor

from src.model import MMEBModel
from src.dataset import EvalDataset, JsonDataset
from src.collator import EvalCollator
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import pickle
import os
from datasets import load_dataset
from evaluation.eval_utils import get_pred, save_results, print_results
from numpy.linalg import norm, pinv, svd

t2i_tasks = [
    "CIRR", "NIGHTS", "EDIS", "MSCOCO_t2i","VisDial","VisualNews_t2i","WebQA", "Wiki-SS-NQ", "OVEN", # retrieval
    "RefCOCO_retrieval", "Place365_retrieval", "SUN397_retrieval", "Country211_retrieval"]
i2t_tasks = [
    "MSCOCO_i2t","VisualNews_i2t", # retrieval
    "ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211" # classification
    ]

def derive_W(u_samples, v_samples):
    """
    Find the optimal transformation matrix W such that Wu approximates v,
    where both u and v are unit vectors.
    
    Args:
        u_samples: List of unit vectors u
        v_samples: List of unit vectors v (corresponding targets)
        
    Returns:
        W: Optimal transformation matrix 
    """
    # Convert to numpy arrays
    U = np.array(u_samples).T  # Each u as a column
    V = np.array(v_samples).T  # Each v as a column
    
    # Initial solution: W = V * U^+
    W_initial = V @ pinv(U)
    
    U_svd, singular_values, Vt_svd = svd(W_initial, full_matrices=False)
    W_orthogonal = U_svd @ Vt_svd
    
    return W_initial, singular_values


def main():
    for arg in sys.argv:
        if arg.startswith("--local-rank="):
            rank = arg.split("=")[1]
            sys.argv.remove(arg)
            sys.argv.append('--local_rank')
            sys.argv.append(rank)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: DataArguments
    training_args: TrainingArguments
    os.makedirs(data_args.encode_output_path, exist_ok=True)
    # prompt_dict = {
    #     # "Original_COCO_retrieval": f'<|image_1|>\nRepresent the given image.\n', 
    #     "SUN397_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What scene is in the image?\n', 
    #     "Place365_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What scene is in the image?\n',
    #     "Country211_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What country is the scene located in the image?\n',
    #     "Visual7W_time_retrieval": f'<|image_1|>\nRepresent the given image with the following question: When is the image taken?\n',
    #     "Visual7W_people_num_retrieval": f'<|image_1|>\nRepresent the given image with the following question: How many people are in the image?\n',
    #     "Visual7W_scene_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What scene is in the image?\n',
    #     "mix_weather_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What is the weather in the image?\n',
    #     "COCO_animal_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What animals are in this image?\n',
    #     "COCO_object_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What objects are in the image?\n',
    #     "COCO_gesture_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What is the person doing in the image?\n',
    #     "COCOStuff_material_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What are the objects made of in the image?\n'
    # }
    gpt_prompt_dict = {
        # "Original_COCO_retrieval": f'<|image_1|>\nRepresent the given image.\n', 
        "SUN397_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What type of location is depicted in this image?\n', 
        "Place365_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What type of location is depicted in this image?\n',
        "Country211_retrieval": f'<|image_1|>\nRepresent the given image with the following question: Which country is shown in this image?\n',
        "Visual7W_time_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What time of day is depicted in this image?\n',
        "Visual7W_people_num_retrieval": f'<|image_1|>\nRepresent the given image with the following question: How many people are present in this image?\n',
        "Visual7W_scene_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What type of location is depicted in this image?\n',
        "mix_weather_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What is the weather like in this image?\n',
        "COCO_animal_retrieval": f'<|image_1|>\nRepresent the given image with the following question: Which animals can be seen in this image?\n',
        "COCO_object_retrieval": f'<|image_1|>\nRepresent the given image with the following question: Which objects are present in this image?\n',
        "COCO_gesture_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What gesture are the people making in this image?\n',
        "COCOStuff_material_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What material are the objects in this image made of?\n'
    }
    prompt = gpt_prompt_dict[data_args.subset_name[0]]

    # ToDo: This part of code is a little bit hacky. Need to refactor later.

    for subset in tqdm(data_args.subset_name, desc="calculate score"):
        encode_qry_path = os.path.join("output/outputs_f", f"{subset}_qry")
        vlm2vec_encode_tgt_path = os.path.join("output/outputs_f", f"{subset}_tgt")
        clip_encode_tgt_path = os.path.join("output/outputs_d", f"{subset}_tgt")
        with open(encode_qry_path, 'rb') as f:
            qry_tensor, qry_index = pickle.load(f)
        with open(vlm2vec_encode_tgt_path, 'rb') as f:
            vlm2vec_tgt_tensor, vlm2vec_tgt_index = pickle.load(f)
        with open(clip_encode_tgt_path, 'rb') as f:
            clip_tgt_tensor, clip_tgt_index = pickle.load(f)

        qry_dict, vlm2vec_tgt_dict, clip_tgt_dict = {}, {}, {}
        for qry_t, tt in zip(qry_tensor, qry_index):
            text, img_path = tt["text"], tt["img_path"]
            qry_dict[(text, img_path)] = qry_t
        for tgt_t, tt in zip(vlm2vec_tgt_tensor, vlm2vec_tgt_index):
            text, img_path = tt["text"], tt["img_path"]
            vlm2vec_tgt_dict[(text, img_path)] = tgt_t
        for tgt_t, tt in zip(clip_tgt_tensor, clip_tgt_index):
            text, img_path = tt["text"], tt["img_path"]
            clip_tgt_dict[(text, img_path)] = tgt_t


        vlm2vec_tgt, clip_tgt = [], []
        for text, img_path in vlm2vec_tgt_dict.keys():
            vlm2vec_tgt.append(vlm2vec_tgt_dict[(text, img_path)])
            clip_tgt.append(clip_tgt_dict[("<|image_1|> Represent the given image.", img_path)])

        sample_num = 100 
        # Randomly sample from vlm2vec_tgt and clip_tgt for W
        indices = np.random.choice(len(vlm2vec_tgt), size=sample_num, replace=False)
        vlm2vec_tgt = [vlm2vec_tgt[i] for i in indices]
        clip_tgt = [clip_tgt[i] for i in indices]
        vlm2vec_tgt = np.stack(vlm2vec_tgt, axis=0) # (num_candidate, dim)
        clip_tgt = np.stack(clip_tgt, axis=0)  # (num_candidate, dim)
            
        output_dim = vlm2vec_tgt.shape[1]
        input_dim = clip_tgt.shape[1]
        # print(output_dim, input_dim) 3072 768

        # Pad clip_tgt to match output_dim
        if clip_tgt.shape[1] < output_dim:
            padding = np.zeros((clip_tgt.shape[0], output_dim - clip_tgt.shape[1]))
            clip_tgt = np.concatenate((clip_tgt, padding), axis=1)

        vlm2vec_tgt_unit = vlm2vec_tgt / np.linalg.norm(vlm2vec_tgt, axis=1, keepdims=True)  # Normalize to unit vectors
        clip_tgt_unit = clip_tgt / np.linalg.norm(clip_tgt, axis=1, keepdims=True)  # Normalize to unit vectors
        W, sv = derive_W(clip_tgt_unit, vlm2vec_tgt_unit)
        
        eval_data = load_dataset(
            'json',
            data_files="../benchmark/"+subset+".json",
            split="train",
        )
        n_correct_1, n_correct_5 = 0, 0
        all_pred = []
        total = 0
        for row in eval_data:
            total += 1

            qry_t = qry_dict[(row["qry_text"], '')]  # (dim,)
            transformed_qry_t = W.T @ qry_t  # Apply W.T to qry_t
            
            tgt_t, vlm2vec_tgt_t, all_candidates = [], [], []
            for tt in row["tgt_img_path"]:
                t = clip_tgt_dict[("<|image_1|> Represent the given image.", tt)]
                tgt_t.append(t)
                
            tgt_t = np.stack(tgt_t, axis=0)  # (num_candidate, dim)

            scores, pred_1 = get_pred(transformed_qry_t, tgt_t, normalization=model_args.normalize)
            if pred_1 == 0:
                n_correct_1 += 1
            scores, pred_5 = get_pred(transformed_qry_t, tgt_t, normalization=model_args.normalize, top_k=5)
            if 0 in pred_5:
                n_correct_5 += 1
            
        print(f"\033[91m{subset} accuracy (recall@1): {n_correct_1/total}\033[0m")
        print(f"\033[91m{subset} accuracy (recall@5): {n_correct_5/total}\033[0m")


if __name__ == "__main__":
    main()
