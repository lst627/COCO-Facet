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

t2i_tasks = [
    "CIRR", "NIGHTS", "EDIS", "MSCOCO_t2i","VisDial","VisualNews_t2i","WebQA", "Wiki-SS-NQ", "OVEN", # retrieval
    "RefCOCO_retrieval", "Place365_retrieval", "SUN397_retrieval", "Country211_retrieval"]
i2t_tasks = [
    "MSCOCO_i2t","VisualNews_i2t", # retrieval
    "ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211" # classification
    ]

def batch_to_device(batch, device):
    _batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            _batch[key] = value.to(device)
        else:
            _batch[key] = value
    return _batch

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
    #     "COCO_object_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What objects are in the image?\n',
    #     "COCO_gesture_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What is the person doing in the image?\n',
    #     "COCOStuff_material_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What are the objects made of in the image?\n'
    # }
    # prompt = prompt_dict[data_args.subset_name[0]]
    gpt_prompt_dict = {
        # "Original_COCO_retrieval": f'<|image_1|>\nRepresent the given image.\n', 
        "SUN397_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What type of location is depicted in this image?\n', 
        "Place365_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What type of location is depicted in this image?\n',
        "Country211_retrieval": f'<|image_1|>\nRepresent the given image with the following question: Which country is shown in this image?\n',
        "Visual7W_time_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What time of day is depicted in this image?\n',
        "Visual7W_people_num_retrieval": f'<|image_1|>\nRepresent the given image with the following question: How many people are present in this image?\n',
        "Visual7W_scene_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What type of location is depicted in this image?\n',
        "mix_weather_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What is the weather like in this image?\n',
        "COCO_object_retrieval": f'<|image_1|>\nRepresent the given image with the following question: Which objects are present in this image?\n',
        "COCO_animal_retrieval": f'<|image_1|>\nRepresent the given image with the following question: Which animals can be seen in this image?\n',
        "COCO_gesture_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What gesture are the people making in this image?\n',
        "COCOStuff_material_retrieval": f'<|image_1|>\nRepresent the given image with the following question: What material are the objects in this image made of?\n'
    }
    prompt = gpt_prompt_dict[data_args.subset_name[0]]

    processor = AutoProcessor.from_pretrained(
        model_args.model_name,
        trust_remote_code=True,
        num_crops=model_args.num_crops,
    )

    processor.tokenizer.padding_side = "right"
    model = MMEBModel.load(model_args)
    model.eval()
    model = model.to(training_args.device, dtype=torch.bfloat16)

    eval_collator = EvalCollator(
        data_args=data_args,
        model_args=model_args,
        processor=processor,
    )

    # ToDo: This part of code is a little bit hacky. Need to refactor later.
    for idx, subset in enumerate(data_args.subset_name):
        score_path = os.path.join(data_args.encode_output_path, f"{subset}_score.json")
        if os.path.exists(score_path):
            try:
                with open(score_path, "r") as f:
                    score_dict = json.load(f)
                print(f"Found previous eval score, skipping {subset}")
                print(score_dict)
            except Exception as e:
                pass

        print(f"\033[91m{idx+1}/{len(data_args.subset_name)}: Processing {subset} now!\033[0m")
        encode_qry_path = os.path.join(data_args.encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(data_args.encode_output_path, f"{subset}_tgt")
        if os.path.exists(encode_qry_path) and os.path.exists(encode_tgt_path):
            continue

        eval_qry_dataset = JsonDataset(
            data_args=data_args,
            model_args=model_args,
            path="../benchmark/"+subset+".json",
            text_field="qry_text",
            img_path_field="qry_img_path",
        )
        eval_tgt_dataset = JsonDataset(
            data_args=data_args,
            model_args=model_args,
            path="../benchmark/"+subset+".json",
            text_field="tgt_text",
            img_path_field="tgt_img_path",
            prompt=prompt
        )

        eval_qry_loader = DataLoader(
            eval_qry_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )
        eval_tgt_loader = DataLoader(
            eval_tgt_dataset,
            batch_size=training_args.per_device_eval_batch_size,
            collate_fn=eval_collator,
            shuffle=False,
            drop_last=False,
            num_workers=training_args.dataloader_num_workers,
        )

        encoded_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_qry_loader, desc="Encode query"):
                batch = batch_to_device(batch, training_args.device)
                with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
                    output = model(qry=batch)
                encoded_tensor.append(output["qry_reps"].cpu().detach().float().numpy())
        encoded_tensor = np.concatenate(encoded_tensor)
        with open(encode_qry_path, 'wb') as f:
            pickle.dump((encoded_tensor, eval_qry_dataset.paired_data), f)

        encoded_tensor = []
        with torch.no_grad():
            for batch in tqdm(eval_tgt_loader, desc="Encode target"):
                batch = batch_to_device(batch, training_args.device)
                output = model(tgt=batch)
                encoded_tensor.append(output["tgt_reps"].cpu().detach().float().numpy())
        encoded_tensor = np.concatenate(encoded_tensor)
        with open(encode_tgt_path, 'wb') as f:
            pickle.dump((encoded_tensor, eval_tgt_dataset.paired_data), f)

    for subset in tqdm(data_args.subset_name, desc="calculate score"):
        encode_qry_path = os.path.join(data_args.encode_output_path, f"{subset}_qry")
        encode_tgt_path = os.path.join(data_args.encode_output_path, f"{subset}_tgt")
        with open(encode_qry_path, 'rb') as f:
            qry_tensor, qry_index = pickle.load(f)
        with open(encode_tgt_path, 'rb') as f:
            tgt_tensor, tgt_index = pickle.load(f)
        qry_dict, tgt_dict = {}, {}
        for qry_t, tt in zip(qry_tensor, qry_index):
            text, img_path = tt["text"], tt["img_path"]
            qry_dict[(text, img_path)] = qry_t
        for tgt_t, tt in zip(tgt_tensor, tgt_index):
            text, img_path = tt["text"], tt["img_path"]
            tgt_dict[(text, img_path)] = tgt_t

        eval_data = load_dataset(
            'json',
            data_files="../benchmark/"+subset+".json",
            split="train",
        )
        n_correct = 0
        all_pred = []
        total = 0
        for row in eval_data:
            total += 1
            qry_t = qry_dict[(row["qry_text"], '')]  # (dim,)
            tgt_t, all_candidates = [], []
            for tt in row["tgt_img_path"]:
                tgt_text = row["tgt_text"] if prompt == None else prompt
                tgt_t.append(tgt_dict[(tgt_text, tt)])
                all_candidates.append((tgt_text, tt))
            tgt_t = np.stack(tgt_t, axis=0)  # (num_candidate, dim)
            scores, pred = get_pred(qry_t, tgt_t, normalization=model_args.normalize)
            if isinstance(pred, list):
                if 0 in pred:
                    n_correct += 1
            else:
                if pred == 0:
                    n_correct += 1
            
            all_pred.append(all_candidates[pred])
        with open(os.path.join(data_args.encode_output_path, f"{subset}_pred.txt"), "w") as f:
            for item in all_pred:
                f.write(f"{item}\n")
        score_path = os.path.join(data_args.encode_output_path, f"{subset}_score.json")
        print(f"Outputting final score to: {score_path}")
        with open(score_path, "w") as f:
            score_dict = {"acc": n_correct/total, "num_correct": n_correct, "num_pred": total}
            json.dump(score_dict, f, indent=4)
        print(f"\033[91m{subset} accuracy: {n_correct/total}\033[0m")


if __name__ == "__main__":
    main()
