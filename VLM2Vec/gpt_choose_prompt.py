import os
os.environ["HF_HOME"] = "/your-path/hf_cache/"
from openai import OpenAI
import time
import json
import re

NUM_SECONDS_TO_SLEEP = 10
client = OpenAI(
    # This is the default and can be omitted
    api_key="your-key",
)
# Define a function to query the OpenAI API and evaluate the answer
def get_answer(question):
    while True:
        try:
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a helpful and precise assistant for choosing the best prompt. Please answer in one letter.'
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
            )

            break
        except Exception as e:
            print("error")
            print(e)
        time.sleep(NUM_SECONDS_TO_SLEEP)

    answer =  completion.choices[0].message.content.replace(".", "")
    yes_no_regex = re.compile(r"^(a|b|c|d|e|f|g|h|i|j)$", re.IGNORECASE)

    if yes_no_regex.match(answer):
        return answer.upper()
    else:
        print("Could not determine the answer. "  + question + " " +answer)
        return "A"

def gpt_choose_answer_for_text(text):
    # prompt_dict = {
    #     "Original_COCO_retrieval": f'<|image_1|>\nRepresent the given image.\n', 
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
    gpt_prompt_dict = {
        "Original_COCO_retrieval": f'<|image_1|>\nRepresent the given image.\n', 
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
    prompts = list(set(gpt_prompt_dict.values()))
    options = {}
    Prompts = ""
    for idx, p in enumerate(prompts):
        options[chr(65 + idx)] = p
        Prompts += chr(65 + idx)+". "+p.replace('<|image_1|>\n', "")

    question4gpt = f'{Prompts} Given the instruction "{text}", choose the most relevant prompt for verifying the results. Please answer in one letter.'
    gpt_answer = get_answer(question4gpt)

    return options[gpt_answer]

def gpt_choose_answer_for_subset(subset):
    # prompt_dict = {
    #     "Original_COCO_retrieval": f'<|image_1|>\nRepresent the given image.\n', 
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
    gpt_prompt_dict = {
        "Original_COCO_retrieval": f'<|image_1|>\nRepresent the given image.\n', 
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
    gt_prompt = gpt_prompt_dict[subset]

    text_file = json.load(open(subset+".json", "r"))
    total, correct = 0, 0
    # selected_prompt_dict = {}
    with open(os.path.join(f"output/outputs_e/{subset}_prompt.json"), "r") as f:
        selected_prompt_dict = json.load(f)
    for id, sample in enumerate(text_file):
        text = sample["qry_text"]
        if text in selected_prompt_dict:
            gpt_answer = selected_prompt_dict[text]
        else:
            gpt_answer = gpt_choose_answer_for_text(text)
            selected_prompt_dict[text] = gpt_answer
        total += 1
        if gpt_answer == gt_prompt:
            correct += 1
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")

if __name__=="__main__":
    for s in [
            "COCO_object_retrieval", 
            "COCO_animal_retrieval",
            "Visual7W_scene_retrieval", 
            "mix_weather_retrieval",
            "COCO_gesture_retrieval",
            "COCOStuff_material_retrieval",
            "Visual7W_people_num_retrieval", 
            "Visual7W_time_retrieval",
            ]:
        gpt_choose_answer_for_subset(s)