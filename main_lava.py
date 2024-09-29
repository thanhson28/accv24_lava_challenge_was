# https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4
import os
import pandas as pd
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser

args_parser = ArgumentParser()
args_parser.add_argument('--data_dir', type=str, default='/home/hvtham/LAVA/son/lava-challenge-2024/LAVA_Challenge_Data')
args_parser.add_argument('--output_dir', type=str, default='/home/hvtham/LAVA/son/output')
args_parser.add_argument('--model_name', type=str, default='Qwen/Qwen2-VL-7B-Instruct')
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

print("CUDA device count:", torch.cuda.device_count())
print("Current CUDA device:", torch.cuda.current_device())
print("CUDA device name:", torch.cuda.get_device_name(0))

# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8")

def process_part(path_dir='/home/hvtham/LAVA/son/lava-challenge-2024/LAVA_Challenge_Data',
                 part='Public',
                 output_dir='/home/hvtham/LAVA/son/output',
                 model_name='Qwen/Qwen2-VL-7B-Instruct'):
    df = pd.read_csv(f'{path_dir}/{part} data/annotation.csv')
    image_folder = f"{path_dir}/{part} data/images"
    model_name = '72B' if '72B' in model_name else '7B'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{part}_qwen_{model_name}_shuffle.csv"
    vote_file = f"{output_dir}/{part}_qwen_{model_name}_vote.csv"
    file_exists = os.path.exists(output_file)
    img_names = []
    if file_exists:
        with open(output_file, 'r') as f:
            lines = f.readlines()
            img_names = [line.split(',')[0] for line in lines[1:]]
            f.closed
    vote_fl = open(vote_file, 'w')

    with open(output_file, 'a' if os.path.exists(output_file) else 'w') as f_out:
        # get all image name in output file
        if not file_exists:
            f_out.write("file_name,answer\n")
        for row in tqdm(df.itertuples(index=False)):
            if row.file_name in img_names:
                continue
            image_path = os.path.join(image_folder, row.file_name)
            question = f"{row.question}"
            option1 = row.option1
            option2 = row.option2
            option3 = row.option3
            option4 = row.option4
            options_dict = {"1": option1, "2": option2, "3": option3, "4": option4}
            options_reverse_dict = {v: k for k, v in options_dict.items()}
            votes = []
            for i in range(5):
                #random swap option
                options = [option1, option2, option3, option4]
                np.random.shuffle(options)
                option1, option2, option3, option4 = options
                shufle_option_dict = {'1': option1, '2': option2, '3': option3, '4': option4}
                options = f"(A) {option1}\n(B) {option2}\n(C) {option3}\n(D) {option4}"
                language = row.language
                prompt = """{question}\n{options}\n
                Answer with the option's letter from the given choices directly."""
                prompt = f'{question}\n{options}\n{prompt}'
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path,
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = inputs.to("cuda")

                generated_ids = model.generate(**inputs, max_new_tokens=128)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                answer1, answer2, answer3, answer4 = row.option1, row.option2, row.option3, row.option4
                answer = output_text[0]
                if answer == "(A)" or answer == 'A':
                    answer = "1"
                elif answer == "(B)" or answer == 'B':
                    answer = "2"
                elif answer == "(C)" or answer == 'C':
                    answer = "3"
                elif answer == "(D)" or answer == 'D':
                    answer = "4"
                if answer not in ["1", "2", "3", "4"]:
                    if answer==answer1 or answer in answer1 or answer1 in answer:
                        answer = "1"
                    elif answer==answer2 or answer in answer2 or answer2 in answer:
                        answer = "2"
                    elif answer==answer3 or answer in answer3 or answer3 in answer:
                        answer = "3"
                    elif answer==answer4 or answer in answer4 or answer4 in answer:
                        answer = "4"
                    else:
                        answer = "1"
                final_answer = options_reverse_dict[shufle_option_dict[answer]]
                votes.append(final_answer)
            print(f"{row.file_name},{votes}")
            vote_fl.write(f"{row.file_name},{votes}\n")
            final_answer = max(set(votes), key = votes.count)
            print(f"{row.file_name},{final_answer}")
            f_out.write(f"{row.file_name},{final_answer}\n")


if __name__ == '__main__':
    #inference process
    args = args_parser.parse_args()

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype='auto',
        attn_implementation="flash_attention_2",
        device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(args.model_name)
    process_part(part='Public', path_dir=args.data_dir, output_dir=args.output_dir)
    process_part(part='Private', path_dir=args.data_dir, output_dir=args.output_dir)