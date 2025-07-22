import argparse
import torch
import os
import json
from tqdm import tqdm
import cv2
from transformers import AutoModelForCausalLM, AutoProcessor
import warnings

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

warnings.filterwarnings("ignore")
device = "cuda"

def remove_answers(sub_qas):
    for sub_qa in sub_qas.values():
        if 'ans' in sub_qa:
            del sub_qa['ans']
    return sub_qas

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = frame_count / fps
    cap.release()
    return duration, fps

def format_question_and_choices(question, choices):
    choices_str = ', '.join([f"{key}: {value}" for key, value in choices.items()])
    return f"Question: {question}, Choices: {choices_str}."

@torch.inference_mode()
def infer(model, processor, conversation):
    inputs = processor(
        conversation=conversation,
        add_system_prompt=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=128)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response

def generate_cot(model, processor, qs, video_path, fps, args,sub_qas=None):
    if args.with_partial_cot:
        cot_prompt = f"{qs}Let's think step by step.Please help me generate sub-questions based on the above question and provide the corresponding answers for analysis of the above question later.Among these sub-questions, the phrase sub questions that I provided below must be included:{remove_answers(sub_qas) if args.remove_ans else sub_qas}"
    else:
        cot_prompt = f"{qs}Let's think step by step.Please help me generate sub-questions based on the above question and provide the corresponding answers for analysis of the above question later."

    conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": fps, "max_frames": 512}},
                    {"type": "text", "text": cot_prompt},
                ]
            },
        ]

    return infer(model, processor, conversation)    



def eval_model(args):
    # Model
    model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    trust_remote_code=True,
    # device_map={"": "cuda:0"},
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # Data
    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")
    questions = []
    with open(os.path.expanduser(args.question_file),"r") as f:
        for eachline in f:
            questions.append(json.loads(eachline))
    
    for line in tqdm(questions, desc="videollama3 eval"):
        src_dataset = line["src_dataset"]
        video_file = line["video_name"]
        video_path = os.path.join(args.video_folder, src_dataset)
        video_path = os.path.join(video_path, video_file)
        duration, _ = get_video_duration(video_path)

        # preprocess the qs
        qs = format_question_and_choices(line["question"],line["choices"])
        cot_prompt = None
        if args.with_partial_cot:
            cot_prompt = generate_cot(model, processor, qs, video_path, args.for_get_frames_num / duration, args, line["sub_qas"])
            cur_prompt = cot_prompt + "Please answer the following question based on the above prompt\n" + qs + "Answer me using only the given options (A,B,C...)" # 拼接 Partial COT 和原始 prompt
        elif args.with_cot:
            cot_prompt = generate_cot(model, processor, qs, video_path, args.for_get_frames_num / duration, args)
            cur_prompt = cot_prompt + "Please answer the following question based on the above prompt\n" + qs + "Answer me using only the given options (A,B,C...)" # 拼接 COT 和原始 prompt
        else:
            cur_prompt = qs + "Answer me using only the given options (A,B,C...)" 
        
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": args.for_get_frames_num / duration, "max_frames": 128}},
                    {"type": "text", "text": cur_prompt},
                ]
            },
        ]
        
        # Save the result
        result_data = {
            "video_id": line["id"],
            "type": line["question_type"],
            "src_dataset": line["src_dataset"],
            "video_name": line["video_name"],
            "prompt": qs,
            "pred_response": infer(model,processor,conversation),
            "ans": line["answer"],
            "aspect": line["aspect"],
        }
        if line["question_type"] == 2:
            result_data["serial number"] = line["serial number"]
        if cot_prompt is not None:
            result_data["cot_prompt"] = cot_prompt
        
        ans_file.write(json.dumps(result_data) + "\n")
        ans_file.flush()

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="", help="Your Model path") 
    parser.add_argument("--video-folder", type=str, default="", help="Your Video path")
    parser.add_argument("--question-file", type=str, default="", help="Your Question path") 
    parser.add_argument("--answer-file", type=str, default="", help="Your Answer path")
    parser.add_argument("--for_get_frames_num", type=int, default=16)
    parser.add_argument('--with-cot', type=bool, default=False)
    parser.add_argument('--remove-ans', type=bool, default=False)
    parser.add_argument('--with-partial-cot', type=bool, default=False)
    args = parser.parse_args()

    eval_model(args)