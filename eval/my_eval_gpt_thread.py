import argparse
import os
import json
import base64
from openai import OpenAI
from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
from copy import deepcopy
from io import BytesIO
from decord import VideoReader, cpu

# Constants
DEFAULT_RETRIES = 5
RATE_LIMIT_DELAY = 30
response_json = {"role": "user", "content": []}

def remove_answers(sub_qas):
    for sub_qa in sub_qas.values():
        if 'ans' in sub_qa:
            del sub_qa['ans']
    return sub_qas

def encode_video(video_path, args):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, args.for_get_frames_num, dtype=int)

    # Ensure the last frame is included
    if total_frame_num - 1 not in uniform_sampled_frames:
        uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()

    base64_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        base64_frames.append(base64_str)

    return base64_frames

def format_question_and_choices(question, choices):
    choices_str = ', '.join([f"{key}: {value}" for key, value in choices.items()])
    return f"Question: {question}, Choices: {choices_str}."

def build_payload(prompt, video_frames, model_path, max_tokens):
    """Construct the API payload for OpenAI requests."""
    payload = {
        "model": model_path,
        "messages": [],
        "max_tokens": max_tokens,
    }
    
    if "<image>" not in prompt:
        payload["messages"].append(deepcopy(response_json))
        payload["messages"][0]["content"].append({"type": "text", "text": prompt})
        for img in video_frames:
            payload["messages"][0]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})
    else:
        prompt = prompt.split("<image>")
        for idx, img in enumerate(video_frames):
            payload["messages"].append(deepcopy(response_json))
            payload["messages"][idx]["content"].append({"type": "text", "text": prompt[idx]})
            payload["messages"][idx]["content"].append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}})

        # If n image tokens are in the contexts
        # contexts will be splitted into n+1 chunks
        # Manually add it into the payload
        payload["messages"].append(deepcopy(response_json))
        payload["messages"][-1]["content"].append({"type": "text", "text": prompt[-1]})
    
    return payload

def call_openai_api(client, payload, max_retries=DEFAULT_RETRIES):
    """Make API call with retry logic and error handling."""
    outputs = None 
    sucess_flag = False
    for retry in range(max_retries):
        try:
            result = client.chat.completions.create(**payload)
            outputs = result.choices[0].message.content
            sucess_flag = True
            break
        except Exception as inst:
            if 'error' in dir(inst):
                if inst.error.code == 'rate_limit_exceeded':
                    if "TPM" in inst.error.message:
                        time.sleep(30)
                        continue
                    else:
                        print(f"Error: {inst.error.message}")
                elif inst.error.code == 'insufficient_quota':
                    print(f'Insufficient quota, exiting.')
                    exit()
                elif inst.error.code == 'content_policy_violation':
                    print(f'Content policy violation, skipping.')
                    break
                print(f'Error: {inst.error.message}, code: {inst.error.code}')
            continue
    if not sucess_flag:
            print(f'Calling OpenAI failed after retrying {max_retries} times.')
            exit()
    
    return outputs

def generate_cot(client, question, video_frames, args, sub_qas=None):
    """Generate Chain-of-Thought reasoning for the question."""
    if args.with_guide_cot:
        guide = remove_answers(sub_qas) if args.remove_ans else sub_qas
        prompt = f"{question}Let's think step by step. Please help me break the above question into separate sub-questions and provide the corresponding answers for analysis of the above question later. Among these sub-questions, the phrase 'sub questions' that I provided must be included: {guide}"
    else:
        prompt = f"{question}Let's think step by step. Please help me break the above question into separate sub-questions and provide the corresponding answers for analysis of the above question later."
    
    payload = build_payload(prompt, video_frames, args.model_path, args.max_tokens)
    output = call_openai_api(client, payload)
    
    return output

def build_result_prompt(question, cot_prompt=None):
    """Construct the final prompt for answer generation."""
    base_prompt = question + "Answer me using only the given options (A,B,C...)"
    
    if cot_prompt:
        return f"{cot_prompt}\nPlease answer the following question based on the above prompt\n{base_prompt}"
    return base_prompt

def process_video(line, args, lock, ans_file):
    """Process a single video and generate answers."""
    # Extract relevant data
    src_dataset = line["src_dataset"]
    video_file = line["video_name"]
    video_path = os.path.join(args.video_folder, src_dataset, video_file)
    
    # Load and preprocess the video
    video_frames = []
    visual = encode_video(video_path, args)
    for img in visual:
        video_frames.append(img)

    question = format_question_and_choices(line["question"],line["choices"])

    # Inference
    client = OpenAI(api_key=args.api_key, base_url=args.api_base)
    
    # Generate CoT if needed
    cot_prompt = None
    if args.with_guide_cot:
        cot_prompt = generate_cot(client, question, video_frames, args, line["sub_qas"])
    elif args.with_cot:
        cot_prompt = generate_cot(client, question, video_frames, args)
    
    # Generate final answer
    final_prompt = build_result_prompt(question, cot_prompt)
    payload = build_payload(final_prompt, video_frames, args.model_path, args.max_tokens)
    output = call_openai_api(client, payload)
    
    # Prepare result data
    result_data = {
        "video_id": line["id"],
        "type": line["question_type"],
        "src_dataset": line["src_dataset"],
        "video_name": line["video_name"],
        "prompt": question,
        "pred_response": output,
        "ans": line["answer"],
        "aspect": line["aspect"],
    }
    
    if line["question_type"] == 2:
        result_data["serial number"] = line["serial number"]
    if cot_prompt:
        result_data["cot_prompt"] = cot_prompt
    
    # Thread-safe writing
    with lock:
        ans_file.write(json.dumps(result_data) + "\n")
        ans_file.flush()

def eval_model(args):
    # Data
    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)

    # Use thread-safe file writing
    with open(answer_file, "w") as ans_file:
        lock = Lock()  # Mutex for safe concurrent writes

        # Load questions
        questions = []
        with open(os.path.expanduser(args.question_file), "r") as f:
            for eachline in f:
                questions.append(json.loads(eachline))
        
        # Set up the thread pool
        with ThreadPoolExecutor(max_workers=args.concurrent_jobs) as executor:
            futures = []
            for line in questions:
                futures.append(executor.submit(process_video, line, args, lock, ans_file))
            
            # Wait for all futures to complete
            for future in as_completed(futures):
                future.result()  # Can raise exceptions here if needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="", help="Your Model path")
    parser.add_argument('--api_key', type=str, default="", help="Your OpenAI API key.")
    parser.add_argument('--api_base', type=str, default="", help="Your API base URL.")
    parser.add_argument("--video-folder", type=str, default="", help="Your Video path")
    parser.add_argument("--question-file", type=str, default="", help="Your Question path") 
    parser.add_argument("--answer-file", type=str, default="", help="Your Answer path")
    parser.add_argument("--for_get_frames_num", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrent-jobs", type=int, default=100, help="Number of concurrent jobs to run.")
    parser.add_argument('--with-cot', type=bool, default=False)
    parser.add_argument('--remove-ans', type=bool, default=False)
    parser.add_argument('--with-guide-cot', type=bool, default=False)
    
    args = parser.parse_args()
    eval_model(args)