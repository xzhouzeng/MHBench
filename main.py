import os, sys
import json
import argparse

import torch
from utils import DisEvalKit,ClsEvalKit

from tqdm import tqdm
from datetime import datetime

now = datetime.now()
file_timestamp = now.strftime("%Y%m%d%H%M%S")

sys.path.append(os.getcwd())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str,
                        default="", 
                        choices=["VideoChatGPT", "Valley2", "Video-LLaMA-2", "VideoChat2", "VideoLLaVA", "LLaMA-VID", "VideoLaVIT", "MiniGPT4-Video", "PLLaVA", "LLaVA-NeXT-Video","VideoChat2-Mistral"])
    parser.add_argument( "--eval_task", type=str, default="classification", choices=["classification", "discrimination"])
    parser.add_argument( "--output_dir", type=str, default="../outputs")
    parser.add_argument( "--restart", action='store_true', help="Restart the evaluation")
    parser.add_argument( "--need_llm_extral", action='store_true')
    parser.add_argument( "--use_mcd",action='store_true',help="Use MotionCD, only for VideoLLaVA, VideoChat2, VideoChat2-Mistral")
    parser.add_argument( "--mcd_alpha", type=float, default=20)
    parser.add_argument( "--mcd_beta", type=float, default=0.1)

    args = parser.parse_args()
    return args

def load_model(TESTING_MODEL,device_id=0):
    if TESTING_MODEL == 'VideoChatGPT':
        from videochatgpt_modeling import VideoChatGPT
        ckpt_path = f"{CKPT_DIR}/Video-ChatGPT-7B"
        model = VideoChatGPT({"model_path": ckpt_path, "device": device_id})
    elif TESTING_MODEL == "Valley2":
        from valley_modeling import Valley
        ckpt_path = f"{CKPT_DIR}/Valley2-7b"
        model = Valley({"model_path": ckpt_path, "device": device_id})
    elif TESTING_MODEL == "Video-LLaMA-2":
        from videollama_modeling import VideoLLaMA
        ckpt_path = f"{CKPT_DIR}/Video-LLaMA-2-7B-Finetuned"
        model = VideoLLaMA({"model_path": ckpt_path, "device": device_id})
    elif TESTING_MODEL == "VideoChat2":
        from videochat_modeling import VideoChat
        ckpt_path = f"{CKPT_DIR}/VideoChat2"
        model = VideoChat({"model_path": ckpt_path, "device": device_id})
    elif TESTING_MODEL == "VideoChat2-Mistral":
        from videochat2_mistral_modeling import VideoChat2_Mistral
        ckpt_path = f"{CKPT_DIR}/VideoChat2-Mistral"
        model = VideoChat2_Mistral({"model_path": ckpt_path, "device": device_id})
    elif TESTING_MODEL == "VideoLLaVA":
        from videollava_modeling_transformer import VideoLLaVA
        ckpt_path = f"{CKPT_DIR}/VideoLLaVA/Video-LLaVA-7B"
        model = VideoLLaVA({"model_path": ckpt_path, "device": device_id})
    elif TESTING_MODEL == "LLaMA-VID":
        from llamavid_modeling import LLaMAVID
        ckpt_path = f"{CKPT_DIR}/LLaMA-VID-7B"
        model = LLaMAVID({"model_path": ckpt_path, "device": device_id})
    
    elif TESTING_MODEL == "VideoLaVIT":
        from videolavit_modeling import VideoLaVIT
        ckpt_path = f"{CKPT_DIR}/Video-LaVIT-v1"
        model = VideoLaVIT({"model_path": ckpt_path, "device": device_id})
    elif TESTING_MODEL == "MiniGPT4-Video":
        from minigpt4video_modeling import MiniGPT4Video
        ckpt_path = f"{CKPT_DIR}/MiniGPT4-Video"
        model = MiniGPT4Video({"model_path": ckpt_path, "device": device_id})
    elif TESTING_MODEL == "PLLaVA":
        from pllava_modeling import PLLaVA
        ckpt_path = f"{CKPT_DIR}/PLLaVA/pllava-7b"
        model = PLLaVA({"model_path": ckpt_path, "device": device_id})

    elif TESTING_MODEL == "LLaVA-NeXT-Video":
        from llavanext_modeling import LLaVANeXT
        ckpt_path = f"{CKPT_DIR}/LLaVA-NeXT-Video/LLaVA-NeXT-Video-7B-DPO"
        model = LLaVANeXT({"model_path": ckpt_path, "device": device_id})

    return model

DATA_DIR = "../dataset/MHBench"
CKPT_DIR = "../checkpoints"


def build_prompt(question,options):
    text_options = "\n".join([
        f"({chr(ord('A')+i)}) {option}" for i, option in enumerate(options)
    ])
    qa_instrution="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question."
    # qa_instrution="Carefully observe the motions that occur in the video, and then choose the best option for the problem."
    qa=f"Question:{question}\nOptions:\n{text_options}\n"
    return f"{qa_instrution}\n{qa} "
        
if __name__ == "__main__":
    args = get_args()

    suffix=""
    if args.use_mcd:
        from video_mcd_sample import evolve_mcd_sampling
        evolve_mcd_sampling()
        suffix="_mcd"

        assert args.model_name in ["VideoLLaVA", "VideoChat2", "VideoChat2-Mistral"], "The code implementation only supports MotionCD for VideoLLaVA, VideoChat2, and VideoChat2_Mistral"

    videollm_type=args.model_name
    
    data_path=f"{DATA_DIR}/{args.eval_task}.json"

    output_root_path=f"{args.output_dir}/{videollm_type}{suffix}/mhbench_{args.eval_task}"
    os.makedirs(output_root_path,exist_ok=True)

    with open(data_path, "r") as f:
        data = json.load(f)

    outputs_path=f"{output_root_path}/response.json"
    results_path=f"{output_root_path}/metrics.json"

    mcd_args={"use_mcd":args.use_mcd,"mcd_alpha":args.mcd_alpha,"mcd_beta":args.mcd_beta}

    if args.eval_task=="classification":
        if not os.path.exists(outputs_path) or args.restart:

            videollm = load_model(videollm_type)

            evalkit=ClsEvalKit()
            answer_prompt="Best option:("
            for item in tqdm(data):
                video_path=f"{DATA_DIR}/videos/{item['video_id']}.mp4"
                if not os.path.exists(video_path):
                    print("Not found ", video_path)
                    raise ValueError
                
                instruction=build_prompt(item['question'],list(item['options'].values()))
                
                with torch.no_grad():
                    if args.use_mcd:
                        outputs = videollm.generate(
                            instruction=instruction,
                            video_path=video_path,
                            answer_prompt=answer_prompt,
                            mcd_args=mcd_args,
                        )
                    else:
                        # some model does not support mcd_args
                        outputs = videollm.generate(
                            instruction=instruction,
                            video_path=video_path,
                            answer_prompt=answer_prompt,
                        )
        
                item['instruction']=instruction
                item['predict'] = outputs
                item['predict_type']=evalkit.recorder(outputs, item['answer'],instruction=instruction,class_id=item['motion_id'])
            
            
            with open(outputs_path, "w") as f:
                json.dump(data, f, indent=4)

        else:
            print("File exists")
            evalkit=ClsEvalKit(output_path=outputs_path)
    else:
        if not os.path.exists(outputs_path) or args.restart:

            videollm = load_model(videollm_type)

            evalkit=DisEvalKit()
            answer_prompt=None
            
            for item in tqdm(data):
                video_path=f"{DATA_DIR}/videos/{item['video_id']}.mp4"
                if not os.path.exists(video_path):
                    print("Not found ", video_path)
                    raise ValueError

                instruction=item['dis_question']
                with torch.no_grad():
                    if args.use_mcd:
                        outputs = videollm.generate(
                            instruction=instruction,
                            video_path=video_path,
                            answer_prompt=answer_prompt,
                            mcd_args=mcd_args,
                        )
                    else:
                        # some model does not support mcd_args
                        outputs = videollm.generate(
                            instruction=instruction,
                            video_path=video_path,
                            answer_prompt=answer_prompt,
                        )
        
                # item['instruction']=instruction
                item['predict'] = outputs
                item['predict_type']=evalkit.recorder(outputs, item['dis_answer'],class_id=item['motion_id'],pair_id=item['id'],instruction=instruction)
            
            
            with open(outputs_path, "w") as f:
                json.dump(data, f, indent=4)

        else:
            print("File exists")
            if videollm_type in ['Video-LLaMA-2','Valley2'] and args.need_llm_extral:
                need_llm_extral=True
            else:
                need_llm_extral=False
            evalkit=DisEvalKit(output_path=outputs_path,need_llm_extral=need_llm_extral,device="cuda:0")

    evalkit.save_result(results_path)
    evalkit.print_result()
    