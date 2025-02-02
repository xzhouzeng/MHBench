import os
import sys
import warnings

import torch
from torch.nn import CrossEntropyLoss
from PIL import Image
import numpy as np
from decord import VideoReader, cpu

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import StoppingCriteria, StoppingCriteriaList


from videochat2_mistral.models import VideoChat2_it_mistral
from videochat2_mistral.utils.easydict import EasyDict

from peft import get_peft_model, LoraConfig, TaskType
from videochat2_mistral.utils.config import Config

from videochat2_mistral.utils.video_transforms import (
    GroupNormalize,
    GroupScale,
    GroupCenterCrop,
    Stack,
    ToTorchFormatTensor,
)


import decord
decord.bridge.set_bridge("torch")

from set_mcd_mistral import evolve_mcd_MistralForCausalLM
evolve_mcd_MistralForCausalLM()

warnings.filterwarnings("ignore")



def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + " " + message + " " + conv.sep
        else:
            ret += role
    return ret


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + " " + message
        else:
            if message:
                ret += role + " " + message + " " + conv.sep
            else:
                ret += role
    return ret


def get_context_emb(conv, model, img_list, answer_prompt=None, print_res=False):
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    if print_res:
        print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')

    if img_list is not None:
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."

    with torch.no_grad():
        seg_tokens = [
            model.mistral_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(model.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.mistral_model.base_model.model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
#         seg_embs = [model.mistral_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    if img_list is None:    # only text
        mixed_embs= seg_embs
    else:
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs


def ask(text, conv):
    conv.messages.append([conv.roles[0], text])
        

class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
    
def answer(conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False,**kwargs):
    stop_words_ids = [
        torch.tensor([2]).to(model.device),
        torch.tensor([29871, 2]).to(model.device)]  # '</s>' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    with torch.no_grad():
        outputs = model.mistral_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            pad_token_id=model.mistral_tokenizer.eos_token_id
        )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
    output_text = model.mistral_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('</s>')[0]  # remove the stop sign </s>
#     output_text = output_text.split('[/INST]')[-1].strip()
    conv.messages[-1][1] = output_text + '</s>'
    return output_text, output_token.cpu().numpy()


def answer_mcd(conv, model, img_list,img_list_mcd, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False,**kwargs):
    stop_words_ids = [
        torch.tensor([2]).to(model.device),
        torch.tensor([29871, 2]).to(model.device)]  # '</s>' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    embs_mcd = get_context_emb(conv, model, img_list_mcd, answer_prompt=answer_prompt, print_res=print_res)
    
    mcd_alpha = kwargs.get("mcd_alpha",1.0)
    mcd_beta = kwargs.get("mcd_beta",0.1)

    with torch.no_grad():
        outputs = model.mistral_model.generate(
            inputs_embeds=embs,
            use_mcd=True,
            inputs_embeds_mcd=embs_mcd,   # add mcd
            mcd_alpha=mcd_alpha,
            mcd_beta=mcd_beta,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            pad_token_id=model.mistral_tokenizer.eos_token_id
        )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
    if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
    output_text = model.mistral_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('</s>')[0]  # remove the stop sign </s>
#     output_text = output_text.split('[/INST]')[-1].strip()
    conv.messages[-1][1] = output_text + '</s>'
    return output_text, output_token.cpu().numpy()


def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    print(f"n_position: {n_position}")
    print(f"pre_n_position: {pre_n_position}")
    
    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        if new_P != 14:
            print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    
    if cur_frame != ckpt_num_frame:
        print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        print(f'Interpolate the position embedding')
        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table


from base import ViLLMBaseModel
class VideoChat2_Mistral(ViLLMBaseModel):


    def __init__(self, model_args,do_sample=False,**kwargs):
        super().__init__(model_args['model_path'], model_args['device'])
        assert(
            "model_path" in model_args
            and "device" in model_args
        )

        config_file = "videochat2_mistral/configs/config_mistral.json"
        cfg = Config.from_file(config_file)
        cfg.model.vision_encoder.num_frames = 4
        cfg.model.vit_blip_model_path = os.path.join(model_args["model_path"], "umt_l16_qformer.pth")
        cfg.model.mistral_model_path = os.path.join(model_args["model_path"], "Mistral-7B-Instruct-v0.2")
        cfg.model.videochat2_model_path = os.path.join(model_args["model_path"], "videochat2_mistral_7b_stage2.pth")
        cfg.model.videochat2_7b_stage3_model_path = os.path.join(model_args["model_path"], "videochat2_mistral_7b_stage3.pth")
        cfg.device = model_args["device"]

        self.cfg = cfg
        self.do_sample = do_sample

        self.kwargs = kwargs

        self.model = VideoChat2_it_mistral(config=self.cfg.model)

        self.model = self.model.to(torch.device(self.cfg.device))

        # add lora to run stage3 model
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, 
            r=16, lora_alpha=32, lora_dropout=0.,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj", "lm_head"
            ]
        )
        self.model.mistral_model = get_peft_model(self.model.mistral_model,
                                                peft_config)

        state_dict = torch.load(self.cfg.model.videochat2_7b_stage3_model_path,
                                "cpu")

        if "model" in state_dict.keys():
            msg = self.model.load_state_dict(state_dict["model"], strict=False)
        else:
            msg = self.model.load_state_dict(state_dict, strict=False)
        # print(msg)

        self.model = self.model.eval()

        #  position embedding
        self.num_segments = 4
        self.resolution = 224
        new_pos_emb = get_sinusoid_encoding_table(
            n_position=(self.resolution // 16)**2 * self.num_segments,
            cur_frame=self.num_segments)
        self.model.vision_encoder.encoder.pos_embed = new_pos_emb

        self.decord_method = {
            "video": self.read_video,
            # 'gif': self.read_gif,
            # 'frame': self.read_frame,
        }

        # transform
        crop_size = self.resolution
        scale_size = self.resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size),
                       interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std),
        ])

        self.video_frames = None
        self.system = "Watch the video and answer the question."

        # if self.model.mistral_tokenizer.pad_token_id is None:
        #     self.model.mistral_tokenizer.pad_token = self.model.mistral_tokenizer.eos_token
        #     self.model.mistral_tokenizer.pad_token_id = self.model.mistral_tokenizer.eos_token_id
            

    def get_index(self, bound):
        if bound is None:
            start_idx, end_idx = 0, len(self.video_frames) - 1
        else:
            start_idx, end_idx = bound[0], bound[1]
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices

    def read_video(self, video_path):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        return vr
    
    def upload_video(self, video_path, video_type="video"):

        decord_method = self.decord_method[video_type]

        self.video_frames = decord_method(video_path)

    def get_visual_emb(self,path,type, bound=None,use_mcd=False):

        images_group = list()
        if type == 'video':
            self.upload_video(path)
            frame_indices = self.get_index(bound)
            if use_mcd:
                frame_indices=frame_indices[::-1]
            
            for frame_index in frame_indices:
                img = Image.fromarray(self.video_frames[frame_index].numpy())
                images_group.append(img)

        elif type == 'image':
            img = Image.open(path)
            # copy 
            for _ in range(self.num_segments):
                images_group.append(img)

        torch_imgs = self.transform(images_group)

        TC, H, W = torch_imgs.shape
        torch_imgs = torch_imgs.reshape(1, TC // 3, 3, H, W)

            
        torch_imgs = torch_imgs.to(self.model.device)

        video_list = []
        with torch.no_grad():
            video_emb, _ = self.model.encode_img(torch_imgs, self.system)

        video_list.append(video_emb)
        return video_list
    
    def get_output(self,path,chat,type='video', bound=None,answer_prompt=None,mcd_args={}):

        # if mcd_args is None:
        #     mcd_args={}

        use_mcd =mcd_args.get("use_mcd",False)

        if use_mcd:
            video_list = self.get_visual_emb(path,type,bound,use_mcd=False)
            video_list_mcd=self.get_visual_emb(path,type,bound,use_mcd=True)   
            output = self.llm_generate_mcd(chat,video_list,video_list_mcd,answer_prompt=answer_prompt,mcd_args=mcd_args)    
        else:
            video_list = self.get_visual_emb(path,type,bound,use_mcd=False)
            output = self.llm_generate(chat,video_list,answer_prompt=answer_prompt)
        
        return output


    def llm_generate(self, chat, video_list,answer_prompt=None):
        
        output = answer(conv=chat, model=self.model, do_sample=self.do_sample, img_list=video_list, max_new_tokens=512,answer_prompt=answer_prompt)[0]
        # remove potential explanation
        output = output.strip().split("\n")[0]

        return output
    
    def llm_generate_mcd(self, chat, video_list,video_list_mcd,answer_prompt=None,mcd_args=None):

        output = answer_mcd(conv=chat, model=self.model, do_sample=self.do_sample, img_list=video_list,img_list_mcd=video_list_mcd, max_new_tokens=512,answer_prompt=answer_prompt,**mcd_args)[0]
        # remove potential explanation
        output = output.strip().split("\n")[0]

        return output

    
    def generate(self,instruction,video_path,type='video', bound=None,answer_prompt=None,mcd_args={}):

        chat = EasyDict({
            "system": "",
            "roles": ("[INST]", "[/INST]"),
            "messages": [],
            "sep": ""
        })

        chat.messages.append([chat.roles[0], "<Video><VideoHere></Video> [/INST]"])
        ask(f"{instruction}", chat)

        output=self.get_output(video_path,chat,type,bound,answer_prompt=answer_prompt,mcd_args=mcd_args)
        return output

