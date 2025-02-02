import av
import numpy as np
from transformers import VideoLlavaProcessor
from PIL import Image
import cv2
from decord import VideoReader, cpu
import torch

# transformers/models/video_llava/modeling_video_llava.py
# from set_mcd_videollava import evolve_mcd_VideoLlava
# evolve_mcd_VideoLlava()
from set_mcd_videollava import VideoLlavaForConditionalGeneration_MotionCD as VideoLlavaForConditionalGeneration

def get_frame_count_opencv(video_path):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    frame_count = len(vr)
    
    return frame_count


def read_video_pyav(all_frames, frame_indices,use_mcd=False):
    if use_mcd:
        frame_indices=frame_indices[::-1]
   
    frames = [all_frames[i] for i in frame_indices]

    assert len(frames) == len(frame_indices)
    
    return np.stack([frame.to_ndarray(format="rgb24") for frame in frames])

class VideoLLaVA:
    def __init__(self, model_args,do_sample=False, **kwargs):
        assert(
            "model_path" in model_args
            and "device" in model_args
        )
        model_path=model_args['model_path']
        self.device = f"cuda:{model_args['device']}" if torch.cuda.is_available() else "cpu"
        self.kwargs = kwargs
        # model_path = '../../weights/Video-LLaVA-7B-hf'
        self.model = VideoLlavaForConditionalGeneration.from_pretrained(model_path, device_map=self.device)
        self.processor = VideoLlavaProcessor.from_pretrained(model_path)

        self.generate_kwargs = {"max_length":1000, "do_sample":do_sample, "top_p":0.9, "top_k":2}

    def genetate_caption(self,path,type='video', bound=None,frame_truncation=None):

        video_prompt = "USER: <video> Detailed description of video content? ASSISTANT:"
        image_prompt = "USER: <image>\nDetailed description of image content? ASSISTANT:"

        prompt={"video":video_prompt,"image":image_prompt}

        inputs = self.get_input_tokens(path, type, prompt, bound, frame_truncation)
        
        return self.llm_generate(inputs)

    def get_input_tokens(self, path, type,prompt, bound, use_mcd=False):

        if type=='video':
            container = av.open(path)
            container.seek(0)
            all_frames = [frame for frame in container.decode(video=0)]


            # sample uniformly 8 frames from the video
            if bound==None:
                # total_frames = container.streams.video[0].frames
                total_frames = len(all_frames)
                indices = np.arange(0, total_frames, total_frames / 8).astype(int)
            else:
                indices = np.arange(bound[0],bound[1],(bound[1]-bound[0])/8).astype(int)
   
            clip = read_video_pyav(all_frames, indices,use_mcd=use_mcd)

            inputs = self.processor(text=prompt['video'], videos=clip, return_tensors="pt").to(self.device)

            # for k in inputs:
            #     inputs[k] = inputs[k].to(next(self.model.parameters()).device)
        elif type=='image':
            image = Image.open(path)
            inputs = self.processor(text=prompt['image'], images=image, return_tensors="pt").to(self.device)

        else:
            raise ValueError('type should be either video or image')
        
        return inputs

    def llm_generate(self, inputs,answer_prefix="ASSISTANT:"):
        generate_ids = self.model.generate(**inputs, **self.generate_kwargs)
        output_text=self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        output_text = output_text.split(answer_prefix)[-1].strip()

        return output_text
    
    def llm_generate_mcd(self, inputs,answer_prefix="ASSISTANT:",mcd_args={}):

        mcd_alpha = mcd_args.get("mcd_alpha")
        mcd_beta = mcd_args.get("mcd_beta")

        generate_ids = self.model.generate(**inputs, 
                                           **self.generate_kwargs,
                                           use_mcd=True,
                                           mcd_alpha=mcd_alpha,
                                           mcd_beta=mcd_beta)
        
        output_text=self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        output_text = output_text.split(answer_prefix)[-1].strip()

        return output_text
    
    def video_true_false_qa(self, path,question, type='video', bound=None,frame_truncation=None,use_mcd=False):
        # system = (
        #     "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. "
        #     "Based on your observation, answer the following true or false questions, being aware that only yes or no can be answered."
        # )

        video_prompt = f"USER: <video> {question} Answer yes or no. ASSISTANT:"
        image_prompt = f"USER: <image>\n{question} Answer yes or no. ASSISTANT:"

        prompt={"video":video_prompt,"image":image_prompt}

        if use_mcd:
            inputs = self.get_input_tokens(path, type, prompt, bound, frame_truncation=None)
            inputs_mcd = self.get_input_tokens(path, type, prompt, bound, frame_truncation)
            inputs['pixel_values_videos_mcd'] = inputs_mcd['pixel_values_videos']
            return self.llm_generate_mcd(inputs)
        else:
            inputs = self.get_input_tokens(path, type, prompt, bound, frame_truncation)
            outputs=self.llm_generate(inputs)

        return outputs

        
    def multioptions_qa(self,path, input_qa, type='video', bound=None,frame_truncation=None,use_mcd=False,answer_prompt=None,**kwargs):

        if answer_prompt is None:
            answer_prompt = "Best option:("
        video_prompt = f"USER: <video> {input_qa} ASSISTANT:{answer_prompt}"
        image_prompt = f"USER: <image>\n{input_qa} ASSISTANT:{answer_prompt}"

        prompt={"video":video_prompt,"image":image_prompt}

        if use_mcd:
            inputs = self.get_input_tokens(path, type, prompt, bound, frame_truncation=None)
            if frame_truncation==None:
                frame_truncation=1
            inputs_mcd = self.get_input_tokens(path, type, prompt, bound, frame_truncation)
            inputs['pixel_values_videos_mcd'] = inputs_mcd['pixel_values_videos']
            outputs =self.llm_generate_mcd(inputs,answer_prefix=f"ASSISTANT:{answer_prompt}")
        else:
            inputs = self.get_input_tokens(path, type, prompt, bound, frame_truncation)
            outputs=self.llm_generate(inputs,answer_prefix=f"ASSISTANT:{answer_prompt}")
            
        return outputs
    
    def generate(self, instruction, video_path,answer_prompt=None, type='video',mcd_args={}, bound=None):
        
        if answer_prompt is None:
            answer_prompt = ""
        video_prompt = f"USER: <video> {instruction} ASSISTANT:{answer_prompt}"
        image_prompt = f"USER: <image>\n{instruction} ASSISTANT:{answer_prompt}"

        prompt={"video":video_prompt,"image":image_prompt}

        use_mcd =mcd_args.get("use_mcd",False)

        if use_mcd:
            inputs = self.get_input_tokens(video_path, type, prompt, bound )
            inputs_mcd = self.get_input_tokens(video_path, type, prompt, bound, use_mcd=True)
            inputs['pixel_values_videos_mcd'] = inputs_mcd['pixel_values_videos']
            outputs =self.llm_generate_mcd(inputs,answer_prefix=f"ASSISTANT:{answer_prompt}",mcd_args=mcd_args)
        else:
            inputs = self.get_input_tokens(video_path, type, prompt, bound)
            outputs=self.llm_generate(inputs,answer_prefix=f"ASSISTANT:{answer_prompt}")
            
        return outputs
