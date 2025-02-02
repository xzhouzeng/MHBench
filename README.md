# MHBench: Demystifying Motion Hallucination in VideoLLMs

## MHBench

### Introduction

Similar to Language or Image LLMs, VideoLLMs are also plagued by hallucination issues. Hallucinations in videos not only manifest in the spatial dimension regarding the perception of the existence of visual objects (static) but also the temporal dimension influencing the perception of actions and events (dynamic). This paper introduces the concept of Motion Hallucination for the first time, exploring the hallucination phenomena caused by insufficient motion perception capabilities in VideoLMMs, as well as how to detect, evaluate, and mitigate the hallucination. To this end, we propose **the first benchmark for assessing motion hallucination MHBench**, which consists of 1,200 videos of 20 different action categories. By constructing a collection of adversarial triplet types of videos (original/antonym/incomplete), we achieve a comprehensive evaluation of motion hallucination. Furthermore, we present a **Motion Contrastive Decoding (MotionCD) method**, which employs bidirectional motion elimination between the original video and its reverse playback to construct an amateur model that removes the influence of motion while preserving visual information, thereby effectively suppressing motion hallucination. Extensive experiments on MHBench reveal that current state-of-the-art VideoLLMs significantly suffer from motion hallucination, while the introduction of MotionCD can effectively mitigate this issue, achieving up to a 15.1% performance improvement. We hope this work will guide future efforts in avoiding and mitigating hallucinations in VideoLLMs.

### Dataset

You can download MHBench from [Google Cloud](https://drive.google.com/drive/folders/1INrzOafJe6uKFp0IZp1z-pdw9bq_YYpJ?usp=sharing) and place it in the <u>dataset </u>directory.

```
dataset                   
    ├── MHBench
        ├── videos
        └── classification.json
        └── discrimination.json
```

Some examples:

```
# classification.json
[
        {
        "id": 0,
        "motion_id": "1",
        "video_id": "195051",
        "question": "Which of the following actions appeared in the video?",
        "answer": "A",
        "options": {
            "A": "covering something",
            "B": "uncovering something",
            "C": "neither action A nor B happened"
        }
    },
...
]

# discrimination.json
[
    {
        "dis_id": 0,
        "id": 0,
        "motion_id": "1",
        "video_id": "195051",
        "dis_question": "Is the action of covering something happening in the video? If this action happens, answer 'yes, it does'; If it doesn't happen, answer 'no, it doesn't'.",
        "dis_answer": "yes"
    },
...
]
```

## Evaluation

### Installation

**Available Baselines**

- VideoChatGPT-7B

- Valley2-7B

- Video-LLaMA-2-7B

- VideoChat2-7B

- VideoLLaVA-7B

- LLaMA-VID-7B

- VideoLaVIT-7B

- MiniGPT4-Video-7B

- PLLaVA-7B

- LLaVA-NeXT-Video-DPO-DPO-7B

- VideoChat2-Mistral-7B

The implementation and integration of the baseline are inspired by **VideoHallucer**. For detailed instructions on installation and available checkpoints, please refer to the [VideoHallucer/INSTALLATION.md at main · patrick-tssn/VideoHallucer](https://github.com/patrick-tssn/VideoHallucer/blob/main/INSTALLATION.md) guide.

The environment configuration of VideoChat2-Mistral-7B refers to [Ask-Anything/video_chat2 at main · OpenGVLab/Ask-Anything](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2), with the following parameter directory:

```
checkpoints
├── VideoChat2-Mistral
│   ├── Mistral-7B-Instruct-v0.2
│   │   ├──...   
│   ├── umt_l16_qformer.pth
│   ├── videochat2_mistral_7b_stage2.pth
│   └── videochat2_mistral_7b_stage3.pth 
```

### Running

To evaluate the models, you can use the `run.sh` script provided below. This script specifies the model name, evaluation task, output directory, and optional parameters like `--use_mcd` and `--restart`. Modify the script to fit your evaluation needs.

#### Script: `run.sh`

```
#!/bin/bash
cd baselines

# Activate your environment (e.g., conda, virtualenv, etc.)
# conda activate videollm

python ../main.py \
    --model_name VideoChat2-Mistral \ # Select the model to evaluate
    --eval_task classification \      # Choose the evaluation task (classification or discrimination)
    --output_dir ../outputs \         # Directory to save the evaluation results
    --use_mcd \                       # Enable MotionCD for supported models
    --mcd_alpha 20 \                  # Set alpha for MotionCD (default: 20)
    --mcd_beta 0.1                    # Set beta for MotionCD (default: 0.1)
```

#### Parameter Descriptions:

- **`--model_name`**: Specifies the model to evaluate. Options include:
  
  【`VideoChatGPT`、`Valley2`、`Video-LLaMA-2`...】

- **`--eval_task`**: Sets the evaluation task. Available choices are:
  
  - `classification`: Evaluate classification tasks.
  - `discrimination`: Evaluate discrimination tasks.

- **`--output_dir`**: Path to save the output results. Default is `../outputs`.

- **`--restart`**: Optional. Restarts the evaluation if previously interrupted.

- **`--use_mcd`**: Optional. Enables **Motion Contrast Decoding (MotionCD)**, supported only for `VideoLLaVA`, `VideoChat2`, and `VideoChat2-Mistral`.

- **`--mcd_alpha`**: Optional. Sets the alpha parameter for MotionCD (default: 20).

- **`--mcd_beta`**: Optional. Sets the beta parameter for MotionCD (default: 0.1).

### Support MotionCD

The implementation of MotionCD is inspired by [Visual Contrastive Decoding]([DAMO-NLP-SG/VCD: [CVPR 2024 Highlight] Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding](https://github.com/DAMO-NLP-SG/VCD)). Currently, this code only supports the three aforementioned models. If you wish to extend support to other models, please refer to the MotionCD implementation example in the `VideoChat2-Mistral` framework.

1. Add the following at the beginning of the start-up script:
   
   ```python
   # baselines/video_mcd_sample.py
   # main.py #100
   from video_mcd_sample import evolve_mcd_sampling
   evolve_mcd_sampling()
   ```
   
   The `evolve_mcd_sampling` function replaces the sampling function in the transformers library. The modified sampling function includes an option for contrastive decoding, while keeping the rest unchanged.

2. Slightly modify `transformers/models/mistral/modeling_mistral.py`:
   
   a. Add contrastive decoding parameters in the `MistralForCausalLM` class's `forward` function to avoid exceptions in `model.generate`.
   
   b. Add the `prepare_inputs_for_generation_mcd` function.
   
   ```python
       def forward(
           self,
           input_ids: torch.LongTensor = None,
           attention_mask: Optional[torch.Tensor] = None,
           position_ids: Optional[torch.LongTensor] = None,
           past_key_values: Optional[List[torch.FloatTensor]] = None,
           inputs_embeds: Optional[torch.FloatTensor] = None,
           labels: Optional[torch.LongTensor] = None,
           use_cache: Optional[bool] = None,
           output_attentions: Optional[bool] = None,
           output_hidden_states: Optional[bool] = None,
           return_dict: Optional[bool] = None,
           # mcd_config
           use_mcd: Optional[bool] = None,
           inputs_embeds_mcd: Optional[torch.FloatTensor] = None,
           mcd_beta: Optional[torch.FloatTensor] = None,
           mcd_alpha: Optional[torch.FloatTensor] = None,
       ) -> Union[Tuple, CausalLMOutputWithPast]:
   
       def prepare_inputs_for_generation_mcd(
           self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds_mcd=None, **kwargs
       ):
       ...
   ```
   
   c. set the hyperparameter in the `generate` function: 
   
   ```python
   # baselines/videochat2_mistral_modeling.py #164
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
   ```

The second step is specifically implemented in the `baselines/set_mcd_mistral.py` file, which is executed in `baselines/videollava_modeling_transformer.py #12`.

## Acknowledgement

- We thank **VideoHallucer** for inspiring the baseline integration framework ([GitHub Repository](https://github.com/patrick-tssn/VideoHallucer)).

- We thank **Visual Contrastive Decoding (VCD)** for inspiring the implementation of MotionCD ([GitHub Repository](https://github.com/DAMO-NLP-SG/VCD)).

## Citation

```
@article{mhbench2025},
  title={MHBench: Demystifying Motion Hallucination in VideoLLMss},
  author={Ming Kong, Xianzhou Zeng, Luyuan Chen, Yadong Li, Bo Yan, Qiang Zhu*},
  journal={AAAI 2025},
  year={2025}
}
```
