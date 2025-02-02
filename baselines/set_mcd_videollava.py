# transformers/models/video_llava/modeling_video_llava.py
from transformers.models.video_llava.modeling_video_llava import *
import transformers

_CONFIG_FOR_DOC = "VideoLlavaConfig"

@add_start_docstrings(
    """The VideoLlava model which consists of a vision backbone and a language model.""",
    VIDEO_LLAVA_START_DOCSTRING,
)
class VideoLlavaForConditionalGeneration_MotionCD(VideoLlavaPreTrainedModel):
    def __init__(self, config: VideoLlavaConfig):
        super().__init__(config)
        self.video_tower = AutoModel.from_config(config.vision_config)
        self.image_tower = AutoModel.from_config(config.vision_config)

        self.multi_modal_projector = VideoLlavaMultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_input_ids_with_visual_features(
        self, visual_features, inputs_embeds, input_ids, attention_mask, labels, num_frames=1
    ):
        num_images, num_image_patches, embed_dim = visual_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        special_vision_token = self.config.video_token_index if num_frames == 8 else self.config.image_token_index

        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == special_vision_token
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_seq_len = (num_special_image_tokens.max() * (num_image_patches * num_frames - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != special_vision_token)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = (
            torch.cumsum((special_image_token_mask * (num_image_patches * num_frames - 1) + 1), dim=-1) - 1
        )
        nb_image_pad = max_seq_len - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        # expand input ids so that the second "merge" with videos does not fail
        final_embedding = torch.zeros(
            batch_size, max_seq_len, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_seq_len, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_seq_len), self.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
        )
        # In case the Vision model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        batch_indices, non_image_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_image_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_image_indices]
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_seq_len), self.config.ignore_index, dtype=input_ids.dtype, device=input_ids.device
            )
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_image_indices]
        else:
            final_labels = None

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.full((batch_size, max_seq_len), True, dtype=torch.bool, device=inputs_embeds.device)
        image_to_overwrite[batch_indices, text_to_overwrite] = False
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None].to(target_device)

        if image_to_overwrite.sum() != visual_features.shape[:-1].numel():
            visual_type = "videos" if num_frames == 8 else "images"
            num_images //= num_frames
            raise ValueError(
                f"The input provided to the model are wrong. The number of {visual_type} tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of {visual_type} given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = visual_features.contiguous().reshape(-1, embed_dim).to(target_device)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids

    def _get_vision_features(
        self,
        pixel_values_images: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        if pixel_values_images is None and pixel_values_videos is None:
            raise ValueError("You have to specify `pixel_values_images` or `pixel_values_videos`")

        # videos do not need to select features and it's always "full" (as it is done in the orig implementation)
        if pixel_values_videos is not None:
            batch_size_vid, num_frames, channels, height, width = pixel_values_videos.shape
            if num_frames != 8:
                raise ValueError(f"Video pixel values should have exactly `8` frames but foung `{num_frames}`")

            pixel_values = pixel_values_videos.reshape(batch_size_vid * num_frames, channels, height, width)
            video_outputs = self.video_tower(pixel_values, output_hidden_states=True)
            video_outputs = video_outputs.hidden_states[vision_feature_layer].squeeze(1)
        else:
            video_outputs = None

        if pixel_values_images is not None:
            image_outputs = self.image_tower(pixel_values_images, output_hidden_states=True)
            image_outputs = image_outputs.hidden_states[vision_feature_layer].squeeze(1)

            if vision_feature_select_strategy == "default":
                image_outputs = image_outputs[:, 1:]
            elif vision_feature_select_strategy == "full":
                image_outputs = image_outputs
            else:
                raise ValueError(f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}")
        else:
            image_outputs = None

        return image_outputs, video_outputs

    @add_start_docstrings_to_model_forward(VIDEO_LLAVA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=VideoLlavaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values_images: torch.FloatTensor = None,
        pixel_values_videos: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,

        # add
        use_mcd: Optional[bool] = None,
        pixel_values_videos_mcd: Optional[torch.FloatTensor] = None,
        mcd_beta: Optional[torch.FloatTensor] = None,
        mcd_alpha: Optional[torch.FloatTensor] = None,

    ) -> Union[Tuple, VideoLlavaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> import numpy as np
        >>> import av
        >>> from huggingface_hub import hf_hub_download
        >>> from transformers import VideoLlavaProcessor, VideoLlavaForConditionalGeneration


        >>> def read_video_pyav(container, indices):
        ...     '''
        ...     Decode the video with PyAV decoder.
        ...     Args:
        ...         container (`av.container.input.InputContainer`): PyAV container.
        ...         indices (`List[int]`): List of frame indices to decode.
        ...     Returns:
        ...         result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        ...     '''
        ...     frames = []
        ...     container.seek(0)
        ...     start_index = indices[0]
        ...     end_index = indices[-1]
        ...     for i, frame in enumerate(container.decode(video=0)):
        ...         if i > end_index:
        ...             break
        ...         if i >= start_index and i in indices:
        ...             frames.append(frame)
        ...     return np.stack([x.to_ndarray(format="rgb24") for x in frames])

        >>> model = VideoLlavaForConditionalGeneration.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")
        >>> processor = VideoLlavaProcessor.from_pretrained("LanguageBind/Video-LLaVA-7B-hf")

        >>> prompt = "USER: <video>Why is this video funny? ASSISTANT:"
        >>> video_path = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")
        >>> container = av.open(video_path)

        >>> # sample uniformly 8 frames from the video
        >>> total_frames = container.streams.video[0].frames
        >>> indices = np.arange(0, total_frames, total_frames / 8).astype(int)
        >>> clip = read_video_pyav(container, indices)

        >>> inputs = processor(text=prompt, videos=clip, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=80)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        'USER:  Why is this video funny? ASSISTANT: The video is funny because the baby is sitting on the bed and reading a book, which is an unusual and amusing sight.Ъ'

        >>> # to generate from image and video mix
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> prompt = [
                "USER: <image> How many cats are there in the image? ASSISTANT:",
                "USER: <video>Why is this video funny? ASSISTANT:"
            ]
        >>> inputs = processor(text=prompt, images=image, videos=clip, padding=True, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=50)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        ['USER:   How many cats are there in the image? ASSISTANT: There are two cats in the image.\nHow many cats are sleeping on the couch?\nThere are', 'USER:  Why is this video funny? ASSISTANT: The video is funny because the baby is sitting on the bed and reading a book, which is an unusual and amusing']
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if (pixel_values_images is not None or pixel_values_videos is not None) and input_ids.shape[1] != 1:
                image_outputs, video_outputs = self._get_vision_features(
                    pixel_values_images=pixel_values_images,
                    pixel_values_videos=pixel_values_videos,
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy,
                )

                # first add image embeds where possible, then expand again and add video embeds
                if image_outputs is not None:
                    visual_features = self.multi_modal_projector(image_outputs)
                    (
                        inputs_embeds,
                        attention_mask,
                        labels,
                        position_ids,
                        input_ids,
                    ) = self._merge_input_ids_with_visual_features(
                        visual_features, inputs_embeds, input_ids, attention_mask, labels
                    )
                if video_outputs is not None:
                    visual_features = self.multi_modal_projector(video_outputs)
                    (
                        inputs_embeds,
                        attention_mask,
                        labels,
                        position_ids,
                        _,
                    ) = self._merge_input_ids_with_visual_features(
                        visual_features,
                        inputs_embeds,
                        input_ids,
                        attention_mask,
                        labels,
                        num_frames=8,
                    )
            else:
                # In case input_ids.shape[1] == 1 & past_key_values != None, we are in the case of
                # generation with cache
                if past_key_values is not None and input_ids.shape[1] == 1:
                    # Retrieve the first layer to inspect the logits and mask out the hidden states
                    # that are set to 0
                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                    # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                    target_length = input_ids.shape[1]
                    past_length = first_layer_past_key_value.shape[-1]

                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], past_length),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )

                    # Filter out only the tokens that can be un-attended, this can happen
                    # if one uses Llava + Fused modules where the cache on the
                    # first iteration is already big enough, or if one passes custom cache
                    valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                    new_batch_index = batch_index[valid_indices]
                    new_non_attended_tokens = non_attended_tokens[valid_indices]

                    # Zero-out the places where we don't need to attend
                    extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                    attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return VideoLlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values_images=None,
        pixel_values_videos=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            else:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

            pixel_values_videos = None
            pixel_values_images = None

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values_videos": pixel_values_videos,
                "pixel_values_images": pixel_values_images,
            }
        )
        return model_inputs
    
    def prepare_inputs_for_generation_mcd(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values_images=None,
        pixel_values_videos_mcd=None,
        attention_mask=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            else:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

            pixel_values_videos = None
            pixel_values_images = None

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values_videos": pixel_values_videos_mcd,
                "pixel_values_images": pixel_values_images,
            }
        )
        return model_inputs
    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
    
def evolve_mcd_VideoLlava():
    transformers.models.video_llava.modeling_video_llava.VideoLlavaForConditionalGeneration=VideoLlavaForConditionalGeneration_MotionCD