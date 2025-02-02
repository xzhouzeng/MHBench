import copy
from typing import Optional, Union

import torch

from torch import nn


from transformers.generation.configuration_utils import GenerationConfig, GenerationMode
from transformers.generation.logits_process import (
    LogitsProcessorList,
)
import transformers
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)

from transformers.generation.utils import GenerateNonBeamOutput,GenerateDecoderOnlyOutput, GenerateEncoderDecoderOutput

from transformers.generation.streamers import BaseStreamer

def sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    logits_warper: Optional[LogitsProcessorList] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
 
    # init values
    pad_token_id = generation_config.pad_token_id
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample
    if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
        raise ValueError(
            "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
            f"{logits_warper})."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size = input_ids.shape[0]
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    # Update: 1. add model_kwargs_mcd
    use_mcd = (model_kwargs.get("use_mcd") == True)

    if use_mcd:
        model_kwargs_mcd=copy.deepcopy(model_kwargs)

        if model_kwargs_mcd.get("inputs_embeds_mcd") is not None:
            if model_kwargs_mcd['inputs_embeds'].shape[1] > model_kwargs_mcd['inputs_embeds_mcd'].shape[1]:
                model_kwargs_mcd['attention_mask'] = model_kwargs_mcd['attention_mask'][:, :model_kwargs_mcd['inputs_embeds_mcd'].shape[1]]
                model_kwargs_mcd['cache_position'] = model_kwargs_mcd['cache_position'][:model_kwargs_mcd['inputs_embeds_mcd'].shape[1]]

        # 清除position_ids
        model_kwargs_mcd.pop("position_ids", None)
        output_attentions_mcd = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
            )
        output_hidden_states_mcd = (
                output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
            )

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # Update: 2. add mcd
        if use_mcd:
            model_inputs_mcd = self.prepare_inputs_for_generation_mcd(input_ids, **model_kwargs_mcd)
            outputs_mcd = self(
                **model_inputs_mcd,
                return_dict=True,
                output_attentions=output_attentions_mcd,
                output_hidden_states=output_hidden_states_mcd,
            )
            next_token_logits_mcd = outputs_mcd.logits[:, -1, :]

            mcd_alpha = model_kwargs.get("mcd_alpha") if model_kwargs.get("mcd_alpha") is not None else 1.0
            mcd_beta = model_kwargs.get("mcd_beta") if model_kwargs.get("mcd_beta") is not None else 0.1
            

            cutoff = torch.log(torch.tensor(mcd_beta)) + next_token_logits.max(dim=-1, keepdim=True).values
            mcd_alpha=mcd_alpha/2
            diffs = (1+mcd_alpha)*next_token_logits - mcd_alpha*next_token_logits_mcd
            mcd_logits = diffs.masked_fill(next_token_logits < cutoff, -float("inf"))

            next_token_logits = mcd_logits

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)
        if do_sample:
            next_token_scores = logits_warper(input_ids, next_token_scores)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )

        # Update: 3. update model_kwargs_mcd
        if use_mcd:
            model_kwargs_mcd = self._update_model_kwargs_for_generation(
                outputs_mcd, model_kwargs_mcd, is_encoder_decoder=self.config.is_encoder_decoder
            )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids
    
def evolve_mcd_sampling():
    print("--------------setting mcd sampling----------------")
    transformers.generation.utils.GenerationMixin._sample = sample