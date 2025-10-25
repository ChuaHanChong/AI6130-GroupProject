from typing import Union
import torch
from transformers import LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from transformers.generation.utils import GenerationMixin, GenerateNonBeamOutput, GenerateDecoderOnlyOutput
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _sled_select_contrast(
    candidate_premature_layers: list[int],
    candidate_premature_logits: dict[int, torch.FloatTensor],
    final_logits: torch.FloatTensor,
    evolution_scale: int = 5,
    evolution_rate: float = 0.3,
    evolution_lower_bound: float = -10.0,
    alpha: float = 0.5,
) -> torch.FloatTensor:
    stacked_premature_layers = torch.stack([candidate_premature_logits[i] for i in candidate_premature_layers], dim=0)
    softmax_mature_layer = F.softmax(final_logits, dim=-1)
    softmax_premature_layers = F.softmax(stacked_premature_layers, dim=-1)
    topk_prob, topk_indices = torch.topk(softmax_mature_layer, evolution_scale)
    topk_indices = topk_indices[0]

    divergence = stacked_premature_layers - final_logits
    candidate_gradients_expanded = softmax_premature_layers.expand(-1, len(topk_indices), -1)
    candidate_mask = torch.zeros_like(candidate_gradients_expanded)
    topk_indices_expanded = topk_indices.unsqueeze(0).unsqueeze(2)
    candidate_mask.scatter_(2, topk_indices_expanded.expand(softmax_premature_layers.size(0), -1, -1), 1)

    candidate_gradients_expanded = candidate_gradients_expanded - candidate_mask
    candidate_gradients_expanded = candidate_gradients_expanded.to(torch.float32)
    layer_divergence_expanded = divergence.to(torch.float32)

    layer_dot_results = F.cosine_similarity(candidate_gradients_expanded, layer_divergence_expanded, dim=2)
    layer_topk_values, layer_topk_indices = torch.topk(layer_dot_results, evolution_scale)
    layer_topk_topk_indices = topk_indices[layer_topk_indices]

    layer_topk_values = (layer_topk_values * (layer_topk_values > 0)) ** 2
    layer_topk_values_sum_layers = torch.sum(layer_topk_values, dim=1).clone()
    non_zero_indices = layer_topk_values_sum_layers != 0
    layer_topk_values[non_zero_indices] /= layer_topk_values_sum_layers[non_zero_indices].unsqueeze(1)
    if layer_topk_values_sum_layers.sum() != 0:
        layer_topk_values_sum_layers = layer_topk_values_sum_layers / layer_topk_values_sum_layers.sum()
    proxy_gradients_tensor_delta = torch.zeros_like(softmax_mature_layer,device=layer_divergence_expanded.device).to(layer_divergence_expanded.dtype).repeat(layer_topk_values.size(0), 1)
    proxy_gradients_tensor_delta.scatter_(1, layer_topk_topk_indices, -layer_topk_values)
    proxy_gradients_tensor_delta = torch.sum(proxy_gradients_tensor_delta * layer_topk_values_sum_layers.unsqueeze(1), dim=0)
    proxy_gradients_tensor_delta = proxy_gradients_tensor_delta.to(softmax_mature_layer.dtype)
    hidden_states_seq_i = final_logits.clone()

    op_T = 1
    evolution_rate_scheduler = [evolution_rate * (1 - i / op_T) for i in range(op_T)]
    for op_t in range(op_T):
        er_t = evolution_rate_scheduler[op_t]
        # Compute softmax only once and update it iteratively if necessary
        softmax_hidden_states_seq_i = F.softmax(hidden_states_seq_i, dim=-1)
        proxy_gradients_tensor = softmax_hidden_states_seq_i + proxy_gradients_tensor_delta
        # In-place update to reduce memory overhead
        hidden_states_seq_i.sub_(er_t * proxy_gradients_tensor)

    sl = torch.stack([candidate_premature_logits[i] for i in candidate_premature_layers], dim=1)  # [B, L, V]
    dist = sl[:, :, topk_indices].permute(0, 2, 1)  # [B, V, L]
    probs = torch.softmax(dist, dim=2)
    entropy = torch.distributions.Categorical(probs=probs, validate_args=False).entropy()  # [B, V]
    hidden_states_seq_i[:, topk_indices] = hidden_states_seq_i[:, topk_indices] + alpha * (-entropy)

    hidden_states_seq_i_new = torch.full_like(hidden_states_seq_i[0], fill_value=evolution_lower_bound, device=hidden_states_seq_i.device, dtype=hidden_states_seq_i.dtype)
    hidden_states_seq_i_new[topk_indices] = hidden_states_seq_i[0, topk_indices]
    next_token_logits = hidden_states_seq_i_new.unsqueeze(dim=0)
    return next_token_logits


def _sled_decoding(
        model,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: "BaseStreamer" = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **dola decoding** and can be
        used for decoder-only text models.
        The method is based on the paper "SELD: Decoding by Contrasting Layers Improves Factuality in Large Language
        Models" (https://huggingface.co/papers/2309.03883) in ICLR 2024.
        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.
        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`]
            or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        evolution_rate: float = generation_config.evolution_rate
        evolution_scale: int = generation_config.evolution_scale
        evolution_lower_bound: float = generation_config.evolution_lower_bound
        alpha: float = generation_config.alpha

        # 1. General sanity checks
        # A few arguments are not allowed, especially arguments that control caches.
        assert evolution_rate is not None, "`evolution_rate` must be set for SELD decoding."
        assert evolution_scale is not None, "`evolution_scale` must be set for SELD decoding."
        assert evolution_lower_bound is not None, "`evolution_lower_bound` must be set for SELD decoding."

        # SELD generation needs num_beams == 1
        if getattr(generation_config, "num_beams", 1) != 1:
            raise ValueError("SELD generation needs num_beams == 1")
        
        if model.config.is_encoder_decoder:
            raise ValueError("SELD decoding is only available for decoder-only models.")

        if generation_config.repetition_penalty < 1.2:
            logger.warning(
                f"`repetition_penalty` is set to a value of {generation_config.repetition_penalty}, which could induce unwanted repetition. "
                "The recommended value for SELD decoding is `repetition_penalty>=1.2`.",
            )

        if getattr(model, "_is_stateful", False):
            # SELD decoding was not designed for stateful models, and would require some changes
            raise ValueError(
                f"SELD decoding is not supported with stateful models, such as {model.__class__.__name__}"
            )

        if model.config.is_encoder_decoder:
            raise ValueError("SELD decoding is only available for decoder-only models.")
        
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        batch_size, cur_length = input_ids.shape[:2]
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = model._get_initial_cache_position(cur_length, input_ids.device, model_kwargs)

        this_peer_finished = False

        # prepare layers for SELD decoding
        final_layer = model.config.get_text_config().num_hidden_layers
        # if the model has tied word embeddings, we skip the word embeddings (0-th) layer and start from the 2nd layer,
        # as the early exit from word embeddings will become identity function
        # if the model is really shallow (<=2 layers), we use the 1st layer if it's not the final layer and the 0-th
        # layer otherwise.
        if not model.config.tie_word_embeddings:
            start_layer = 0
        elif final_layer > 2:
            start_layer = 2
        elif final_layer == 2:
            start_layer = 1
        else:
            start_layer = 0

        candidate_premature_layers = list(range(start_layer, final_layer))

        lm_head = model.get_output_embeddings()
        if lm_head is None:
            raise ValueError("SELD is not supported for models that don't have output embeddings.")

        while model._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = model(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=True,
            )

            # .float() is needed to retain precision for later logits manipulations
            final_layer_next_token_logits = outputs.logits[:, -1, :].detach().to(copy=True, dtype=torch.float32)
            final_logits = outputs.logits[:, -1, :].float()
            candidate_premature_logits = {}
            for candidate_premature_layer in candidate_premature_layers:
                candidate_premature_logits[candidate_premature_layer] = lm_head(outputs.hidden_states[candidate_premature_layer][:, -1, :]).to(final_logits.device)

            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = model._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=model.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            next_token_logits = _sled_select_contrast(
                candidate_premature_layers, 
                candidate_premature_logits, 
                final_logits,
                evolution_scale=evolution_scale,
                evolution_rate=evolution_rate,
                evolution_lower_bound=evolution_lower_bound,
                alpha=alpha
            )
            next_token_logits = next_token_logits.to(input_ids.device)
            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (final_layer_next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            if do_sample:  # sample
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:  # argmax
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            # stop when each sentence is finished
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
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


def generate(model, *args, **kwargs):
    """Custom generate function for SELD decoding.
    Args:
        model (`PreTrainedModel`):
            The model to generate from.
        evolution_rate (`float`):
            The evolution rate for SELD decoding.
        evolution_scale (`int`):
            The evolution scale for SELD decoding.
        evolution_lower_bound (`float`):
            The evolution lower bound for SELD decoding.
    """
    generation_outputs = GenerationMixin.generate(model, *args, custom_generate=_sled_decoding, **kwargs)
    return generation_outputs
