# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""ChatGLM-3 model."""

import torch

from megatron import get_args
from megatron.core import mpu, tensor_parallel, sequence_parallel
from .module import MegatronModule, fp32_to_float16, float16_to_fp32

from .enums import AttnMaskType
from .language_model import parallel_lm_logits
from .language_model import get_language_model
from .utils import init_method_normal
from .utils import scaled_init_method_normal

from megatron.model import LayerNorm
from .language_model import EmbeddingPipe
from .transformer import ParallelTransformerLayerPipe, LMHeadPipe
from deepspeed.pipe import PipelineModule, LayerSpec, TiedLayerSpec

try:
    from apex.normalization import MixedFusedRMSNorm
except ImportError:
    MixedFusedRMSNorm = None

try:         
    from deepspeed.checkpoint import (
        VOCABULARY_PARAMETER_PATTERNS,
        PIPELINE_REPLICATED_PARAMETER_PATTERNS,
        TP_REPLICATED_PARAMETER_PATTERNS,
        PARAMETER_WITH_ROW_PARALLELISM_PATTERNS,
    )
    DS_UNIVERSAL_CHECKPOINT_INFO = True 
except ImportError:
    DS_UNIVERSAL_CHECKPOINT_INFO = False  


def post_language_model_processing(lm_output, labels, logit_weights,
                                   parallel_output,
                                   fp16_lm_cross_entropy):

    # Output. Format [s b h]
    output = parallel_lm_logits(
        lm_output,
        logit_weights,
        parallel_output)

    if labels is None:
        # [s b h] => [b s h]
        return output.transpose(0,1).contiguous()
    else:
        # [b s] => [s b]
        labels = labels.transpose(0,1).contiguous()
        cross_entropy = sequence_parallel.vocab_sequence_parallel_cross_entropy if mpu.get_sequence_parallel_world_size() > 1 \
            else tensor_parallel.vocab_parallel_cross_entropy
        if fp16_lm_cross_entropy:
            assert output.dtype == torch.half
            loss = cross_entropy(output, labels)
        else:
            loss = cross_entropy(output.float(), labels)

        # [s b] => [b, s]
        loss = loss.transpose(0,1).contiguous()
        return loss


class ChatGLM3Model(MegatronModule):
    """ChatGLM-3 Language model."""

    def __init__(self,
                 config,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 return_moe_loss=True):
        args = get_args()
        super().__init__(config=config, share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights)

        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.return_moe_loss = return_moe_loss
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights

        self.language_model, self._language_model_key = get_language_model(
            config=config,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            encoder_attn_mask_type=AttnMaskType.causal,
            pre_process=self.pre_process,
            post_process=self.post_process,
            num_experts=args.num_experts)

        if not args.untie_embeddings_and_output_weights:
            self.initialize_word_embeddings()

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.language_model.set_input_tensor(input_tensor)

    def forward(self, input_ids, position_ids, attention_mask,
                retriever_input_ids=None,
                retriever_position_ids=None,
                retriever_attn_mask=None,
                labels=None, tokentype_ids=None, inference_params=None,
                curriculum_seqlen=None):
        args = get_args()
        if curriculum_seqlen is not None:
            args.curriculum_seqlen = curriculum_seqlen
            if curriculum_seqlen < input_ids.size()[1]:
                # seqlen-based curriculum learning
                # input_ids, position_ids, labels have size [batch size, seqlen]
                input_ids = input_ids[:, :curriculum_seqlen].contiguous()
                position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                if labels is not None:
                    labels = labels[:, :curriculum_seqlen].contiguous()

                # attention_mask has size [1, 1, seqlen, seqlen]
                attention_mask = attention_mask[:, :, :curriculum_seqlen, :curriculum_seqlen].contiguous()
        else:
            if args.curriculum_learning_legacy:
                # If got a None input, need to reset curriculum_seqlen on user side
                args.curriculum_seqlen = args.seq_length

        lm_output, moe_losses = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            retriever_input_ids=retriever_input_ids,
            retriever_position_ids=retriever_position_ids,
            retriever_attn_mask=retriever_attn_mask,
            inference_params=inference_params)

        if self.post_process:
            lm_output = post_language_model_processing(
                lm_output, labels,
                self.language_model.output_layer.weight if self.untie_embeddings_and_output_weights else self.shared_embedding_or_output_weight(),
                self.parallel_output,
                self.fp16_lm_cross_entropy)

        return lm_output, moe_losses if self.return_moe_loss else lm_output

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):

        state_dict_ = {}
        language_model_state_dict = self.language_model.state_dict_for_save_checkpoint(
                prefix=prefix, keep_vars=keep_vars)
        # MoE states need to be handled separately by DeepSpeed engine, thus
        # moving them to the top level dictionary
        if "moe_state_dict" in language_model_state_dict:
            for key in list(language_model_state_dict["moe_state_dict"].keys()):
                state_dict_[key] = language_model_state_dict["moe_state_dict"].pop(key)
            del language_model_state_dict["moe_state_dict"]
        state_dict_[self._language_model_key] = language_model_state_dict
        # Save word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            state_dict_[self._word_embeddings_for_head_key] \
                = self.word_embeddings.state_dict(prefix=prefix,
                                                  keep_vars=keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Load word_embeddings.
        if self.post_process and not self.pre_process and not self.untie_embeddings_and_output_weights:
            self.word_embeddings.load_state_dict(
                state_dict[self._word_embeddings_for_head_key], strict=strict)
        # Gather MoE states and move under language model
        moe_state_dict = {}
        for key in list(state_dict.keys()):
            if 'expert' in key and 'moe.gate.wg.weight' not in key:
                moe_state_dict[key] = state_dict.pop(key)
        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        if len(moe_state_dict) > 0:
            state_dict["moe_state_dict"] = moe_state_dict
        self.language_model.load_state_dict(state_dict, strict=strict)

    def universal_checkpoint_info(self):
        info = dict()
        if DS_UNIVERSAL_CHECKPOINT_INFO:
            # Vocabulary parameters (embeddings) that require special handling due to padding.
            info[VOCABULARY_PARAMETER_PATTERNS] = [
                r"tied_modules.embed.word_embeddings.weight"
            ]

            # Parameter slices that should be averaged not concatenated.
            info[TP_REPLICATED_PARAMETER_PATTERNS] = [
                r"tied_modules.embed.position_embeddings.weight",
                r"\d+.input_layernorm.weight",
                r"\d+.input_layernorm.bias",
                r"\d+.post_attention_layernorm.weight",
                r"\d+.post_attention_layernorm.bias",
                r"\d+.self_attention.dense.bias",
                r"\d+.mlp.dense_4h_to_h.bias",
                r"\d+.weight",
                r"\d+.bias",
            ]

            # Parameter that are sliced on the row dimension
            info[PARAMETER_WITH_ROW_PARALLELISM_PATTERNS] = [
                r"\d+.mlp.dense_4h_to_h.weight",
                r"\d+.self_attention.dense.weight",
            ]

        return info
    
def CrossEntropy(output, labels):
    labels, loss_mask = labels[0], labels[1]

    args = get_args()

    # [b s] => [s b]
    labels = labels.transpose(0, 1).contiguous()
    losses = tensor_parallel.vocab_parallel_cross_entropy(output.contiguous().float(), labels)
    # [s b] => [b, s]
    losses = losses.transpose(0, 1).contiguous()
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    return loss
