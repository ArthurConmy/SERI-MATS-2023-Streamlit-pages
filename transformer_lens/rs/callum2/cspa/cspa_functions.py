# This file contains functions to get results from my model. I use them to generate the visualisations found on the Streamlit pages.

# Make sure explore_prompts is in path (it will be by default in Streamlit)
import sys, os
from typing import Dict, Any, Tuple, List, Optional, Literal, Union
from transformer_lens import HookedTransformer, utils
from functools import partial
import einops
from dataclasses import dataclass
from transformer_lens.hook_points import HookPoint
from jaxtyping import Int, Float, Bool
import torch as t
from collections import defaultdict
import time
from torch import Tensor
import pandas as pd
import numpy as np
from copy import copy
import math
from tqdm import tqdm
import pickle

from transformer_lens.rs.callum2.cspa.cspa_semantic_similarity import (
    concat_lists,
    make_list_correct_length,
    get_list_with_no_repetitions,
)
from transformer_lens.rs.callum2.utils import (
    ST_HTML_PATH,
    devices_are_equal,
    first_occurrence_2d,
    concat_dicts,
    kl_div,
)

Head = Tuple[int, int]


FUNCTION_STR_TOKS =  [
    '\x00',
    '\x01',
    '\x02',
    '\x03',
    '\x04',
    '\x05',
    '\x06',
    '\x07',
    '\x08',
    '\t',
    '\n',
    '\x0b',
    '\x0c',
    '\r',
    '\x0e',
    '\x0f',
    '\x10',
    '\x11',
    '\x12',
    '\x13',
    '\x14',
    '\x15',
    '\x16',
    '\x17',
    '\x18',
    '\x19',
    '\x1a',
    '\x1b',
    '\x1c',
    '\x1d',
    '\x1e',
    '\x1f',
    '\x7f',
    ' a',
    ' the',
    ' an',
    ' to',
    ' of',
    ' in',
    ' that',
    ' for',
    ' it',
    ' with',
    ' as',
    ' was',
    ' The',
    ' are',
    ' by',
    ' have',
    ' this',
    'The',
    ' will',
    ' they',
    ' their',
    ' which',
    ' about',
    ' In',
    ' like',
    ' them',
    ' some',
    ' when',
    ' It',
    ' what',
    ' its',
    ' only',
    ' how',
    ' most',
    ' This',
    ' these',
    ' very',
    ' much',
    ' those',
    ' such',
    ' But',
    ' You',
    ' If',
    ' take',
    'It',
    ' As',
    ' For',
    ' They',
    'the',
    ' That',
    'But',
    ' When',
    ' To',
    'As',
    ' almost',
    ' With',
    ' However',
    'When',
    ' These',
    'That',
    'To',
    ' By',
    ' takes',
    ' While',
    ' whose',
    'With',
    ' Of',
    ' THE',
    ' From',
    ' aren',
    'While',
    ' perhaps',
    'By',
    ' whatever',
    'with',
    ' Like',
    'However',
    'Of',
    ' Their',
    ' Those',
    ' Its',
    ' Thus',
    ' Such',
    'Those',
    ' Much',
    ' Between',
    'Their',
    ' meanwhile',
    ' nevertheless',
    'Its',
    ' at', ' of', 'to', ' now', "'s", 'The', ".", ",", "?", "!", " '", "'", ' "', '"', 'you', 'your', ' you', ' your', ' once', ' and', ' all', 'Now', ' He',
    ' her', ' my', ' your',
    '<|endoftext|>',
    # Need to go through these words and add more to them, I suspect this list is minimal
]






def gram_schmidt(basis: Float[Tensor, "... d num"]) -> Float[Tensor, "... d num"]:
    '''
    Performs Gram-Schmidt orthonormalization on a batch of vectors, returning a basis.

    `basis` is a batch of vectors. If it was 2D, then it would be `num` vectors each with length
    `d`, and I'd want a basis for these vectors in d-dimensional space. If it has more dimensions 
    at the start, then I want to do the same thing for all of them (i.e. get multiple independent
    bases).

    If the vectors aren't linearly independent, then some of the basis vectors will be zero (this is
    so we can still batch our projections, even if the subspace rank for each individual projection
    is not equal.
    '''
    # Make a copy of the vectors
    basis = basis.clone()
    num_vectors = basis.shape[-1]
    
    # Iterate over each vector in the batch, starting from the zeroth
    for i in range(num_vectors):

        # Project the i-th vector onto the space orthogonal to the previous ones
        for j in range(i):
            projection = einops.einsum(basis[..., i], basis[..., j], "... d, ... d -> ...")
            basis[..., i] = basis[..., i] - einops.einsum(projection, basis[..., j], "..., ... d -> ... d")
        
        # Normalize this vector (we can set it to zero if it's not linearly independent)
        basis_vec_norm = basis[..., i].norm(dim=-1, keepdim=True)
        basis[..., i] = t.where(
            basis_vec_norm > 1e-6,
            basis[..., i] / basis_vec_norm,
            0.0,
        )
    
    return basis



def project(
    vectors: Float[Tensor, "... d"],
    proj_directions: Float[Tensor, "... d num"],
    only_keep: Optional[Literal["pos", "neg"]] = None,
    gs: bool = True,
    return_coeffs: bool = False,
):
    '''
    `vectors` is a batch of vectors, with last dimension `d` and all earlier dimensions as batch dims.

    `proj_directions` is either the same shape as `vectors`, or has an extra dim at the end.

    If they have the same shape, we project each vector in `vectors` onto the corresponding direction
    in `proj_directions`. If `proj_directions` has an extra dim, then the last dimension is another 
    batch dim, i.e. we're projecting each vector onto a subspace rather than a single vector.
    '''
    # If we're only projecting onto one direction, add a dim at the end (for consistency)
    if proj_directions.shape == vectors.shape:
        proj_directions = proj_directions.unsqueeze(-1)
    # Check shapes
    assert proj_directions.shape[:-1] == vectors.shape
    assert not((proj_directions.shape[-1] > 20) and gs), "Shouldn't do too many vectors, GS orth might be computationally heavy I think"

    # We might want to have done G-S orthonormalization first
    proj_directions_basis = gram_schmidt(proj_directions) if gs else proj_directions

    components_in_proj_dir = einops.einsum(
        vectors, proj_directions_basis,
        "... d, ... d num -> ... num"
    )
    if return_coeffs: return components_in_proj_dir
    
    if only_keep == "pos":
        components_in_proj_dir = t.where(components_in_proj_dir > 0, components_in_proj_dir, 0.0)
    elif only_keep == "neg":
        components_in_proj_dir = t.where(components_in_proj_dir < 0, components_in_proj_dir, 0.0)

    vectors_projected = einops.einsum(
        components_in_proj_dir,
        proj_directions_basis,
        "... num, ... d num -> ... d"
    )

    return vectors_projected



def model_fwd_pass_from_resid_pre(
    model: HookedTransformer, 
    resid_pre: Float[Tensor, "batch seq d_model"],
    layer: int,
    return_type: Literal["logits", "resid_post", "loss"] = "logits",
    toks: Optional[Int[Tensor, "batch seq"]] = None,
) -> Float[Tensor, "batch seq d_vocab"]:
    '''
    Performs a forward pass starting from an intermediate point.

    For instance, if layer=10, this will apply the TransformerBlocks from
    layers 10, 11 respectively, then ln_final and unembed.

    Also, if return_resid_post = True, then it just returns the final value
    of the residual stream, i.e. omitting ln_final and unembed.
    '''
    assert return_type in ["logits", "resid_post", "loss"]

    resid = resid_pre
    for i in range(layer, model.cfg.n_layers):
        resid = model.blocks[i](resid)
    
    if (return_type == "resid_post"): return resid

    resid_scaled = model.ln_final(resid)
    logits = model.unembed(resid_scaled)

    if return_type == "logits":
        return logits
    elif return_type == "resid_post":
        return resid
    elif return_type == "loss":
        assert toks is not None
        return model.loss_fn(logits, toks, per_token=True)
    


def get_effective_embedding(model: HookedTransformer) -> Float[Tensor, "d_vocab d_model"]:

    # TODO - make this consistent (i.e. change the func in `generate_bag_of_words_quad_plot` to also return W_U and W_E separately)

    W_E = model.W_E.clone()
    W_U = model.W_U.clone()
    # t.testing.assert_close(W_E[:3, :3], W_U[:3, :3].T)  NOT TRUE, because of the center unembed part, and the folded weights

    resid_pre = W_E.unsqueeze(0)
    pre_attention = model.blocks[0].ln1(resid_pre)
    attn_out = einops.einsum(
        pre_attention, 
        model.W_V[0],
        model.W_O[0],
        "b s d_model, num_heads d_model d_head, num_heads d_head d_model_out -> b s d_model_out",
    )
    resid_mid = attn_out + resid_pre
    normalized_resid_mid = model.blocks[0].ln2(resid_mid)
    mlp_out = model.blocks[0].mlp(normalized_resid_mid)
    
    W_EE = mlp_out.squeeze()
    W_EE_full = resid_mid.squeeze() + mlp_out.squeeze()

    t.cuda.empty_cache()

    return {
        "W_E (no MLPs)": W_E,
        "W_U": W_U.T,
        # "W_E (raw, no MLPs)": W_E,
        "W_E (including MLPs)": W_EE_full,
        "W_E (only MLPs)": W_EE
    }




    




def get_cspa_results(
    model: HookedTransformer,
    toks: Int[Tensor, "batch seq"],
    negative_head: Tuple[int, int],
    interventions: List[str],
    K_unembeddings: Optional[Union[int, float]] = None,
    K_semantic: int = 10,
    only_keep_negative_components: bool = False,
    semantic_dict: dict = {},
    result_mean: Optional[Dict[Tuple[int, int], Float[Tensor, "seq_plus d_model"]]] = None,
    return_dla: bool = False,
    return_logits: bool = False,
    verbose: bool = False,
    keep_self_attn: bool = True,
) -> Tuple[
        Dict[str, Float[Tensor, "batch seq-1"]],
        Int[Tensor, "n 4"],
        Float[Tensor, "n"],
        Int[Tensor, "batch seqK K_semantic"],
        float,
    ]:
    '''
    Short explpanation of the copy-suppression preserving ablation (hereby CSPA), with
    the following arguments:

        interventions = ["qk", "ov"]
        K_semantic = 3
        K_unembeddings = 5

        For every souce token s, we pick the 3 tokens which it's most semantically
        related to - call this set S*.

        For every destination token d, we look at the union of all semantically similar
        tokens S* for the source tokens before it in context, and pick the 5 which
        are most predicted (i.e. the highest logit lens). This gives us 5 pairs (s, s*).

        For each source token s which is in one of these pairs, we take the result vector
        which is moved from s -> d, and project it onto the unembeddings of all s* which
        it appears in a pair with. If a source token doesn't appear in one of these pairs,
        we mean ablate that vector.

            Summary: information is moved from s -> d if and only if s is semantically similar 
            to some token s* which is being predicted at d, and in this case the information
            which is moved is restricted to the subspace of the unembedding of s*.

        If instead we just had interventions = ["qk"], then we'd filter for the pairs (s, s*)
        where s* is predicted at d, but we wouldn't project their output vectors onto
        unembeddings. If instead we just had ["ov"], then we'd project all vectors from s onto 
        the span of unembeddings of their s*, but we wouldn't also filter for pairs (s, s*) 
        where s* is predicted.

    
    A few notes / explanations:

        > result_mean is the vector we'll use for ablation, if supplied. It'll map e.g. 
          (10, 7) to the mean result vector for each seqpos (hopefully seqpos is larger than 
          that for toks).
        > If K_unembeddings is a float rather than an int, it's converted to an int as follows:
          ceil(K_unembeddings * destination_position).

    Return type:

        > A dictionary of the results, with "loss", "loss_ablated", and "loss_cspa", plus the
          logits & KL divergences, as keys.
        > A dict mapping (batch_idx, d) to the list of (s, s*) which we preserve in our
          ablation.
    '''

    # ====================================================================
    # ! STEP 0: Define setup vars, move things onto the correct device, get semantically similar toks
    # ====================================================================

    LAYER, HEAD = negative_head
    batch_size, seq_len = toks.shape
    
    model.reset_hooks(including_permanent=True)
    t.cuda.empty_cache()
    
    W_U = model.W_U
    FUNCTION_TOKS = model.to_tokens(FUNCTION_STR_TOKS, prepend_bos=False).squeeze()


    if K_semantic > 1:
        # Flatten all tokens, and convert them into str toks
        str_toks_flat = model.to_str_tokens(toks.flatten(), prepend_bos=False)
        # Use our dictionary to find the top `K_semantic` semantically similar tokens for each one
        semantically_similar_str_toks = concat_lists([make_list_correct_length(concat_lists(semantic_dict[str_tok]), K_semantic, pad_tok='======') for str_tok in str_toks_flat])
        # Turn these into tokens
        semantically_similar_toks = model.to_tokens(semantically_similar_str_toks, prepend_bos=False)
        semantically_similar_toks = semantically_similar_toks.reshape(batch_size, seq_len, K_semantic) # [batch seqK K_semantic]
    else:
        semantically_similar_toks = toks.unsqueeze(-1) # [batch seqK 1]



    # ====================================================================
    # ! STEP 1: Perform clean and ablated model forward passes (while caching clean activations, for use later)
    # ====================================================================
    
    # Get all hook names
    # TODO - change names to "pattern_hook_name" and "pattern_hook_fn" etc
    hook_name_resid = utils.get_act_name("resid_pre", LAYER)
    hook_name_resid_final = utils.get_act_name("resid_post", model.cfg.n_layers - 1)
    hook_name_result = utils.get_act_name("result", LAYER)
    hook_name_v = utils.get_act_name("v", LAYER)
    hook_name_pattern = utils.get_act_name("pattern", LAYER)
    hook_name_scale = utils.get_act_name("scale", LAYER, "ln1")
    hook_name_scale_final = utils.get_act_name("scale")
    hook_names_to_cache = [hook_name_scale, hook_name_v, hook_name_pattern, hook_name_scale_final, hook_name_resid, hook_name_resid_final, hook_name_result]

    t_clean_and_ablated = time.time()

    # * Get clean results (also use this to get residual stream before layer 10)
    model.reset_hooks()
    logits, cache = model.run_with_cache(
        toks,
        return_type = "logits",
        names_filter = lambda name: name in hook_names_to_cache
    )
    loss = model.loss_fn(logits, toks, per_token=True)
    resid_post_final = cache[hook_name_resid_final] # [batch seqQ d_model]
    resid_pre = cache[hook_name_resid] # [batch seqK d_model]
    # TODO odd thing; sometimes it seems like scale isn't recorded as different across heads. why?
    scale = cache[hook_name_scale]
    if scale.ndim == 4:
        scale = cache[hook_name_scale][:, :, HEAD] # [batch seqK 1]
    head_result_orig = cache[hook_name_result][:, :, HEAD] # [batch seqQ d_model]
    final_scale = cache[hook_name_scale_final] # [batch seq 1]
    v = cache[hook_name_v][:, :, HEAD] # [batch seqK d_head]
    pattern = cache[hook_name_pattern][:, HEAD] # [batch seqQ seqK]      
    del cache

    # * Perform complete ablation (via a direct calculation)
    if batch_size * seq_len < 1000:
        assert result_mean is not None, "You should be using an externally supplied mean ablation vector for such a small dataset."
    if result_mean is None:
        head_result_orig_mean_ablated = einops.reduce(head_result_orig, "batch seqQ d_model -> seqQ d_model", "mean")
    else:
        head_result_orig_mean_ablated = result_mean[(LAYER, HEAD)][:seq_len]
    resid_post_final_mean_ablated = resid_post_final + (head_result_orig_mean_ablated - head_result_orig) # [batch seq d_model]
    logits_mean_ablated = (resid_post_final_mean_ablated / final_scale) @ model.W_U + model.b_U
    loss_mean_ablated = model.loss_fn(logits_mean_ablated, toks, per_token=True)
    model.reset_hooks()

    t_clean_and_ablated = time.time() - t_clean_and_ablated



    # ====================================================================
    # ! STEP 3: Get CSPA results (this is the hard part!)
    # ====================================================================

    # Multiply by output matrix, then by attention probabilities
    output = v @ model.W_O[LAYER, HEAD] # [batch seqK d_model]
    output_attn = einops.einsum(output, pattern, "batch seqK d_model, batch seqQ seqK -> batch seqQ seqK d_model")
    
    # We might want to use the results supplied for mean ablation
    if result_mean is None:
        output_attn_mean_ablated = einops.reduce(output_attn, "batch seqQ seqK d_model -> seqQ 1 d_model", "mean")
    else:
        # output_attn_pre_mean_ablation = einops.einsum(
        #     result_mean[(LAYER, HEAD)][:seq_len], pattern,
        #     "seqQ d_model, batch seqQ seqK -> batch seqQ seqK d_model"
        # )
        # # TODO - which of these two below is more principled? Doesn't really matter; they get approximately the same results.
        # # output_attn_mean_ablated = einops.reduce(output_attn_pre_mean_ablation, "batch seqQ seqK d_model -> seqQ 1 d_model", "mean")
        # output_attn_mean_ablated = einops.reduce(output_attn_pre_mean_ablation, "batch seqQ seqK d_model -> seqQ seqK d_model", "mean")

        output_attn_mean_ablated = einops.einsum(
            result_mean[(LAYER, HEAD)][:seq_len], pattern,
            "seqQ d_model, batch seqQ seqK -> batch seqQ seqK d_model"
        )

    # If we ablate QK, this means we need to get the top predicted semantically similar tokens
    if "qk" in interventions:
        # Get the unembeddings we'll be projecting onto (also get the dict of (s, s*) pairs and store in context)
        # Most of the elements in `semantically_similar_unembeddings` will be zero
        t0 = time.time()
        semantically_similar_unembeddings, top_K_and_Ksem_per_dest_token, logit_lens_for_top_K_Ksem, top_K_and_Ksem_mask = get_top_predicted_semantically_similar_tokens(
            toks=toks,
            resid_pre=resid_pre,
            semantically_similar_toks=semantically_similar_toks,
            K_unembeddings=K_unembeddings,
            function_toks=FUNCTION_TOKS,
            model=model,
            final_scale=final_scale,
            keep_self_attn=keep_self_attn,
        )
        if verbose: print(f"Fraction of unembeddings we keep = {(semantically_similar_unembeddings.abs() > 1e-6).float().mean():.4f}")
        time_for_sstar = time.time() - t0
    else:
        time_for_sstar = 0.0
        # In this case, s_s_u will be like the thing above, but without basically all of its elements masked to zero
        semantically_similar_unembeddings = einops.repeat(
            W_U.T[semantically_similar_toks],
            "batch seqK K_semantic d_model -> batch seqQ seqK d_model K_semantic",
            seqQ = seq_len,
        )

    if "ov" in interventions:
        # We project the output onto the unembeddings we got from the code above (which will either be all unembeddings,
        # or those which were filtered for being predicted on the destination side).
        if only_keep_negative_components:
            assert K_semantic == 1, "Can't use semantic similarity if we're only keeping negative components."
        output_attn_cspa = project(
            vectors = output_attn - output_attn_mean_ablated,
            proj_directions = semantically_similar_unembeddings,
            only_keep = "neg" if only_keep_negative_components else None
        ) + output_attn_mean_ablated
    else:
        # In this case, we assume we are filtering for QK (cause we're doing at least one). We want to set the output to be the mean-ablated
        # output at all source positions which are not in the top predicted semantically similar tokens.
        def any_reduction(tensor: Tensor, dims: tuple):
            assert dims == (3,)
            return tensor.any(dims[0])
        top_K_and_Ksem_mask_any = einops.reduce(
            top_K_and_Ksem_mask, 
            "batch seqQ seqK K_semantic -> batch seqQ seqK",
            reduction = any_reduction # dims will be supplied as (3,) I think
        )
        output_attn_cspa = t.where(
            top_K_and_Ksem_mask_any.unsqueeze(-1),
            output_attn,
            output_attn_mean_ablated,
        )

    # Sum over key-side vectors to get new head result
    # ? (don't override the BOS token attention, because it's more appropriate to preserve this information I think)
    # output_attn_cspa[:, :, 0, :] = output_attn[:, :, 0, :]
    head_result_cspa = einops.reduce(output_attn_cspa, "batch seqQ seqK d_model -> batch seqQ d_model", "sum")


    # Get DLA, logits, and loss
    dla_cspa = ((head_result_cspa - head_result_orig_mean_ablated) / final_scale) @ model.W_U
    resid_post_final_cspa = resid_post_final + (head_result_cspa - head_result_orig) # [batch seq d_model]
    logits_cspa = (resid_post_final_cspa / final_scale) @ model.W_U + model.b_U
    loss_cspa = model.loss_fn(logits_cspa, toks, per_token=True)

    cspa_results = {
        "loss": loss,
        "loss_cspa": loss_cspa,
        "loss_ablated": loss_mean_ablated,
        # "dla": dla_cspa,
        # "logits": logits_cspa,
        # "logits_orig": logits,
        # "logits_ablated": logits_mean_ablated,
        "kl_div_ablated_to_orig": kl_div(logits, logits_mean_ablated),
        "kl_div_cspa_to_orig": kl_div(logits, logits_cspa),
    }
    if return_dla: 
        cspa_results["dla"] = dla_cspa
    if return_logits:
        cspa_results["logits_cspa"] = logits_cspa
        cspa_results["logits_orig"] = logits
        cspa_results["logits_ablated"] = logits_mean_ablated

    return cspa_results, top_K_and_Ksem_per_dest_token, logit_lens_for_top_K_Ksem, semantically_similar_toks, time_for_sstar



def get_cspa_results_batched(
    model: HookedTransformer,
    toks: Int[Tensor, "batch seq"],
    max_batch_size: int,
    negative_head: Tuple[int, int],
    interventions: List[str],
    K_unembeddings: Optional[Union[int, float]] = None,
    K_semantic: int = 10,
    only_keep_negative_components: bool = False,
    semantic_dict: dict = {},
    result_mean: Optional[Dict[Tuple[int, int], Float[Tensor, "seq_plus d_model"]]] = None,
    use_cuda: bool = False,
    verbose: bool = False,
    compute_s_sstar_dict: bool = False,
    return_dla: bool = False,
    return_logits: bool = False,
    keep_self_attn: bool = True,
) -> Dict[str, Float[Tensor, "batch seq-1"]]:
    '''
    Gets results from CSPA, by splitting the tokens along batch dimension and running it several 
    times. This allows me to use batch sizes of 1000+ without getting CUDA errors.

    See the `get_cspa_results` docstring for more info.
    '''
    batch_size, seq_len = toks.shape
    chunks = toks.shape[0] // max_batch_size

    device = t.device("cuda" if use_cuda else "cpu")
    result_mean = {k: v.to(device) for k, v in result_mean.items()}

    orig_model_device = str(next(iter(model.parameters())).device)
    orig_toks_device = str(toks.device)
    target_device = "cuda" if use_cuda else "cpu"
    if not devices_are_equal(orig_model_device, target_device):
        model = model.to(target_device)
    if not devices_are_equal(orig_toks_device, target_device):
        toks = toks.to(target_device)

    CSPA_RESULTS = {}
    TOP_K_AND_KSEM_PER_DEST_TOKEN = t.empty((0, 4), dtype=t.long, device=target_device)
    LOGIT_LENS_FOR_TOP_K_KSEM = t.empty((0,), dtype=t.float, device=target_device)
    SEMANTICALLY_SIMILAR_TOKS = t.empty((0, seq_len, K_semantic), dtype=t.long, device=target_device)


    for i, _toks in enumerate(t.chunk(toks, chunks=chunks)):
        if verbose and i == 0:
            bar = tqdm(total=chunks, desc=f"Batch {i+1}/{chunks}, shape {_toks.shape}")
        
        # Get new results
        t_get = time.time()
        cspa_results, top_K_and_Ksem_per_dest_token, logit_lens_for_top_K_Ksem, semantically_similar_toks, t_sstar = get_cspa_results(
            model = model,
            toks = _toks,
            negative_head = negative_head,
            interventions = interventions,
            K_unembeddings = K_unembeddings,
            K_semantic = K_semantic,
            only_keep_negative_components = only_keep_negative_components,
            semantic_dict = semantic_dict,
            result_mean = result_mean,
            keep_self_attn = keep_self_attn,
            return_dla = return_dla,
            return_logits = return_logits,
        )
        t_get = time.time() - t_get

        # Add them to all accumulated results
        t_agg = time.time()
        CSPA_RESULTS = concat_dicts(CSPA_RESULTS, cspa_results)
        TOP_K_AND_KSEM_PER_DEST_TOKEN = t.cat([TOP_K_AND_KSEM_PER_DEST_TOKEN, top_K_and_Ksem_per_dest_token], dim=0)
        LOGIT_LENS_FOR_TOP_K_KSEM = t.cat([LOGIT_LENS_FOR_TOP_K_KSEM, logit_lens_for_top_K_Ksem], dim=0)
        SEMANTICALLY_SIMILAR_TOKS = t.cat([SEMANTICALLY_SIMILAR_TOKS, semantically_similar_toks], dim=0)
        del cspa_results, top_K_and_Ksem_per_dest_token, logit_lens_for_top_K_Ksem, semantically_similar_toks
        t.cuda.empty_cache()
        t_agg = time.time() - t_agg

        if verbose:
            bar.update()
            bar.set_description(f"Batch {i+1}/{chunks}, shape = {tuple(_toks.shape)}, times = [get = {t_get-t_sstar:.2f}, s* = {t_sstar:.2f}, aggregate = {t_agg:.2f}]")

    if compute_s_sstar_dict:
        if verbose: print("Converting top K and Ksem to dict ...", end="\r"); t0 = time.time()
        S_SSTAR_PAIRS = convert_top_K_and_Ksem_to_dict(
            top_K_and_Ksem_per_dest_token = TOP_K_AND_KSEM_PER_DEST_TOKEN,
            logit_lens_for_top_K_Ksem = LOGIT_LENS_FOR_TOP_K_KSEM,
            toks = toks,
            semantically_similar_toks = SEMANTICALLY_SIMILAR_TOKS,
            model = model,
        )
        if verbose: print(f"Converting top K and Ksem to dict ... {time.time()-t0:.2f}")
        to_return = (CSPA_RESULTS, S_SSTAR_PAIRS)
    else:
        to_return = CSPA_RESULTS

    if not devices_are_equal(orig_model_device, target_device):
        model = model.to(orig_model_device)
    if not devices_are_equal(orig_toks_device, target_device):
        toks = toks.to(orig_toks_device)
    
    return to_return



def get_top_predicted_semantically_similar_tokens(
    toks: Int[Tensor, "batch seq"],
    resid_pre: Float[Tensor, "batch seqK d_model"],
    semantically_similar_toks: Int[Tensor, "batch seq K_semantic"],
    K_unembeddings: Optional[Union[int, float]],
    function_toks: Int[Tensor, "tok"],
    model: HookedTransformer,
    final_scale: Optional[Float[Tensor, "batch seqQ 1"]] = None,
    keep_self_attn: bool = True,
) -> Tuple[
    Float[Tensor, "batch seqQ seqK d_model K_semantic"], 
    Int[Tensor, "n 4"], 
    Float[Tensor, "n"], 
    Bool[Tensor, "batch seqQ seqK K_semantic"],
]:
    '''
    Arguments:

        toks: [batch seq]
            The source tokens.
        
        resid_pre: [batch seqK d_model]
            The residual stream before the head, which we'll use to figure out which tokens in `toks`
            are predicted with highest probability.

        semantically_similar_toks: [batch seqK K_semantic]
            Semantically similar tokens for each source token.
        
        function_toks: [tok]
            1D array of function tokens (we filter for pairs (s, s*) where s is not a function token).

    Returns:

        semantically_similar_unembeddings: [batch seqQ seqK d_model K_semantic]
            The unembeddings of the semantically similar tokens, with all the vectors except the
            ones we're actually using set to zero.
        
        top_K_and_Ksem_per_dest_token: [n 4]
            The i-th row is the (b, sQ, sK, K_s) indices of the i-th top predicted semantically similar
            token.

        logit_lens_for_top_K_and_Ksem_per_dest_token: [n]
            The i-th element is the logits for the corresponding prediction in the previous tensor. This
            is used to make sure that the eventual s_star dictionary we create is sorted correctly.
        
        mask: [batch seqQ seqK K_semantic]
            The mask which we'll be applying once we project the unembeddings.
    '''
    semantically_similar_unembeddings = model.W_U.T[semantically_similar_toks].transpose(-1, -2) # [batch seqK d_model K_semantic]
    batch_size, seq_len = toks.shape
    
    # If K_unembeddings is None, then we have no restrictions, and we use all the top K_semantic toks for each seqK
    if K_unembeddings is None:
        semantically_similar_unembeddings = einops.repeat(
            semantically_similar_unembeddings,
            "batch seqK d_model K_semantic -> batch seqQ seqK d_model K_semantic",
            seqQ = seq_len,
        )

    # Else, we filter down the sem similar unembeddings to only those that are predicted
    else:
        # Get logit lens from current value of residual stream
        # logit_lens[b, sQ, sK, K_s] = prediction at destination token (b, sQ), for the K_s'th semantically similar token to source token (b, sK)
        resid_pre_scaled = resid_pre if (final_scale is None) else resid_pre / final_scale
        logit_lens = einops.einsum(
            resid_pre_scaled, semantically_similar_unembeddings,
            "batch seqQ d_model, batch seqK d_model K_semantic -> batch seqQ seqK K_semantic",
        )

        # * MASK: make sure function words are never the source token (because we've observed that the QK circuit has managed to learn this)
        is_fn_word = (toks[:, :, None] == function_toks).any(dim=-1) # [batch seqK]
        logit_lens = t.where(einops.repeat(is_fn_word, "batch seqK -> batch 1 seqK 1"), -1e9, logit_lens)
        # * MASK: apply causal mask (this might be strict if keep_self_attn is False)
        seqQ_idx = einops.repeat(t.arange(seq_len), "seqQ -> 1 seqQ 1 1").to(logit_lens.device)
        seqK_idx = einops.repeat(t.arange(seq_len), "seqK -> 1 1 seqK 1").to(logit_lens.device)
        causal_mask = (seqQ_idx < seqK_idx) if keep_self_attn else (seqQ_idx <= seqK_idx)
        logit_lens = t.where(causal_mask, -1e9, logit_lens)
        # * MASK: each source token should only be counted at its first instance
        # Note, we apply this mask to get our topk values (so no repetitions), but we don't want to apply it when we're choosing pairs to keep
        first_occurrence_mask = einops.repeat(first_occurrence_2d(toks), "batch seqK -> batch 1 seqK 1")
        logit_lens_for_topk = t.where(~first_occurrence_mask, -1e9, logit_lens)

        # Get the top predicted src-semantic-neighbours s* for each destination token d
        # (this might be different for each destination posn, if K_unembeddings is a float)
        if isinstance(K_unembeddings, int):
            top_K_and_Ksem_per_dest_token_values = logit_lens_for_topk.flatten(-2, -1).topk(K_unembeddings, dim=-1).values[..., [[-1]]] # [batch seqQ 1 1]
        else:
            top_K_and_Ksem_per_dest_token_values = t.full((batch_size, seq_len, 1, 1), fill_value=-float("inf"), device=logit_lens.device)
            for dest_posn in range(seq_len):
                K_u_dest = math.ceil(K_unembeddings * (dest_posn + 1))
                top_K_and_Ksem_per_dest_token_values[:, dest_posn] = logit_lens_for_topk[:, dest_posn].flatten(-2, -1).topk(K_u_dest, dim=-1).values[..., [[-1]]]
        
        # Later we'll be computing the list of (s, s*) for analysis afterwards
        top_K_and_Ksem_mask = (logit_lens + 1e-6 >= top_K_and_Ksem_per_dest_token_values) # [batch seqQ seqK K_s]
        top_K_and_Ksem_per_dest_token = t.nonzero(top_K_and_Ksem_mask) # [n 4 = (batch, seqQ, seqK, K_s)], n >= batch * seqQ * K_u (more if we're double-counting source tokens)
        b, sQ, sK, K_s = top_K_and_Ksem_per_dest_token.T.tolist()
        logit_lens_for_top_K_and_Ksem_per_dest_token = logit_lens[b, sQ, sK, K_s]

        # Use this boolean mask to set some of the unembedding vectors to zero
        unembeddings = einops.repeat(semantically_similar_unembeddings, "batch seqK d_model K_semantic -> batch 1 seqK d_model K_semantic")
        top_K_and_Ksem_mask_repeated = einops.repeat(top_K_and_Ksem_mask, "batch seqQ seqK K_semantic -> batch seqQ seqK 1 K_semantic")
        semantically_similar_unembeddings = unembeddings * top_K_and_Ksem_mask_repeated.float()

    return semantically_similar_unembeddings, top_K_and_Ksem_per_dest_token, logit_lens_for_top_K_and_Ksem_per_dest_token, top_K_and_Ksem_mask




def convert_top_K_and_Ksem_to_dict(
    top_K_and_Ksem_per_dest_token: Int[Tensor, "n 4"], # each row is (batch, seqQ, seqK, K_s)
    logit_lens_for_top_K_Ksem: Float[Tensor, "n"], # each element is logits (we keep it for sorting purposes)
    toks: Int[Tensor, "batch seq"],
    semantically_similar_toks: Int[Tensor, "batch seq s_K"],
    model: HookedTransformer,
):
    '''
    Making this function because it's more efficient to do this all at once (model.to_str_tokens is slow!).
    '''
    s_sstar_pairs = defaultdict(list)

    # Get all batch indices, dest pos indices, src pos indices, and semantically similar indices
    b, sQ, sK, K_s = top_K_and_Ksem_per_dest_token.T.tolist()

    # Get the string representations of (s, s*) that we'll be using in the html viz (s comes with its position)
    s_str_toks = model.to_str_tokens(toks[b, sK], prepend_bos=False)
    s_repr = [f"[{s_posn}] {repr(s_str_tok)}" for s_posn, s_str_tok in zip(sK, s_str_toks)]
    sstar_str_toks = model.to_str_tokens(semantically_similar_toks[b, sK, K_s], prepend_bos=False)
    sstar_repr = [repr(sstar_str_tok) for sstar_str_tok in sstar_str_toks]

    # Add them all to our dict
    for _b, _sQ, _s, _sstar, _LL in zip(b, sQ, s_repr, sstar_repr, logit_lens_for_top_K_Ksem):
        s_sstar_pairs[(_b, _sQ)].append((_LL, _s, _sstar))
    
    # Make sure we rank order the entries in each dictionary by how much they're being predicted
    for (b, sQ), s_star_list in s_sstar_pairs.items():
        s_sstar_pairs[(b, sQ)] = sorted(s_star_list, key = lambda x: x[0], reverse=True)
    
    return s_sstar_pairs



# ? Removed this because it didn't work well
# # Add hooks to project the value input along the source tokens' effective embeddings
# def hook_fn_project_v(
#     v_input: Float[Tensor, "batch seqK head d_model"],
#     hook: HookPoint,
# ):
#     '''
#     Projects the value input onto the effective embedding for the source token.
#     '''
#     v_input_head = v_input[:, :, HEAD]
#     v_input_head_mean = einops.reduce(v_input_head, "batch seqK d_model -> seqK d_model")

#     v_input[:, :, HEAD] = project(
#         vectors = v_input_head - v_input_head_mean,
#         proj_directions = effective_embeddings,
#     ) + v_input_head_mean
    
#     return v_input
# if "v" in interventions:
#     model.add_hook(hook_name_v_input, hook_fn_project_v)
#     model.add_hook(hook_name_scale, partial(hook_fn_freeze_scale, frozen_scale=scale, component="v"))


# ? This code doesn't work super well; projecting queries is meh.
# def hook_fn_project_q_via_pattern(
#     pattern: Float[Tensor, "batch head seqQ seqK"],
#     hook: HookPoint,
#     resid_pre: Float[Tensor, "batch seqK d_model"],
# ):
#     '''
#     Projects the query input onto the effective embedding for the source token.
#     '''
#     # Apply LN scale
#     scale = resid_pre.std(dim=-1, keepdim=True)
#     # Get the thing we'll be projecting along the sematic unembeddings for each src token
#     query_input = einops.repeat(
#         resid_pre,
#         "batch seqQ d_model -> batch seqQ seqK d_model",
#         seqK = seq_len,
#     )
#     query_input_mean = query_input.mean(dim=0, keepdim=True)
#     semantically_similar_unembeddings_repeated = einops.repeat(
#         semantically_similar_unembeddings,
#         "batch seqK d_model K_semantic -> batch seqQ seqK d_model K_semantic",
#         seqQ = seq_len,
#     )
#     # Project onto the appropriate semantically similar tokens
#     query_input_projected = project(
#         vectors = query_input - query_input_mean,
#         proj_directions = semantically_similar_unembeddings_repeated,
#     ) # [batch seqQ seqK d_model] #  + query_input_mean
#     # Rescale by norm of source tokens (this is how we control for function words!)
#     # TODO - this is definitely not the optimal way to do it, not sure what is optimal yet
#     query_input_projected = query_input_projected * einops.repeat(
#         unembeddings.norm(dim=-1), #  / unembeddings.norm(dim=-1).max(),
#         "batch seqK -> batch seqQ seqK d_model",
#         seqQ = seq_len,
#         d_model = query_input_projected.shape[-1],
#     )
#     # Add mean back in
#     query_input_projected = query_input_projected + query_input_mean

#     # Now we can get keys and queries, and calculate our new attention scores (scaled and masked!)
#     keys = (resid_pre / scale) @ model.W_K[LAYER, HEAD] + model.b_K[LAYER, HEAD]
#     queries = (query_input_projected / scale.unsqueeze(-1)) @ model.W_Q[LAYER, HEAD] + model.b_Q[LAYER, HEAD]
#     new_attn_scores = einops.einsum(
#         queries, keys,
#         "batch seqQ seqK d_head, batch seqK d_head -> batch seqQ seqK",
#     ) / (model.cfg.d_head ** 0.5)
#     new_attn_scores.masked_fill_(t.triu(t.ones_like(new_attn_scores), diagonal=1).bool(), -float("inf"))
#     new_pattern = new_attn_scores.softmax(dim=-1)

#     # We want to make sure that attention prob to the zeroth token is same as before (as a baseline)
#     # e.g. if original attn to 0 was very high, we'll be scaling down the new not-to-0 attn probs
#     new_pattern[:, 1:, 1:] *= (1 - pattern[:, HEAD, 1:, [0]]) / (1 - new_pattern[:, 1:, [0]])
#     new_pattern[:, :, 0] = pattern[:, HEAD, :, 0]
#     # t.testing.assert_close(new_pattern.sum(dim=-1), t.ones_like(new_pattern.sum(dim=-1)))

#     hook.ctx["info"] = (pattern[:, HEAD].clone(), new_attn_scores.clone(), new_pattern.clone())

#     pattern[:, HEAD] = new_pattern
#     return pattern

