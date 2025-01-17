# %% [markdown] [4]:

"""
Runs an experiment where we see that unembedding for *one* token is a decent percentage of the usage of 
direct effect of NMS
"""

from transformer_lens.cautils.notebook import *
from transformer_lens.rs.callum.keys_fixed import project, get_effective_embedding_2
from transformer_lens.rs.arthurs_notebooks.arthurs_utils import *
import argparse

#%%

model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=False,
)
model.set_use_attn_result(True)

# %%

MAX_SEQ_LEN = 512
BATCH_SIZE = 30
batched_tokens, targets = get_filtered_webtext(model, batch_size=BATCH_SIZE, seed=1717, device="cuda", max_seq_len=MAX_SEQ_LEN)
effective_embeddings = get_effective_embedding_2(model)

# %%

# Find the top 5% of things by importance
# Do this crap
# See change in loss

NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX = NEG_HEADS[model.cfg.model_name]
NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX = 10, 7
# for NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX in [(10, 0), (10, 7), (9, 9), (11, 10)] + list(itertools.product(range(11, -1, -1), range(12))):

END_STATE_HOOK = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
# warnings.warn("Changed to scores for a diff comparison")
# attention_pattern_hook_name = get_act_name("attn_scores", NEGATIVE_LAYER_IDX)
attention_pattern_hook_name = get_act_name("pattern", NEGATIVE_LAYER_IDX)
names_filter1 = (
    lambda name: name == END_STATE_HOOK
    or name==get_act_name("resid_pre", 1)
    or name==f"blocks.{NEGATIVE_LAYER_IDX}.hook_resid_pre"
    or name==f"blocks.{NEGATIVE_LAYER_IDX}.attn.hook_result"
    or name==attention_pattern_hook_name
)
logits, cache = model.run_with_cache(
    batched_tokens,
    names_filter=names_filter1,
)
gc.collect()
torch.cuda.empty_cache()

# %%

original_end_state = cache[END_STATE_HOOK]

batched_tokens_loss = get_metric_from_end_state(
    model=model,
    end_state=original_end_state,
    targets=targets,
)

#%%

head_output = cache[get_act_name("result", NEGATIVE_LAYER_IDX)][:, :, NEGATIVE_HEAD_IDX]
assert head_output.shape == (BATCH_SIZE, MAX_SEQ_LEN, model.cfg.d_model)

#%%

if ipython is not None:
    unembed = einops.einsum(
        head_output, 
        model.W_U,
        "b s d_model, d_model d_vocab -> b s d_vocab",
    )

#%% 

if ipython is not None:
    the_topk = torch.topk(
        -unembed.cpu(),
        k=10,
        dim=-1,
    ).indices

#%%

mean_head_output = einops.reduce(head_output, "b s d -> d", reduction="mean")

#%%

mean_ablated_end_states = cache[get_act_name("resid_post", model.cfg.n_layers-1)] - head_output + einops.repeat(mean_head_output, "d -> b s d", b=BATCH_SIZE, s=MAX_SEQ_LEN)
mean_ablated_loss = get_metric_from_end_state(
    model=model,
    end_state=mean_ablated_end_states,
    targets=targets,
)

# %%

max_importance_examples = sorted(
    [
        (
            batch_idx,
            seq_idx,
            (mean_ablated_loss-batched_tokens_loss)[batch_idx, seq_idx].item(),
        )
        for batch_idx, seq_idx in itertools.product(
            range(BATCH_SIZE), range(MAX_SEQ_LEN)
        )
    ],
    key=lambda x: x[2],
    reverse=True,
)

# %%

# Get the top 5% of things by importance
all_top_5_percent = max_importance_examples[: len(max_importance_examples)//20]

np.random.seed(799)
# warnings.warn("No shuffle!!!")
np.random.shuffle(all_top_5_percent)
top_5_percent = all_top_5_percent[: BATCH_SIZE]

top5p_batch_indices = [x[0] for x in top_5_percent]
top5p_seq_indices = [x[1] for x in top_5_percent]

#%%

if ipython is not None:
    top5p_topks = the_topk[top5p_batch_indices, top5p_seq_indices]

#%% [markdown]
# <h2> Making some helpful datasets </h2>

my_dict = {}
works = 0
if ipython is not None:
    for idx in range(len(top5p_batch_indices)):
        current_tokens = batched_tokens[top5p_batch_indices[idx], :top5p_seq_indices[idx]+1].tolist()

        top_negs = top5p_topks[idx].tolist()

        is_in_top_negs = {
            i: int(top_negs[i] in current_tokens) for i in range(3)
        }
        if sum(list(is_in_top_negs.values()))==1:
            the_tokens = current_tokens + [top_negs[i] for i in range(3) if is_in_top_negs[i]]
            assert len(the_tokens) == len(current_tokens) + 1

            while the_tokens.count(the_tokens[-1]) > 2:
                the_tokens = the_tokens[1:]

            # # Uncomment if things are pretty cursed for doing mean ablations if the copied word is super near the end
            # if the_tokens.index(the_tokens[-1]) >= len(the_tokens) - 3:
            #     print("FAIL")
            #     continue

            print("Success! Here's the prompt where we added the copy suppressed thing as the last token:")
            print(model.to_string(the_tokens))
            key_token = model.to_string(the_tokens[-1:])
            my_dict[key_token] = model.to_string(the_tokens)
            
        else:
            print("FAIL")

        print("-"*50)

#%%

with open("/code/TransformerLens/transformer_lens/rs/arthur/json_data/and_even_more_top5p_examples.json", "w") as f:
    json.dump(my_dict, f)

# And things are done!

#%%

top5p_tokens = batched_tokens[top5p_batch_indices]
top5p_targets = torch.LongTensor([targets[top5p_batch_idx, top5p_seq_idx] for top5p_batch_idx, top5p_seq_idx in zip(top5p_batch_indices, top5p_seq_indices)])

#%%

top5p_losses = batched_tokens_loss[top5p_batch_indices, top5p_seq_indices]

# %%

# Do the key-side thing where we project onto W_U
# selected_unembeddings = cache[get_act_name("resid_pre", NEGATIVE_LAYER_IDX)][torch.tensor(top5p_batch_indices), torch.tensor(top5p_seq_indices)]

keyside_projections = t.zeros((BATCH_SIZE, MAX_SEQ_LEN, model.cfg.d_model))
keyside_orthogonals = t.zeros((BATCH_SIZE, MAX_SEQ_LEN, model.cfg.d_model))

for batch_idx, seq_idx in tqdm(list(itertools.product(range(BATCH_SIZE), range(MAX_SEQ_LEN)))):
    keyside_vector, keyside_orthogonal = project(
        normalize(cache[get_act_name("resid_pre", NEGATIVE_LAYER_IDX)][batch_idx, seq_idx]) * np.sqrt(model.cfg.d_model), # simulate LN
        cache[get_act_name("resid_pre", 1)][batch_idx, seq_idx],
    )
    keyside_projections[batch_idx, seq_idx] = keyside_vector
    keyside_orthogonals[batch_idx, seq_idx] = keyside_orthogonal

warnings.warn("Reversing BOS directions...")
keyside_projections[:, 0] *= - 1.0

#%%

new_k_input = t.zeros((BATCH_SIZE, MAX_SEQ_LEN, model.cfg.d_model))

np.random.seed(433)
for batch_batch_idx, batch_idx in enumerate(top5p_batch_indices):

    # warnings.warn("Writing as literally the unembed...")

    # new_k_input[batch_batch_idx] = torch.stack([
    #     effective_embeddings["W_E (including MLPs)"][batched_tokens[batch_idx, seq_idx]] for seq_idx in range(MAX_SEQ_LEN)
    # ])

    rand_batch_indices = [np.random.randint(0, BATCH_SIZE) for _ in range(MAX_SEQ_LEN)]
    rand_seq_indices = [np.random.randint(0, MAX_SEQ_LEN) for _ in range(MAX_SEQ_LEN)]

    new_k_input[batch_batch_idx] = torch.stack([
        keyside_projections[batch_idx, seq_idx] + 0.0*keyside_orthogonals[rand_batch_idx][rand_seq_idx] for seq_idx, rand_batch_idx, rand_seq_idx in zip(range(MAX_SEQ_LEN), rand_batch_indices, rand_seq_indices, strict=True)
    ])

#%%

model.set_use_split_qkv_input(True)
model.set_use_split_qkv_normalized_input(True)
model.reset_hooks()
model.add_hook(
    get_act_name("k_normalized_input", NEGATIVE_LAYER_IDX),
    partial(set_to_value, head_idx=NEGATIVE_HEAD_IDX, new_value=new_k_input.to("cuda")),
    level=1,
)
# model.add_hook(
#     get_act_name("q_input", NEGATIVE_LAYER_IDX),
#     partial(set_to_value, head_idx=NEGATIVE_HEAD_IDX, seq_indices = top5p_seq_indices, new_value=queryside_vectors.to("cuda")),
#     level=1,
# )
model.to("cuda")
logits, top_5p_cache = model.run_with_cache(
    top5p_tokens.to("cuda"),
    names_filter = lambda name: name in [get_act_name("result", NEGATIVE_LAYER_IDX), get_act_name("resid_post", model.cfg.n_layers-1), attention_pattern_hook_name],
    device="cuda:0"
)
model.reset_hooks()
model.to("cuda:0")
new_head_out = top_5p_cache[get_act_name("result", NEGATIVE_LAYER_IDX)][torch.arange(len(top5p_tokens)), top5p_seq_indices, NEGATIVE_HEAD_IDX].unsqueeze(0)

relevant_head_outs = []
for batch_idx, seq_idx in zip(top5p_batch_indices, top5p_seq_indices):
    relevant_head_outs.append(
        head_output[batch_idx, seq_idx]
    )
relevant_head_outs = t.stack(relevant_head_outs).unsqueeze(0)

#%%

# let's compare the patterns

my_idx = 0
initial_pattern = cache[attention_pattern_hook_name][top5p_batch_indices[my_idx], NEGATIVE_HEAD_IDX, top5p_seq_indices[my_idx], :top5p_seq_indices[my_idx]+1]
current_pattern = top_5p_cache[attention_pattern_hook_name][my_idx, NEGATIVE_HEAD_IDX, top5p_seq_indices[my_idx], :top5p_seq_indices[my_idx]+1]
# for pattern in [initial_pattern, current_pattern]:
#     px.bar(
#         x = [str(x) for x in enumerate(model.to_str_tokens(top5p_tokens[0])[:len(initial_pattern)])],
#         y = pattern.cpu().numpy(),
#     ).show()

#%%

if ipython is not None and False:
    new_head_unembed = einops.einsum(
        new_head_out[0],
        model.W_U,
        "seq_len dim, dim vocab -> seq_len vocab",
    )
    new_head_topk = t.topk(-new_head_unembed, k=10, dim=-1).indices

#%%

top_5p_end_state = original_end_state[top5p_batch_indices, top5p_seq_indices].unsqueeze(0)
assert top_5p_end_state.shape == relevant_head_outs.shape == new_head_out.shape, (top_5p_end_state.shape, relevant_head_outs.shape, new_head_out.shape)

top_5p_end_state -= relevant_head_outs
top_5p_end_state += new_head_out

#%%

loss = get_metric_from_end_state(
    end_state=top_5p_end_state,
    model=model,
    logits=None,
    targets=top5p_targets.unsqueeze(0),
)

#%%

mean_ablation_loss = get_metric_from_end_state(
    end_state=top_5p_end_state-new_head_out+mean_head_output,
    logits=None,
    targets=top5p_targets.unsqueeze(0),
    model=model,
)
torch.testing.assert_close(
    mean_ablation_loss[0],
    mean_ablated_loss[top5p_batch_indices, top5p_seq_indices],
    rtol=1e-3,
    atol=1e-3,
)
the_mean_ablated_loss = [mean_ablated_loss[batch_idx, seq_idx].item() for batch_idx, seq_idx in zip(top5p_batch_indices, top5p_seq_indices)]

#%%

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x = top5p_losses.cpu().tolist(),
        y = loss.cpu().squeeze(0).tolist(),
        mode = "markers",
        text = [f"Batch {batch_idx}, Seq {seq_idx}" for batch_idx, seq_idx in zip(top5p_batch_indices, top5p_seq_indices)],
        name = "Projected Keys Losses",
    )
)
fig.add_trace(
    go.Scatter(
        x = top5p_losses.cpu().tolist(),
        y = torch.tensor(the_mean_ablated_loss),
        mode = "markers",
        text = [f"Batch {batch_idx}, Seq {seq_idx}" for batch_idx, seq_idx in zip(top5p_batch_indices, top5p_seq_indices)],
        name = "Mean Ablated Loss",
    )
)
# add y=x line
fig.add_trace(
    go.Scatter(
        x = [min(top5p_losses).item(), max(top5p_losses).item()],
        y = [min(top5p_losses).item(), max(top5p_losses).item()],
        mode = "lines",
        name = "y=x",
    )
)
fig.show()

# %%

change_in_loss = (loss.cpu() - top5p_losses.cpu()).tolist()[0]
change_in_loss_mean = sum(change_in_loss)/len(change_in_loss)
top5pdata = top5p_losses.cpu().tolist()
lossdata = loss.cpu().tolist()

MY_FNAME = "../arthur/json_data/try_and_make_loss_recov_work.json"
try:
    # # read the existing data
    with open(MY_FNAME, "r") as f:
        cur_json = json.load(f)

except FileNotFoundError:
    cur_json = {}

mean_ablation_bad = (mean_ablated_loss[top5p_batch_indices, top5p_seq_indices] - top5p_losses).tolist()

# update the data
cur_json[str((NEGATIVE_LAYER_IDX, NEGATIVE_HEAD_IDX))] = {
    "layer_idx": NEGATIVE_LAYER_IDX,
    "head_idx": NEGATIVE_HEAD_IDX,
    "top5p_losses": top5pdata,
    "losses": lossdata,
    "change_in_loss": change_in_loss,
    "change_in_loss_mean": change_in_loss_mean,
    "mean_ablated_loss": the_mean_ablated_loss,
    "time": ctime()+"_remember_sometimes_this_is_an_hour_too_early",
    "how_bad_is_mean_ablation": mean_ablation_bad,
    "how_bad_is_mean_ablation_mean": sum(mean_ablation_bad)/len(mean_ablation_bad),
}    

# write the data
# write the updated data (don't overwrite!)
with open(MY_FNAME, "w") as f:
    f.write(json.dumps(cur_json, indent=4))

# %%

# now also try the q projection
# %%

# also import cautils
import json 
with open (MY_FNAME, "r") as f:
    cur_json = json.load(f)

# %%

text = [f"Layer {x['layer_idx']}, Head {x['head_idx']}" for x in cur_json.values()]

fig = px.bar(
    x = text,
    # x=[x["change_in_loss_mean"] for x in cur_json.values()],
    y=[x["how_bad_is_mean_ablation_mean"]/x["change_in_loss_mean"] for x in cur_json.values()],
    # labels={
    #     "x": "How much loss does the key lock get?",
    #     "y": "How bad is mean ablation?",
    # },
    title="Ratio (mean increase in loss by mean ablating) / (mean increase in loss by projecting keys to W_E (including MLPs) and projecting the query onto the unembedding directions for all tokens in context). Datapoints sampled from top 5% of direct effect datapoints per head",
)            

# fig.add_shape(
#     type="line",
#     x0=-0.1,
#     y0=-0.1,
#     x1=5.1,
#     y1=5.1,
# )

fig.show()

# %%

fig = go.Figure()

for layer, head in [(10, 7), (11, 10), (9, 9), (11, 6), (11, 3)]:
    fig.add_trace(
        go.Scatter(
            x = cur_json[str((layer, head))]["top5p_losses"],
            y = cur_json[str((layer, head))]["change_in_loss"],
            mode = "markers",
            name = f"{layer}.{head}",
        )
    )

fig.show()
# %%
