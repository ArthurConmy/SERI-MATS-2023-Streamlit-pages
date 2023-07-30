#%%

"""
NOTE: this is cribbed from explore_prompts.ipynb, Arthur finds it much easier to use + debug py files
"""


# # Explore Prompts
# 
# This is the notebook I use to test out the functions in this directory, and generate the plots in the Streamlit page.

# ## Setup

# In[4]:

from transformer_lens.cautils.notebook import *
import gzip

from transformer_lens.rs.callum2.explore_prompts.model_results_3 import (
    get_model_results,
    HeadResults,
    LayerResults,
    DictOfHeadResults,
    ModelResults,
    first_occurrence,
    project,
    model_fwd_pass_from_resid_pre,
)
from transformer_lens.rs.callum2.explore_prompts.explore_prompts_utils import (
    create_title_and_subtitles,
    parse_str,
    parse_str_tok_for_printing,
    parse_str_toks_for_printing,
    topk_of_Nd_tensor,
    ST_HTML_PATH,
)
from transformer_lens.rs.callum2.explore_prompts.copy_suppression_classification import (
    generate_scatter,
    generate_hist,
    plot_logit_lens,
    plot_full_matrix_histogram,
)

from transformer_lens.rs.arthurs_notebooks.arthur_utils import get_metric_from_end_state, dot_with_query, set_to_value

from transformer_lens.rs.callum2.keys_fixed import project as original_project

clear_output()

# In[5]:


def get_effective_embedding_2(model: HookedTransformer) -> Float[Tensor, "d_vocab d_model"]:

    W_E = model.W_E.clone()
    W_U = model.W_U.clone()
    # t.testing.assert_close(W_E[:10, :10], W_U[:10, :10].T)  NOT TRUE, because of the center unembed part!

    resid = W_E.unsqueeze(0)

    for i in range(10):
        pre_attention = model.blocks[i].ln1(resid)
        attn_out = einops.einsum(
            pre_attention, 
            model.W_V[i],
            model.W_O[i],
            "b s d_model, num_heads d_model d_head, num_heads d_head d_model_out -> b s d_model_out",
        )
        resid_mid = attn_out + resid
        normalized_resid_mid = model.blocks[i].ln2(resid_mid)
        mlp_out = model.blocks[i].mlp(normalized_resid_mid)
        resid = resid_mid + mlp_out

        if i == 0:
            W_EE = mlp_out.squeeze()
            W_EE_full = resid.squeeze()

    W_EE_stacked = resid.squeeze()

    t.cuda.empty_cache()

    return {
        "W_E (no MLPs)": W_E,
        "W_U": W_U.T,
        # "W_E (raw, no MLPs)": W_E,
        "W_E (including MLPs)": W_EE_full,
        "W_E (only MLPs)": W_EE,
        "W_E (including MLPs, first 9 layers)": W_EE_stacked
    }


# In[6]:


model = HookedTransformer.from_pretrained(
    "gpt2-small",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    device="cuda"
    # fold value bizas?
)
model.set_use_split_qkv_input(False)
model.set_use_attn_result(True)

clear_output()



# In[7]:


W_EE_dict = get_effective_embedding_2(model)


# ## Getting model results

# In[8]:


BATCH_SIZE = 20 # Smaller on Arthur's machine
SEQ_LEN = 100 # 70 for viz (no more, because attn)
TESTING = False

NEGATIVE_HEADS = [(10, 7)]

def process_webtext(
    seed: int = 6,
    batch_size: int = BATCH_SIZE,
    indices: Optional[List[int]] = None,
    seq_len: int = SEQ_LEN,
    verbose: bool = False,
):
    DATA_STR = get_webtext(seed=seed)
    if indices is None:
        DATA_STR = DATA_STR[:batch_size]
    else:
        DATA_STR = [DATA_STR[i] for i in indices]
    DATA_STR = [parse_str(s) for s in DATA_STR]

    DATA_TOKS = model.to_tokens(DATA_STR)
    DATA_STR_TOKS = model.to_str_tokens(DATA_STR)

    if seq_len < 1024:
        DATA_TOKS = DATA_TOKS[:, :seq_len]
        DATA_STR_TOKS = [str_toks[:seq_len] for str_toks in DATA_STR_TOKS]

    DATA_STR_TOKS_PARSED = list(map(parse_str_toks_for_printing, DATA_STR_TOKS))

    clear_output()
    if verbose:
        print(f"Shape = {DATA_TOKS.shape}\n")
        print("First prompt:\n" + "".join(DATA_STR_TOKS[0]))

    return DATA_TOKS, DATA_STR_TOKS_PARSED


DATA_TOKS, DATA_STR_TOKS_PARSED = process_webtext(verbose=True) # indices=list(range(32, 40))
BATCH_SIZE, SEQ_LEN = DATA_TOKS.shape

NUM_MINIBATCHES = 1 # previouly 3
DO_KEYSIDE_PROJECTIONS = False
DO_QUERYSIDE_PROJECTIONS = True
PROJECT_MODE: Literal["unembeddings", "name_movers"] = "name_movers" # name_movers is Neel's idea

MINIBATCH_SIZE = BATCH_SIZE // NUM_MINIBATCHES
MINIBATCH_DATA_TOKS = [DATA_TOKS[i*MINIBATCH_SIZE:(i+1)*MINIBATCH_SIZE] for i in range(NUM_MINIBATCHES)]
MINIBATCH_DATA_STR_TOKS_PARSED = [DATA_STR_TOKS_PARSED[i*MINIBATCH_SIZE:(i+1)*MINIBATCH_SIZE] for i in range(NUM_MINIBATCHES)]

# In[13]:

K_semantic = 10
K_unembed = 10

ICS_list = []
HTML_list = []

assert NUM_MINIBATCHES == 1, "Deprecating support for several minibatches"

_DATA_TOKS = MINIBATCH_DATA_TOKS[0]
DATA_STR_TOKS_PARSED= MINIBATCH_DATA_STR_TOKS_PARSED[0]
i = 0
#%%

model.to("cpu")
MODEL_RESULTS = get_model_results(
    model,
    toks=_DATA_TOKS.to("cpu"),
    negative_heads=NEGATIVE_HEADS,
    verbose=True,
    K_semantic=K_semantic,
    K_unembed=K_unembed,
    use_cuda=False,
    effective_embedding="W_E (including MLPs)",
    include_qk = True,
)
model=model.to("cuda")

#%%

logits_for_E_sq_QK = MODEL_RESULTS.misc["logits_for_E_sq_QK"] # this has the future stuff screened off
E_sq_QK: Float[torch.Tensor, "batch seq_len K"] = MODEL_RESULTS.E_sq_QK[10, 7]

_logits_for_E_sq = MODEL_RESULTS.misc["logits_for_E_sq"] 
_E_sq = MODEL_RESULTS.E_sq[10, 7]

#%%

"""
The goal is to make a BATCH_SIZE x SEQ_LEN-1 list of losses here 

Let's decompose the goal
1. Firstly reproduce that mean ablating the direct effect of 10.7 gives points that are exclusively on the y=x line DONE
2. Use your get_metric_from_end_state methinks : ) DONE
3. Let the experiments begin DONE

Brainstorm of some ideas for how to QK approximations:

1) Set query to the unembedding component, only?
2) [While doing ??? with BOS? Making its attention score such that the attention is the same?]
Okay we're trading off simplicity of implementation vs something that could actually work
Callum's implementation has too many side conditions, let's do some hacky things fast and look at failures

Okay so the implementation with various topk things is really bad
Let's do 
i) Do Callum's embedding idea
ii) Choose versions of these that are predicted maximally 
iii) Project onto these unembeddings
(still leaves possibility that some orthogonal matters more)

Update: query is still fucked, try Neel quick hack use Layer 9 Name Movers then do some more comlicated things

""" 

#%%

model.reset_hooks()
final_ln_scale_hook_name = "ln_final.hook_scale"
resid_pre_name = get_act_name("resid_pre", 10)
resid_pre1_name = get_act_name("resid_pre", 5)

logits, cache = model.run_with_cache(
    _DATA_TOKS[:, :-1],
    names_filter = lambda name: name in [get_act_name("result", 10), get_act_name("result", 9), get_act_name("resid_post", 11), final_ln_scale_hook_name, resid_pre_name, resid_pre1_name, get_act_name("attn_scores", 10)],
)

pre_state = cache[get_act_name("resid_pre", 10)]
end_state = cache[get_act_name("resid_post", 11)]
head_out = cache[get_act_name("result", 10)][:, :, 7].clone()
scale = cache[final_ln_scale_hook_name]
resid_pre1 = cache[resid_pre1_name]
layer_nine_outs = cache[get_act_name("result", 10)]
layer_nine_name_movers = [0, 6, 7, 9] # unsure if we'll use these...

attn_scores = cache[get_act_name("attn_scores", 10)][:, 7].clone()

del cache
gc.collect()
t.cuda.empty_cache()

#%%

mean_head_output = einops.reduce(
    head_out,
    "b s d_head -> d_head",
    reduction="mean",
)

#%%

head_logit_lens = einops.einsum(
    head_out / head_out.norm(dim=-1, keepdim=True),
    model.W_U,
    "b s d_model, d_model d_vocab -> b s d_vocab",
)

#%%

top_answers = torch.topk(
    head_logit_lens,
    dim=-1,
    k=20, # really we want to do semantic similarity I think?
).indices

#%%

head_loss = get_metric_from_end_state(
    model = model,
    end_state = end_state - head_out + mean_head_output.unsqueeze(0).unsqueeze(0).clone(),
    frozen_ln_scale = scale,
    targets = _DATA_TOKS[:, 1:],
)

#%%

# Compute the output of Head 10.7 manually?

if TESTING: # This is actually fairly slow, a bit of a problem for the 
    manual_head_output = t.zeros(BATCH_SIZE, SEQ_LEN, model.cfg.d_model).cuda()
    for batch_idx, seq_idx in tqdm(list(itertools.product(range(BATCH_SIZE), range(SEQ_LEN-1)))):    
        model.reset_hooks()
        current_attention_scores = dot_with_query(
            unnormalized_keys = pre_state[batch_idx, :seq_idx+1, :],
            unnormalized_queries = einops.repeat(
                pre_state[batch_idx, seq_idx, :],
                "d_model -> seq_len d_model",
                seq_len = seq_idx+1,
            ),
            model = model,
            layer_idx = 10,
            head_idx = 7,
            use_tqdm = False,
            key_bias = False,
            query_bias = False,
        )
        t.testing.assert_close(
            current_attention_scores,
            attn_scores[batch_idx, seq_idx, :seq_idx+1],
            atol=5e-3,
            rtol=5e-3,
        ), f"batch_idx={batch_idx}, seq_idx={seq_idx} failure"

#%%

my_random_tokens = model.to_tokens("Here are some random words:" + " Reboot| Telegram| deregulation| 7000| asses| IPM|bats| scoreboard| shrouded| volleyball|acan|earcher| buttocks|adies| Giovanni| Jesuit| Sheen|reverse|ruits|".replace("|", ""))[0]
my_random_tokens = model.to_tokens("The")[0]

my_embeddings = t.zeros(BATCH_SIZE, SEQ_LEN-1, model.cfg.d_model).cuda()

for batch_idx in range(BATCH_SIZE):
    current_prompt = t.cat([
        einops.repeat(my_random_tokens, "random_seq_len -> cur_seq_len random_seq_len", cur_seq_len=SEQ_LEN-1).clone(),
        _DATA_TOKS[batch_idx, :-1].unsqueeze(-1).clone(),
    ],dim=1)
    current_embeddings = model.run_with_cache(
        current_prompt,
        names_filter = lambda name: name==get_act_name("resid_pre", 10),
    )[1][get_act_name("resid_pre", 10)][torch.arange(SEQ_LEN-1), -1]
    my_embeddings[batch_idx] = current_embeddings

#%%

# Cribbed from `venn_diagrams_loss_recovered.py`

keyside_projections = t.zeros((BATCH_SIZE, SEQ_LEN-1, model.cfg.d_model)).to(model.cfg.device)
keyside_orthogonals = t.zeros((BATCH_SIZE, SEQ_LEN-1, model.cfg.d_model)).to(model.cfg.device)

if DO_KEYSIDE_PROJECTIONS:
    for batch_idx, seq_idx in tqdm(list(itertools.product(range(BATCH_SIZE), range(SEQ_LEN-1)))):
        
        project_onto = None
        project_onto = my_embeddings[batch_idx, seq_idx]

        keyside_vector, keyside_orthogonal = original_project(
            normalize(pre_state[batch_idx, seq_idx]) * np.sqrt(model.cfg.d_model), # simulate LN
            project_onto,
            test = False,
        )

        if seq_idx != 0:
            keyside_projections[batch_idx, seq_idx] = keyside_vector
            keyside_orthogonals[batch_idx, seq_idx] = keyside_orthogonal

        else: # BOS seems weird
            keyside_projections[batch_idx, seq_idx] = keyside_vector + keyside_orthogonal
            keyside_orthogonals[batch_idx, seq_idx] = 0.0

else:
    keyside_projections[:] = pre_state.clone()
    keyside_orthogonals[:] = 0.0

#%% 

resid_pre_mean = einops.reduce(
    pre_state, 
    "b s d_model -> d_model",
    reduction="mean",
)

#%%

attention_score_projections = t.zeros((BATCH_SIZE, SEQ_LEN-1, SEQ_LEN-1)).to(model.cfg.device)
attention_score_projections[:] = attn_scores.clone()

if DO_QUERYSIDE_PROJECTIONS:

    attention_score_projections[:] = -100_000

    for batch_idx, seq_idx in tqdm(list(itertools.product(range(BATCH_SIZE), range(1, SEQ_LEN-1)))):  # preserve BOS attention score
        model.reset_hooks()

        warnings.warn("We're using 2* lol")

        normalized_queries = einops.repeat(
            2 * normalize(pre_state[batch_idx, seq_idx, :]) * np.sqrt(model.cfg.d_model),
            "d_model -> seq_len d_model",
            seq_len = seq_idx,
        )

        if PROJECT_MODE == "unembeddings":
            # project each onto the relevant unembedding
            normalized_queries, _ = original_project(
                normalized_queries,
                list(einops.rearrange(model.W_U.T[E_sq_QK[batch_idx, 1:seq_idx+1]], "seq_len ten d_model -> ten seq_len d_model")),
                test=False,
            )
        elif PROJECT_MODE == "name_movers":
            normalized_queries, _ = original_project(
                normalized_queries,
                list(einops.repeat(layer_nine_outs[batch_idx, seq_idx], "head d_model -> head seq d_model", seq=seq_idx)), # for now project onto all layer 9 heads
                test=False,
            )

        else:
            raise ValueError(f"Unknown project mode {PROJECT_MODE}")

        cur_attn_scores = dot_with_query(
            unnormalized_keys = keyside_projections[batch_idx, 1:seq_idx+1, :],
            unnormalized_queries = normalized_queries,
            model = model,
            layer_idx = 10,
            head_idx = 7,
            use_tqdm = False,
            normalize_queries = False, 
            normalize_keys = True,
            add_query_bias = True, 
            add_key_bias = True,
        )

        attention_score_projections[batch_idx, seq_idx, 1:seq_idx+1] = cur_attn_scores

        # if batch_idx > 2:
        #     warnings.warn("Early")
        #     break

true_attention_pattern = attn_scores.clone().softmax(dim=-1)
our_attention_scores = attention_score_projections.clone()
# our_attention_scores *= 0.5 
our_attention_scores[:, :, 0] = -100_000 # temporarily kill BOS
our_attention_pattern = our_attention_scores.softmax(dim=-1)
our_attention_pattern *= (-true_attention_pattern[:, :, 0] + 1.0).unsqueeze(-1) # so that BOS equal to original value
our_attention_pattern[:, :, 0] = true_attention_pattern[:, :, 0] # rows still have attention 1.0

#%%

CUTOFF = 50
BATCH_INDEX = 2 # 2 is great!

# for name, attention_pattern in zip(["true", "ours"], [true_attention_pattern, our_attention_pattern], strict=True):
# set range -10 10
for name, attention_pattern in zip(["true", "ours"], [attn_scores, attention_score_projections], strict=True):  
    imshow(
        attention_pattern[BATCH_INDEX, :CUTOFF, :CUTOFF],
        x = model.to_str_tokens(_DATA_TOKS[BATCH_INDEX, :CUTOFF]),   
        y = model.to_str_tokens(_DATA_TOKS[BATCH_INDEX, :CUTOFF]),   
        title = name,
        zmin = -10, 
        zmax = 10,
    )

#%%

model.set_use_split_qkv_input(True)
model.set_use_split_qkv_normalized_input(True)
model.reset_hooks()

# # Add the hook approximation

if DO_QUERYSIDE_PROJECTIONS:
    model.add_hook(
        get_act_name("pattern", 10),
        partial(set_to_value, head_idx=7, new_value=our_attention_pattern.cuda()),
        level=1,
    )

elif DO_KEYSIDE_PROJECTIONS:
    model.add_hook(
        get_act_name("k_normalized_input", 10),
        partial(set_to_value, head_idx=7, new_value=keyside_projections.cuda()),
        level=1,
    )

projected_head_output = model.run_with_cache(_DATA_TOKS[:, :-1], names_filter = lambda name: name==get_act_name("result", 10))[1][get_act_name("result", 10)][:, :, 7]
model.reset_hooks()

projected_loss = get_metric_from_end_state(
    model = model,
    end_state = end_state - head_out + projected_head_output,
    frozen_ln_scale = scale,
    targets = _DATA_TOKS[:, 1:],
)

ICS: dict = MODEL_RESULTS.is_copy_suppression[("direct", "frozen", "mean")][10, 7]
ICS_list.append(ICS)
new_ICS = deepcopy(ICS)
new_ICS["L_CS"] = projected_loss.cpu()

scatter, results, df = generate_scatter(
    ICS=new_ICS,
    DATA_STR_TOKS_PARSED=list(itertools.chain(*MINIBATCH_DATA_STR_TOKS_PARSED)),
)

#%%
