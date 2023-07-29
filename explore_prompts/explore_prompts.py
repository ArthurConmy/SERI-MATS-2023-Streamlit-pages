#%%

"""
NOTE: this is cribbed from explore_prompts.ipynb, Arthur finds it much easier to use + debug py files
"""


# # Explore Prompts
# 
# This is the notebook I use to test out the functions in this directory, and generate the plots in the Streamlit page.

# ## Setup

# In[4]:


# import torch as t
# from transformer_lens import HookedTransformer

# model = HookedTransformer.from_pretrained(
#     "gpt2-small",
#     center_unembed=True,
#     center_writing_weights=True,
#     fold_ln=True,
#     device="cpu"
#     # refactor_factored_attn_matrices=True,
# )
# model.set_use_split_qkv_input(False)
# model.set_use_attn_result(True)

# t.save(model.half(), "gpt2-small.pt")

from transformer_lens.cautils.notebook import *
import gzip

from generate_html import (
    CSS,
    generate_4_html_plots,
    generate_html_for_DLA_plot,
    generate_html_for_logit_plot,
    generate_html_for_loss_plot,
    generate_html_for_unembedding_components_plot,
    attn_filter,
    _get_color,
)
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

from transformer_lens.rs.arthurs_notebooks.arthur_utils import get_metric_from_end_state

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
    device="cpu" # "cuda"
    # fold value bias?
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
batch_idx = 36

NEGATIVE_HEADS = [(10, 7), (11, 10)]

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

MINIBATCH_SIZE = BATCH_SIZE // NUM_MINIBATCHES
MINIBATCH_DATA_TOKS = [DATA_TOKS[i*MINIBATCH_SIZE:(i+1)*MINIBATCH_SIZE] for i in range(NUM_MINIBATCHES)]
MINIBATCH_DATA_STR_TOKS_PARSED = [DATA_STR_TOKS_PARSED[i*MINIBATCH_SIZE:(i+1)*MINIBATCH_SIZE] for i in range(NUM_MINIBATCHES)]

# In[13]:

K_semantic = 5
K_unembed = 10

ICS_list = []
HTML_list = []

assert NUM_MINIBATCHES == 1, "Deprecating support for several minibatches"

_DATA_TOKS = MINIBATCH_DATA_TOKS[0]
DATA_STR_TOKS_PARSED= MINIBATCH_DATA_STR_TOKS_PARSED[0]
i = 0

#%%

# toks=_DATA_TOKS
# negative_heads=NEGATIVE_HEADS
# verbose=True
# K_semantic=K_semantic
# K_unembed=K_unembed
# use_cuda=False
# effective_embedding="W_E (including MLPs)"

MODEL_RESULTS = get_model_results(
    model,
    toks=_DATA_TOKS,
    negative_heads=NEGATIVE_HEADS,
    verbose=True,
    K_semantic=K_semantic,
    K_unembed=K_unembed,
    use_cuda=False,
    effective_embedding="W_E (including MLPs)",
)

#%%

# The goal is to make a BATCH_SIZE x SEQ_LEN-1 list of losses here 

# Let's decompose the goal
# 1. Firstly reproduce that mean ablating the direct effect of 10.7 gives points that are exclusively on the y=x line
# 2. Use your get_metric_from_end_state methinks : ) 
# 3. Let the experiments begin

#%%

model.reset_hooks()
final_ln_scale_hook_name = "ln_final.hook_scale"

logits, cache = model.run_with_cache(
    _DATA_TOKS[:, :-1],
    names_filter = lambda name: name in [get_act_name("result", 10), get_act_name("resid_post", 11), final_ln_scale_hook_name],
)

end_state = cache[get_act_name("resid_post", 11)]
head_out = cache[get_act_name("result", 10)][:, :, 7].clone()
scale = cache[final_ln_scale_hook_name]
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

head_loss = get_metric_from_end_state(
    model = model,
    end_state = end_state - head_out + mean_head_output.unsqueeze(0).unsqueeze(0).clone(),
    frozen_ln_scale = scale,
    targets = _DATA_TOKS[:, 1:],
)

#%%

ICS: dict = MODEL_RESULTS.is_copy_suppression[("direct", "frozen", "mean")][10, 7]
ICS_list.append(ICS)

# In[ ]:

new_ICS = deepcopy(ICS)
new_ICS["L_CS"] = head_loss

# In[ ]:

scatter, results, df = generate_scatter(
    ICS=new_ICS,
    DATA_STR_TOKS_PARSED=list(itertools.chain(*MINIBATCH_DATA_STR_TOKS_PARSED)),
)

#%%

hist1 = generate_hist(ICS, threshold=0.05)
hist2 = generate_hist(ICS, threshold=0.025)

with gzip.open(_ST_HTML_PATH / f"CS_CLASSIFICATION.pkl", "wb") as f:
    pickle.dump({"hist1": hist1, "hist2": hist2, "scatter": scatter}, f)


# In[ ]:

(df["y"].abs() > 0.5).sum(), (df["x"].abs() > 0.5).sum()

#%%