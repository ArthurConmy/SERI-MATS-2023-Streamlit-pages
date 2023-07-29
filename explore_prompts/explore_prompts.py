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

# DATA_STR_TOKS_PARSED_2 = DATA_STR_TOKS_PARSED[MINIBATCH_SIZE:2*MINIBATCH_SIZE]
# DATA_STR_TOKS_PARSED_3 = DATA_STR_TOKS_PARSED[2*MINIBATCH_SIZE:]
# DATA_TOKS_MINI = DATA_TOKS[[32, 36], :60]
# DATA_STR_TOKS_PARSED_MINI = [DATA_STR_TOKS_PARSED[32][:60], DATA_STR_TOKS_PARSED[36][:60]]

# In[13]:

K_semantic = 5
K_unembed = 10

ICS_list = []
HTML_list = []

for i, (_DATA_TOKS, DATA_STR_TOKS_PARSED) in list(enumerate(zip(
    MINIBATCH_DATA_TOKS,
    MINIBATCH_DATA_STR_TOKS_PARSED,
))):
    print(f"Minibatch {i} of {NUM_MINIBATCHES}")

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


    ICS: dict = MODEL_RESULTS.is_copy_suppression[("direct", "frozen", "mean")][10, 7]
    ICS_list.append(ICS)

    HTML_PLOTS = generate_4_html_plots(
        model,
        _DATA_TOKS,
        DATA_STR_TOKS_PARSED,
        NEGATIVE_HEADS,
        save_files = False,
        model_results = MODEL_RESULTS,
        # restrict_computation = ["UNEMBEDDINGS"]
    )
    HTML_list.append(HTML_PLOTS)

    # use os.path.expanduser so this works on any machine where TL is cloned in the home directory
    _ST_HTML_PATH = Path(os.path.expanduser('~/TransformerLens/transformer_lens/rs/callum2/explore_prompts/media/'))

    with open(_ST_HTML_PATH / f"ICS_{i}.pkl", "wb") as f:
        pickle.dump(ICS, f)

    with gzip.open(_ST_HTML_PATH / f"GZIP_HTML_PLOTS_{i}.pkl", "wb") as f:
        pickle.dump(HTML_PLOTS, f)

# In[ ]:


ICS_list = [pickle.load(open(_ST_HTML_PATH / f"ICS_{i}.pkl", "rb")) for i in range(NUM_MINIBATCHES)]
ICS = {k: t.concat([ICS_list[i][k] for i in range(NUM_MINIBATCHES)], dim=0) for k in ICS_list[0].keys()}
with open(_ST_HTML_PATH / f"ICS.pkl", "wb") as f:
    pickle.dump(ICS, f)


HTML_list: List[dict] = [pickle.load(gzip.open(_ST_HTML_PATH / f"GZIP_HTML_PLOTS_{i}.pkl", "rb")) for i in range(NUM_MINIBATCHES)]
HTML = HTML_list[0]
for html in HTML_list[1:]:
    first_batch_idx = len(HTML["LOGITS_ORIG"])
    for k, v in html.items():
        for (batch_idx, *other_args), html_str in v.items():
            HTML[k][(first_batch_idx + batch_idx, *other_args)] = html_str
with gzip.open(_ST_HTML_PATH / f"GZIPPED_HTML_PLOTS.pkl", "wb") as f:
    pickle.dump(HTML, f)


# In[ ]:




scatter, results, df = generate_scatter(
    ICS=ICS,
    DATA_STR_TOKS_PARSED=list(itertools.chain(*MINIBATCH_DATA_STR_TOKS_PARSED)),
)
hist1 = generate_hist(ICS, threshold=0.05)
hist2 = generate_hist(ICS, threshold=0.025)

with gzip.open(_ST_HTML_PATH / f"CS_CLASSIFICATION.pkl", "wb") as f:
    pickle.dump({"hist1": hist1, "hist2": hist2, "scatter": scatter}, f)


# In[ ]:

(df["y"].abs() > 0.5).sum(), (df["x"].abs() > 0.5).sum()

#%%