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


BATCH_SIZE = 91 # 80 for viz?
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

MINIBATCH_SIZE = BATCH_SIZE // 3
DATA_TOKS_1 = DATA_TOKS[:MINIBATCH_SIZE]
DATA_TOKS_2 = DATA_TOKS[MINIBATCH_SIZE:2*MINIBATCH_SIZE]
DATA_TOKS_3 = DATA_TOKS[2*MINIBATCH_SIZE:]
DATA_STR_TOKS_PARSED_1 = DATA_STR_TOKS_PARSED[:MINIBATCH_SIZE]
DATA_STR_TOKS_PARSED_2 = DATA_STR_TOKS_PARSED[MINIBATCH_SIZE:2*MINIBATCH_SIZE]
DATA_STR_TOKS_PARSED_3 = DATA_STR_TOKS_PARSED[2*MINIBATCH_SIZE:]

# DATA_TOKS_MINI = DATA_TOKS[[32, 36], :60]
# DATA_STR_TOKS_PARSED_MINI = [DATA_STR_TOKS_PARSED[32][:60], DATA_STR_TOKS_PARSED[36][:60]]


# In[9]:


# model.reset_hooks(including_permanent=True)

# prompt = "All's fair in love and war"
# toks = model.to_tokens(prompt)
# str_toks = model.to_str_tokens(toks)
# if isinstance(str_toks[0], str): str_toks = [str_toks]
# str_toks_parsed = [list(map(parse_str_tok_for_printing, s)) for s in str_toks]

# MODEL_RESULTS = get_model_results(model, toks, NEGATIVE_HEADS)

# HTML_PLOTS = generate_4_html_plots(
#     model_results = MODEL_RESULTS,
#     model = model,
#     data_toks = toks,
#     data_str_toks_parsed = str_toks_parsed,
#     negative_heads = NEGATIVE_HEADS,
#     save_files = False,
# )

# for k, v in HTML_PLOTS.items():
#     print(k)
#     for k2 in v.keys(): print(f"-> {k2}")

# display(HTML(CSS + HTML_PLOTS["LOSS"][(0, "10.7", "direct+unfrozen+mean", True)] + "<br>" * 5))

# p = Path('/home/ubuntu/TransformerLens/transformer_lens/rs/callum2/explore_prompts/media/')
# ICS_list = [
#     pickle.load(open(p / f"ICS_0{i}.pkl", "rb"))
#     for i in range(3)
# ]


# ## Results from  `model_results_3.py`
# 
# This is the current best version of the copy-suppression ablation algorithm. It has the following basic structure:
# 
# * For each source token, pick `K_semantic` semantically related tokens, defined as ones which are also suppressed when we apply the OV circuit
#     * e.g. this usually contains the token itself, but also things like plurals/capitals, or tokenization weirdness e.g. `" Berkeley"` will be related to `"keley"`
# * This gives us `seqK * K_semantic` pairs of tokens for each destination token - denote each pair `(S, Q)`
# * We pick the best `K_unembed` pairs (ordered by how much the unembedding for the second of the pair appears in the query-side residual stream)
# * We mean-ablate all attention except for each of the `(S, Q)` pairs we've chosen
#     * i.e. for each source token `S`, we project the vector moved from `S` to `D` onto the space spanned by the unembedding vectors of `Q` for each pair `(S, Q)` we've chosen
# 
# #### Why is this different from previous versions?
# 
# * The first version only used "pure copying", i.e. only `(S, S)` for each token
#     * This captured some of the effect, but in practice the model often does things like "attend to `" Pier"`, and use that to suppress both `" Pier"` and `" pier"` (because different circuitry deals with whether the token is capitalized or not). And maybe suppressing `" pier"` was the thing that actually helped decrease loss.
#     * This is reflected when we look at the OV circuit: when `" Pier"` is attended to, both `" pier"` and `" Pier"` are suppressed (see first image below).<br><br>
# * The second version only looked at e.g. the top 100 unembedding tokens on the query-side, then saw if any of those were semantically related to the source tokens (in the QK circuit)
#     * Again, this captured some of the effect (sometimes the token that gets suppressed is already the top predicted)
#     * But other times, it's only suppressed because it's **higher than the other tokens in context**, not because it's one of the absolute best predictions. E.g. it might be 4 logits below the top prediction, which puts it at e.g. rank 400, which wouldn't get captured by this method
#     * Note that, in these cases, it's usually a head in layer 10 or 11 which ends up boosting this word to be one of the top predictions - otherwise this couldn't be an example in which head 10.7 affected the loss
# 
# Key points:
# 
# * The first version didn't work because **it didn't capture the idea that the model can suppress several variants of a token at once, not literally the same token.**
#     * <details><summary>Example image which shows failure mode</summary><br><img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/s1.png" width="800"></details><br>
# * The second version didn't work because **it didn't capture the idea that the model can suppress a token because it's predicted more than anything else in context, not only if it's the top prediction.**
#     * <details><summary>Example image which shows failure mode</summary><br><img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/kara2.png" width="550"><br><img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/kara_attn.png" width="450"><br>After layer 10, <code>" Kara"</code> becomes the second-most predicted token (which explains why 10.7 pushing <code>" Kara"</code> down does affect the loss), but <code>" Kara"</code> doesn't need to be one of the most predicted tokens before 10.7.</details>
# 
# 
# #### A few other notes:
# 
# * Why can't we look at the QK circuit to judge semantic similarity with this method?
#     * Answer - it's not necessarily the case that "what attends most to `src` is semantically related to `src`. Maybe there's some weird token that happens to have a high attn score for `src`; higher than anything else. I saw an example of this, I can't remember the src token, but I remember that the 5 things which attended most to it were all years e.g. `" 1973"`, despite this token just being a regular word with nothing to do with years.
#         * When I find this example, I'll put a screenshot below. But there are lots!<br>
# * This metric still isn't perfect, i.e. it doesn't explain 100%. But that makes sense, because this assumes that the number of useful pairs `(S, Q)` is finite; in fact it's pretty hard to isolate this distributed effect.
# 

# In[10]:


10 / (768 * 35)


# In[13]:


K_semantic = 5
K_unembed = 10

ICS_list = []
HTML_list = []

for i, (_DATA_TOKS, DATA_STR_TOKS_PARSED) in list(enumerate(zip(
    [DATA_TOKS_1, DATA_TOKS_2, DATA_TOKS_3],
    [DATA_STR_TOKS_PARSED_1, DATA_STR_TOKS_PARSED_2, DATA_STR_TOKS_PARSED_3],
))):

    if i == 0: continue

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


ICS_list = [pickle.load(open(_ST_HTML_PATH / f"ICS_{i}.pkl", "rb")) for i in range(3)]
ICS = {k: t.concat([ICS_list[i][k] for i in range(3)], dim=0) for k in ICS_list[0].keys()}
with open(_ST_HTML_PATH / f"ICS.pkl", "wb") as f:
    pickle.dump(ICS, f)


HTML_list: List[dict] = [pickle.load(gzip.open(_ST_HTML_PATH / f"GZIP_HTML_PLOTS_{i}.pkl", "rb")) for i in range(3)]
HTML = HTML_list[0]
for html in HTML_list[1:]:
    first_batch_idx = len(HTML["LOGITS_ORIG"])
    for k, v in html.items():
        for (batch_idx, *other_args), html_str in v.items():
            HTML[k][(first_batch_idx + batch_idx, *other_args)] = html_str
with gzip.open(_ST_HTML_PATH / f"GZIPPED_HTML_PLOTS.pkl", "wb") as f:
    pickle.dump(HTML, f)


# In[ ]:


# ICS: dict = MODEL_RESULTS.is_copy_suppression[("direct", "frozen", "mean")][11, 10]

# (with smaller seq len)
# K_semantic = 05, K_unembed = 10, result is 0.168/0.063 and 0.013/0.002
# K_semantic = 20, K_unembed = 20, result is 0.168/0.059 and 0.013/0.002

# I think K_semantic = 5, K_unembed = 10 works best

scatter, results, df = generate_scatter(
    ICS=ICS,
    DATA_STR_TOKS_PARSED=DATA_STR_TOKS_PARSED_1 + DATA_STR_TOKS_PARSED_2 + DATA_STR_TOKS_PARSED_3,
)
hist1 = generate_hist(ICS, threshold=0.05)
hist2 = generate_hist(ICS, threshold=0.025)

with gzip.open(_ST_HTML_PATH / f"CS_CLASSIFICATION.pkl", "wb") as f:
    pickle.dump({"hist1": hist1, "hist2": hist2, "scatter": scatter}, f)


# In[ ]:


(df["y"].abs() > 0.5).sum(), (df["x"].abs() > 0.5).sum()


# In[ ]:





# In[ ]:


ICS = pickle.load(open(_ST_HTML_PATH / f"ICS_Ks20_Ku20.pkl", "rb"))

results, df = generate_scatter(
    ICS=ICS,
    DATA_STR_TOKS_PARSED=DATA_STR_TOKS_PARSED_1 + DATA_STR_TOKS_PARSED_2 + DATA_STR_TOKS_PARSED_3,
)
generate_hist(ICS, threshold=0.05)
generate_hist(ICS, threshold=0.025)


# ### Some examples which mess up this metric, and explanations for why

# #### What are the failure modes of this metric?
# 
# There aren't really that many egregious failures. Also, a nice property of the graph above - the more important this head's effect on the model's loss, the more likely it is that copy-suppression explains most/all of the head's behaviour.
# 
# Here are a few examples further broken down. Note that I'm focusing on situations where **this head was important, and the CS ablation failed to capture this importance**, because these are more interesting than the model accidentally doing something dumb.
# 
# #### `(1, 68)` - `"understaffing and (violence)"`
# 
# I don't understand what's forming the attention patterns here. `" overcrowd"` is being predicted. Presumably this is some kind of bigram thing, where the `"ing"` from `" overstaffing"` contains information about the `" overstaffing"` word, and that's being negatively copied.
# 
# <img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/cex_attn1.png" width="200">
# 
# #### `(8, 36)` - `" beyond requiring a (business)`
# 
# The attn is quite distributed. It's mostly over `" rentals"`, but also a little over 4 other words. The main effect here as far as I can tell is attending back to `" rentals"` (and words like it) and suppressing `" permit"` (and words like it). But it's no wonder copy-suppression metric failed here, this doesn't seem clean as I'm reading it.
# 
# #### `(27, 36)` - `"\n\n (Hor) rend ous"`
# 
# The second line break is attending to `" Speed"` (because it's predicting `"Speed"`), and it's suppressing `"Speed"`. This seems like a pretty clear example of copy-suppression, but I suspect our choice of K-params was just a bit too strict, and this didn't make the cut.
# 
# #### `(67, 47)` - `Attwater's (voice)`
# 
# Attention is pretty distributed here. Most of it goes to `" case"` (which is highly predicted). Not sure why this isn't picked up by my metric, since (see printout below) it seems like this is one of the pairs I'm preserving. My guess is that it's just because things are distributed.
# 
# #### `(24, 73)` - `another (round)`
# 
# `" another"` is attending strongly to `" negotiations"`, and not much to anything else. Again, not entirely sure why this isn't picked up on (see below).

# In[14]:


# (B, D) = (67, 47)
# (B, D) = (27, 36)
(B, D) = (24, 73)


toks = DATA_TOKS[[B]]

MODEL_RESULTS = get_model_results(
    model,
    toks=toks,
    negative_heads=NEGATIVE_HEADS,
    verbose=True,
    K_semantic=5,
    K_unembed=5,
    use_cuda=False,
    effective_embedding="W_E (including MLPs)",
)

# ! Sanity checking for the whole semantic similarity thing - this prints out exactly what pairs (S, Q) we're finding.

top_toks_for_E_sq: Int[Tensor, "N 4"] = MODEL_RESULTS.misc[(10, "top_toks_for_E_sq")]
logits_for_E_sq: Int[Tensor, "batch seqQ seqK K_semantic"] = MODEL_RESULTS.misc[(10, "logits_for_E_sq")]
logit_lens: Int[Tensor, "batch seqQ d_vocab"] = MODEL_RESULTS.misc[(10, "logit_lens")]
E_sq: Int[Tensor, "batch seqK K_semantic"] = MODEL_RESULTS.E_sq[10, 7]

from rich import print as rprint
from rich.table import Table
table = Table("Prefix", "Map (being predicted)", "Src (being attended to)")
query = "".join([
    repr("".join(DATA_STR_TOKS_PARSED[B][max(0, D-5): D+1])),
    " -> ",
    (repr(DATA_STR_TOKS_PARSED[B][D+1]) if D < SEQ_LEN-1 else "...?")
])
top_toks_for_E_sq_filtered = [row[2:] for row in top_toks_for_E_sq if (row[0].item() == 0) and (row[1].item() == D)]
for sK, K_sem in top_toks_for_E_sq_filtered:
    sK_token = toks[0, sK].item()
    sU_token = E_sq[0, sK, K_sem].item()
    thing_that_is_predicted = repr(parse_str_tok_for_printing(model.to_single_str_token(sU_token))) + f" ({D})"
    thing_that_is_attended_to = repr(parse_str_tok_for_printing(model.to_single_str_token(sK_token))) + f" ({sK})"
    table.add_row(query + "\n", thing_that_is_predicted, thing_that_is_attended_to)
rprint(table)


# ## Testing `U_p` and `E_ps` from `model_results_2.py` (no longer using this)

# In[ ]:


Kp = 25
Ks = 5

for i, data_toks in enumerate([DATA_TOKS_1, DATA_TOKS_2, DATA_TOKS_3]):

    MODEL_RESULTS = get_model_results(
        model,
        toks=data_toks,
        negative_heads=NEGATIVE_HEADS,
        verbose=True,
        Kp=Kp,
        Ks=Ks,
        use_cuda=False,
    )

    # Note - the complexity increases by so much when you increase sequence length, seems more than quadratic

    # %load_ext line_profiler
    # %lprun -f get_model_results get_model_results(model, DATA_TOKS, negative_heads=NEGATIVE_HEADS, verbose=True, num_top_src_tokens=5, loss_ratio_threshold=0.5)

    ICS = MODEL_RESULTS.is_copy_suppression[("direct", "frozen", "mean")][10, 7]
    p = Path("/home/ubuntu/TransformerLens/transformer_lens/rs/callum2/explore_prompts/media") / f"ICS_0{i}.pkl"
    with open(p.resolve(), "wb") as f:
        pickle.dump(ICS, f)
    del MODEL_RESULTS

    # (batch, seq, Kp, Ks)
    # = (40, 60, 20, 5) took 4.5 minutes
    # = (40*3, 70, 25, 5) took 8.5 minutes a-piece


# In[ ]:


U_p = MODEL_RESULTS.U_p[10, 7]
E_ps = MODEL_RESULTS.E_ps[10, 7]
U_p_repeated = MODEL_RESULTS.misc["U_p_repeated"]
assert U_p_repeated.shape == (BATCH_SIZE, SEQ_LEN, SEQ_LEN, Kp)
assert U_p.shape == (BATCH_SIZE, SEQ_LEN, Kp)
assert E_ps.shape == (BATCH_SIZE, SEQ_LEN, Kp, Ks)


# In[ ]:


predictions_after_eater = model.to_str_tokens(U_p[1, 46])
things_that_this_attends_to = E_ps[1, 46]

for Up, Eps in zip(predictions_after_eater, things_that_this_attends_to):
    print(f"{Up!r} -> {model.to_str_tokens(Eps)}")


# In[ ]:


eater: Int[Tensor, "seqK Kp"] = U_p_repeated[1, 46]

for i in range(eater.shape[0]):
    if eater[i].sum() > 0:
        print(i, eater[i])


# In[ ]:


non_ablated_attention: Int[Tensor, "instances b_sQ_sK_Kp"] = (U_p_repeated >= 0).nonzero()
print(non_ablated_attention.shape)


# In[ ]:


from rich import print as rprint
from rich.table import Table

for b in range(2):
        
    table = Table("Prefix", "Predicting...", "Attending back to...", title=f"Batch {b}")

    for _b, sQ, sK, Kp in non_ablated_attention[non_ablated_attention[:, 0] == b]:
        query = repr("".join(DATA_STR_TOKS_PARSED[b][max(0, sQ-5): sQ+1])) + " -> " + (repr(DATA_STR_TOKS_PARSED[b][sQ+1]) if sQ < SEQ_LEN-1 else "...?")
        thing_that_is_predicted = repr(parse_str_tok_for_printing(model.to_single_str_token(U_p[b, sQ, Kp].item()))) + f" ({sQ})"
        thing_that_is_attended_to = repr(DATA_STR_TOKS_PARSED[b][sK]) + f" ({sK})"
        table.add_row(query + "\n", thing_that_is_predicted, thing_that_is_attended_to)

    rprint(table)


# ## Histograms: logit lens

# In[43]:


k = 15
neg = False
all_ranks = []

model.reset_hooks()
logits, cache = model.run_with_cache(DATA_TOKS_2)


# In[52]:


# points_to_plot = [
#     (35, 39, " About"),
#     (67, 21, " delays"),
#     (8, 35, " rentals"),
#     (8, 54, " require"),
#     (53, 18, [" San", " Francisco"]),
#     (33, 9, " Hollywood"),
#     (49, 7, " Home"),
#     (71, 34, " sound"),
#     (14, 56, " Kara"),
# ]
# points_to_plot = [
#     (45, 42, [" editorial"]),
#     (45, 58, [" stadium", " Stadium", " stadiums"]),
#     (43, 56, [" Biden"]),
#     (43, 44, [" interview", " campaign"]),
#     (38, 54, [" Mary", " Catholics"]),
#     (33, 29, " Hollywood"),
#     (33, 42, " BlackBerry"),
#     (31, 33, [" Church", " churches"]),
#     (28, 53, [" mobile", " phone", " device"]),
#     (25, 32, [" abstraction", " abstract", " Abstract"]),
#     (18, 25, ["TPP", " Lee"]),
#     (10, 52, [" Italy", " mafia"]),
#     (10, 52, [" Italy", " mafia"]),
#     (10, 35, [" Italy", " mafia"]),
#     (10, 25, [" Italian", " Italy"]),
#     (6, 52, [" landfill", " waste"]),
#     (4, 52, " jury"),
# ]
points_to_plot = [
    # (14, 56, " Kara"),
    # (67, 47, " case"),
    # (24, 73, " negotiation"),
    (2, 35, [" Berkeley", "keley"]),
]

resid_pre_head = (cache["resid_pre", 10]) / cache["scale", 10, "ln1"]  #  - cache["resid_pre", 1]

plot_logit_lens(points_to_plot, resid_pre_head, model, DATA_STR_TOKS_PARSED_2, k=15, title="Predictions at token ' of', before head 10.7")


# ## Histograms: QK and OV circuit

# In[32]:


def plot_both(dest, src, focus_on: Literal["src", "dest"]):
    plot_full_matrix_histogram(W_EE_dict, src, dest, model, k=15, circuit="OV", neg=True, head=(10, 7), flip=(focus_on=="dest"))
    plot_full_matrix_histogram(W_EE_dict, src, dest, model, k=15, circuit="QK", neg=False, head=(10, 7), flip=(focus_on=="src"))

plot_both(dest=" Berkeley", src="keley", focus_on="src")


# In[95]:


def plot_both(dest, src, focus_on: Literal["src", "dest"]):
    plot_full_matrix_histogram(W_EE_dict, src, dest, model, k=15, circuit="OV", neg=True, head=(10, 7), flip=(focus_on=="dest"))
    plot_full_matrix_histogram(W_EE_dict, src, dest, model, k=15, circuit="QK", neg=False, head=(10, 7), flip=(focus_on=="src"))

plot_both(dest=" negotiation", src=" negotiations", focus_on="dest")


# In[ ]:


logprobs_orig = MODEL_RESULTS.logits_orig[32, 19].log_softmax(-1)
logprobs_abl = MODEL_RESULTS.logits[("direct", "frozen", "mean")][10, 7][32, 19].log_softmax(-1)

logprobs_orig_topk = logprobs_orig.topk(10, dim=-1, largest=True)
y_orig = logprobs_orig_topk.values.tolist()
x = logprobs_orig_topk.indices
y_abl = logprobs_abl[x].tolist()
x = list(map(repr, model.to_str_tokens(x)))

orig_colors = ["darkblue"] * len(x)
abl_colors = ["blue"] * len(x)

correct_next_str_tok = " heated"
correct_next_token = model.to_single_token(" heated")
# if repr(correct_next_str_tok) in x:
#     idx = x.index(repr(correct_next_str_tok))
#     orig_colors[idx] = "darkgreen"
#     abl_colors[idx] = "green"

x.append(repr(correct_next_str_tok))
y_orig.append(logprobs_orig[correct_next_token].item())
y_abl.append(logprobs_abl[correct_next_token].item())
orig_colors.append("darkgreen")
abl_colors.append("green")

fig = go.Figure(
    data = [
        go.Bar(x=x, y=y_orig, name='Original', marker_color=["#FF7700"] * (len(x)-1) + ["#024B7A"]), # 7A30AB
        go.Bar(x=x, y=y_abl, name='Ablated', marker_color=["#FFAE49"] * (len(x)-1) + ["#44A5C2"]), # D44BFA
    ],
    # data = [
    #     go.Bar(x=x[:-1], y=y_orig[:-1], name='Original', marker_color="#FF7700", legendgroup="group1"),
    #     go.Bar(x=x[:-1], y=y_abl[:-1], name='Ablated', marker_color="#FFAE49", legendgroup="group1"),
    #     go.Bar(x=[x[-1]], y=[y_orig[-1]], name='Original (correct token)', marker_color="#024B7A", legendgroup="group2"),
    #     go.Bar(x=[x[-1]], y=[y_abl[-1]], name='Ablated (correct token)', marker_color="#44A5C2", legendgroup="group2"),
    # ],
    layout = dict(
        barmode='group',
        xaxis_tickangle=30,
        title="Logprobs: original vs ablated",
        xaxis_title_text="Predicted next token",
        yaxis_title_text="Logprob",
        width=800,
        bargap=0.35,
    )
)
fig.data = fig.data #+ ({"name": "New"},)
fig.show()


# In[ ]:


plot_full_matrix_histogram(W_EE_dict, " device", k=10, include=[" devices"], circuit="OV", neg=True, head=(10, 7))
plot_full_matrix_histogram(W_EE_dict, " devices", k=10, include=[" device"], circuit="QK", neg=False, head=(10, 7))


# In[ ]:


W_EE = W_EE_dict["W_E (including MLPs)"]
W_EE = W_EE_dict["W_E (only MLPs)"]
W_U = W_EE_dict["W_U"].T

tok_strs = ["pier"]
for i in range(len(tok_strs)): tok_strs.append(tok_strs[i].capitalize())
for i in range(len(tok_strs)): tok_strs.append(tok_strs[i] + "s")
for i in range(len(tok_strs)): tok_strs.append(" " + tok_strs[i])
tok_strs = [s for s in tok_strs if model.to_tokens(s, prepend_bos=False).squeeze().ndim == 0]

toks = model.to_tokens(tok_strs, prepend_bos=False).squeeze()

W_EE_toks = W_EE[toks]
W_EE_normed = W_EE_toks / W_EE_toks.norm(dim=-1, keepdim=True)
cos_sim_embeddings = W_EE_normed @ W_EE_normed.T

W_U_toks = W_U.T[toks]
W_U_normed = W_U_toks / W_U_toks.norm(dim=-1, keepdim=True)
cos_sim_unembeddings = W_U_normed @ W_U_normed.T

W_EE_OV_toks_107 = W_EE_toks @ model.W_V[10, 7] @ model.W_O[10, 7]
W_EE_OV_toks_99 = W_EE_toks @ model.W_V[9, 9] @ model.W_O[9, 9]
W_EE_OV_toks_107_normed = W_EE_OV_toks_107 / W_EE_OV_toks_107.norm(dim=-1, keepdim=True)
W_EE_OV_toks_99_normed = W_EE_OV_toks_99 / W_EE_OV_toks_99.norm(dim=-1, keepdim=True)
cos_sim_107 = W_EE_OV_toks_107_normed @ W_EE_OV_toks_107_normed.T
cos_sim_99 = W_EE_OV_toks_99_normed @ W_EE_OV_toks_99_normed.T

imshow(
    t.stack([
        cos_sim_embeddings,
        cos_sim_unembeddings,
        cos_sim_107,
        cos_sim_99,
    ]),
    x = list(map(repr, tok_strs)),
    y = list(map(repr, tok_strs)),
    title = "Cosine similarity of variants of ' pier'",
    facet_col = 0,
    facet_labels = ["Effective embeddings", "Unembeddings", "W_OV output (10.7)", "W_OV output (9.9)"],
    border = True,
    width=1200,
)

# W_EE_OV_normed = W_EE_OV / W_EE_OV.std(dim=-1, keepdim=True)


# ## Create OV and QK circuits Streamlit page
# 
# I need to save the following things:
# 
# * The QK and OV matrices for head 10.7 and 11.10
# * The extended embedding and unembedding matrices
# * The tokenizer

# In[40]:


dict_to_store = {
    "tokenizer": model.tokenizer,
    "W_V_107": model.W_V[10, 7],
    "W_O_107": model.W_O[10, 7],
    "W_V_1110": model.W_V[11, 10],
    "W_O_1110": model.W_O[11, 10],
    "W_Q_107": model.W_Q[10, 7],
    "W_K_107": model.W_K[10, 7],
    "W_Q_1110": model.W_Q[11, 10],
    "W_K_1110": model.W_K[11, 10],
    "b_Q_107": model.b_Q[10, 7],
    "b_K_107": model.b_K[10, 7],
    "b_Q_1110": model.b_Q[11, 10],
    "b_K_1110": model.b_K[11, 10],
    "W_EE": W_EE_dict["W_E (including MLPs)"],
    "W_U": model.W_U,
}
dict_to_store = {k: v.half() if isinstance(v, t.Tensor) else v for k, v in dict_to_store.items()}

with gzip.open(_ST_HTML_PATH / f"OV_QK_circuits.pkl", "wb") as f:
    pickle.dump(dict_to_store, f)


# ## Generate `explore_prompts` HTML plots for Streamlit page

# In[ ]:


HTML_PLOTS = generate_4_html_plots(
    model_results = MODEL_RESULTS,
    model = model,
    data_toks = DATA_TOKS,
    data_str_toks_parsed = DATA_STR_TOKS_PARSED,
    negative_heads = NEGATIVE_HEADS,
    save_files = True,
    progress_bar = True,
    restrict_computation = ["LOSS"]
)


# In[ ]:


(
    model.W_U.T[model.to_single_token(" pier")].norm().item(), 
    model.W_U.T[model.to_single_token(" Pier")].norm().item(),
)


# In[ ]:


t.cosine_similarity(
    model.W_U.T[model.to_single_token(" pier")],
    model.W_U.T[model.to_single_token(" Pier")],
    dim=-1
).item()


pier = model.W_U.T[model.to_single_token(" pier")]
Pier = model.W_U.T[model.to_single_token(" Pier")]
pier /= pier.norm()
Pier /= Pier.norm()
print(pier @ Pier)


# In[ ]:


def W_U(s):
    return model.W_U.T[model.to_single_token(s)]
def W_EE0(s):
    return W_EE_dict["W_E (only MLPs)"][model.to_single_token(s)]

def cos_sim(v1, v2):
    return v1 @ v2 / (v1.norm() * v2.norm())

print(f"Unembeddings cosine similarity (Berkeley) = {cos_sim(W_U('keley'), W_U(' Berkeley')):.3f}") 
print(f"Embeddings cosine similarity (Berkeley)   = {cos_sim(W_EE0('keley'), W_EE0(' Berkeley')):.3f}") 
print("")
print(f"Unembeddings cosine similarity (pier) = {cos_sim(W_U(' pier'), W_U(' Pier')):.3f}") 
print(f"Embeddings cosine similarity (pier)   = {cos_sim(W_EE0(' pier'), W_EE0(' Pier')):.3f}") 


# In[ ]:


t.cosine_similarity(
    W_EE0(" screen") - W_EE0(" screens"),
    W_EE0(" device") - W_EE0(" devices"),
    dim=-1
).item()


# In[ ]:


t.cosine_similarity(
    W_EE(" computer") - W_EE(" computers"),
    W_EE(" sign") - W_EE(" signs"),
    dim=-1
).item()


# ## Scatter plots of copy-suppression classification

# In[ ]:





# In[ ]:





# In[ ]:


ICS = MODEL_RESULTS.is_copy_suppression[("direct", "frozen", "mean")][10, 7]
z = ICS["CS"]
ratio = ICS["LR"]
l_orig = ICS["L_ORIG"]
l_cs = ICS["L_CS"]
l_abl = ICS["L_ABL"]
# Get the 2.5% cutoff examples, and the 5% cutoff examples
l_abl_minus_orig = l_abl - l_orig

non_extreme_05pct = (l_abl_minus_orig < l_abl_minus_orig.quantile(0.95)) & (l_abl_minus_orig > l_abl_minus_orig.quantile(0.05))
non_extreme_025pct = (l_abl_minus_orig < l_abl_minus_orig.quantile(0.975)) & (l_abl_minus_orig > l_abl_minus_orig.quantile(0.025))

z_05pct = t.where(non_extreme_05pct, 2, z)
z_025pct = t.where(non_extreme_025pct, 2, z)

ratio_05pct = ratio.flatten()[z_05pct.flatten() != 2]
ratio_025pct = ratio.flatten()[z_025pct.flatten() != 2]

fig = hist(
    ratio_05pct,
    template="simple_white",
    title=create_title_and_subtitles(
        title="Cross-entropy loss, when ablating everything except copy-suppression mechanism",
        subtitles=[
            "One = same loss as original (no ablation)",
            "Zero = same loss as complete ablation",
            "Only looking at top/bottom 5% of loss-affecting examples"
        ]
    ),
    labels={"x": "Cross entropy loss (post-affine transformation)"},
    return_fig=True,
)
fig.update_layout(title_y=0.92, margin_t=150, height=500, width=800)
fig.add_vline(opacity=1.0, x=ratio_05pct.median(), line=dict(width=3, color="red"), annotation_text=f"Median = {ratio_05pct.median():.3f} ", annotation_position="top left")
fig.show()


# In[ ]:


fig = hist(
    ratio_025pct,
    template="simple_white",
    title=create_title_and_subtitles(
        title="Cross-entropy loss, when ablating everything except copy-suppression mechanism",
        subtitles=[
            "One = same loss as original (no ablation)",
            "Zero = same loss as complete ablation",
            "Only looking at top/bottom 2.5% of loss-affecting examples"
        ]
    ),
    labels={"x": "Cross entropy loss (post-affine transformation)"},
    return_fig=True,
)
fig.update_layout(title_y=0.92, margin_t=150, height=500, width=800)
fig.add_vline(opacity=1.0, x=ratio_025pct.median(), line=dict(width=3, color="red"), annotation_text=f"Median = {ratio_025pct.median():.3f} ", annotation_position="top left")
fig.show()


# ## [02] Copy Suppression Classification

# In[ ]:


p = Path(os.path.expanduser('~/TransformerLens/transformer_lens/rs/callum2/explore_prompts/media/'))
ICS_list = [
    pickle.load(open(p / f"ICS_0{i}.pkl", "rb"))
    for i in range(3)
]
# ICS_list = [MODEL_RESULTS.is_copy_suppression[("direct", "frozen", "mean")][10, 7]]

generate_scatter(
    ICS_list=ICS_list,
    DATA_STR_TOKS_PARSED_list=[DATA_STR_TOKS_PARSED_1, DATA_STR_TOKS_PARSED_2, DATA_STR_TOKS_PARSED_3],
    head=(10, 7),
)


# %%
