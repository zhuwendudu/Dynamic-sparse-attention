# Dynamic Sparse Attention

This repository implements a *complexity-conditioned dynamic temperature attention mechanism*.  
When the input signal is **complex and ambiguous**, the model **keeps attention broad** (exploration).  
When the signal becomes **differentiated and structured**, the attention **automatically sharpens** (exploitation).

This allows the model to:
- avoid premature over-focusing
- prevent collapse into incorrect local minima
- shift attention **in sync with information clarity**

### Key Idea
We dynamically adjust the softmax temperature τ based on **structural variance** in the attention logits.
τ = τ₀ / (1 + Var(attention logits))
This means:
| Situation | τ Value | Attention Behavior |
|---|---|---|
| Inputs unclear / many competing features | High τ | Broad exploration |
| Clear dominant pattern emerges | Low τ | Focused exploitation |

### Code
See: `dynamic_sparse_attention.py`

### Coming Next
- Example notebook
- Experimental comparisons
- Paper draft (NeurIPS / ICLR style)
