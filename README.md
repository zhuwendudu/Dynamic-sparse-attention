# Dynamic Sparse Attention (Explore → Focus)

This repository implements a **complexity-conditioned dynamic temperature attention mechanism**.

When the input is **ambiguous** (many competing features), the attention stays **broad** — this corresponds to **exploration**.  
When a **dominant structure** emerges, the attention **sharpens** — this corresponds to **exploitation**.

This matches how humans think:

> 先观察 → 看清状况 → 再集中注意力做决定

---

## Key Idea

We adjust the softmax temperature **τ** based on the *structural variance* of the attention logits.

| Situation | τ Value | Attention Behavior |
|---|---|---|
| Inputs unclear / noisy / many competing hypotheses | **High τ** | Broad exploration |
| Clear dominant pattern emerges | **Low τ** | Focused exploitation |

This avoids:

- premature over-focusing
- collapsing into incorrect local minima
- losing flexibility too early

---

## Code

The core logic is implemented in:
