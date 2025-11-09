# Dynamic Sparse Attention (Complexity-Conditioned Temperature)

This repository provides a **dynamic temperature attention mechanism** that adapts the softmax temperature τ based on feature complexity.

When representations are **ambiguous**, attention remains **broad** (exploration).  
When a **dominant pattern emerges**, attention **sharpens** automatically (exploitation).

This allows attention to **shift modes**:
**explore → detect structure → focus**.

---

## Core Mechanism

Standard attention:

\[
\text{softmax}(QK^\top / \tau)
\]

uses a **fixed τ**, forcing a single focus level.

We compute τ dynamically:

\[
\tau = g(\mathrm{Var}(QK^\top))
\]

- **Low variance** → structure unclear → **τ increases** → broad attention  
- **High variance** → structure clear → **τ decreases** → focused attention  

This provides a **continuous, interpretable phase shift** in attention.

---

## Why This Matters

| Situation | Fixed Attention | Dynamic Sparse Attention |
|---|---|---|
| Uncertain / noisy input | Over-focus too early | Maintains exploration |
| Clear structure emerges | Remains diffuse | Automatically sharpens |
| Robustness | Sensitive to scaling | Stable across conditions |
| Agent / LLM reasoning | No mode shift | Supports search → commit dynamics |

This mechanism is useful for:
- **Large Language Models**
- **Vision transformers**
- **Reinforcement-learning agents / planning models**

---

## Files
