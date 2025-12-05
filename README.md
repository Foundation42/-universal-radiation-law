# Exploring Radiation Scaling Through an Information-Theoretic Lens

## A Computational Exploration (Not a New Theory)

*December 2025*

*Collaboration: Christian Beaumont (Human), Claude Code, Claude Chat, DeepSeek (AI)*

---

## What This Is

This repository contains simulations exploring radiation scaling laws (ω², ω⁴, ω⁶) through the lens of a thought experiment:

> *"What if the vacuum has informational viscosity — a resistance to changes it can't track smoothly?"*

After sharing this work online, physicists correctly pointed out that we **overclaimed** what we'd found. This README is an honest reassessment.

**What we did:** Built simulations that reproduce known physics and wrapped them in an information-theoretic narrative.

**What we didn't do:** Derive new physics or make testable predictions that differ from established theory.

**What remains open:** Whether the "information" framing offers genuine conceptual insight, or is just a rebranding of existing physics.

---

## The Thought Experiment

The original idea:

1. The vacuum carries information about field configurations
2. It has a maximum update rate (the speed of light)
3. When matter changes state faster than the vacuum can track, "friction" occurs
4. Different types of fields require different types of tracking → different scaling laws

This *predicts* (or rather, *is consistent with*) the known hierarchy:

| Field Type | What it "tracks" | Radiation Scaling |
|------------|------------------|-------------------|
| Scalar | Position | ω² |
| Vector (EM) | Phase/connection | ω⁴ |
| Tensor (Gravity) | Curvature | ω⁶ |

---

## What The Simulations Show

### 1. The Hierarchy is Reproduced

Our simulations correctly reproduce:
- Velocity coupling → ω²
- Acceleration coupling → ω⁴
- Jerk coupling → ω⁶

**Caveat:** We built these couplings in. The simulations confirm consistency, not discovery.

### 2. Finite-Size Effects Are Real

When we varied charge size (σ), the measured exponent smoothly transitioned:

| Charge Size σ | Measured Exponent |
|---------------|-------------------|
| 0.80 (smeared) | 2.34 |
| 0.50 | 3.02 |
| 0.25 | 3.77 |
| 0.18 (point-like) | 4.03 |

This is **known physics** — finite-size charge distributions don't radiate like point charges. But it's satisfying that the simulations capture it correctly.

### 3. The GPU Maxwell Solver Works

The self-consistent 3D electromagnetic simulation (`self_consistent_em.py`) correctly evolves fields according to Maxwell's equations. This is good code, even if it's not new physics.

---

## Honest Assessment

### What Was Overclaimed

- Calling this a "Universal Law of Radiation" — it's a reframing, not a new law
- Implying we derived the hierarchy — we assumed couplings that produce it
- The "information" language — not rigorously defined

### What's Still Interesting

- The thought experiment *is consistent* with known physics — that's not nothing
- The intuition "vacuum resists changes it can't track" gives a mental model for *why* ω⁴ and ω⁶ arise
- The simulations correctly reproduce real scaling behavior
- The σ → 0 limit recovering point-charge physics is pedagogically valuable

### What Would Make This Real Physics

1. **Rigorously define "information"** — Shannon entropy? Fisher information? Holographic bounds?
2. **Derive the couplings** from first principles rather than assuming them
3. **Make predictions that differ from standard physics** — and propose experiments
4. **Dimensional consistency** throughout (the original had errors)
5. **Connect to QFT/renormalization** — does this framing illuminate anything new?

---

## The Lesson Learned

It's easy to:
- Build simulations that confirm what you're looking for
- Wrap results in compelling narrative
- Convince yourself (and AI collaborators) of profundity

The Reddit physics community correctly pushed back on overclaiming. But they also noted: consistency with known physics isn't worthless — it's just not the same as new physics.

> *"The first principle is that you must not fool yourself — and you are the easiest person to fool."*
> — Richard Feynman

---

## What You Can Learn From This Repo

### Working Code

- **`self_consistent_em.py`** — GPU-accelerated 3D Maxwell solver (Taichi)
- **`gaussian_theory_test.py`** — Finite-size charge effects on radiation
- **`radiation_reaction.py`** — Power-law scaling measurements
- **`universality_test.py`** — Comparison across coupling types

### Pedagogical Value

- How to set up FDTD electromagnetic simulations
- Power-law fitting and log-log analysis
- GPU acceleration with Taichi
- Visualization of field dynamics

### A Cautionary Tale

- How easy it is to overclaim computational results
- The difference between consistency and derivation
- Why peer review (even harsh Reddit comments) matters

---

## Files

### Simulations
| File | Description |
|------|-------------|
| `corrugated_vacuum_sim.py` | Original toy model |
| `self_consistent_em.py` | GPU Maxwell solver |
| `gaussian_theory_test.py` | Charge size effects |
| `radiation_reaction.py` | Scaling law tests |
| `universality_test.py` | Multi-coupling comparison |

### Documentation
| File | Description |
|------|-------------|
| `chat.md` | Full conversation log |
| `NextSteps.md` | Original research directions (overly ambitious) |

### Visualizations
Various `.png` and `.gif` files showing simulation outputs.

---

## Status

**The thought experiment:** Still interesting, not validated or invalidated

**The simulations:** Working code, reproduces known physics

**The claims:** Walked back from "universal law" to "pedagogical exploration"

**The journey:** Worth documenting honestly

---

## Acknowledgments

- **Reddit r/LLMPhysics** — for the reality check
- **Claude, DeepSeek** — useful for coding, not for physics validation
- **The process** — sometimes the lesson is about intellectual honesty

---

*"Making simulations go brrrr on a GPU isn't the same as doing physics. But it's a start."*
