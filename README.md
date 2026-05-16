# running-the-table

Pool term for clearing all balls without a miss. Also: running the simulation. Also: what hyperfocus feels like. Same algorithm, three substrates: a pool table, a neural network, and you.

---

## What This Is

A three-part written work plus an interactive reinforcement-learning demo that maps **ADHD neuroscience** onto **Q-learning** onto **the physics of a pool table** — and argues they are the same loop:

> belief → guess → move → reality → surprise → sharper belief

The thesis: your brain is a prediction engine. Dopamine is the reward prediction error signal. Narrative is the value function. Personality is deepened forest trails (neural paths carved by repetition). The quality of your life depends on the quality of the simulation running silently underneath your behavior — and on whether the engine ever learns to notice when its own training data has run out.

---

## The Pieces

### `blog.md` — Calling the Shot

The flagship essay. ~25,000 words. Starts with a Staff Engineer playing pool alone at 3 AM and unfolds into:

- **INCU** (Interest, Novelty, Challenge, Urgency) as the neurochemical spec sheet for the ADHD nervous system
- **Q-learning** as the formal name for how expertise is built — in code, in pool, in life
- **Reward Prediction Error** as the mechanism behind dopamine, flow, and the "prototyper's high"
- **Hallucination** as the universal failure mode when any prediction engine leaves its training data
- **Appendices** that scale the same pattern to nations, religions, generational trauma, executive dysfunction, and the closed-model trap

Key mapping (the Rosetta Stone of the text):

| Human Experience | CS Vocabulary | Neuroscience |
|---|---|---|
| Narrative / Story | Value Function | Predictive Coding Model |
| Dopamine spike | Reward Prediction Error (RPE) | δ signal (Schultz) |
| Personality / Instinct | Deepened Q-values (argmax) | Myelinated neural pathways |
| "Gut feeling" | Policy π(s) | Compiled experience |
| Confident hallucination | Out-of-distribution extrapolation | Confabulation |
| Practice | Bellman updates | Neuroplasticity |
| Pool shot | Function call with 12 parameters | Motor planning + prediction |

### `blog2.md` — Training Data

The prequel. ~8,000 words. About the **stories you tell yourself** — not as self-help, but as literal predictive models stored in the nervous system:

- Where your stories came from (training distribution: parents, school, culture, biology)
- The confirmation engine (stories are self-sealing)
- Stories as strategy (narrative prescribes behavior, not just describes it)
- Why external success never dislodges a core limiting belief

### `blog3.md` — The Dopamine Narrator

The sequel. ~10,000 words. Specific to the **ADHD brain as a story-machine**:

- The interest-based nervous system as a narrative machine (story-gate vs. will-gate)
- The five-step attention loop (scan → register → collapse → sustain → transition)
- The shame narrative as the most expensive story an ADHD adult runs
- Dopamine debt and the crash cycle
- Practical framework for architecting narrative dopamine

### `demo.py` — The Interactive Proof

A single-file FastAPI app that brings the theory to life. A Q-learning agent learns a hidden pool-table geometry while you watch:

- **Act I** — Blank slate (Q-table initialized to zero)
- **Act II** — The Dopamine Loop (600 training episodes, RPE pulses visible)
- **Act III** — When the Map Becomes the Voice (policy crystallizes into "instinct")
- **Act IV** — Hallucination at the Edge (out-of-distribution queries answered with full confidence)
- **Act V** — Player B Walks Up (same algorithm, different training data, opposite conclusions)
- **Epilogue** — Carrying the loop back into your actual life

Run it:

```bash
pip install fastapi uvicorn
python demo.py
# opens http://127.0.0.1:8765
```

Environment toggles:
- `FAST_MODE=1` — skip dramatic pauses
- `NO_OPEN=1` — don't auto-open browser
- `PORT=8765` — server port (default)

### `bg.png`

The shared header image used across all three essays.

---

## Reading Order

There is no wrong order, but the intended arc:

1. **blog2.md** (Training Data) — understand what stories are, mechanically
2. **blog3.md** (The Dopamine Narrator) — understand how the ADHD brain runs on them
3. **blog.md** (Calling the Shot) — the full synthesis: neuroscience + CS + pool + life
4. **demo.py** — watch the math prove itself in 60 seconds

Or just start with `demo.py` if you want to *feel* the thesis before you read it.

---

## Who This Is For

- The neurodivergent adult who has been told they have "so much potential, if only..."
- The engineer who ships impossible prototypes and then drowns in maintenance
- The ML researcher who forgot the Bellman equation describes her own brain
- The parent of an ADHD kid trying to understand why the homework didn't get done
- The person who plays pool alone at 2 AM and has never quite known why it works
- Anyone who has ever been completely wrong with total certainty

---

## Core Vocabulary

| Term | What it means here |
|---|---|
| **INCU** | Interest, Novelty, Challenge, Urgency — the four dopamine triggers for the ADHD nervous system (Dodson) |
| **Q-table** | The internal map of "in states like this, actions like that tend to produce rewards like this much" |
| **RPE (δ)** | Reward Prediction Error — the only signal that ever updates a belief |
| **Training distribution** | The experiences that shaped your priors — childhood, culture, biology, luck |
| **Out-of-distribution** | When reality asks a question your model has no data for |
| **Hallucination** | What every prediction engine does outside its training data: answers confidently from the nearest neighbor |
| **Story-gate** | The ADHD brain's activation mechanism — pass through it with narrative, or the engine won't start |
| **The closed-model trap** | When a mind treats contradicting signals as noise instead of information |

---

## License

MIT
