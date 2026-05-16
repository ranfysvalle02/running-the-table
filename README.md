# Running the Table

Pool term for clearing all balls without a miss. Also: running the simulation. Also: what hyperfocus feels like. Same algorithm, three substrates: a pool table, a neural network, and you.

---

You are a prediction engine. So is everyone else. So is every culture, every religion, every LLM humming on a server somewhere. The quality of your life depends on the quality of the simulation running silently underneath your behavior — and on whether the engine ever learns to notice when its own training data has run out.

This project is a written series and an interactive demo that traces one idea across neuroscience, computer science, pool physics, and the stories you've been telling yourself since childhood:

> **belief → guess → move → reality → surprise → sharper belief**

It starts with the invisible stories running your life — where they came from, why they feel like facts instead of hypotheses, how they compound. Then it looks at the ADHD brain specifically: a high-threshold narrative engine that produces extraordinary output on the right fuel and extraordinary paralysis on the wrong one. Then it pulls everything together at a pool table at 3 AM — mapping dopamine onto reward prediction error, personality onto deepened Q-values, hyperfocus onto full narrative capture, and confident hallucination onto out-of-distribution extrapolation. And then it zooms all the way out: the scientific method, humanity's long history of getting less wrong, and why the story you tell yourself about what's possible is the initial condition of every experiment you'll ever run.

The demo lets you watch it happen in real time. A Q-learning agent discovers pool-table physics from scratch while its dopamine signal pulses, its instincts crystallize, and — the moment you push it past its training data — it hallucinates with full confidence. Same loop as your brain. Same failure mode too.

---

## Run the Demo

```bash
pip install fastapi uvicorn
python demo.py
# opens http://127.0.0.1:8765
```

Environment toggles:
- `FAST_MODE=1` — skip dramatic pauses
- `NO_OPEN=1` — don't auto-open browser
- `PORT=8765` — server port (default)

---

## License

MIT
