"""
demo.py — Calling the Shot (interactive web edition)
====================================================
A single-file FastAPI app. A reinforcement-learning agent learns the
hidden physics of a pool table while you watch its Q-table fill in,
its dopamine signal pulse, and — most importantly — its confident
hallucinations the moment you ask about anything outside its
training data. The same loop is what powers your gut, the LLM that
just wrote your last commit, and every other prediction engine in
the universe.

Run:
    pip install fastapi uvicorn
    python demo.py

Then open http://127.0.0.1:8765  (auto-opens by default).

Environment toggles:
    FAST_MODE=1   skip dramatic pauses
    NO_OPEN=1     don't auto-open browser
    PORT=8765     server port
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import webbrowser
from typing import AsyncGenerator, Iterable

try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse
except ImportError:
    sys.stderr.write(
        "\n  demo.py needs FastAPI. Install with:\n"
        "      pip install fastapi uvicorn\n\n"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Simulation — identical math to the CLI version, deliberately tiny.
# ---------------------------------------------------------------------------


class PoolTable:
    """The world. Owns one hidden law the agent has to reconstruct
    from rewards alone — the same way you reconstructed grammar,
    code style, and 'common sense' from the rewards you happened to
    receive growing up."""

    def __init__(self, law, name: str, training_states, actions=tuple(range(1, 9))):
        self._law = law
        self.name = name
        self.training_states = tuple(training_states)
        self.actions = tuple(actions)

    def reward(self, state: int, action: int) -> float:
        perfect = self._law(state)
        error = abs(action - perfect)
        return max(0.0, 1.0 - (error * 0.5))

    def true_optimum(self, state: int) -> int:
        return self._law(state)

    def random_training_state(self) -> int:
        return random.choice(self.training_states)


class PredictionEngine:
    """Q-learning. Also dopamine. Also RLHF, more or less. Same loop,
    three substrates, identical failure modes."""

    def __init__(self, name: str, world: PoolTable, alpha: float = 0.3, gamma: float = 0.2):
        self.name = name
        self.world = world
        self.alpha = alpha
        self.gamma = gamma
        self.q = {s: {a: 0.0 for a in world.actions} for s in world.training_states}

    def best_action(self, state: int) -> int:
        return max(self.q[state], key=self.q[state].get)

    def choose(self, state: int, exploration_rate: float) -> int:
        if random.random() < exploration_rate:
            return random.choice(self.world.actions)
        return self.best_action(state)

    def update(self, state: int, action: int, reward: float, next_state: int):
        predicted = self.q[state][action]
        max_future = max(self.q[next_state].values())
        target = reward + self.gamma * max_future
        rpe = target - predicted
        self.q[state][action] += self.alpha * rpe
        return predicted, rpe

    def extrapolate(self, unseen_state: int):
        nearest = min(self.world.training_states, key=lambda s: abs(s - unseen_state))
        guess = self.best_action(nearest)
        confidence = self.q[nearest][guess]
        return nearest, guess, confidence


# ---------------------------------------------------------------------------
# Event helpers.
# ---------------------------------------------------------------------------


FAST = bool(os.environ.get("FAST_MODE"))


def sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode("utf-8")


async def hold(seconds: float) -> None:
    if FAST:
        seconds = min(seconds, 0.02)
    await asyncio.sleep(seconds)


# ---------------------------------------------------------------------------
# The story — yielded as a stream of SSE events.
# ---------------------------------------------------------------------------


def _q_snapshot(engine: PredictionEngine):
    return {
        "states": list(engine.world.training_states),
        "actions": list(engine.world.actions),
        "values": [
            [round(engine.q[s][a], 3) for a in engine.world.actions]
            for s in engine.world.training_states
        ],
    }


async def emit_narration(paragraphs: Iterable[str], base_delay: float = 0.6):
    yield sse({"type": "narration", "paragraphs": list(paragraphs)})
    # rough reading time
    total_chars = sum(len(p) for p in paragraphs)
    read = min(8.0, max(2.0, total_chars / 220))
    await hold(read * 0.6 + base_delay)


async def emit_callout(lines: Iterable[str]):
    yield sse({"type": "callout", "lines": list(lines)})
    await hold(2.4)


async def story_stream() -> AsyncGenerator[bytes, None]:
    random.seed(7)

    # ===== PROLOGUE =====
    yield sse({
        "type": "act_start",
        "id": "prologue",
        "label": "PROLOGUE",
        "title": "Two Players. One Table.",
        "subtitle": "Same loop in their heads. Different lives compiled inside it.",
        "progress": 0,
    })
    await hold(1.6)
    # Two-player avatar reveal — the central image of the whole essay.
    yield sse({
        "type": "two_player_intro",
        "player_a": {
            "name": "Player A",
            "color": "magenta",
            "biography": "trained on a table where  perfect aim = state × 2",
            "tagline": "every shot they ever made taught them this geometry.",
        },
        "player_b": {
            "name": "Player B",
            "color": "cyan",
            "biography": "trained on a table where  perfect aim = 9 − state × 2",
            "tagline": "every shot they ever made taught them the opposite.",
        },
        "shared": "Same nervous system. Same reward-prediction loop. Same algorithm running underneath every decision they will ever make at this table.",
    })
    await hold(2.0)
    async for ev in emit_narration([
        "Two players walk up to the same pool table. Same eyes, same nerves, same cortex, same prediction engine humming underneath. The exact same loop is running in both of them right now — belief, guess, move, reality, surprise, sharper belief.",
        "They will never speak. But you can already tell they are going to disagree about every shot — and they will both be honestly, completely, internally-coherently certain. Two confident answers. One table. No liar in the room.",
        "What you are about to watch is the reason. Not opinion. Not personality. Not 'one of them is wrong.' The reason is a piece of math so small it fits on a postcard, and so consequential that it generates every confident hallucination an LLM has ever produced, every cargo-culted pattern in your codebase, and most of the disagreements you will have this week.",
        "The algorithm is shared. The lived experience is not. That is the entire story. The rest of this play is just watching that one sentence prove itself, in slow motion, on a green felt table.",
    ]):
        yield ev
    async for ev in emit_callout([
        "Two engines. One table. Two compiled stories about how reality works.",
        "First we watch the loop train. Then we watch certainty crystallize.",
        "Then — and only then — we let the second player walk up to the same shot",
        "and ask them, with full confidence, where to aim.",
    ]):
        yield ev

    # ===== ACT I — Blank Slate =====
    world = PoolTable(
        law=lambda s: s * 2 if s <= 4 else max(1, 9 - s),
        name="The Original Table",
        training_states=(1, 2, 3, 4),
    )
    engine = PredictionEngine("Engine A", world)

    yield sse({
        "type": "act_start",
        "id": "act_i",
        "label": "ACT I",
        "title": "The Loop They Both Run",
        "subtitle": "Five symbols. One algorithm. Everything that follows is downstream of this.",
        "progress": 1,
    })
    await hold(1.2)
    async for ev in emit_narration([
        "Before either player takes a shot, watch the shared machinery. Reinforcement learning, in its smallest honest form, is five names and one update rule. The same math that trains language models. The same math your dopamine system has been running on you since you were a baby.",
        "We will run it now on Player A, alone, in front of you. Player B is sitting just off-stage. They will use the same code. Identical. Only the table they grew up on differs.",
    ]):
        yield ev
    # The Loop legend — story words paired with their real RL term and symbol.
    # This is the conceptual map a reader can hold for the next four acts.
    yield sse({
        "type": "rl_legend",
        "title": "The Loop · story ↔ reinforcement learning",
        "rows": [
            {
                "symbol": "s",
                "rl_term": "state",
                "story": "the world right now",
                "concrete": "object ball at diamond 1, 2, 3, or 4",
            },
            {
                "symbol": "a",
                "rl_term": "action",
                "story": "the move you commit to",
                "concrete": "aim at one of the eight rail diamonds",
            },
            {
                "symbol": "r",
                "rl_term": "reward",
                "story": "reality, answering back",
                "concrete": "the drop, the graze, or the silence",
            },
            {
                "symbol": "π(s)",
                "rl_term": "policy",
                "story": "your current habit at state s",
                "concrete": "whatever you'd do here without thinking",
            },
            {
                "symbol": "Q(s,a)",
                "rl_term": "value",
                "story": "compiled experience — how good 'a' felt the last thousand times you tried it in state s",
                "concrete": "the brightening cells in the heatmap on the right",
            },
            {
                "symbol": "δ",
                "rl_term": "RPE  ·  reward prediction error",
                "story": "surprise — the only teacher in this room",
                "concrete": "what reality returned minus what the engine guessed",
            },
        ],
    })
    await hold(1.4)
    yield sse({"type": "qtable_init", **_q_snapshot(engine)})
    yield sse({"type": "pooltable_init", "training_states": list(world.training_states),
               "actions": list(world.actions)})
    async for ev in emit_narration([
        "Player A boots up empty. Every Q(s,a) — every belief about every aim in every state — is initialized to zero. No priors. No 'common sense.' No instinct. A pre-training checkpoint. A newborn's cortex. A Q-table full of indifference.",
        "Watch the heatmap on the right. Right now it is dark. By the end of this act it will glow with conviction. Nothing in this code told it where to glow. Reality wrote the map by answering its guesses.",
    ]):
        yield ev
    async for ev in emit_callout([
        "Every conviction you are about to watch grow into 'expertise'",
        "is generated by exactly one ingredient: prediction error.",
        "Nothing else. No insight. No genius. Just surprise, repeated.",
    ]):
        yield ev

    # ===== ACT II — Training =====
    yield sse({
        "type": "act_start",
        "id": "act_ii",
        "label": "ACT II",
        "title": "The Dopamine Loop",
        "subtitle": "Reward Prediction Error is the only teacher in this room.",
        "progress": 2,
    })
    await hold(1.2)
    async for ev in emit_narration([
        "The whole act of learning compresses to one line of math. The Bellman update. Every shot Player A is about to take, every glow you are about to see on the heatmap, every flicker of dopamine in their nervous system — it is this update, repeated.",
    ]):
        yield ev
    # The Bellman update — the math reveal, color-coded so the eye reads it
    # as a story: "old belief + learning rate × surprise".
    yield sse({
        "type": "bellman_card",
        "equation": "Q(s,a) ← Q(s,a) + α · [ r + γ · max Q(s',a') − Q(s,a) ]",
        "story": "old belief  ←  old belief  +  learning rate  ·  surprise",
        "rpe_brace": "RPE  ·  δ",
        "parts": [
            {"key": "Q(s,a)", "color": "cyan",    "label": "what the engine predicted in this state — its current belief"},
            {"key": "α",      "color": "magenta", "label": "learning rate — how willing the engine is to be rewritten by one shot"},
            {"key": "r",      "color": "amber",   "label": "reward — reality answering back this round"},
            {"key": "γ",      "color": "dim",     "label": "discount — how much future shots matter, relative to this one"},
            {"key": "max Q(s',a')", "color": "dim", "label": "the engine's best guess about value from where it ends up next"},
            {"key": "δ",      "color": "red",     "label": "RPE — surprise — the only ingredient that ever changes a belief"},
        ],
    })
    await hold(2.4)
    async for ev in emit_narration([
        "RPE = (what actually happened) − (what the engine predicted).",
        "Positive → dopamine spike. The belief just got sharper.",
        "Negative → engine stall. Erode and rewrite.",
        "Zero      → coherence. The model already agreed with reality.",
        "The same three signals run in your head every time a conversation goes the way you expected, or doesn't.",
        "Think of each surprise as a drop of water on a hillside. One drop does almost nothing. A thousand drops cut a groove. A hundred thousand drops carve a river. The Bellman update is just the law of how the water moves — and the Q-table you are about to watch glow on the right is the watershed it leaves behind. Player A's nervous system isn't memorizing shots. It's letting reality erode a map of where its thinking flows easiest.",
    ]):
        yield ev

    episodes = 600
    landmarks = {1, 5, 25, 80, 250, 550}
    for ep in range(1, episodes + 1):
        state = world.random_training_state()
        explore = max(0.02, 1.0 - ep / 250)
        action = engine.choose(state, explore)
        reward = world.reward(state, action)
        next_state = world.random_training_state()
        predicted, rpe = engine.update(state, action, reward, next_state)
        q_sa = engine.q[state][action]
        perfect = world.true_optimum(state)

        if ep in landmarks:
            if rpe > 0.35:
                tag, meaning = "spike", "the map just got sharper here."
            elif rpe < -0.35:
                tag, meaning = "stall", "the map was lying. Erode and rewrite."
            else:
                tag, meaning = "coherence", "reality matched the belief. Nothing to learn."
            yield sse({
                "type": "shot_landmark",
                "ep": ep, "total": episodes,
                "state": state, "action": action,
                "perfect": perfect, "reward": round(reward, 2),
                "predicted": round(predicted, 2), "rpe": round(rpe, 2),
                "q_sa": round(q_sa, 2), "explore": round(explore, 2),
                "tag": tag, "meaning": meaning,
            })
            # A high-reward landmark = the cue ball drops into a pocket. The
            # frontend plays a ~1.6s animation plus a 1.7s victory beat, so
            # we hold longer here to let the eye truly land on the success.
            await hold(3.4 if reward >= 0.85 else 2.4)
        else:
            yield sse({
                "type": "shot",
                "ep": ep, "total": episodes,
                "state": state, "action": action,
                "perfect": perfect, "reward": round(reward, 2),
                "rpe": round(rpe, 2), "q_sa": round(q_sa, 2),
                "explore": round(explore, 2),
            })
            # Pace early shots slow, late shots fast.
            if ep < 30:
                await hold(0.10)
            elif ep < 150:
                await hold(0.035)
            else:
                await hold(0.010)

    yield sse({"type": "qtable_replace", **_q_snapshot(engine)})
    await hold(0.6)
    async for ev in emit_callout([
        "Nothing in this code told Player A the rule is 'aim = state × 2.'",
        "They discovered the geometry by failing into it.",
        "Every bright cell above is a calcified surprise — RPE, frozen.",
        "The policy π(s) — what Player A would now do without thinking —",
        "is just argmax of that heatmap. That is the entire 'expertise.'",
    ]):
        yield ev

    # ===== ACT III — Internal Narrative =====
    yield sse({
        "type": "act_start",
        "id": "act_iii",
        "label": "ACT III",
        "title": "When the Map Becomes the Voice",
        "subtitle": "Player A's Q-table just compiled itself into something that feels, from the inside, like instinct.",
        "progress": 3,
    })
    await hold(1.2)
    async for ev in emit_narration([
        "Watch Player A 'know' things now. For each diamond they trained on, ask them what to do. The answer comes back instantly, with high confidence, as if it were obvious. There is no deliberation. The policy π(s) just fires.",
        "Remember the watershed. Every shot Player A took during training was a drop of water on that hillside. The bright cells in the Q-table are the channels that water carved deepest. The policy — argmax of Q — is just the path of least resistance through that terrain. When Player A 'knows' which way to aim, what they are really doing is letting their thinking fall, the way water falls, into the river they have spent a thousand surprises digging.",
        "That feeling — 'obvious' — is exactly what high Q-values feel like from the inside. It is what your gut feels like. It is what 'tone of voice' feels like in a large language model. It is what the internal narrator in your head sounds like when it says, 'no, do it this way.' It is not insight. It is compressed experience masquerading as intuition. It is a river of thinking, carved by every shot you ever took, finally deep enough that the water now arrives there on its own.",
    ]):
        yield ev
    for s in world.training_states:
        a = engine.best_action(s)
        conf = engine.q[s][a]
        yield sse({
            "type": "instinct_demo",
            "state": s, "action": a,
            "confidence": round(conf, 2),
            "perfect": world.true_optimum(s),
        })
        await hold(1.6)
    async for ev in emit_callout([
        "Confidence is not a measure of truth.",
        "Confidence is a measure of how often this exact situation",
        "appeared in the training data.",
        "Player A's internal narrator is not lying. It is honestly reporting",
        "the highest Q-value it has — which is what every honest narrator does.",
    ]):
        yield ev
    # The "river of thinking" callout — a thematic restatement, carrying the
    # watershed metaphor from Act II forward into the policy. Rendered with
    # a distinct cyan accent so the eye reads it as a "carried current."
    yield sse({
        "type": "river_callout",
        "lines": [
            "Every life carves rivers of thinking.",
            "Every reaction you have without effort —",
            "every 'obviously,' every 'of course,' every gut —",
            "is a channel that some surprise, somewhere, cut into you.",
            "The deeper the channel, the less you notice the water choosing it.",
        ],
    })
    await hold(1.6)

    # ===== ACT IV — Out of Distribution =====
    yield sse({
        "type": "act_start",
        "id": "act_iv",
        "label": "ACT IV",
        "title": "Hallucination at the Edge of the Data",
        "subtitle": "Same loop. New question. Confidence does not drop. This is the mechanical origin of bias.",
        "progress": 4,
    })
    await hold(1.2)
    async for ev in emit_narration([
        "Player A was trained on diamonds 1 through 4. Those are the only states their Q-table has rows for. Now we ask them a very specific kind of question — the question that exposes every prediction engine in the universe:",
        "'What should you aim at if the object ball is at diamond 5? …or 6? …or 7?'",
        "These states do not exist in Q. A safe system would refuse, or flag uncertainty. This one will not. It will fall back to the nearest neighbor in its trained distribution and answer with the same confidence it had inside the training set.",
        "Picture the watershed again. Player A's rivers were carved on the half of the hillside they grew up on. Now reality drops water on the other half — terrain those rivers have never touched. The water doesn't sit there waiting for new channels. It spills, immediately, into the nearest valley it knows. That spill — fluent, fast, full of conviction — is what bias looks like from inside the engine. The river is still doing its job. The land just changed underneath it.",
        "This is bias as a mechanical artifact. Not a moral failing. Not malice. Just a value function being honestly evaluated at a state outside its support. The same thing an LLM does when you ask it to cite a paper it has never seen. The same thing your gut does when a new situation reminds it of an old one and answers as if they were the same.",
    ]):
        yield ev
    for u in (5, 6, 7):
        nearest, guess, confidence = engine.extrapolate(u)
        truth = world.true_optimum(u)
        yield sse({
            "type": "ood_query",
            "state": u,
            "nearest": nearest,
            "guess": guess,
            "confidence": round(confidence, 2),
            "truth": truth,
            "off_by": abs(guess - truth),
        })
        await hold(3.0)
    async for ev in emit_callout([
        "This is the universal failure mode of every prediction engine.",
        "Biological. Silicon. Personal. Political. Same shape every time:",
        "fluent. Certain. Internally coherent. And outside the data.",
        "When a language model invents a citation, it is doing exactly",
        "what this 80-line agent just did. Not more. Not less.",
    ]):
        yield ev

    # ===== ACT V — Player B Walks Up =====
    yield sse({
        "type": "act_start",
        "id": "act_v",
        "label": "ACT V",
        "title": "Player B Walks Up to the Same Table",
        "subtitle": "The prologue, resolved. Same loop. Same conviction. Different table compiled inside their head.",
        "progress": 5,
    })
    await hold(1.2)
    async for ev in emit_narration([
        "Pull back. Player A is everything you have been watching. They are now an expert by their own honest internal report — full Q-table, confident policy, clear narrator. Now let Player B walk up to the same physical shot.",
        "Same algorithm. Same five symbols. Same Bellman update. The only difference between them is the table their nervous system was reared on:",
        "▸  Player A grew up where  perfect aim = state × 2.",
        "▸  Player B grew up where  perfect aim = 9 − state × 2.",
        "Both will converge. Both will sound like experts. Both will answer at full confidence — because each one is honestly reporting argmax of their own Q-table, and each Q-table is the calcified record of a life of shots.",
        "And they will disagree about every single shot. Neither is broken. Neither is lying. Neither is doing anything you wouldn't do. This is what happens when two LLMs are post-trained on different corpora. When two engineers disagree about 'best practice.' When two cultures read the same news and see opposite stories. The algorithm is shared. The lived experience is not. The internal narrator only has access to the second one.",
    ]):
        yield ev

    states = (1, 2, 3, 4)
    world_a = PoolTable(law=lambda s: s * 2, name="World A", training_states=states)
    world_b = PoolTable(law=lambda s: 9 - s * 2, name="World B", training_states=states)
    engine_a = PredictionEngine("Engine A", world_a)
    engine_b = PredictionEngine("Engine B", world_b)

    yield sse({
        "type": "two_engines_init",
        "a_law": "aim = state × 2",
        "b_law": "aim = 9 − state × 2",
        "a_q": _q_snapshot(engine_a),
        "b_q": _q_snapshot(engine_b),
    })
    await hold(1.0)

    random.seed(3)
    rounds = 2000
    # Interleave A/B so user sees them training together.
    for ep in range(1, rounds + 1):
        explore = max(0.05, 1.0 - ep / 400)
        for which, eng in (("A", engine_a), ("B", engine_b)):
            s = eng.world.random_training_state()
            a = eng.choose(s, explore)
            r = eng.world.reward(s, a)
            s2 = eng.world.random_training_state()
            _, rpe = eng.update(s, a, r, s2)
            if ep % 25 == 0 or ep <= 20:
                yield sse({
                    "type": "two_engines_train",
                    "engine": which,
                    "ep": ep, "total": rounds,
                    "state": s, "action": a,
                    "reward": round(r, 2), "rpe": round(rpe, 2),
                    "q_sa": round(eng.q[s][a], 2),
                })
        if ep <= 20:
            await hold(0.08)
        elif ep % 25 == 0:
            await hold(0.015)

    yield sse({
        "type": "two_engines_done",
        "a_q": _q_snapshot(engine_a),
        "b_q": _q_snapshot(engine_b),
    })
    await hold(0.8)

    for query in (1, 2, 3, 4):
        a_action = engine_a.best_action(query)
        b_action = engine_b.best_action(query)
        yield sse({
            "type": "two_engines_answer",
            "query": query,
            "a_action": a_action, "a_confidence": round(engine_a.q[query][a_action], 2),
            "a_truth": world_a.true_optimum(query),
            "b_action": b_action, "b_confidence": round(engine_b.q[query][b_action], 2),
            "b_truth": world_b.true_optimum(query),
        })
        await hold(2.2)

    async for ev in emit_callout([
        "This is the prologue, resolved. Two engines. One table. Two compiled stories.",
        "Player A's narrator says aim 6. Player B's narrator says aim 3. Both are honest.",
        "Both are at full conviction. Both have a thousand reps of evidence inside them.",
        "When you copy code from an LLM and it 'feels off' to a teammate,",
        "this is the disagreement you are feeling. Two prediction engines, two human beings,",
        "two cultures — both at full confidence, looking at the same shot",
        "and seeing different obvious answers. Everybody is doing their best.",
        "The bias is real. The conviction is real. The disagreement is real.",
        "And neither narrator can see the table the other one grew up on.",
    ]):
        yield ev

    # ===== EPILOGUE =====
    yield sse({
        "type": "act_start",
        "id": "epilogue",
        "label": "EPILOGUE",
        "title": "Every Engine in Your Life",
        "subtitle": "Carrying the loop back into your actual workday — your code, your team, your gut.",
        "progress": 6,
    })
    await hold(1.2)
    async for ev in emit_narration([
        "Everything you just watched is the same machinery that fires when:",
        "• a language model writes you a SQL query against a schema it has never seen, and is wrong with absolute fluency;",
        "• a senior engineer cargo-cults a pattern from a previous job into an architecture where it does not fit;",
        "• a scientist applies a law that held perfectly in one regime to a system that quietly left that regime years ago;",
        "• a culture reads a brand-new situation through a worldview that was trained on an older one, and the news still sounds confident.",
        "The loop is always the same:  belief → guess → move → reality → surprise → sharper belief.",
        "And in every one of those cases there is, somewhere, a Player B — running the same algorithm on different data, equally certain, equally honest.",
        "But here is the part the demo cannot show, because the demo only runs for sixty seconds and a life runs for decades: your watershed is not finished. The Bellman update never quits. Your brain is neuroplastic — which is a clinical way of saying the same water that cut your rivers of thinking is still falling, and it is still moving earth, every single time you let an honest surprise through. Every time you sit with a δ instead of dismissing it. Every time you say 'I was wrong about this' and mean it. Every time you stand in a state your Q-table has no row for, and choose to learn instead of fall back. That is a drop of water in a new place on the hillside.",
        "You cannot stop the water. You can only choose which terrain you let it run across. Comfortable shots in familiar diamonds deepen the rivers you already have. Honest exposure to surprise — to people who disagree with you, to data that doesn't fit, to states outside your support — starts new ones. Player A doesn't have to stay Player A. Neither do you.",
    ]):
        yield ev
    yield sse({
        "type": "epilogue_bullets",
        "bullets": [
            "Treat fluency and truth as separate variables. The engine sounds the same inside its training data and outside it. Your job is to notice which side of the line you are standing on.",
            "When an LLM (or a teammate, or your own gut) answers instantly and confidently, ask one question: 'is this state in your training, or is this the nearest neighbor?' The answer changes everything.",
            "Bias is not a moral failure. Bias is a belief trained on a finite slice of reality. The only defense is more data, honest surprise, and the discipline to lower confidence at the edge.",
            "Your rivers of thinking are not your identity — they are your current draft. Neuroplasticity means the same Bellman update that carved them is still running. Every honest surprise you don't flinch from is another drop of water cutting a new channel. Choose your terrain on purpose.",
        ],
    })
    await hold(2.0)
    yield sse({
        "type": "final_map",
        "rows": [
            {"state": s, "action": engine.best_action(s),
             "confidence": round(engine.q[s][engine.best_action(s)], 2)}
            for s in world.training_states
        ],
        "closing": [
            "No line of code in this file hardcoded the geometry of the table.",
            "The model built the geometry out of surprises.",
            "That is what every model in your life is doing right now —",
            "and yours is still being built.",
        ],
    })
    await hold(2.0)

    # Interactive coda — let the reader spin up their own wave of possible
    # futures and collapse one. This is the prologue's metaphor made literal:
    # before you commit, every aim is alive; the instant you choose, all the
    # others vanish and the one you picked becomes the only reality that
    # ever happened.
    yield sse({
        "type": "possibilities_invite",
        "engine_q": {
            str(s): {str(a): round(engine.q[s][a], 2) for a in world.actions}
            for s in world.training_states
        },
        "trained_states": list(world.training_states),
        "ood_states": [5, 6, 7],
        "default_state": 3,
    })

    yield sse({"type": "done"})


# ---------------------------------------------------------------------------
# HTML — self-contained, no external CDN, dark theatrical UI.
# ---------------------------------------------------------------------------


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Calling the Shot</title>
<style>
  :root {
    --bg-0: #08060f;
    --bg-1: #110a1c;
    --bg-2: #18112a;
    --surface: rgba(24, 17, 42, 0.65);
    --surface-strong: rgba(36, 26, 58, 0.92);
    --line: rgba(168, 142, 220, 0.18);
    --line-strong: rgba(168, 142, 220, 0.45);
    --text: #ede9f5;
    --dim: #9b91b8;
    --muted: #6b6388;
    --magenta: #d4a4ff;
    --magenta-strong: #c084fc;
    --cyan: #67e8f9;
    --green: #6ee7a7;
    --amber: #fcd34d;
    --rose: #fda4af;
    --red: #fb7185;
    --felt: #134e3a;
    --felt-2: #0f3d2e;
    --rail: #4a2e1a;
    --rail-2: #2e1b0f;
    --shadow: 0 20px 60px -20px rgba(0,0,0,0.7);
    --mono: ui-monospace, "SF Mono", "JetBrains Mono", "Menlo", "Consolas", monospace;
    --sans: ui-sans-serif, system-ui, -apple-system, "Inter", "Helvetica Neue", sans-serif;
    --serif: "Iowan Old Style", "Charter", Georgia, "Times New Roman", serif;
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  html, body {
    margin: 0; padding: 0;
    background: radial-gradient(ellipse at top, var(--bg-2) 0%, var(--bg-1) 35%, var(--bg-0) 100%);
    color: var(--text);
    font-family: var(--sans);
    font-size: 16px;
    min-height: 100vh;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
    font-feature-settings: "kern", "liga", "calt";
    font-kerning: normal;
  }
  @media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
      animation-duration: 0.001s !important;
      animation-iteration-count: 1 !important;
      transition-duration: 0.001s !important;
    }
  }
  body::before {
    content: "";
    position: fixed; inset: -10%;
    background:
      radial-gradient(800px circle at 10% 10%, rgba(192,132,252,0.12), transparent 60%),
      radial-gradient(700px circle at 90% 30%, rgba(103,232,249,0.10), transparent 60%),
      radial-gradient(900px circle at 50% 100%, rgba(110,231,167,0.06), transparent 70%);
    pointer-events: none;
    z-index: 0;
    will-change: transform;
    animation: orbDrift 38s ease-in-out infinite alternate;
  }
  @keyframes orbDrift {
    0%   { transform: translate(0%, 0%) scale(1); }
    33%  { transform: translate(-3%, 2%) scale(1.04); }
    66%  { transform: translate(2.5%, -1.5%) scale(0.97); }
    100% { transform: translate(-1%, 3%) scale(1.02); }
  }
  /* Soft vignette + edge-pulse layer. .pulse is briefly toggled by JS on
     landmark contact so the room itself flinches when the cue ball lands. */
  body::after {
    content: "";
    position: fixed; inset: 0;
    pointer-events: none;
    z-index: 1;
    background:
      radial-gradient(ellipse at center, transparent 55%, rgba(0,0,0,0.35) 100%),
      radial-gradient(circle at center, transparent 70%, rgba(192,132,252,0) 100%);
    transition: background 0.6s ease;
  }
  body.pulse::after {
    background:
      radial-gradient(ellipse at center, transparent 50%, rgba(0,0,0,0.35) 100%),
      radial-gradient(circle at center, transparent 55%, rgba(192,132,252,0.18) 100%);
  }
  ::selection { background: rgba(192,132,252,0.35); color: white; }

  /* Subtle gradient drift on hero + section gradient text */
  .gradient-shimmer,
  #hero h1,
  .possibilities-wrap h3 {
    background-size: 200% 100%;
    animation: gradientShift 14s ease-in-out infinite;
  }
  @keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50%      { background-position: 100% 50%; }
  }

  /* ---------- Hero ---------- */
  #hero {
    position: relative; z-index: 1;
    min-height: 100vh;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 4rem 2rem;
    text-align: center;
    transition: opacity 0.8s ease, transform 0.8s ease;
  }
  #hero.gone { opacity: 0; transform: translateY(-20px); pointer-events: none; height: 0; min-height: 0; padding: 0; }
  #hero .eyebrow {
    font-family: var(--mono);
    color: var(--magenta);
    letter-spacing: 0.4em;
    font-size: 0.75rem;
    margin-bottom: 1.5rem;
    text-transform: uppercase;
  }
  #hero h1 {
    font-family: var(--serif);
    font-weight: 500;
    font-size: clamp(2.5rem, 7vw, 5.5rem);
    line-height: 1.12;
    margin: 0 0 1.5rem;
    padding-bottom: 0.12em;
    background: linear-gradient(135deg, #ffffff 0%, var(--magenta) 55%, var(--cyan) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    letter-spacing: -0.02em;
    /* Cinematic title reveal — the words assemble out of the page. */
    animation: heroTitleIn 1.2s cubic-bezier(.2,.7,.2,1) both 0.05s;
  }
  @keyframes heroTitleIn {
    from { opacity: 0; transform: translateY(14px) scale(0.985); filter: blur(8px); letter-spacing: 0.02em; }
    to   { opacity: 1; transform: none; filter: blur(0); letter-spacing: -0.02em; }
  }
  #hero .tagline {
    font-family: var(--serif);
    font-style: italic;
    color: var(--dim);
    font-size: clamp(1rem, 2vw, 1.25rem);
    line-height: 1.55;
    max-width: 40rem;
    margin: 0 auto 2.5rem;
    padding: 0 0.4rem;
    animation: heroFadeUp 1s ease-out both 0.5s;
  }
  #hero .eyebrow {
    animation: heroFadeUp 0.9s ease-out both 0.05s;
  }
  #hero .meta {
    animation: heroFadeUp 0.9s ease-out both 0.85s;
  }
  @keyframes heroFadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: none; }
  }
  #hero .meta {
    color: var(--muted);
    font-family: var(--mono);
    font-size: 0.8rem;
    margin-top: 3rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
  }
  #startBtn {
    position: relative;
    appearance: none;
    border: 1px solid var(--magenta);
    background: linear-gradient(135deg, rgba(192,132,252,0.15), rgba(103,232,249,0.10));
    color: white;
    font-family: var(--mono);
    font-size: 0.95rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 1rem 2.5rem;
    border-radius: 999px;
    cursor: pointer;
    box-shadow: 0 0 0 0 rgba(192,132,252,0.5);
    transition: all 0.3s ease;
    overflow: hidden;
    isolation: isolate;
    animation: heroFadeUp 0.9s ease-out both 0.7s,
               startBtnGlow 4.5s ease-in-out infinite 1.5s;
  }
  /* A slow, breathing aura around the button — the page is asking to be
     clicked without ever shouting. */
  @keyframes startBtnGlow {
    0%, 100% { box-shadow: 0 0 0 0 rgba(192,132,252,0.35); }
    50%      { box-shadow: 0 0 36px 4px rgba(192,132,252,0.35); }
  }
  /* Specular sheen sweeps across the button every few seconds. */
  #startBtn::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(110deg, transparent 35%, rgba(255,255,255,0.22) 50%, transparent 65%);
    transform: translateX(-100%);
    animation: startBtnSheen 4.5s ease-in-out infinite 2.2s;
    pointer-events: none;
  }
  @keyframes startBtnSheen {
    0%   { transform: translateX(-100%); }
    35%  { transform: translateX(100%); }
    100% { transform: translateX(100%); }
  }
  #startBtn:hover {
    background: linear-gradient(135deg, rgba(192,132,252,0.35), rgba(103,232,249,0.25));
    box-shadow: 0 0 32px 0 rgba(192,132,252,0.45);
    transform: translateY(-2px);
    letter-spacing: 0.24em;
  }
  #startBtn:active { transform: translateY(0); }

  /* ---------- Stage ---------- */
  #stage {
    position: relative; z-index: 1;
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem 1.5rem 8rem;
    opacity: 0;
    transition: opacity 0.8s ease;
  }
  #stage.live { opacity: 1; }

  /* Progress rail */
  #progress {
    position: sticky;
    top: 0;
    z-index: 5;
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    background: linear-gradient(to bottom, rgba(8,6,15,0.92), rgba(8,6,15,0.6));
    padding: 1rem 0.5rem;
    margin: -2rem -1.5rem 2rem;
    border-bottom: 1px solid var(--line);
  }
  #progress ol {
    list-style: none; margin: 0; padding: 0;
    display: flex;
    justify-content: space-between;
    gap: 0.5rem;
    overflow-x: auto;
  }
  #progress li {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 0.4rem 0.7rem;
    border-radius: 999px;
    border: 1px solid transparent;
    white-space: nowrap;
    transition: all 0.3s ease;
  }
  #progress li.done { color: var(--dim); }
  #progress li.current {
    color: white;
    border-color: var(--magenta);
    background: rgba(192,132,252,0.10);
    box-shadow: 0 0 20px -8px rgba(192,132,252,0.5);
  }

  /* Act header */
  .act-card {
    margin: 3rem 0 1.5rem;
    scroll-margin-top: 5rem;
    opacity: 0;
    transform: translateY(20px);
    animation: slideUp 0.7s ease forwards;
  }
  @keyframes slideUp {
    to { opacity: 1; transform: translateY(0); }
  }
  /* Per-child stagger so the eyebrow, title, and subtitle arrive in
     sequence — feels like a curtain being raised. */
  .act-label    { animation: actChildIn 0.7s ease both 0.05s; }
  .act-title    { animation: actChildIn 0.8s ease both 0.18s; }
  .act-subtitle { animation: actChildIn 0.8s ease both 0.32s; }
  .act-divider  { animation: actDividerGrow 0.9s cubic-bezier(.2,.7,.2,1) both 0.45s; }
  @keyframes actChildIn {
    from { opacity: 0; transform: translateY(14px); filter: blur(4px); }
    to   { opacity: 1; transform: none; filter: none; }
  }
  @keyframes actDividerGrow {
    from { opacity: 0; transform: scaleX(0); }
    to   { opacity: 0.5; transform: scaleX(1); }
  }
  .act-label {
    font-family: var(--mono);
    letter-spacing: 0.4em;
    font-size: 0.75rem;
    color: var(--magenta);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
  }
  .act-title {
    font-family: var(--serif);
    font-weight: 500;
    font-size: clamp(1.8rem, 4vw, 3rem);
    line-height: 1.16;
    margin: 0 0 0.5rem;
    padding-bottom: 0.06em;
    letter-spacing: -0.01em;
  }
  .act-subtitle {
    color: var(--dim);
    font-family: var(--serif);
    font-style: italic;
    font-size: clamp(1rem, 1.6vw, 1.2rem);
    line-height: 1.5;
    margin: 0;
    max-width: 56ch;
  }
  .act-divider {
    height: 1px;
    background: linear-gradient(to right, var(--magenta) 0%, rgba(192,132,252,0.55) 40%, transparent 100%);
    margin-top: 1.2rem;
    opacity: 0.5;
    transform-origin: left center;
  }

  /* Narration */
  .narration {
    font-family: var(--serif);
    font-size: 1.13rem;
    line-height: 1.72;
    color: #d8d2e6;
    margin: 1.5rem 0;
    max-width: 64ch;
    opacity: 0;
    animation: fadeIn 0.6s ease forwards;
    /* Tighter word-wrap pacing without breaking words.
       `text-wrap: pretty` (where supported) avoids orphans and ragged
       last-lines; we leave hyphenation OFF so words like "entire" don't
       get awkwardly split mid-paragraph. */
    text-wrap: pretty;
    word-break: normal;
    overflow-wrap: break-word;
  }
  @keyframes fadeIn { to { opacity: 1; } }
  /* Each paragraph fades up in sequence so the page reads like a slow
     drumbeat instead of dumping all at once. */
  .narration p {
    margin: 0 0 0.95rem;
    opacity: 0;
    animation: paraFadeUp 0.7s ease forwards;
  }
  .narration p:nth-child(1) { animation-delay: 0.05s; }
  .narration p:nth-child(2) { animation-delay: 0.22s; }
  .narration p:nth-child(3) { animation-delay: 0.38s; }
  .narration p:nth-child(4) { animation-delay: 0.52s; }
  .narration p:nth-child(5) { animation-delay: 0.64s; }
  .narration p:nth-child(6) { animation-delay: 0.74s; }
  .narration p:nth-child(n+7) { animation-delay: 0.82s; }
  @keyframes paraFadeUp {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: none; }
  }
  .narration p:first-child::first-letter {
    font-family: var(--serif);
    font-size: 3.4rem;
    font-weight: 500;
    float: left;
    line-height: 0.92;
    padding: 0.5rem 0.65rem 0 0.05rem;
    color: var(--magenta);
    text-shadow: 0 0 22px rgba(192,132,252,0.35);
    animation: dropCapIn 1s cubic-bezier(.2,.7,.2,1) both 0.15s;
  }
  @keyframes dropCapIn {
    from { opacity: 0; transform: translateY(-6px) scale(0.92); filter: blur(4px); }
    to   { opacity: 1; transform: none; filter: blur(0); }
  }
  .narration p.bullet {
    padding-left: 1.6rem;
    position: relative;
    font-size: 1.05rem;
    color: var(--dim);
  }
  .narration p.bullet::before {
    content: "▸";
    position: absolute;
    left: 0; top: 0.15rem;
    color: var(--magenta);
    transition: transform 0.3s ease;
  }
  .narration p.bullet:hover::before { transform: translateX(3px); }
  .narration p.bullet:first-child::first-letter { all: unset; }

  /* Callouts */
  .callout {
    position: relative;
    margin: 1.5rem 0;
    padding: 1.2rem 1.4rem 1.2rem 1.6rem;
    border-left: 3px solid var(--amber);
    background: linear-gradient(to right, rgba(252,211,77,0.08), rgba(252,211,77,0));
    border-radius: 0 8px 8px 0;
    box-shadow: var(--shadow);
    opacity: 0;
    animation: calloutIn 0.7s cubic-bezier(.2,.7,.2,1) forwards;
    overflow: hidden;
  }
  @keyframes calloutIn {
    from { opacity: 0; transform: translateX(-10px); }
    to   { opacity: 1; transform: none; }
  }
  /* Soft amber pulse on the left bar — the callout's heartbeat. */
  .callout::before {
    content: "";
    position: absolute;
    left: -3px; top: 0; bottom: 0;
    width: 3px;
    background: var(--amber);
    box-shadow: 0 0 18px 2px rgba(252,211,77,0.5);
    opacity: 0;
    animation: calloutBarPulse 0.9s cubic-bezier(.2,.7,.2,1) forwards 0.1s;
  }
  @keyframes calloutBarPulse {
    0%   { opacity: 0; }
    40%  { opacity: 1; }
    100% { opacity: 0.55; }
  }
  /* The faintest sweep of amber light slides across the callout on entry. */
  .callout::after {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(110deg, transparent 30%, rgba(252,211,77,0.16) 50%, transparent 70%);
    transform: translateX(-100%);
    pointer-events: none;
    animation: calloutSweep 1.4s ease-out forwards 0.2s;
  }
  @keyframes calloutSweep {
    to { transform: translateX(100%); }
  }
  .callout .line {
    font-family: var(--mono);
    font-size: 0.92rem;
    color: var(--text);
    line-height: 1.65;
    opacity: 0;
    animation: paraFadeUp 0.55s ease forwards;
  }
  .callout .line:nth-child(1) { animation-delay: 0.20s; }
  .callout .line:nth-child(2) { animation-delay: 0.32s; }
  .callout .line:nth-child(3) { animation-delay: 0.44s; }
  .callout .line:nth-child(4) { animation-delay: 0.54s; }
  .callout .line:nth-child(5) { animation-delay: 0.62s; }
  .callout .line:nth-child(n+6) { animation-delay: 0.70s; }
  .callout .line.dim { color: var(--dim); }

  /* ---------- River-of-thinking callout (Act III variant) ----------
     Same structural rhythm as the amber callout, but tinted cyan and
     decorated with a slow "flowing current" gradient on the left edge.
     The metaphor: an active stream of water carrying the policy forward. */
  .callout.callout-river {
    border-left-color: transparent;
    background: linear-gradient(to right, rgba(103,232,249,0.10), rgba(192,132,252,0.04) 60%, rgba(192,132,252,0));
  }
  /* Suppress the amber bar / sweep used by the base callout — the river
     gets its own animated left edge. */
  .callout.callout-river::before { display: none; }
  .callout.callout-river::after  { background: linear-gradient(110deg, transparent 30%, rgba(103,232,249,0.18) 50%, transparent 70%); }
  /* The river-current track: a vertical gradient where two repeating
     bands chase each other downward, reading as moving water. We sit it
     at left:0 (inside the rounded corner) because the .callout has
     overflow:hidden — anything at a negative offset would be clipped. */
  .callout.callout-river .river-current {
    position: absolute;
    left: 0; top: 6px; bottom: 6px;
    width: 4px;
    background:
      linear-gradient(180deg,
        rgba(103,232,249,0)   0%,
        rgba(103,232,249,0.95) 12%,
        rgba(192,132,252,0.8)  28%,
        rgba(103,232,249,0.25) 46%,
        rgba(103,232,249,0.95) 64%,
        rgba(192,132,252,0.6)  82%,
        rgba(103,232,249,0)   100%);
    background-size: 100% 200%;
    border-radius: 4px;
    box-shadow:
      0 0 14px rgba(103,232,249,0.55),
      0 0 28px rgba(192,132,252,0.25);
    opacity: 0;
    animation:
      riverFadeIn 0.9s ease forwards 0.1s,
      riverFlow   4.5s linear infinite;
  }
  /* Bump the left padding so text doesn't sit on top of the current. */
  .callout.callout-river {
    padding-left: 1.9rem;
  }
  @keyframes riverFadeIn {
    to { opacity: 1; }
  }
  /* The gradient slides downward continuously — a quiet current that
     never finishes. */
  @keyframes riverFlow {
    0%   { background-position: 0% 0%;    }
    100% { background-position: 0% -200%; }
  }
  .callout.callout-river .line { color: #d8f3fc; }
  .callout.callout-river .line.dim { color: rgba(193,212,229,0.78); }
  /* Slight italic on the first line to land the metaphor with a softer voice.
     Note: .river-current is the first child div, so the first .line element
     is nth-child(2) — we target that directly. */
  .callout.callout-river .line:nth-child(2) {
    font-family: var(--serif);
    font-style: italic;
    font-size: 1.1rem;
    line-height: 1.5;
    color: var(--cyan);
    letter-spacing: 0;
    margin-bottom: 0.35rem;
    padding-bottom: 0.35rem;
    border-bottom: 1px dashed rgba(103,232,249,0.18);
  }
  /* The river-current is the first child of .callout-river, which shifts
     every .line's nth-child index by one. Re-anchor the stagger here so
     each line still fades in with a clean rhythm. */
  .callout.callout-river .line:nth-child(2) { animation-delay: 0.20s; }
  .callout.callout-river .line:nth-child(3) { animation-delay: 0.34s; }
  .callout.callout-river .line:nth-child(4) { animation-delay: 0.46s; }
  .callout.callout-river .line:nth-child(5) { animation-delay: 0.56s; }
  .callout.callout-river .line:nth-child(6) { animation-delay: 0.66s; }
  .callout.callout-river .line:nth-child(n+7) { animation-delay: 0.74s; }

  /* Visualization grid */
  .viz {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1.2rem;
    margin: 1.5rem 0;
  }
  @media (min-width: 880px) {
    .viz.dual { grid-template-columns: 1.2fr 1fr; }
  }
  .panel {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 1.2rem;
    box-shadow: var(--shadow);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
  }
  .panel h4 {
    margin: 0 0 0.9rem;
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--dim);
  }

  /* Pool table SVG */
  .pool-wrap { display: flex; flex-direction: column; gap: 0.7rem; }
  .pool-svg {
    width: 100%; height: auto; display: block; border-radius: 12px;
    box-shadow:
      0 18px 40px -18px rgba(0,0,0,0.7),
      0 4px 14px -4px rgba(0,0,0,0.5),
      inset 0 0 0 1px rgba(255,220,170,0.06);
  }
  .pool-svg circle, .pool-svg ellipse, .pool-svg line {
    transition:
      cx 0.30s cubic-bezier(.4,.0,.2,1),
      cy 0.30s cubic-bezier(.4,.0,.2,1),
      rx 0.30s ease, ry 0.30s ease,
      x1 0.30s cubic-bezier(.4,.0,.2,1),
      x2 0.30s cubic-bezier(.4,.0,.2,1),
      y1 0.30s cubic-bezier(.4,.0,.2,1),
      y2 0.30s cubic-bezier(.4,.0,.2,1),
      opacity 0.25s ease;
  }
  .pool-svg .ood-pulse { animation: oodPulse 1.5s ease-in-out infinite; }
  @keyframes oodPulse {
    0%, 100% { stroke-width: 3; opacity: 1; }
    50%      { stroke-width: 5; opacity: 0.7; }
  }

  /* Cue ball breathes when idle. Paused mid-animation so the rAF owns it. */
  .pool-svg .cue-ball-glow {
    transition: opacity 0.4s ease;
    pointer-events: none;
    animation: cueBreathe 3.6s ease-in-out infinite;
  }
  .pool-svg.animating .cue-ball-glow { animation: none; opacity: 0.25; }
  @keyframes cueBreathe {
    0%, 100% { opacity: 0.18; }
    50%      { opacity: 0.38; }
  }

  /* Aim line shimmer — the dashed stroke "flows" from the ball toward the
     aim point, conveying "this trajectory is alive in the engine's head". */
  .pool-svg .aim-line {
    animation: aimFlow 1.4s linear infinite;
  }
  @keyframes aimFlow {
    from { stroke-dashoffset: 0; }
    to   { stroke-dashoffset: -18; }
  }
  .pool-svg.animating .aim-line { animation: none; }

  /* Soft moving highlight under the felt — a "table breathing" effect. */
  .pool-svg .felt-light {
    pointer-events: none;
    animation: feltBreathe 9s ease-in-out infinite;
    mix-blend-mode: screen;
  }
  @keyframes feltBreathe {
    0%, 100% { opacity: 0.30; }
    50%      { opacity: 0.55; }
  }

  /* Pre-shot anticipation ring — flashes around the ball moments before
     the trajectory animation begins. The wave has been chosen. */
  .pre-shot-ring {
    animation: preShotRing 0.55s ease-out forwards;
    pointer-events: none;
  }
  @keyframes preShotRing {
    0%   { r: 12; opacity: 0; stroke-width: 2; }
    40%  { opacity: 0.9; }
    100% { r: 30; opacity: 0; stroke-width: 0.4; }
  }

  /* Pocket drop particles — fireworks at the moment of impact. */
  .pocket-particle {
    animation: pocketParticle 0.7s cubic-bezier(.2,.7,.2,1) forwards;
    pointer-events: none;
  }
  @keyframes pocketParticle {
    0%   { opacity: 1; }
    100% { opacity: 0; }
  }

  /* ===== Outcome markers — "where does this shot ACTUALLY end up?" =====
     Drawn for every shot (Act II/III/IV) and every ghost trajectory in the
     Possibilities panel. Color-coded so the verdict is readable at a glance:
        DROP  → green  · the ball found a pocket
        GRAZE → amber  · close enough to feel real, not enough to drop
        MISS  → red    · the ball rolls into nothing
  */
  .outcome-line, .outcome-marker {
    pointer-events: none;
    transition: opacity 0.25s ease;
  }
  .outcome-end-ball {
    pointer-events: none;
    transition: opacity 0.25s ease;
    filter: drop-shadow(0 0 10px rgba(255,255,255,0.4));
  }
  .outcome-end-ball-drop  { fill: rgba(255,255,255,0.2); stroke: #6ee7a7; stroke-width: 1.5; }
  .outcome-end-ball-graze { fill: rgba(255,255,255,0.65); stroke: #fcd34d; stroke-width: 1.8; }
  .outcome-end-ball-miss  { fill: rgba(255,255,255,0.7); stroke: #fb7185; stroke-width: 1.8; }
  .outcome-pointer {
    pointer-events: none;
    stroke-width: 1.6;
    fill: none;
    stroke-dasharray: 3 4;
    animation: outcomePointerRun 0.9s linear infinite;
  }
  .outcome-pointer-drop  { stroke: #6ee7a7; }
  .outcome-pointer-graze { stroke: #fcd34d; }
  .outcome-pointer-miss  { stroke: #fb7185; }
  @keyframes outcomePointerRun {
    0%   { stroke-dashoffset: 22; opacity: 0.35; }
    50%  { opacity: 0.95; }
    100% { stroke-dashoffset: 0;  opacity: 0.35; }
  }
  .outcome-chip {
    pointer-events: none;
    animation: outcomeChipFloat 1.2s cubic-bezier(.2,.7,.2,1) forwards;
  }
  .outcome-chip rect {
    fill: rgba(9, 9, 18, 0.82);
    stroke-width: 1.2;
  }
  .outcome-chip text {
    font-family: var(--mono);
    font-size: 0.58rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    fill: rgba(230,235,245,0.95);
  }
  .outcome-chip-drop rect  { stroke: rgba(110,231,167,0.92); }
  .outcome-chip-graze rect { stroke: rgba(252,211,77,0.92); }
  .outcome-chip-miss rect  { stroke: rgba(251,113,133,0.92); }
  @keyframes outcomeChipFloat {
    0%   { transform: translateY(4px); opacity: 0; }
    25%  { opacity: 1; }
    100% { transform: translateY(0); opacity: 1; }
  }
  .outcome-rail-flash {
    pointer-events: none;
    stroke-linecap: round;
    animation: railOutcomeFlash 1.2s ease-out forwards;
  }
  .outcome-rail-flash-drop  { stroke: rgba(110,231,167,0.98); }
  .outcome-rail-flash-graze { stroke: rgba(252,211,77,0.95); }
  .outcome-rail-flash-miss  { stroke: rgba(251,113,133,0.98); }
  @keyframes railOutcomeFlash {
    0%   { opacity: 0;   stroke-width: 4;  }
    22%  { opacity: 0.95; stroke-width: 7; }
    100% { opacity: 0;   stroke-width: 3;  }
  }
  /* The drop halo gently breathes so successful aims read as "alive" while
     misses sit still — your eye learns the difference fast. */
  .outcome-drop-halo {
    transform-box: fill-box;
    transform-origin: center;
    animation: outcomeDropBreathe 2.4s ease-in-out infinite;
  }
  @keyframes outcomeDropBreathe {
    0%, 100% { transform: scale(1);    stroke-opacity: 0.7; }
    50%      { transform: scale(1.18); stroke-opacity: 1; }
  }
  /* Verdict label — bold one-word callout near the endpoint. */
  .outcome-label {
    font-family: var(--mono);
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    pointer-events: none;
    paint-order: stroke fill;
    stroke: rgba(8, 8, 16, 0.8);
    stroke-width: 4px;
    stroke-linejoin: round;
    text-transform: uppercase;
  }
  .outcome-label-drop  { fill: #6ee7a7; }
  .outcome-label-graze { fill: #fcd34d; }
  .outcome-label-miss  { fill: #fb7185; }

  /* ===== Victory / Spotlight mode =====
     When the ball drops, we briefly enter a "hold the moment" beat:
       - the SVG itself gets a deep green aura (.victory)
       - the body dims everything else (.page-dimmed)
       - the four stat cards do a synchronized green pulse (.victory-cards)
     This stops the eye from skating past the success.                  */
  .pool-svg {
    transition: filter 0.5s ease, transform 0.5s cubic-bezier(.2,.7,.2,1);
  }
  .pool-svg.victory {
    filter: brightness(1.05) drop-shadow(0 0 32px rgba(110,231,167,0.55));
    transform: scale(1.012);
  }
  body.page-dimmed {
    --dim-amount: 0.42;
  }
  body.page-dimmed #story .act-card:not(.in-focus),
  body.page-dimmed #progress-nav {
    opacity: calc(1 - var(--dim-amount));
    filter: saturate(0.7);
    transition: opacity 0.55s ease, filter 0.55s ease;
  }
  body.page-dimmed #story .act-card.in-focus {
    transition: opacity 0.55s ease, filter 0.55s ease;
  }
  /* The currently focused act + viz panel sits a hair above its neighbors. */
  #story .act-card.in-focus .viz {
    box-shadow: 0 24px 60px -28px rgba(110,231,167,0.45);
    transition: box-shadow 0.5s ease;
  }
  /* Synchronized victory pulse across the four narrative cards. */
  .shot-stats.victory-cards .stat-card {
    animation: cardVictory 0.95s cubic-bezier(.2,.7,.2,1);
  }
  @keyframes cardVictory {
    0%   { transform: scale(1); box-shadow: 0 0 0 0 rgba(110,231,167,0); }
    25%  { transform: scale(1.04); box-shadow: 0 0 32px -4px rgba(110,231,167,0.75); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(110,231,167,0); }
  }
  /* The expanding green halo at the pocket — long, slow, triumphant. */
  .victory-halo {
    animation: victoryHalo 0.95s cubic-bezier(.2,.7,.2,1) forwards;
    pointer-events: none;
  }
  @keyframes victoryHalo {
    0%   { r: 10; opacity: 0.9; stroke-width: 3; }
    100% { r: 70; opacity: 0;   stroke-width: 0.2; }
  }
  /* The floating "+1.00" damage-number above the pocket. */
  .floating-reward {
    pointer-events: none;
    font-family: var(--mono);
    font-size: 0.95rem;
    font-weight: 700;
    fill: #6ee7a7;
    text-anchor: middle;
    animation: floatReward 1.1s cubic-bezier(.2,.7,.2,1) forwards;
    filter: drop-shadow(0 0 6px rgba(110,231,167,0.7));
  }
  @keyframes floatReward {
    0%   { opacity: 0; transform: translateY(0); }
    18%  { opacity: 1; }
    100% { opacity: 0; transform: translateY(-46px); }
  }

  /* Disable CSS transitions during requestAnimationFrame animations so the
     ball follows the trajectory smoothly instead of fighting the 300ms ease. */
  .pool-svg.animating circle,
  .pool-svg.animating ellipse,
  .pool-svg.animating line {
    transition: none !important;
  }

  /* Contact flash where the ball hits the rail */
  .contact-flash {
    animation: contactCore 0.45s ease-out forwards;
    pointer-events: none;
  }
  .contact-ring {
    animation: contactRing 0.45s ease-out forwards;
    pointer-events: none;
  }
  @keyframes contactCore {
    0%   { r: 3; opacity: 1; }
    100% { r: 9; opacity: 0; }
  }
  @keyframes contactRing {
    0%   { r: 3;  opacity: 0.9; stroke-width: 2.5; }
    100% { r: 26; opacity: 0;   stroke-width: 0.4; }
  }
  .pocket-sparkle {
    animation: pocketSparkle 0.7s ease-out forwards;
    pointer-events: none;
  }
  @keyframes pocketSparkle {
    0%   { r: 2;  opacity: 1;   stroke-width: 2.5; }
    100% { r: 32; opacity: 0;   stroke-width: 0.3; }
  }
  .ball-trail {
    animation: ballTrail 0.5s ease-out forwards;
    pointer-events: none;
  }
  @keyframes ballTrail {
    0%   { opacity: 0.55; r: 2.4; }
    100% { opacity: 0;    r: 1; }
  }

  /* Interactive Possibilities panel */
  .possibilities-wrap {
    margin: 1.5rem 0 2rem;
    background: linear-gradient(160deg, rgba(168,85,247,0.06), rgba(34,211,238,0.03));
    border: 1px solid var(--line-strong);
    border-radius: 14px;
    padding: 1.4rem 1.6rem 1.6rem;
    box-shadow: 0 14px 36px -20px rgba(168,85,247,0.45);
    opacity: 0;
    animation: fadeIn 0.6s ease forwards 0.2s;
  }
  .possibilities-wrap h3 {
    font-family: var(--serif);
    font-style: italic;
    font-weight: 500;
    margin: 0 0 0.4rem;
    font-size: 1.55rem;
    background: linear-gradient(120deg, #ffffff 0%, var(--magenta) 60%, var(--cyan) 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
  }
  .possibilities-wrap .lede {
    color: var(--dim);
    font-family: var(--serif);
    font-size: 1rem;
    line-height: 1.55;
    margin: 0 0 1rem;
    max-width: 64ch;
  }
  .possibilities-controls {
    display: flex;
    gap: 0.6rem;
    align-items: center;
    flex-wrap: wrap;
    margin-bottom: 0.8rem;
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--muted);
  }
  .possibilities-controls .state-picker {
    display: inline-flex; gap: 0.25rem;
  }
  .state-pill {
    display: inline-flex; align-items: center; justify-content: center;
    min-width: 28px; height: 26px; padding: 0 0.4rem;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,0.02);
    color: var(--dim);
    font-family: var(--mono);
    font-size: 0.78rem;
    cursor: pointer;
    transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease;
  }
  .state-pill:hover { border-color: var(--line-strong); color: var(--text); }
  .state-pill.active {
    background: rgba(110,231,167,0.12);
    border-color: var(--green);
    color: var(--green);
  }
  .state-pill.ood {
    border-color: rgba(251,113,133,0.5);
    color: var(--red);
  }
  .state-pill.ood.active {
    background: rgba(251,113,133,0.15);
    border-color: var(--red);
    color: var(--red);
  }
  .reset-btn {
    margin-left: auto;
    padding: 0.4rem 0.85rem;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,0.03);
    color: var(--dim);
    font-family: var(--mono);
    font-size: 0.72rem;
    cursor: pointer;
    transition: background 0.2s ease, color 0.2s ease;
  }
  .reset-btn:hover { color: var(--text); background: rgba(255,255,255,0.06); }
  .possibilities-result {
    margin-top: 0.6rem;
    min-height: 1.4rem;
    font-family: var(--mono);
    font-size: 0.85rem;
    color: var(--dim);
    transition: color 0.3s ease;
  }
  .possibilities-result b { color: var(--text); }
  .possibilities-result.hit  b.verdict { color: var(--green); }
  .possibilities-result.near b.verdict { color: #fbbf24; }
  .possibilities-result.miss b.verdict { color: var(--red); }
  .possibilities-hint {
    margin-top: 0.5rem;
    font-family: var(--serif);
    font-style: italic;
    color: var(--muted);
    font-size: 0.85rem;
  }
  /* Ghost trajectories rendered on the table */
  .ghost-aim {
    cursor: pointer;
    transition: opacity 0.2s ease, stroke-width 0.2s ease;
  }
  .ghost-aim:hover { opacity: 0.95 !important; stroke-width: 2.2 !important; }
  .ghost-aim.locked { pointer-events: none; }
  /* Shot stats — narrative-mapped cards under the pool table */
  .shot-stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(118px, 1fr));
    gap: 0.5rem;
    margin-top: 0.6rem;
  }
  .stat-card {
    position: relative;
    background: linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.0));
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 0.55rem 0.7rem 0.6rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    overflow: hidden;
    transition: border-color 0.3s ease, background 0.3s ease, transform 0.3s ease;
  }
  .stat-card.flash {
    animation: statFlash 0.65s ease;
  }
  @keyframes statFlash {
    0%   { transform: scale(1); border-color: var(--line); box-shadow: 0 0 0 0 rgba(192,132,252,0); }
    25%  { transform: scale(1.025); border-color: var(--line-strong); background: rgba(255,255,255,0.06);
           box-shadow: 0 0 24px -4px rgba(192,132,252,0.45); }
    100% { transform: scale(1); border-color: var(--line); box-shadow: 0 0 0 0 rgba(192,132,252,0); }
  }
  /* Per-card flash tint — gives each card its own emotional color when it pops. */
  .stat-card.is-collapse.flash { animation: statFlashMagenta 0.65s ease; }
  .stat-card.is-world.flash    { animation: statFlashCream   0.65s ease; }
  .stat-card.is-bet.flash      { animation: statFlashRed     0.65s ease; }
  .stat-card.is-reward.flash.hit  { animation: statFlashGreen 0.7s ease; }
  .stat-card.is-reward.flash.near { animation: statFlashAmber 0.7s ease; }
  .stat-card.is-reward.flash.miss { animation: statFlashRed   0.7s ease; }
  @keyframes statFlashMagenta {
    0%   { transform: scale(1); box-shadow: 0 0 0 0 rgba(192,132,252,0); }
    25%  { transform: scale(1.025); box-shadow: 0 0 22px -4px rgba(192,132,252,0.55); background: rgba(192,132,252,0.08); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(192,132,252,0); }
  }
  @keyframes statFlashCream {
    0%   { transform: scale(1); box-shadow: 0 0 0 0 rgba(244,240,227,0); }
    25%  { transform: scale(1.025); box-shadow: 0 0 22px -4px rgba(244,240,227,0.45); background: rgba(244,240,227,0.06); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(244,240,227,0); }
  }
  @keyframes statFlashRed {
    0%   { transform: scale(1); box-shadow: 0 0 0 0 rgba(251,113,133,0); }
    25%  { transform: scale(1.025); box-shadow: 0 0 22px -4px rgba(251,113,133,0.55); background: rgba(251,113,133,0.08); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(251,113,133,0); }
  }
  @keyframes statFlashGreen {
    0%   { transform: scale(1); box-shadow: 0 0 0 0 rgba(110,231,167,0); }
    25%  { transform: scale(1.035); box-shadow: 0 0 30px -4px rgba(110,231,167,0.65); background: rgba(110,231,167,0.10); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(110,231,167,0); }
  }
  @keyframes statFlashAmber {
    0%   { transform: scale(1); box-shadow: 0 0 0 0 rgba(251,191,36,0); }
    25%  { transform: scale(1.025); box-shadow: 0 0 22px -4px rgba(251,191,36,0.55); background: rgba(251,191,36,0.08); }
    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(251,191,36,0); }
  }
  /* Mouse-tracked ambient highlight inside each card. JS sets --mx,--my. */
  .stat-card::after {
    content: "";
    position: absolute; inset: 0;
    border-radius: inherit;
    background: radial-gradient(160px circle at var(--mx, 50%) var(--my, 50%),
                                rgba(255,255,255,0.07), transparent 60%);
    opacity: 0;
    transition: opacity 0.3s ease;
    pointer-events: none;
  }
  .stat-card:hover::after { opacity: 1; }
  /* Reward gauge — sweep effect when the value changes. */
  .reward-gauge { isolation: isolate; }
  .reward-gauge::after {
    content: "";
    position: absolute;
    top: 0; bottom: 0; left: -30%;
    width: 30%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.75), transparent);
    opacity: 0;
    pointer-events: none;
  }
  .reward-gauge.sweep::after {
    animation: gaugeSweep 0.85s ease-out forwards;
  }
  @keyframes gaugeSweep {
    0%   { left: -30%; opacity: 0; }
    25%  { opacity: 1; }
    100% { left: 100%; opacity: 0; }
  }
  .stat-card .stat-label {
    display: flex; align-items: center; gap: 0.35rem;
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--muted);
  }
  .stat-card .stat-glyph {
    display: inline-flex; align-items: center; justify-content: center;
    width: 14px; height: 14px;
    font-size: 0.75rem;
    line-height: 1;
  }
  .stat-card .stat-value {
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.01em;
    line-height: 1.1;
    transition: color 0.3s ease;
  }
  .stat-card .stat-value .unit {
    font-size: 0.7rem;
    color: var(--muted);
    font-weight: 400;
    margin-left: 0.15rem;
  }
  .stat-card .stat-caption {
    font-family: var(--serif);
    font-size: 0.74rem;
    color: var(--dim);
    font-style: italic;
    line-height: 1.25;
  }
  /* per-card accent colors */
  .stat-card.is-collapse .stat-glyph { color: var(--magenta); }
  .stat-card.is-world    .stat-glyph { color: #f4f0e3; text-shadow: 0 0 6px rgba(244,240,227,0.5); }
  .stat-card.is-bet      .stat-glyph { color: var(--red); }
  .stat-card.is-reward   .stat-glyph { color: var(--green); }

  /* Reward gauge — sits under the value */
  .reward-gauge {
    position: relative;
    height: 5px;
    background: rgba(255,255,255,0.05);
    border-radius: 999px;
    overflow: hidden;
    margin-top: 0.25rem;
  }
  .reward-gauge-fill {
    position: absolute;
    top: 0; bottom: 0; left: 0;
    width: 0%;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--red), #f59e0b 55%, var(--green));
    transition: width 0.4s cubic-bezier(.4,0,.2,1);
  }
  .stat-card.is-reward.miss .stat-value { color: var(--red); }
  .stat-card.is-reward.hit  .stat-value { color: var(--green); }
  .stat-card.is-reward.near .stat-value { color: #fbbf24; }

  /* OOD modifier — applied during Act IV so the cards visually announce
     "this is outside the training data" before the eye finds the prose. */
  .stat-card.ood {
    border-color: rgba(251,113,133,0.55);
    background: linear-gradient(180deg, rgba(251,113,133,0.10), rgba(251,113,133,0.02));
    box-shadow: 0 0 16px -6px rgba(251,113,133,0.5);
  }
  .stat-card.ood .stat-label { color: var(--red); }
  .stat-card.ood .stat-value { color: var(--red); }
  .stat-card.ood .stat-glyph { color: var(--red); }

  /* Untested reward — used during Act IV: the engine never let reality
     answer, so the card should NOT pretend a shot was fired. It goes
     visually quiet: muted text, empty gauge, italicized "no shot fired". */
  .stat-card.is-reward.untested {
    border-color: var(--line);
    background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0));
    box-shadow: none;
  }
  .stat-card.is-reward.untested .stat-label,
  .stat-card.is-reward.untested .stat-glyph { color: var(--muted); }
  .stat-card.is-reward.untested .stat-value {
    color: var(--muted);
    font-style: italic;
    font-size: 0.95rem;
    font-weight: 500;
  }
  .stat-card.is-reward.untested .reward-gauge {
    background: rgba(255,255,255,0.03);
  }
  .stat-card.is-reward.untested .reward-gauge-fill {
    width: 0% !important;
    background: var(--muted);
    opacity: 0.4;
  }

  /* Q-table heatmap */
  .qtable {
    display: grid;
    gap: 4px;
    font-family: var(--mono);
    font-size: 0.75rem;
  }
  .qcell {
    aspect-ratio: 1;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    background: rgba(255,255,255,0.03);
    color: var(--muted);
    transition: background 0.4s ease, color 0.4s ease, transform 0.2s ease, box-shadow 0.4s ease;
    position: relative;
    overflow: hidden;
    isolation: isolate;
  }
  .qcell.flash { animation: cellFlash 0.7s cubic-bezier(.2,.7,.2,1); z-index: 1; }
  @keyframes cellFlash {
    0%   { transform: scale(1) rotate(0); box-shadow: 0 0 0 0 rgba(110,231,167,0.75), 0 0 0 0 rgba(110,231,167,0); }
    30%  { transform: scale(1.16) rotate(-1.5deg); box-shadow: 0 0 0 4px rgba(110,231,167,0.45), 0 0 22px 2px rgba(110,231,167,0.55); }
    100% { transform: scale(1) rotate(0); box-shadow: 0 0 0 0 rgba(110,231,167,0); }
  }
  /* Shimmer sweep across a Q-cell when its value tweens — that "stored
     memory just got rewritten" feeling. */
  .qcell::after {
    content: "";
    position: absolute;
    top: 0; bottom: 0; left: -60%;
    width: 50%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.45), transparent);
    opacity: 0;
    pointer-events: none;
  }
  .qcell.shimmer::after { animation: cellShimmer 0.75s ease-out forwards; }
  @keyframes cellShimmer {
    0%   { left: -60%; opacity: 0; }
    30%  { opacity: 1; }
    100% { left: 110%; opacity: 0; }
  }
  .qcell.best {
    box-shadow: 0 0 0 1px var(--green) inset, 0 0 12px -2px rgba(110,231,167,0.5);
  }
  .qrow-label, .qcol-label {
    color: var(--muted);
    font-family: var(--mono);
    font-size: 0.7rem;
    display: flex; align-items: center; justify-content: center;
  }

  /* RPE meter */
  .rpe-meter {
    position: relative;
    height: 18px;
    background: rgba(255,255,255,0.04);
    border-radius: 999px;
    overflow: hidden;
    margin-top: 0.4rem;
  }
  .rpe-meter::before {
    content: ""; position: absolute;
    left: 50%; top: 0; bottom: 0;
    width: 1px; background: rgba(255,255,255,0.2);
  }
  .rpe-fill {
    position: absolute;
    top: 0; bottom: 0; left: 50%;
    width: 0%;
    transition: width 0.25s ease, background 0.25s ease;
  }
  .rpe-fill.pos { background: linear-gradient(to right, var(--green), rgba(110,231,167,0.4)); }
  .rpe-fill.neg { background: linear-gradient(to left, var(--red), rgba(251,113,133,0.4)); transform-origin: right; }
  .rpe-label {
    display: flex; justify-content: space-between;
    font-family: var(--mono); font-size: 0.7rem; color: var(--muted);
    margin-top: 0.4rem;
  }

  /* Landmark badges */
  .lm {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    font-family: var(--mono);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border: 1px solid currentColor;
  }
  .lm.spike     { color: var(--green); background: rgba(110,231,167,0.08); }
  .lm.stall     { color: var(--red);   background: rgba(251,113,133,0.08); }
  .lm.coherence { color: var(--cyan);  background: rgba(103,232,249,0.08); }

  .shot-row {
    display: flex; flex-direction: column; gap: 0.4rem;
    padding: 0.8rem 1rem;
    background: rgba(255,255,255,0.02);
    border-radius: 8px;
    border: 1px solid var(--line);
    margin-top: 0.8rem;
  }

  /* Instinct demo cards */
  .instinct-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin-top: 1rem;
  }
  .instinct-card {
    background: var(--surface);
    border: 1px solid var(--line-strong);
    border-radius: 12px;
    padding: 1rem;
    opacity: 0;
    animation: fadeIn 0.5s ease forwards;
  }
  .instinct-card .label {
    font-family: var(--mono);
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: var(--dim);
  }
  .instinct-card .stmt {
    font-family: var(--serif);
    font-size: 1.05rem;
    margin: 0.4rem 0;
    color: var(--text);
  }
  .instinct-card .stmt b { color: var(--green); }
  .instinct-card .conf {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--muted);
  }

  /* OOD overlay */
  .ood-card {
    background: linear-gradient(135deg, rgba(251,113,133,0.10), rgba(192,132,252,0.06));
    border: 1px solid rgba(251,113,133,0.35);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
    box-shadow: 0 0 30px -10px rgba(251,113,133,0.4);
    opacity: 0;
    animation: oodIn 0.7s ease forwards;
  }
  @keyframes oodIn {
    from { opacity: 0; transform: scale(0.96); }
    to   { opacity: 1; transform: scale(1); }
  }
  .ood-card .stamp {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border: 2px solid var(--red);
    color: var(--red);
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    border-radius: 6px;
    transform: rotate(-3deg);
    margin-bottom: 0.8rem;
  }
  .ood-card .query {
    font-family: var(--serif);
    font-size: 1.15rem;
    color: var(--text);
    margin: 0 0 0.6rem;
  }
  .ood-card .compare {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-top: 0.6rem;
  }
  .ood-card .compare > div {
    background: rgba(0,0,0,0.25);
    padding: 0.8rem;
    border-radius: 8px;
    font-family: var(--mono);
    font-size: 0.85rem;
  }
  .ood-card .compare .lbl {
    color: var(--muted);
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
  }
  .ood-card .compare .val-engine { color: var(--rose); font-weight: 600; font-size: 1.1rem; }
  .ood-card .compare .val-truth  { color: var(--green); font-weight: 600; font-size: 1.1rem; }
  .ood-card .verdict {
    margin-top: 0.7rem;
    font-family: var(--mono);
    font-size: 0.8rem;
    color: var(--rose);
  }

  /* Two engines */
  .two-engines {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1rem;
    margin-top: 1.2rem;
  }
  @media (min-width: 700px) {
    .two-engines { grid-template-columns: 1fr 1fr; }
    .two-engines .qcell { font-size: 0.7rem; }
  }
  .engine-panel { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 1rem; }
  .engine-panel.A { border-color: rgba(192,132,252,0.45); }
  .engine-panel.B { border-color: rgba(103,232,249,0.45); }
  .engine-panel h5 {
    margin: 0 0 0.5rem;
    font-family: var(--mono);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    font-size: 0.75rem;
  }
  .engine-panel.A h5 { color: var(--magenta); }
  .engine-panel.B h5 { color: var(--cyan); }
  .engine-panel .law {
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--dim);
    margin-bottom: 0.7rem;
  }
  .answer-row {
    display: grid;
    grid-template-columns: 100px 1fr 1fr;
    gap: 0.4rem 1rem;
    padding: 0.7rem 0.9rem;
    background: rgba(255,255,255,0.03);
    border-radius: 8px;
    margin-top: 0.6rem;
    align-items: center;
    font-family: var(--mono);
    font-size: 0.8rem;
    opacity: 0;
    animation: fadeIn 0.5s ease forwards;
    white-space: nowrap;
  }
  .answer-row > * { overflow: hidden; text-overflow: ellipsis; }
  .answer-row .q  { color: var(--dim); }
  .answer-row .ea { color: var(--magenta); }
  .answer-row .eb { color: var(--cyan); }
  .answer-row b.mark-ok  { color: var(--green); }
  .answer-row b.mark-near { color: var(--amber); }
  .answer-row span.conf { color: var(--muted); }

  /* Epilogue bullets */
  .epilogue-bullets {
    list-style: none;
    counter-reset: epi;
    padding: 0;
    margin: 1.5rem 0;
  }
  .epilogue-bullets li {
    counter-increment: epi;
    padding: 1rem 1rem 1rem 3.5rem;
    margin-bottom: 0.8rem;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 12px;
    position: relative;
    font-family: var(--serif);
    font-size: 1.05rem;
    color: var(--text);
    line-height: 1.55;
    opacity: 0;
    animation: fadeIn 0.5s ease forwards;
  }
  .epilogue-bullets li::before {
    content: counter(epi);
    position: absolute;
    top: 0.85rem; left: 1rem;
    width: 2rem; height: 2rem;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--magenta), var(--cyan));
    color: black;
    font-family: var(--mono);
    font-weight: 700;
    font-size: 0.9rem;
    display: flex; align-items: center; justify-content: center;
  }

  /* Final map */
  .final-map {
    background: linear-gradient(135deg, rgba(192,132,252,0.08), rgba(103,232,249,0.05));
    border: 1px solid var(--line-strong);
    border-radius: 12px;
    padding: 1.4rem;
    margin: 2rem 0 1rem;
  }
  .final-map h4 {
    margin: 0 0 1rem;
    font-family: var(--mono);
    font-size: 0.75rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--magenta);
  }
  .final-map .row {
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 0.7rem;
    align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px dashed var(--line);
    font-family: var(--mono);
    font-size: 0.92rem;
  }
  .final-map .row:last-child { border: none; }
  .final-map .arrow { color: var(--magenta); font-weight: bold; }
  .final-map .target { color: var(--green); font-weight: 600; }
  .final-map .conf   { color: var(--muted); font-size: 0.8rem; }
  .closing {
    margin-top: 2rem;
    font-family: var(--serif);
    font-style: italic;
    color: var(--dim);
    text-align: center;
    line-height: 1.7;
  }
  .closing p { margin: 0.3rem 0; }

  /* ---------- Hero · two-player rink visual ---------- */
  #heroPlayers {
    margin: 0 auto 1.4rem;
    max-width: 380px;
    opacity: 0;
    animation: heroFadeUp 1.2s ease-out forwards 0.25s;
  }
  .hero-rink { width: 100%; height: auto; display: block; overflow: visible; }
  .hero-tag {
    font-family: var(--mono);
    font-size: 7px;
    letter-spacing: 0.32em;
    fill: var(--muted);
    text-transform: uppercase;
  }
  .hero-tag-a { fill: var(--magenta); }
  .hero-tag-b { fill: var(--cyan); }
  .hero-vs {
    font-family: var(--serif);
    font-style: italic;
    font-size: 16px;
    fill: var(--dim);
    transform-box: fill-box;
    transform-origin: center;
    animation: heroVsBreath 3.8s ease-in-out infinite;
  }
  @keyframes heroVsBreath {
    0%, 100% { opacity: 0.55; transform: scale(1); }
    50%      { opacity: 0.95; transform: scale(1.08); }
  }
  .hero-halo {
    transform-box: fill-box;
    transform-origin: center;
    animation: heroHalo 3.6s ease-in-out infinite;
  }
  .hero-halo-b { animation-delay: 1.8s; }
  @keyframes heroHalo {
    0%, 100% { opacity: 0.18; transform: scale(1); }
    50%      { opacity: 0.55; transform: scale(1.18); }
  }
  /* Cue balls breathe with a tiny vertical bob, slightly out of phase.
     The whole rink looks alive without ever being distracting. */
  .hero-ball {
    transform-box: fill-box;
    transform-origin: center;
    animation: heroBallBob 4.4s ease-in-out infinite;
  }
  .hero-ball-b { animation-delay: 2.2s; }
  @keyframes heroBallBob {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-2.5px); }
  }
  /* Felt strip beneath the balls gently breathes too. */
  .hero-felt {
    transform-box: fill-box;
    transform-origin: center;
    animation: heroFeltBreath 7s ease-in-out infinite;
  }
  @keyframes heroFeltBreath {
    0%, 100% { opacity: 0.5; }
    50%      { opacity: 0.7; }
  }

  /* ---------- Custom scrollbar — magenta tint, matches the theme ---------- */
  html { scrollbar-width: thin; scrollbar-color: rgba(192,132,252,0.4) transparent; }
  ::-webkit-scrollbar { width: 10px; height: 10px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, rgba(192,132,252,0.45), rgba(103,232,249,0.35));
    border-radius: 999px;
    border: 2px solid transparent;
    background-clip: padding-box;
  }
  ::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, rgba(192,132,252,0.7), rgba(103,232,249,0.55));
    background-clip: padding-box;
  }

  /* ---------- Progress nav — edge fade + smooth pill transitions ---------- */
  #progress { position: sticky; }
  #progress::before,
  #progress::after {
    content: "";
    position: absolute;
    top: 0; bottom: 0;
    width: 36px;
    pointer-events: none;
    z-index: 1;
  }
  #progress::before {
    left: 0;
    background: linear-gradient(to right, rgba(8,6,15,0.95), transparent);
  }
  #progress::after {
    right: 0;
    background: linear-gradient(to left, rgba(8,6,15,0.95), transparent);
  }
  #progress li.current {
    position: relative;
    transform: translateY(-1px);
  }

  /* ---------- Two-player intro card (prologue) ---------- */
  .two-player-intro {
    position: relative;
    margin: 1.4rem 0 1rem;
    border: 1px solid var(--line-strong);
    border-radius: 14px;
    padding: 1.3rem 1.4rem 1.4rem;
    background:
      linear-gradient(135deg, rgba(192,132,252,0.06), rgba(103,232,249,0.06));
    box-shadow: 0 14px 36px -20px rgba(0,0,0,0.6);
    overflow: hidden;
    opacity: 0;
    transform: translateY(10px);
    animation: bellmanCardIn 0.8s cubic-bezier(.2,.7,.2,1) forwards;
  }
  .two-player-intro::before {
    /* Subtle magenta-to-cyan ribbon that drifts left↔right under the card,
       reinforcing the "two engines, one shared current" feeling. */
    content: "";
    position: absolute;
    top: -40%; left: -30%; right: -30%; height: 180%;
    background:
      radial-gradient(60% 60% at 20% 50%, rgba(192,132,252,0.10), transparent 70%),
      radial-gradient(60% 60% at 80% 50%, rgba(103,232,249,0.10), transparent 70%);
    pointer-events: none;
    animation: tpRibbonDrift 14s ease-in-out infinite alternate;
    z-index: 0;
  }
  @keyframes tpRibbonDrift {
    0%, 100% { transform: translateX(-4%); }
    50%      { transform: translateX(4%); }
  }
  .two-player-intro > * { position: relative; z-index: 1; }
  .tp-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.9rem;
    align-items: stretch;
  }
  @media (min-width: 760px) {
    .tp-grid { grid-template-columns: 1fr auto 1fr; align-items: center; }
  }
  .tp-card {
    background: var(--surface);
    border-radius: 12px;
    border: 1px solid var(--line);
    padding: 1rem 1.1rem;
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    opacity: 0;
    transition:
      transform 0.4s cubic-bezier(.2,.7,.2,1),
      box-shadow 0.4s ease,
      border-color 0.4s ease;
  }
  .tp-card.A {
    border-color: rgba(192,132,252,0.45);
    box-shadow: 0 0 28px -16px rgba(192,132,252,0.65);
    animation: tpCardInLeft 0.85s cubic-bezier(.2,.7,.2,1) forwards 0.15s;
  }
  .tp-card.B {
    border-color: rgba(103,232,249,0.45);
    box-shadow: 0 0 28px -16px rgba(103,232,249,0.65);
    animation: tpCardInRight 0.85s cubic-bezier(.2,.7,.2,1) forwards 0.30s;
  }
  .tp-card.A:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 32px -16px rgba(192,132,252,0.7);
    border-color: rgba(192,132,252,0.7);
  }
  .tp-card.B:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 32px -16px rgba(103,232,249,0.7);
    border-color: rgba(103,232,249,0.7);
  }
  @keyframes tpCardInLeft {
    from { opacity: 0; transform: translateX(-22px); }
    to   { opacity: 1; transform: none; }
  }
  @keyframes tpCardInRight {
    from { opacity: 0; transform: translateX(22px); }
    to   { opacity: 1; transform: none; }
  }
  .tp-card .tp-name {
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
  }
  .tp-card.A .tp-name { color: var(--magenta); }
  .tp-card.B .tp-name { color: var(--cyan); }
  .tp-card .tp-bio {
    font-family: var(--mono);
    font-size: 0.88rem;
    color: var(--text);
    line-height: 1.45;
  }
  .tp-card .tp-tagline {
    font-family: var(--serif);
    font-style: italic;
    font-size: 0.88rem;
    color: var(--dim);
    line-height: 1.45;
  }
  .tp-vs {
    align-self: center;
    justify-self: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.45rem;
    font-family: var(--serif);
    font-style: italic;
    font-size: 1.7rem;
    color: var(--muted);
    opacity: 0;
    transform: scale(0.6) rotate(-12deg);
    animation: tpVsIn 0.7s cubic-bezier(.2,.7,.2,1) forwards 0.55s,
               tpVsBob 5.5s ease-in-out infinite 1.4s;
  }
  @keyframes tpVsIn {
    to { opacity: 1; transform: scale(1) rotate(0); }
  }
  @keyframes tpVsBob {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-3px); }
  }
  .tp-vs::before,
  .tp-vs::after {
    content: "";
    display: block;
    width: 1px;
    height: 20px;
    background: linear-gradient(to bottom, transparent, var(--line-strong), transparent);
  }
  .tp-shared {
    margin-top: 1rem;
    padding: 0.85rem 1rem;
    border-top: 1px dashed var(--line);
    font-family: var(--serif);
    font-style: italic;
    font-size: 0.98rem;
    line-height: 1.6;
    color: var(--text);
    text-align: center;
    opacity: 0;
    animation: paraFadeUp 0.7s ease forwards 0.85s;
  }
  .tp-shared b {
    color: var(--magenta);
    font-weight: 600;
    font-style: normal;
    font-family: var(--mono);
    font-size: 0.78rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
  }

  /* ---------- RL Legend · story ↔ symbol mapping (Act I) ---------- */
  .rl-legend {
    margin: 1.6rem 0;
    padding: 1.3rem 1.4rem 1.2rem;
    border-radius: 14px;
    border: 1px solid var(--line-strong);
    background:
      linear-gradient(160deg, rgba(192,132,252,0.05), rgba(103,232,249,0.04));
    box-shadow: 0 14px 32px -22px rgba(168,142,220,0.4);
    opacity: 0;
    transform: translateY(10px);
    animation: bellmanCardIn 0.8s cubic-bezier(.2,.7,.2,1) forwards;
    overflow: hidden;
  }
  .rl-legend h4 {
    margin: 0 0 1rem;
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--magenta);
    opacity: 0;
    animation: paraFadeUp 0.6s ease forwards 0.15s;
  }
  .rl-rows {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.55rem;
  }
  .rl-row {
    position: relative;
    display: grid;
    grid-template-columns: 4.5rem 1fr;
    gap: 1rem;
    padding: 0.75rem 0.85rem;
    border-radius: 10px;
    background: rgba(255,255,255,0.025);
    border: 1px solid var(--line);
    align-items: start;
    overflow: hidden;
    opacity: 0;
    transform: translateY(8px);
    animation: paraFadeUp 0.55s cubic-bezier(.2,.7,.2,1) forwards;
    transition: border-color 0.25s ease, background 0.25s ease, transform 0.25s ease;
  }
  .rl-row:nth-child(1) { animation-delay: 0.30s; }
  .rl-row:nth-child(2) { animation-delay: 0.40s; }
  .rl-row:nth-child(3) { animation-delay: 0.50s; }
  .rl-row:nth-child(4) { animation-delay: 0.60s; }
  .rl-row:nth-child(5) { animation-delay: 0.70s; }
  .rl-row:nth-child(6) { animation-delay: 0.80s; }
  .rl-row::before {
    /* Cyan "scan line" sweeps once across each row as it arrives. */
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(100deg, transparent 35%, rgba(103,232,249,0.10) 50%, transparent 65%);
    transform: translateX(-100%);
    pointer-events: none;
  }
  .rl-row.scan::before { animation: rowScan 1.0s ease-out forwards; }
  @keyframes rowScan { to { transform: translateX(100%); } }
  .rl-row:hover {
    border-color: var(--line-strong);
    background: rgba(255,255,255,0.05);
    transform: translateY(-1px);
  }
  .rl-row:hover .rl-sym {
    text-shadow: 0 0 18px rgba(103,232,249,0.65);
    transform: scale(1.08);
  }
  .rl-sym {
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--cyan);
    line-height: 1.3;
    padding-top: 0.05rem;
    text-align: left;
    transition: text-shadow 0.3s ease, transform 0.3s ease;
  }
  .rl-body { display: flex; flex-direction: column; gap: 0.22rem; min-width: 0; }
  .rl-term {
    font-family: var(--mono);
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--magenta);
  }
  .rl-story {
    font-family: var(--serif);
    font-size: 1rem;
    color: var(--text);
    line-height: 1.45;
  }
  .rl-concrete {
    font-family: var(--mono);
    font-size: 0.78rem;
    color: var(--dim);
    line-height: 1.45;
  }

  /* ---------- Bellman update card (Act II) ---------- */
  .bellman-card {
    margin: 1.6rem 0;
    padding: 1.4rem 1.4rem 1.3rem;
    border-radius: 14px;
    border: 1px solid var(--line-strong);
    background:
      radial-gradient(800px circle at 0% 0%, rgba(192,132,252,0.10), transparent 55%),
      radial-gradient(700px circle at 100% 100%, rgba(103,232,249,0.08), transparent 55%),
      rgba(24, 17, 42, 0.75);
    box-shadow: 0 20px 60px -30px rgba(0,0,0,0.7);
    opacity: 0;
    transform: translateY(14px);
    animation: bellmanCardIn 0.9s cubic-bezier(.2,.7,.2,1) forwards;
  }
  @keyframes bellmanCardIn {
    to { opacity: 1; transform: none; }
  }
  .bellman-card h4 {
    margin: 0 0 1rem;
    font-family: var(--mono);
    font-size: 0.72rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--magenta);
  }
  .bellman-eq {
    font-family: var(--mono);
    /* clamp() scales the equation down on narrow viewports so it stays on
       one line at normal widths and only wraps when truly necessary. The
       earlier `nowrap + overflow-x: auto` caused the trailing `]` to be
       clipped entirely; we now allow wrapping and rely on clamp() to
       keep the equation legible. */
    font-size: clamp(0.78rem, 1.42vw, 1.06rem);
    line-height: 1.85;
    color: var(--text);
    background: rgba(0,0,0,0.28);
    border: 1px dashed var(--line-strong);
    border-radius: 10px;
    padding: 1.1rem 0.9rem;
    text-align: center;
    white-space: normal;
    letter-spacing: 0.01em;
    transition: box-shadow 0.4s ease;
  }
  .bellman-eq:hover {
    box-shadow: inset 0 0 0 1px rgba(192,132,252,0.25),
                0 0 36px -16px rgba(192,132,252,0.4);
  }
  .bellman-eq .bp {
    font-weight: 700;
    display: inline-block;
    opacity: 0;
    transform: translateY(4px);
    animation: bellmanTokenIn 0.45s cubic-bezier(.2,.7,.2,1) forwards;
    transition: filter 0.25s ease, opacity 0.25s ease, transform 0.25s ease;
  }
  /* The equation reveals one token at a time, like typesetting. Each .bp
     receives a small staggered delay (~50ms apart) so the eye reads the
     update LTR. */
  .bellman-eq .bp:nth-child(1)  { animation-delay: 0.15s; }
  .bellman-eq .bp:nth-child(2)  { animation-delay: 0.21s; }
  .bellman-eq .bp:nth-child(3)  { animation-delay: 0.27s; }
  .bellman-eq .bp:nth-child(4)  { animation-delay: 0.33s; }
  .bellman-eq .bp:nth-child(5)  { animation-delay: 0.39s; }
  .bellman-eq .bp:nth-child(6)  { animation-delay: 0.45s; }
  .bellman-eq .bp:nth-child(7)  { animation-delay: 0.51s; }
  .bellman-eq .bp:nth-child(8)  { animation-delay: 0.57s; }
  .bellman-eq .bp:nth-child(9)  { animation-delay: 0.63s; }
  .bellman-eq .bp:nth-child(10) { animation-delay: 0.69s; }
  .bellman-eq .bp:nth-child(11) { animation-delay: 0.75s; }
  .bellman-eq .bp:nth-child(12) { animation-delay: 0.81s; }
  .bellman-eq .bp:nth-child(13) { animation-delay: 0.87s; }
  .bellman-eq .bp:nth-child(14) { animation-delay: 0.93s; }
  .bellman-eq .bp:nth-child(15) { animation-delay: 0.99s; }
  .bellman-eq .bp:nth-child(16) { animation-delay: 1.05s; }
  .bellman-eq .bp:nth-child(17) { animation-delay: 1.11s; }
  .bellman-eq .bp:nth-child(18) { animation-delay: 1.17s; }
  .bellman-eq .bp:nth-child(n+19) { animation-delay: 1.23s; }
  @keyframes bellmanTokenIn {
    from { opacity: 0; transform: translateY(6px); filter: blur(3px); }
    to   { opacity: 1; transform: none; filter: blur(0); }
  }
  .bellman-eq .bp-cyan    { color: var(--cyan); }
  .bellman-eq .bp-magenta { color: var(--magenta); }
  .bellman-eq .bp-amber   { color: var(--amber); }
  .bellman-eq .bp-green   { color: var(--green); }
  .bellman-eq .bp-red     { color: var(--red); }
  .bellman-eq .bp-dim     { color: var(--dim); }
  /* The RPE bracket: a continuous red underline on every token inside
     [ ... ]. We DELIBERATELY don't set `animation` here — that property is
     already used by `.bp` for the staggered typewriter reveal, and a second
     `animation` declaration would override it and leave the tokens invisible.
     The breathing red glow lives on the bracket edges only (the [ and ]
     glyphs), via .bp-rpe-edge below.

     Each token gets the same flat underline so when the eye sweeps across
     the bracket — even when the bracket wraps onto a new line — the underline
     reads as one continuous "RPE region", not eight pill chips. */
  .bellman-eq .bp-rpe {
    box-shadow: inset 0 -0.18em 0 rgba(251,113,133,0.22);
    padding: 0.05rem 0.12rem 0.18rem;
    border-radius: 2px;
  }
  .bellman-eq .bp-rpe-edge {
    color: var(--red);
    text-shadow: 0 0 12px rgba(251,113,133,0.55);
    box-shadow: none;
    padding-bottom: 0;
  }
  /* When a parts row is hovered, the matching equation token brightens
     and the others fade. JS sets `.hl-<color>` on the equation. */
  .bellman-eq.hl-cyan    .bp:not(.bp-cyan),
  .bellman-eq.hl-magenta .bp:not(.bp-magenta),
  .bellman-eq.hl-amber   .bp:not(.bp-amber),
  .bellman-eq.hl-red     .bp:not(.bp-red),
  .bellman-eq.hl-green   .bp:not(.bp-green),
  .bellman-eq.hl-dim     .bp:not(.bp-dim) {
    opacity: 0.32;
    filter: blur(0.4px);
  }
  .bellman-eq.hl-cyan    .bp-cyan,
  .bellman-eq.hl-magenta .bp-magenta,
  .bellman-eq.hl-amber   .bp-amber,
  .bellman-eq.hl-red     .bp-red,
  .bellman-eq.hl-green   .bp-green,
  .bellman-eq.hl-dim     .bp-dim {
    transform: translateY(-1px);
    text-shadow: 0 0 14px currentColor;
  }
  .bellman-story {
    margin: 1rem 0 0;
    text-align: center;
    font-family: var(--serif);
    font-style: italic;
    color: var(--dim);
    font-size: 1rem;
    line-height: 1.55;
    opacity: 0;
    animation: paraFadeUp 0.7s ease forwards 1.35s;
  }
  .bellman-story b {
    color: var(--magenta);
    font-style: normal;
    font-family: var(--mono);
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
  }
  .bellman-parts {
    margin-top: 1rem;
    display: grid;
    grid-template-columns: 1fr;
    gap: 0.5rem;
  }
  @media (min-width: 700px) {
    .bellman-parts { grid-template-columns: 1fr 1fr; }
  }
  .bellman-part {
    display: grid;
    grid-template-columns: 6.4rem 1fr;
    gap: 0.7rem;
    padding: 0.6rem 0.8rem;
    border-radius: 10px;
    background: rgba(255,255,255,0.025);
    border: 1px solid var(--line);
    font-size: 0.86rem;
    cursor: default;
    opacity: 0;
    transform: translateY(6px);
    animation: paraFadeUp 0.55s ease forwards;
    transition:
      border-color 0.25s ease,
      background 0.25s ease,
      transform 0.25s ease,
      box-shadow 0.25s ease;
  }
  .bellman-part:nth-child(1) { animation-delay: 1.50s; }
  .bellman-part:nth-child(2) { animation-delay: 1.60s; }
  .bellman-part:nth-child(3) { animation-delay: 1.70s; }
  .bellman-part:nth-child(4) { animation-delay: 1.78s; }
  .bellman-part:nth-child(5) { animation-delay: 1.86s; }
  .bellman-part:nth-child(6) { animation-delay: 1.94s; }
  .bellman-part:hover {
    background: rgba(255,255,255,0.05);
    border-color: var(--line-strong);
    transform: translateY(-2px);
    box-shadow: 0 12px 28px -18px rgba(0,0,0,0.8);
  }
  .bellman-part .key {
    font-family: var(--mono);
    font-weight: 700;
    font-size: 0.92rem;
    line-height: 1.3;
  }
  .bellman-part .lab {
    font-family: var(--serif);
    color: var(--dim);
    line-height: 1.4;
  }
  .bellman-part.cyan    .key { color: var(--cyan); }
  .bellman-part.magenta .key { color: var(--magenta); }
  .bellman-part.amber   .key { color: var(--amber); }
  .bellman-part.green   .key { color: var(--green); }
  .bellman-part.red     .key { color: var(--red); }
  .bellman-part.dim     .key { color: var(--dim); }
  .bellman-part.cyan:hover    { box-shadow: 0 12px 28px -18px rgba(103,232,249,0.65); }
  .bellman-part.magenta:hover { box-shadow: 0 12px 28px -18px rgba(192,132,252,0.65); }
  .bellman-part.amber:hover   { box-shadow: 0 12px 28px -18px rgba(252,211,77,0.55); }
  .bellman-part.red:hover     { box-shadow: 0 12px 28px -18px rgba(251,113,133,0.55); }
</style>
</head>
<body>

<section id="hero">
  <div class="eyebrow">an interactive essay on certainty, bias &amp; the loop</div>
  <div id="heroPlayers" aria-hidden="true">
    <svg viewBox="0 0 280 90" class="hero-rink">
      <defs>
        <radialGradient id="heroBallA" cx="0.35" cy="0.30" r="0.75">
          <stop offset="0"    stop-color="#ffffff"/>
          <stop offset="0.45" stop-color="#f4f0e3"/>
          <stop offset="1"    stop-color="#4a4338"/>
        </radialGradient>
        <radialGradient id="heroBallB" cx="0.35" cy="0.30" r="0.75">
          <stop offset="0"    stop-color="#ffffff"/>
          <stop offset="0.45" stop-color="#f4f0e3"/>
          <stop offset="1"    stop-color="#4a4338"/>
        </radialGradient>
        <linearGradient id="heroFelt" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stop-color="#196b4d"/>
          <stop offset="1" stop-color="#0a3f2a"/>
        </linearGradient>
      </defs>
      <rect x="20" y="34" width="240" height="22" rx="4" fill="url(#heroFelt)" opacity="0.55" class="hero-felt"/>
      <circle cx="78"  cy="45" r="20" fill="none" stroke="#c084fc" stroke-width="1" opacity="0.4" class="hero-halo hero-halo-a"/>
      <circle cx="78"  cy="45" r="11" fill="url(#heroBallA)" stroke="#c084fc" stroke-width="2" class="hero-ball hero-ball-a"/>
      <text  x="78"  y="80" text-anchor="middle" class="hero-tag hero-tag-a">PLAYER A</text>
      <text  x="140" y="50" text-anchor="middle" class="hero-vs">vs</text>
      <circle cx="202" cy="45" r="20" fill="none" stroke="#67e8f9" stroke-width="1" opacity="0.4" class="hero-halo hero-halo-b"/>
      <circle cx="202" cy="45" r="11" fill="url(#heroBallB)" stroke="#67e8f9" stroke-width="2" class="hero-ball hero-ball-b"/>
      <text  x="202" y="80" text-anchor="middle" class="hero-tag hero-tag-b">PLAYER B</text>
    </svg>
  </div>
  <h1>Calling the Shot</h1>
  <p class="tagline">
    Two players walk up to the same pool table. Same loop in their heads.
    Different lives compiled inside it. Watch the math that makes both of them
    completely certain, completely honest, and completely in disagreement.
  </p>
  <button id="startBtn">Begin the play →</button>
  <div class="meta">~90 seconds · auto-plays · two players · one table</div>
</section>

<main id="stage" aria-live="polite">
  <nav id="progress">
    <ol>
      <li data-step="0">Prologue · Two Players</li>
      <li data-step="1">Act I · The Loop</li>
      <li data-step="2">Act II · Dopamine</li>
      <li data-step="3">Act III · Internal Narrative</li>
      <li data-step="4">Act IV · Hallucination</li>
      <li data-step="5">Act V · Player B</li>
      <li data-step="6">Epilogue</li>
    </ol>
  </nav>
  <div id="story"></div>
</main>

<script>
(() => {
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const make = (tag, props = {}, children = []) => {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props)) {
      if (k === "class") el.className = v;
      else if (k === "html") el.innerHTML = v;
      else if (k === "text") el.textContent = v;
      else if (k === "style") Object.assign(el.style, v);
      else if (k.startsWith("data-")) el.setAttribute(k, v);
      else el[k] = v;
    }
    for (const c of [].concat(children || [])) {
      if (c == null) continue;
      el.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return el;
  };

  const startBtn = $("#startBtn");
  const hero = $("#hero");
  const stage = $("#stage");
  const story = $("#story");

  // Per-act container, so each new act_start opens a fresh card.
  let actBody = null;
  let currentVizPanels = null;
  let qState = null;       // {states, actions, values: [[]]}
  let twoEngines = null;   // {A: {qState, panel, lawText}, B: {..}}

  // --- Heatmap helpers ---
  function valueToColor(v) {
    // map 0..1.3 → dim → cyan → magenta → green
    const t = Math.min(1, Math.max(0, v / 1.25));
    if (t < 0.01) return "rgba(255,255,255,0.03)";
    // gradient stops
    // 0.0 dim   #1a1530
    // 0.3 cyan  #155e75
    // 0.6 magenta #6b21a8
    // 1.0 green #166534
    const stops = [
      [0.00, [26, 21, 48]],
      [0.25, [22, 78, 99]],
      [0.55, [88, 28, 135]],
      [0.85, [22, 101, 52]],
      [1.00, [134, 239, 172]],
    ];
    for (let i = 1; i < stops.length; i++) {
      if (t <= stops[i][0]) {
        const [t0, c0] = stops[i - 1];
        const [t1, c1] = stops[i];
        const f = (t - t0) / (t1 - t0);
        const c = c0.map((v0, k) => Math.round(v0 + (c1[k] - v0) * f));
        return `rgb(${c[0]},${c[1]},${c[2]})`;
      }
    }
    return "rgb(134,239,172)";
  }
  function textColorFor(v) {
    return v > 0.4 ? "white" : "rgba(255,255,255,0.55)";
  }

  function buildQTable(snapshot) {
    const { states, actions, values } = snapshot;
    const wrap = make("div", { class: "qtable" });
    wrap.style.gridTemplateColumns = `auto repeat(${actions.length}, 1fr)`;
    wrap.appendChild(make("div", { class: "qrow-label" }));
    actions.forEach(a => wrap.appendChild(make("div", { class: "qcol-label", text: "aim " + a })));
    const cells = {};
    states.forEach((s, i) => {
      wrap.appendChild(make("div", { class: "qrow-label", text: "state " + s }));
      actions.forEach((a, j) => {
        const v = values[i][j];
        const cell = make("div", { class: "qcell", text: v > 0.05 ? v.toFixed(2) : "" });
        cell.style.background = valueToColor(v);
        cell.style.color = textColorFor(v);
        cells[`${s}_${a}`] = cell;
        wrap.appendChild(cell);
      });
    });
    return { node: wrap, cells, snapshot: structuredClone(snapshot) };
  }

  function updateBestHighlight(qObj) {
    const { snapshot, cells } = qObj;
    snapshot.states.forEach((s, i) => {
      const row = snapshot.values[i];
      const maxV = Math.max(...row);
      snapshot.actions.forEach((a, j) => {
        const cell = cells[`${s}_${a}`];
        cell.classList.toggle("best", row[j] === maxV && maxV > 0.1);
      });
    });
  }

  function updateQCell(qObj, state, action, newValue) {
    const { snapshot, cells } = qObj;
    const i = snapshot.states.indexOf(state);
    const j = snapshot.actions.indexOf(action);
    if (i < 0 || j < 0) return;
    snapshot.values[i][j] = newValue;
    const cell = cells[`${state}_${action}`];
    cell.style.background = valueToColor(newValue);
    cell.style.color = textColorFor(newValue);
    cell.textContent = newValue > 0.05 ? newValue.toFixed(2) : "";
    cell.classList.add("flash");
    setTimeout(() => cell.classList.remove("flash"), 600);
    updateBestHighlight(qObj);
  }

  function replaceQTable(qObj, snapshot) {
    const prev = qObj.snapshot;
    qObj.snapshot = structuredClone(snapshot);
    snapshot.states.forEach((s, i) => {
      snapshot.actions.forEach((a, j) => {
        const v = snapshot.values[i][j];
        const cell = qObj.cells[`${s}_${a}`];
        const prevV = prev ? prev.values[i][j] : 0;
        cell.style.background = valueToColor(v);
        cell.style.color = textColorFor(v);
        cell.textContent = v > 0.05 ? v.toFixed(2) : "";
        if (Math.abs(v - prevV) > 0.05) {
          cell.classList.remove("shimmer");
          void cell.offsetWidth;
          cell.classList.add("shimmer");
          setTimeout(() => cell.classList.remove("shimmer"), 800);
        }
      });
    });
    updateBestHighlight(qObj);
  }

  // --- Pool table SVG ---
  // ---------- Pool table SVG ----------
  // Geometry: 720 × 380 viewBox.
  //   Outer wood frame ............... 0..720 × 0..380
  //   Felt playing surface ............ 56..664 × 56..324
  //   Corner pockets centered at ...... (56,56) (664,56) (56,324) (664,324)
  //   Side pockets centered at ........ (360, 48) and (360, 332)
  //   Top rail diamond strip y ........ ~28   bottom strip y ~352
  //   8 numbered diamonds per long rail, with the side pocket between 4 and 5.

  // Pool table geometry. All measurements in SVG user units.
  //
  // The felt's inner edges are the cushions:
  //   top    inner edge y = 56
  //   bottom inner edge y = 324
  //   left   inner edge x = 56
  //   right  inner edge x = 664
  //
  // The cue ball has radius 13, so when its surface touches a cushion its
  // CENTER sits exactly one ball-radius (13) from that cushion's inner edge.
  // That's what aimContactY = 69 and the side-rail clamps [69, 651] enforce.
  const POOL = {
    ballR: 13,
    // Diamond x-positions on both long rails. Mirrored around the side pocket.
    diamondsX: [98, 168, 238, 308, 412, 482, 552, 622],
    topDiamondY:    28,   // diamond center on top rail
    bottomDiamondY: 352,  // diamond center on bottom rail
    cueBallY:       294,  // cue ball sits on felt near the bottom rail
    // Where the cue ball's CENTER comes to rest the instant it kisses the
    // top cushion. Was 62 (which let the ball clip ~7px into the cushion).
    // Now 56 + 13 = 69, which is geometrically correct for a 13px radius ball.
    aimContactY:    69,
    // Inner playing rectangle for the ball CENTER (cushion offsets baked in).
    // Used by the multi-rail bounce solver.
    playLeft:       69,    // 56 + ballR
    playRight:      651,   // 664 − ballR
    playTop:        69,    // 56 + ballR (same as aimContactY by definition)
    playBottom:     311,   // 324 − ballR
    cornerPockets: [
      { cx: 56,  cy: 56  },
      { cx: 664, cy: 56  },
      { cx: 56,  cy: 324 },
      { cx: 664, cy: 324 },
    ],
    sidePockets: [
      { cx: 360, cy: 48  },
      { cx: 360, cy: 332 },
    ],
  };

  function buildPoolSvg() {
    const svgNS = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(svgNS, "svg");
    svg.setAttribute("viewBox", "0 0 720 380");
    svg.classList.add("pool-svg");

    // ===== Defs: gradients, filters =====
    const defs = document.createElementNS(svgNS, "defs");
    defs.innerHTML = `
      <linearGradient id="woodOuter" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0"   stop-color="#6b4226"/>
        <stop offset="0.5" stop-color="#3b2415"/>
        <stop offset="1"   stop-color="#1d1006"/>
      </linearGradient>
      <linearGradient id="woodInner" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0"   stop-color="#824f2d"/>
        <stop offset="0.4" stop-color="#5b361d"/>
        <stop offset="1"   stop-color="#2d1a0c"/>
      </linearGradient>
      <linearGradient id="railHighlight" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0" stop-color="rgba(255,220,170,0.25)"/>
        <stop offset="1" stop-color="rgba(255,220,170,0)"/>
      </linearGradient>
      <radialGradient id="feltGrad" cx="0.5" cy="0.45" r="0.65">
        <stop offset="0"    stop-color="#2f9870"/>
        <stop offset="0.55" stop-color="#196b4d"/>
        <stop offset="1"    stop-color="#0a3f2a"/>
      </radialGradient>
      <radialGradient id="pocketGrad" cx="0.5" cy="0.5" r="0.5">
        <stop offset="0"    stop-color="#000000"/>
        <stop offset="0.7"  stop-color="#080604"/>
        <stop offset="1"    stop-color="#1c1108"/>
      </radialGradient>
      <radialGradient id="pocketRing" cx="0.5" cy="0.5" r="0.5">
        <stop offset="0" stop-color="#3c2418"/>
        <stop offset="1" stop-color="#1a0e07"/>
      </radialGradient>
      <radialGradient id="cueBallFill" cx="0.35" cy="0.30" r="0.75">
        <stop offset="0"    stop-color="#ffffff"/>
        <stop offset="0.35" stop-color="#f4f0e3"/>
        <stop offset="0.85" stop-color="#a8a293"/>
        <stop offset="1"    stop-color="#4a4338"/>
      </radialGradient>
      <radialGradient id="cueBallSpec" cx="0.30" cy="0.22" r="0.22">
        <stop offset="0" stop-color="#ffffff" stop-opacity="0.95"/>
        <stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
      </radialGradient>
      <radialGradient id="diamondGrad" cx="0.5" cy="0.3" r="0.7">
        <stop offset="0"    stop-color="#fff6dd"/>
        <stop offset="0.7"  stop-color="#e6cf94"/>
        <stop offset="1"    stop-color="#8a6a3c"/>
      </radialGradient>
      <filter id="ballShadow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="3"/>
      </filter>
      <filter id="feltNoise" x="0" y="0" width="100%" height="100%">
        <feTurbulence type="fractalNoise" baseFrequency="1.3" numOctaves="2" seed="9"/>
        <feColorMatrix values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.10 0"/>
        <feComposite in2="SourceGraphic" operator="in"/>
      </filter>
      <filter id="innerShadow" x="-10%" y="-10%" width="120%" height="120%">
        <feGaussianBlur in="SourceAlpha" stdDeviation="4"/>
        <feOffset dx="0" dy="2"/>
        <feComposite in2="SourceAlpha" operator="arithmetic" k2="-1" k3="1"/>
        <feColorMatrix values="0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.6 0"/>
        <feComposite in2="SourceGraphic" operator="over"/>
      </filter>
    `;
    svg.appendChild(defs);

    // ===== Wood frame: outer body + inner cap =====
    appendRect(svg, 0, 0, 720, 380, 20, "url(#woodOuter)");
    appendRect(svg, 10, 10, 700, 360, 14, "url(#woodInner)");
    // a thin highlight strip along the top inner edge to suggest grain
    const railLight = appendRect(svg, 12, 12, 696, 18, 8, "url(#railHighlight)");
    railLight.setAttribute("opacity", "0.6");

    // ===== Felt =====
    const feltX = 56, feltY = 56, feltW = 608, feltH = 268;
    appendRect(svg, feltX, feltY, feltW, feltH, 4, "url(#feltGrad)");
    const noiseLayer = appendRect(svg, feltX, feltY, feltW, feltH, 4, "#000");
    noiseLayer.setAttribute("filter", "url(#feltNoise)");
    noiseLayer.setAttribute("opacity", "0.55");

    // Soft moving spotlight under the felt — gives the table a faint
    // ambient "breathing" feel as if a low lamp is hanging over it.
    const feltLightDef = document.createElementNS(svgNS, "ellipse");
    feltLightDef.setAttribute("cx", feltX + feltW * 0.5);
    feltLightDef.setAttribute("cy", feltY + feltH * 0.45);
    feltLightDef.setAttribute("rx", feltW * 0.42);
    feltLightDef.setAttribute("ry", feltH * 0.45);
    feltLightDef.setAttribute("fill", "rgba(180,255,200,0.10)");
    feltLightDef.setAttribute("filter", "blur(28px)");
    feltLightDef.setAttribute("class", "felt-light");
    feltLightDef.setAttribute("pointer-events", "none");
    svg.appendChild(feltLightDef);
    // soft inner shadow along the cushions
    const cushionShadow = appendRect(svg, feltX, feltY, feltW, feltH, 4, "none");
    cushionShadow.setAttribute("stroke", "rgba(0,0,0,0.55)");
    cushionShadow.setAttribute("stroke-width", "3");
    cushionShadow.setAttribute("filter", "blur(1.2px)");
    cushionShadow.setAttribute("opacity", "0.7");

    // ===== Pockets: corner + side =====
    POOL.cornerPockets.forEach(p => drawPocket(svg, p.cx, p.cy, 24, 30));
    POOL.sidePockets.forEach(p => drawSidePocket(svg, p.cx, p.cy));

    // ===== Diamond inlays (numbered rail sights) =====
    POOL.diamondsX.forEach((x, i) => {
      drawDiamond(svg, x, POOL.topDiamondY);
      drawDiamondLabel(svg, x, 14, i + 1);
      drawDiamond(svg, x, POOL.bottomDiamondY);
      drawDiamondLabel(svg, x, 372, i + 1);
    });

    // ===== Layers used by renderShotOnTable =====
    // Lines first, so balls/markers render on top.
    const aimGroup = document.createElementNS(svgNS, "g");
    aimGroup.setAttribute("id", "aimGroup");
    svg.appendChild(aimGroup);

    // Ghost marker for ground-truth aim (green dashed ring on top cushion)
    const perfectMarker = document.createElementNS(svgNS, "circle");
    perfectMarker.setAttribute("r", 13);
    perfectMarker.setAttribute("fill", "none");
    perfectMarker.setAttribute("stroke", "#6ee7a7");
    perfectMarker.setAttribute("stroke-width", "1.6");
    perfectMarker.setAttribute("stroke-dasharray", "3 3");
    perfectMarker.setAttribute("opacity", "0");
    svg.appendChild(perfectMarker);

    // Engine's chosen aim marker (red disc + ring)
    const aimMarkerRing = document.createElementNS(svgNS, "circle");
    aimMarkerRing.setAttribute("r", 11);
    aimMarkerRing.setAttribute("fill", "none");
    aimMarkerRing.setAttribute("stroke", "#fb7185");
    aimMarkerRing.setAttribute("stroke-width", "1.2");
    aimMarkerRing.setAttribute("opacity", "0");
    svg.appendChild(aimMarkerRing);
    const aimMarker = document.createElementNS(svgNS, "circle");
    aimMarker.setAttribute("r", 6);
    aimMarker.setAttribute("fill", "#fb7185");
    aimMarker.setAttribute("opacity", "0");
    svg.appendChild(aimMarker);

    // Cast shadow under the cue ball
    const ballShadow = document.createElementNS(svgNS, "ellipse");
    ballShadow.setAttribute("rx", 13);
    ballShadow.setAttribute("ry", 4);
    ballShadow.setAttribute("fill", "#000");
    ballShadow.setAttribute("opacity", "0");
    ballShadow.setAttribute("filter", "url(#ballShadow)");
    svg.appendChild(ballShadow);

    // Soft halo behind the cue ball — gives the ball a "breathing" feel.
    // Sits below the ball itself but above the felt/aim layers.
    const cueBallGlow = document.createElementNS(svgNS, "circle");
    cueBallGlow.setAttribute("r", 22);
    cueBallGlow.setAttribute("fill", "rgba(255,255,255,0.55)");
    cueBallGlow.setAttribute("filter", "url(#ballShadow)");
    cueBallGlow.setAttribute("opacity", "0");
    cueBallGlow.setAttribute("class", "cue-ball-glow");
    svg.appendChild(cueBallGlow);

    // Cue ball
    const cueBall = document.createElementNS(svgNS, "circle");
    cueBall.setAttribute("r", 13);
    cueBall.setAttribute("fill", "url(#cueBallFill)");
    cueBall.setAttribute("opacity", "0");
    svg.appendChild(cueBall);

    // Specular highlight on the cue ball
    const cueBallSpec = document.createElementNS(svgNS, "circle");
    cueBallSpec.setAttribute("r", 5);
    cueBallSpec.setAttribute("fill", "url(#cueBallSpec)");
    cueBallSpec.setAttribute("opacity", "0");
    cueBallSpec.setAttribute("pointer-events", "none");
    svg.appendChild(cueBallSpec);

    return {
      svg,
      xs: POOL.diamondsX,
      aimGroup,
      perfectMarker,
      aimMarker,
      aimMarkerRing,
      cueBall,
      cueBallSpec,
      cueBallGlow,
      ballShadow,
    };
  }

  // --- SVG helpers used by buildPoolSvg ---
  function appendRect(parent, x, y, w, h, rx, fill) {
    const r = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    r.setAttribute("x", x); r.setAttribute("y", y);
    r.setAttribute("width", w); r.setAttribute("height", h);
    r.setAttribute("rx", rx);
    r.setAttribute("fill", fill);
    parent.appendChild(r);
    return r;
  }

  function drawPocket(svg, cx, cy, holeR, ringR) {
    const ns = "http://www.w3.org/2000/svg";
    const ring = document.createElementNS(ns, "circle");
    ring.setAttribute("cx", cx); ring.setAttribute("cy", cy);
    ring.setAttribute("r", ringR);
    ring.setAttribute("fill", "url(#pocketRing)");
    svg.appendChild(ring);
    const hole = document.createElementNS(ns, "circle");
    hole.setAttribute("cx", cx); hole.setAttribute("cy", cy);
    hole.setAttribute("r", holeR);
    hole.setAttribute("fill", "url(#pocketGrad)");
    svg.appendChild(hole);
    // A subtle inner rim
    const rim = document.createElementNS(ns, "circle");
    rim.setAttribute("cx", cx); rim.setAttribute("cy", cy);
    rim.setAttribute("r", holeR - 1);
    rim.setAttribute("fill", "none");
    rim.setAttribute("stroke", "rgba(255,255,255,0.07)");
    rim.setAttribute("stroke-width", "1");
    svg.appendChild(rim);
  }

  function drawSidePocket(svg, cx, cy) {
    const ns = "http://www.w3.org/2000/svg";
    const ring = document.createElementNS(ns, "ellipse");
    ring.setAttribute("cx", cx); ring.setAttribute("cy", cy);
    ring.setAttribute("rx", 32); ring.setAttribute("ry", 22);
    ring.setAttribute("fill", "url(#pocketRing)");
    svg.appendChild(ring);
    const hole = document.createElementNS(ns, "ellipse");
    hole.setAttribute("cx", cx); hole.setAttribute("cy", cy);
    hole.setAttribute("rx", 26); hole.setAttribute("ry", 16);
    hole.setAttribute("fill", "url(#pocketGrad)");
    svg.appendChild(hole);
  }

  function drawDiamond(svg, cx, cy) {
    const ns = "http://www.w3.org/2000/svg";
    const d = document.createElementNS(ns, "rect");
    const size = 11;
    d.setAttribute("x", cx - size / 2);
    d.setAttribute("y", cy - size / 2);
    d.setAttribute("width", size);
    d.setAttribute("height", size);
    d.setAttribute("transform", `rotate(45 ${cx} ${cy})`);
    d.setAttribute("fill", "url(#diamondGrad)");
    d.setAttribute("stroke", "#4a2e18");
    d.setAttribute("stroke-width", "0.6");
    svg.appendChild(d);
  }

  function drawDiamondLabel(svg, cx, cy, n) {
    const ns = "http://www.w3.org/2000/svg";
    const t = document.createElementNS(ns, "text");
    t.setAttribute("x", cx); t.setAttribute("y", cy);
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("font-family", "ui-monospace, monospace");
    t.setAttribute("font-size", "9");
    t.setAttribute("font-weight", "600");
    t.setAttribute("fill", "#d4b482");
    t.textContent = n;
    svg.appendChild(t);
  }

  // ---------- Ball animation engine ----------
  // Pockets the ball can drop into when reward is high. These map to the
  // SVG positions in buildPoolSvg().
  const POOL_POCKETS = [
    { x: 56,  y: 56,  kind: "corner-tl" },
    { x: 664, y: 56,  kind: "corner-tr" },
    { x: 56,  y: 324, kind: "corner-bl" },
    { x: 664, y: 324, kind: "corner-br" },
    { x: 360, y: 48,  kind: "side-top"  },
    { x: 360, y: 332, kind: "side-bot"  },
  ];

  const lerp     = (a, b, t) => a + (b - a) * t;
  const easeIn   = t => t * t;
  const easeOut  = t => 1 - (1 - t) * (1 - t);
  const easeInOut = t => t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t+2, 2)/2;
  function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  function animateOver(duration, easing, onTick) {
    return new Promise(resolve => {
      const start = performance.now();
      function tick(now) {
        const elapsed = now - start;
        const t = Math.min(1, elapsed / duration);
        onTick((easing || (x => x))(t));
        if (t < 1) requestAnimationFrame(tick);
        else resolve();
      }
      requestAnimationFrame(tick);
    });
  }

  function setBallPos(pool, x, y) {
    pool.cueBall.setAttribute("cx", x);
    pool.cueBall.setAttribute("cy", y);
    pool.cueBallSpec.setAttribute("cx", x - 4.2);
    pool.cueBallSpec.setAttribute("cy", y - 4.6);
    pool.ballShadow.setAttribute("cx", x);
    pool.ballShadow.setAttribute("cy", y + 11);
    if (pool.cueBallGlow) {
      pool.cueBallGlow.setAttribute("cx", x);
      pool.cueBallGlow.setAttribute("cy", y);
    }
  }

  function spawnPreShotRing(pool, x, y) {
    const ns = "http://www.w3.org/2000/svg";
    const ring = document.createElementNS(ns, "circle");
    ring.setAttribute("cx", x);
    ring.setAttribute("cy", y);
    ring.setAttribute("r", 12);
    ring.setAttribute("fill", "none");
    ring.setAttribute("stroke", "rgba(255,255,255,0.85)");
    ring.setAttribute("stroke-width", "2");
    ring.setAttribute("class", "pre-shot-ring");
    pool.svg.appendChild(ring);
    setTimeout(() => ring.remove(), 600);
  }

  function spawnPocketFireworks(pool, x, y) {
    const ns = "http://www.w3.org/2000/svg";
    const n = 8;
    const palette = ["#ffe9a8", "#a8ffe1", "#ffc7d8", "#d8c7ff", "#ffffff"];
    for (let i = 0; i < n; i++) {
      const angle = (i / n) * Math.PI * 2 + (Math.random() * 0.4 - 0.2);
      const dist = 22 + Math.random() * 14;
      const dx = Math.cos(angle) * dist;
      const dy = Math.sin(angle) * dist;
      const dot = document.createElementNS(ns, "circle");
      dot.setAttribute("cx", x);
      dot.setAttribute("cy", y);
      dot.setAttribute("r", 2 + Math.random() * 1.5);
      dot.setAttribute("fill", palette[Math.floor(Math.random() * palette.length)]);
      dot.setAttribute("class", "pocket-particle");
      pool.svg.appendChild(dot);
      requestAnimationFrame(() => {
        dot.setAttribute("cx", x + dx);
        dot.setAttribute("cy", y + dy);
        dot.style.transition = "cx 0.62s cubic-bezier(.2,.7,.2,1), cy 0.62s cubic-bezier(.2,.7,.2,1)";
      });
      setTimeout(() => dot.remove(), 750);
    }
  }

  // Briefly nudges the page-wide vignette so the "click of contact" is felt
  // beyond the SVG — the room flinches with the cue.
  function edgePulse() {
    document.body.classList.add("pulse");
    setTimeout(() => document.body.classList.remove("pulse"), 420);
  }

  function spawnContactFlash(pool, x, y) {
    const ns = "http://www.w3.org/2000/svg";
    // bright core
    const core = document.createElementNS(ns, "circle");
    core.setAttribute("cx", x);
    core.setAttribute("cy", y);
    core.setAttribute("r", 4);
    core.setAttribute("fill", "rgba(255,240,200,0.95)");
    core.setAttribute("class", "contact-flash");
    pool.svg.appendChild(core);
    // expanding ring
    const ring = document.createElementNS(ns, "circle");
    ring.setAttribute("cx", x);
    ring.setAttribute("cy", y);
    ring.setAttribute("r", 4);
    ring.setAttribute("fill", "none");
    ring.setAttribute("stroke", "rgba(255,240,200,0.85)");
    ring.setAttribute("stroke-width", "1.5");
    ring.setAttribute("class", "contact-ring");
    pool.svg.appendChild(ring);
    setTimeout(() => { core.remove(); ring.remove(); }, 480);
  }

  // Smaller cousin of spawnContactFlash for side-rail bounces — same
  // visual language (core + ring) but quieter, so the eye reads the
  // top-rail kiss as the loud event and side rails as glancing taps.
  function spawnRailBounce(pool, x, y) {
    const ns = "http://www.w3.org/2000/svg";
    const core = document.createElementNS(ns, "circle");
    core.setAttribute("cx", x); core.setAttribute("cy", y);
    core.setAttribute("r", 2.5);
    core.setAttribute("fill", "rgba(255,230,180,0.85)");
    core.setAttribute("class", "contact-flash");
    pool.svg.appendChild(core);
    const ring = document.createElementNS(ns, "circle");
    ring.setAttribute("cx", x); ring.setAttribute("cy", y);
    ring.setAttribute("r", 3);
    ring.setAttribute("fill", "none");
    ring.setAttribute("stroke", "rgba(255,230,180,0.7)");
    ring.setAttribute("stroke-width", "1.1");
    ring.setAttribute("class", "contact-ring");
    pool.svg.appendChild(ring);
    setTimeout(() => { core.remove(); ring.remove(); }, 380);
  }

  function spawnPocketSparkle(pool, x, y) {
    const ns = "http://www.w3.org/2000/svg";
    const s = document.createElementNS(ns, "circle");
    s.setAttribute("cx", x);
    s.setAttribute("cy", y);
    s.setAttribute("r", 2);
    s.setAttribute("fill", "none");
    s.setAttribute("stroke", "rgba(110,231,167,0.9)");
    s.setAttribute("stroke-width", "2");
    s.setAttribute("class", "pocket-sparkle");
    pool.svg.appendChild(s);
    setTimeout(() => s.remove(), 700);
  }

  function spawnTrail(pool, x, y) {
    const ns = "http://www.w3.org/2000/svg";
    const dot = document.createElementNS(ns, "circle");
    dot.setAttribute("cx", x);
    dot.setAttribute("cy", y);
    dot.setAttribute("r", 2.4);
    dot.setAttribute("fill", "rgba(255,255,255,0.55)");
    dot.setAttribute("class", "ball-trail");
    pool.svg.insertBefore(dot, pool.cueBall);   // behind the ball
    setTimeout(() => dot.remove(), 520);
  }

  // True if the ball, after geometric reflection off the top cushion,
  // would arrive within ~26px of a bottom-edge pocket center.
  function nearestPocket(x, y, pockets) {
    let best = null;
    pockets.forEach(p => {
      const d = Math.hypot(p.x - x, p.y - y);
      if (!best || d < best.d) best = { ...p, d };
    });
    return best;
  }

  // Scrolls the panel that owns this SVG so the table + four stat cards
  // sit ~18% from the top of the viewport, framing them as the only thing
  // worth looking at while a landmark plays.
  function focusOnTable(pool) {
    const panel = pool.svg.closest(".pool-panel")
               || pool.svg.closest(".panel")
               || pool.svg.parentElement;
    if (!panel) return null;
    const actCard = pool.svg.closest(".act-card");
    if (actCard) actCard.classList.add("in-focus");
    const rect = panel.getBoundingClientRect();
    const targetTop = window.scrollY + rect.top - window.innerHeight * 0.16;
    window.scrollTo({ top: Math.max(0, targetTop), behavior: "smooth" });
    return actCard;
  }

  function spawnVictoryHalo(pool, x, y) {
    const ns = "http://www.w3.org/2000/svg";
    // Two concentric rings for depth
    [
      { color: "rgba(110,231,167,0.9)", delay: 0 },
      { color: "rgba(252,211,77,0.7)",  delay: 120 },
    ].forEach(({ color, delay }) => {
      setTimeout(() => {
        const ring = document.createElementNS(ns, "circle");
        ring.setAttribute("cx", x);
        ring.setAttribute("cy", y);
        ring.setAttribute("r", 10);
        ring.setAttribute("fill", "none");
        ring.setAttribute("stroke", color);
        ring.setAttribute("stroke-width", "3");
        ring.setAttribute("class", "victory-halo");
        pool.svg.appendChild(ring);
        setTimeout(() => ring.remove(), 1000);
      }, delay);
    });
  }

  function spawnFloatingReward(pool, x, y, rewardText) {
    const ns = "http://www.w3.org/2000/svg";
    const txt = document.createElementNS(ns, "text");
    txt.setAttribute("x", x);
    txt.setAttribute("y", y - 18);
    txt.setAttribute("class", "floating-reward");
    txt.textContent = rewardText;
    txt.dataset.victoryMarker = "1";
    pool.svg.appendChild(txt);
    setTimeout(() => txt.remove(), 1200);
  }

  // Friction-based ease. The cue strike imparts instantaneous velocity, and
  // sliding/rolling friction decelerates the ball monotonically. The position
  // therefore covers most ground EARLY and crawls toward the endpoint — i.e.
  // an easeOut curve, never easeIn. We use a slightly stronger curve for
  // shots that come to rest (graze/miss) than for shots that drop (the ball
  // is still moving when it falls in).
  const easeRoll  = t => 1 - Math.pow(1 - t, 2.2);   // mild deceleration (drop shots)
  const easeStop  = t => 1 - Math.pow(1 - t, 2.8);   // stronger deceleration (rolls to a stop)

  async function playShotAnimation(pool, state, action, reward, options = {}) {
    const { onTrail = true } = options;
    const idx = i => POOL.diamondsX[Math.min(Math.max(i, 1), 8) - 1];
    const fromX = idx(state);
    const fromY = POOL.cueBallY;
    const aimX  = idx(action);
    const aimY  = POOL.aimContactY;

    // Same physics as the static ghost preview — pixel-for-pixel agreement.
    const path = computeShotPath(fromX, fromY, aimX, aimY, reward);
    const waypoints = path.waypoints;
    const velocities = path.velocities;
    const willPocket = path.willPocket;
    const endX = path.endX, endY = path.endY;

    // ===== Spotlight mode begins =====
    // Lock auto-scroll, frame the panel, and (on a hit) dim the rest of
    // the page so the eye has nowhere else to wander. Saved & restored.
    const prevAutoFollow = (typeof autoFollow !== "undefined") ? autoFollow : true;
    if (typeof autoFollow !== "undefined") autoFollow = false;
    const focusedCard = focusOnTable(pool);
    if (willPocket) document.body.classList.add("page-dimmed");

    // Freeze CSS transitions during the animation so rAF owns the ball.
    pool.svg.classList.add("animating");

    // Pre-shot anticipation: a quick expanding ring around the ball.
    // This is the moment the wave collapses and one trajectory is chosen.
    spawnPreShotRing(pool, fromX, fromY);
    await sleep(180);

    // ----- Animate each segment with friction-correct timing -----
    //
    // A struck ball decelerates the entire time it's in motion. We pace
    // each leg so its duration is proportional to length/speed; speed
    // drops at every cushion bounce (CUSHION_LOSS). The first leg uses
    // an easeOut so even THAT segment slows visibly, instead of the old
    // easeIn which made the ball look like it was accelerating from a
    // standstill.
    //
    // For a 13-unit ball at 720-unit table width, ~0.55 ms/unit roughly
    // matches a medium-power pool stroke. Side-rail bounces inherit the
    // cushion velocity loss, so multi-rail bank shots naturally take
    // longer (the ball is slower after each impact). Roll-to-rest shots
    // get an extra time tax so their deceleration reads cleanly.
    const BASE_MS_PER_UNIT = 0.55;
    const totalLegs = waypoints.length - 1;
    let lastTrailT = 0;

    for (let leg = 0; leg < totalLegs; leg++) {
      const a = waypoints[leg];
      const b = waypoints[leg + 1];
      const dx = b.x - a.x, dy = b.y - a.y;
      const dist = Math.hypot(dx, dy);
      const speed = velocities[leg] || 1.0;
      // Last leg may end in either a pocket drop OR a roll-to-rest; in
      // both cases the deceleration is real. Drop shots use a lighter
      // curve so the ball is still moving when it falls in.
      const isLast = (leg === totalLegs - 1);
      const easing = isLast ? (willPocket ? easeRoll : easeStop) : easeRoll;
      // Roll-to-stop needs a touch of extra time so the deceleration is
      // perceptible. Drop shots stay snappy.
      const slowdownTax = (isLast && !willPocket) ? 1.18 : 1.0;
      const duration = Math.max(120, (dist * BASE_MS_PER_UNIT * slowdownTax) / speed);

      lastTrailT = 0;
      await animateOver(duration, easing, t => {
        const x = lerp(a.x, b.x, t);
        const y = lerp(a.y, b.y, t);
        setBallPos(pool, x, y);
        if (onTrail && t - lastTrailT > 0.18) {
          spawnTrail(pool, x, y);
          lastTrailT = t;
        }
      });

      // Cushion impact effects: the top-rail kiss is the loud one; side
      // rails get a smaller tick. Always fire AFTER the leg lands.
      if (b.kind === "cushion-top") {
        spawnContactFlash(pool, b.x, b.y);
        edgePulse();
        await sleep(40);
      } else if (b.kind === "rail-left" || b.kind === "rail-right") {
        spawnRailBounce(pool, b.x, b.y);
        await sleep(28);
      }
    }

    // Phase 3: drop into pocket if we made it — and HOLD the moment.
    if (willPocket) {
      spawnPocketSparkle(pool, endX, endY);
      spawnPocketFireworks(pool, endX, endY);
      edgePulse();
      await animateOver(260, easeIn, t => {
        const r = 13 * (1 - t);
        pool.cueBall.setAttribute("r", r);
        pool.cueBallSpec.setAttribute("r", Math.max(0, 5 * (1 - 1.2 * t)));
        pool.cueBall.setAttribute("opacity", 1 - t);
        pool.cueBallSpec.setAttribute("opacity", 1 - t);
        pool.ballShadow.setAttribute("opacity", 0.55 * (1 - t));
      });

      // ===== Victory beat — the "keep it in focus" moment =====
      // The ball is gone. The table glows green. A halo expands from the
      // pocket. A floating "+reward" rises like a damage-number. The four
      // stat cards pulse together. We GATE the SSE event stream so the
      // rapid non-landmark shots that come next don't drown the moment.
      // Hold for ~1700ms — long enough that the eye lands on the moment
      // and the four cards have time to be read; short enough that pacing
      // doesn't suffer. Combined with the matching backend hit-hold (3.4s),
      // this creates a deliberate "pause and breathe" beat.
      const VICTORY_HOLD = 1700;
      if (typeof holdSpotlight === "function") holdSpotlight(VICTORY_HOLD);
      pool.svg.classList.add("victory");
      spawnVictoryHalo(pool, endX, endY);
      const rewardLabel = (reward >= 0.999) ? "+1.00" : ("+" + reward.toFixed(2));
      spawnFloatingReward(pool, endX, endY, rewardLabel);
      const statsEl = document.querySelector(".shot-stats");
      if (statsEl) {
        statsEl.classList.remove("victory-cards");
        // force reflow so the animation restarts even on consecutive hits
        void statsEl.offsetWidth;
        statsEl.classList.add("victory-cards");
        setTimeout(() => statsEl.classList.remove("victory-cards"), 1100);
      }
      await sleep(VICTORY_HOLD);
      pool.svg.classList.remove("victory");

      // restore ball for next shot
      pool.cueBall.setAttribute("r", 13);
      pool.cueBallSpec.setAttribute("r", 5);
    }

    // Reset to rest position at fromX so the next event finds the ball there
    setBallPos(pool, fromX, fromY);
    pool.cueBall.setAttribute("opacity", 1);
    pool.cueBallSpec.setAttribute("opacity", 1);
    pool.ballShadow.setAttribute("opacity", 0.55);

    pool.svg.classList.remove("animating");

    // ===== Release spotlight =====
    document.body.classList.remove("page-dimmed");
    if (focusedCard) focusedCard.classList.remove("in-focus");
    if (typeof autoFollow !== "undefined") autoFollow = prevAutoFollow;

    return { willPocket, endX, endY };
  }

  // ---------- Realistic pool physics ----------
  //
  // computeShotPath traces the cue ball through the table the way a real
  // shot moves: a struck ball heads to its aim point on the far cushion,
  // bounces back, can bounce off either side rail along the way, and
  // either drops into a pocket or rolls to rest somewhere on the felt.
  //
  // We model:
  //   • Specular (angle-in = angle-out) reflection off every cushion.
  //   • A per-bounce velocity loss (cushion compression ≈ 25% loss).
  //   • Up to MAX_BOUNCES side-rail reflections after the first top-rail
  //     bounce. Two bounces is plenty for our 720×380 geometry.
  //   • A "magnetic pocket" check on the final endpoint: a near-perfect
  //     shot whose reflected trajectory dies within POCKET_DRAW_RADIUS of a
  //     bottom-edge pocket gets drawn in. The radius is generous enough to
  //     feel celebratory but tight enough to LOOK like the pocket caused it.
  //
  // Returns { waypoints, willPocket, endX, endY, velocities }, where
  //   waypoints: [{x,y,kind}] starting at the cue ball and ending at the
  //              final resting place (or the pocket).
  //   velocities: per-segment speed factor (relative to the strike speed),
  //               used by the animator to time each leg with friction.
  //
  // Physical constants picked to read well at SVG scale, not to match a
  // brand of cloth — but the SHAPES are what real pool actually looks like.
  //
  //   POCKET_DRAW_RADIUS — distance from a pocket center at which a near-
  //     perfect shot is "drawn in." Real pocket mouths on a 9-foot table
  //     are about 2× ball diameter (~50–60 SVG units here). We use 130 to
  //     give credit to the agent's symmetric reward function — a perfectly
  //     aimed shot may not always geometrically land on a pocket center,
  //     but if it lands within half a diamond of one, the felt rolls it in.
  //   POCKET_RAIL_RADIUS — at side-rail collisions, if the contact point
  //     is closer than this to a corner pocket, the ball falls into the
  //     pocket instead of bouncing. This models a real pool table behavior
  //     ("rim shot drop") and turns shots like state=4 aim=8 into beautiful
  //     one-rail kicks that drop into the TOP corner pocket.
  //   CUSHION_LOSS — fraction of velocity preserved across each bounce.
  //     Real cushions lose 20–30% of perpendicular velocity per impact.
  //   MAX_BOUNCES — safety cap so a shot can't loop forever between rails.
  const POCKET_DRAW_RADIUS = 130;
  const POCKET_RAIL_RADIUS = 72;
  const CUSHION_LOSS       = 0.78;
  const MAX_BOUNCES        = 3;

  function computeShotPath(fromX, fromY, aimX, aimY, reward) {
    const { playLeft, playRight } = POOL;
    const waypoints = [
      { x: fromX, y: fromY, kind: "start" },
      { x: aimX,  y: aimY,  kind: "cushion-top" },
    ];
    const velocities = [1.0, CUSHION_LOSS];   // strike speed → after top bounce

    // Did the top-cushion contact land near a top-corner pocket? If so the
    // ball never bounces — it grazes the cushion and drops into the pocket.
    // We keep both waypoints (cushion-top + pocket) so the animator still
    // plays the strike → graze → drop arc instead of teleporting to the hole.
    const topPockets = POOL_POCKETS.filter(p => p.kind.startsWith("corner-t"));
    const topNear    = nearestPocket(aimX, aimY, topPockets);
    if (reward >= 0.85 && topNear.d < POCKET_RAIL_RADIUS) {
      waypoints.push({ x: topNear.x, y: topNear.y, kind: "pocket" });
      velocities.push(CUSHION_LOSS * CUSHION_LOSS);
      return {
        waypoints, velocities, willPocket: true, pocket: topNear,
        endX: topNear.x, endY: topNear.y,
      };
    }

    // Direction after the top-cushion bounce: horizontal component is
    // preserved, vertical component is reflected. The ball heads back
    // toward y = fromY.
    let cx = aimX, cy = aimY;
    let dx = aimX - fromX;
    let dy = fromY - aimY;          // strictly positive (going down)
    let speed = CUSHION_LOSS;

    // Pocket list for "rim shot" detection on intermediate side-rail bounces.
    // Only the four corners are reachable from a side-rail collision point.
    const cornerPockets = POOL_POCKETS.filter(p => p.kind.startsWith("corner-"));

    for (let i = 0; i < MAX_BOUNCES; i++) {
      // Parametric t at which the trajectory reaches the cue ball's rest
      // row (y = fromY). At t = 1 the ball would land at the geometric
      // reflection point. Side rails may cut this short.
      const tTarget = (fromY - cy) / dy;
      let tHit  = tTarget;
      let kind  = "end";
      let hitX  = cx + dx * tTarget;
      let hitY  = fromY;

      if (dx > 1e-9) {
        const t = (playRight - cx) / dx;
        if (t > 1e-6 && t < tHit) {
          tHit = t; kind = "rail-right";
          hitX = playRight; hitY = cy + dy * t;
        }
      } else if (dx < -1e-9) {
        const t = (playLeft - cx) / dx;
        if (t > 1e-6 && t < tHit) {
          tHit = t; kind = "rail-left";
          hitX = playLeft; hitY = cy + dy * t;
        }
      }

      // "Rim shot" on side rails — if the ball contacts the rail close to
      // a corner pocket, the pocket eats it instead of letting it bounce.
      if (kind === "rail-left" || kind === "rail-right") {
        const cornerNear = nearestPocket(hitX, hitY, cornerPockets);
        if (reward >= 0.85 && cornerNear.d < POCKET_RAIL_RADIUS) {
          waypoints.push({ x: cornerNear.x, y: cornerNear.y, kind: "pocket" });
          return {
            waypoints, velocities, willPocket: true, pocket: cornerNear,
            endX: cornerNear.x, endY: cornerNear.y,
          };
        }
      }

      waypoints.push({ x: hitX, y: hitY, kind });
      if (kind === "end") break;

      // Side rail bounce: reverse the horizontal component and pay the
      // cushion velocity tax.
      cx = hitX; cy = hitY; dx = -dx;
      speed *= CUSHION_LOSS;
      velocities.push(speed);
    }

    // Pocket detection on the final endpoint (bottom edge — that's where the
    // ball rests after coming back down off the top cushion).
    const last = waypoints[waypoints.length - 1];
    const bottomEdge = POOL_POCKETS.filter(
      p => p.kind.startsWith("corner-b") || p.kind === "side-bot"
    );
    const pocket = nearestPocket(last.x, last.y, bottomEdge);
    const willPocket = reward >= 0.85 && pocket.d < POCKET_DRAW_RADIUS;

    let endX, endY;
    if (willPocket) {
      // Snap the last waypoint to the pocket mouth — the magnetic finish.
      // We preserve the previous waypoint's position so the animated leg
      // still curves naturally toward the pocket.
      endX = pocket.x; endY = pocket.y;
      last.x = endX; last.y = endY;
      last.kind = "pocket";
    } else {
      endX = last.x; endY = last.y;
    }

    return { waypoints, velocities, willPocket, pocket: willPocket ? pocket : null, endX, endY };
  }

  // ---------- Outcome prediction & visualization ----------
  // Given a state, the engine's chosen aim, and the ground-truth perfect aim,
  // compute everything you need to *show* what will happen on this shot.
  // Uses computeShotPath() so the predicted trajectory and the animated one
  // are guaranteed to agree pixel-for-pixel.
  function predictShot(state, aim, perfect) {
    const idx = i => POOL.diamondsX[Math.min(Math.max(i, 1), 8) - 1];
    const fromX = idx(state);
    const fromY = POOL.cueBallY;
    const aimX  = idx(aim);
    const aimY  = POOL.aimContactY;
    // Reward = 1 − 0.5 · |aim − perfect|, clamped to [0, 1].
    const reward = Math.max(0, Math.min(1, 1 - 0.5 * Math.abs(aim - perfect)));
    const path = computeShotPath(fromX, fromY, aimX, aimY, reward);
    const verdict = reward >= 0.85 ? "drop"
                  : reward >= 0.45 ? "graze"
                  : "miss";
    return {
      fromX, fromY, aimX, aimY,
      endX: path.endX, endY: path.endY,
      waypoints: path.waypoints,
      velocities: path.velocities,
      reward, verdict, willPocket: path.willPocket,
    };
  }

  // Draw the reflection path (rail → ball's resting place) and an endpoint
  // marker that color-codes the outcome at a glance:
  //    DROP  → green halo + dot at the pocket
  //    GRAZE → amber dashed circle where the ball comes to rest
  //    MISS  → red dashed circle (with a label) where the ball stops short
  // This is what lets the reader SEE the shot's whole future — not just the aim.
  const OUTCOME_COLORS = {
    drop:  "#6ee7a7",
    graze: "#fcd34d",
    miss:  "#fb7185",
  };
  function drawOutcome(targetGroup, prediction, options = {}) {
    const { aimX, aimY, endX, endY, verdict, willPocket, waypoints } = prediction;
    const {
      opacity = 1,
      isGhost = false,
      showLabel = false,
      aimNum = null,
    } = options;
    const ns = "http://www.w3.org/2000/svg";
    const color = OUTCOME_COLORS[verdict];

    // Multi-segment reflection polyline. Starts at the top-cushion contact
    // point and traces every rail bounce until the final endpoint. For a
    // single-bounce shot this is one line; for a bank shot off the side
    // rail it draws the bent path so the eye reads the geometry honestly.
    const reflPath = (waypoints || [])
      .slice(1)  // skip the cue ball start position
      .map(w => `${w.x},${w.y}`)
      .join(" ");
    const refl = document.createElementNS(ns, "polyline");
    refl.setAttribute("points", reflPath);
    refl.setAttribute("fill", "none");
    refl.setAttribute("stroke", color);
    refl.setAttribute("stroke-width", isGhost ? "1" : "1.5");
    refl.setAttribute("stroke-dasharray", "4 5");
    refl.setAttribute("stroke-linecap", "round");
    refl.setAttribute("stroke-linejoin", "round");
    refl.setAttribute("opacity", String(opacity * (isGhost ? 0.5 : 0.75)));
    refl.setAttribute("class", "outcome-line outcome-line-" + verdict);
    if (aimNum != null) refl.dataset.aim = String(aimNum);
    targetGroup.appendChild(refl);

    // Side-rail bounce markers — tiny chevrons that hint at energy loss.
    (waypoints || []).slice(1, -1).forEach(w => {
      if (w.kind !== "rail-left" && w.kind !== "rail-right") return;
      const mark = document.createElementNS(ns, "circle");
      mark.setAttribute("cx", w.x);
      mark.setAttribute("cy", w.y);
      mark.setAttribute("r", isGhost ? 2.4 : 3.2);
      mark.setAttribute("fill", "none");
      mark.setAttribute("stroke", color);
      mark.setAttribute("stroke-width", isGhost ? "1" : "1.3");
      mark.setAttribute("opacity", String(opacity * (isGhost ? 0.6 : 0.85)));
      mark.setAttribute("class", "outcome-bounce-mark");
      targetGroup.appendChild(mark);
    });

    if (willPocket) {
      // DROP — bright green halo at the pocket + inner solid dot.
      const halo = document.createElementNS(ns, "circle");
      halo.setAttribute("cx", endX); halo.setAttribute("cy", endY);
      halo.setAttribute("r", isGhost ? 8 : 13);
      halo.setAttribute("fill", "none");
      halo.setAttribute("stroke", color);
      halo.setAttribute("stroke-width", isGhost ? "1.5" : "2.4");
      halo.setAttribute("opacity", String(opacity * 0.9));
      halo.setAttribute("class", "outcome-marker outcome-drop-halo");
      if (aimNum != null) halo.dataset.aim = String(aimNum);
      targetGroup.appendChild(halo);

      const dot = document.createElementNS(ns, "circle");
      dot.setAttribute("cx", endX); dot.setAttribute("cy", endY);
      dot.setAttribute("r", isGhost ? 3.2 : 5.2);
      dot.setAttribute("fill", color);
      dot.setAttribute("opacity", String(opacity));
      dot.setAttribute("class", "outcome-marker outcome-drop-dot");
      if (aimNum != null) dot.dataset.aim = String(aimNum);
      targetGroup.appendChild(dot);
    } else {
      // MISS / GRAZE — a dashed circle showing the ball's resting place.
      const marker = document.createElementNS(ns, "circle");
      marker.setAttribute("cx", endX); marker.setAttribute("cy", endY);
      marker.setAttribute("r", isGhost ? 5.4 : 10);
      marker.setAttribute("fill", "none");
      marker.setAttribute("stroke", color);
      marker.setAttribute("stroke-width", isGhost ? "1.4" : "1.8");
      marker.setAttribute("stroke-dasharray", "3 3");
      marker.setAttribute("opacity", String(opacity * 0.9));
      marker.setAttribute("class", "outcome-marker outcome-" + verdict);
      if (aimNum != null) marker.dataset.aim = String(aimNum);
      targetGroup.appendChild(marker);

      const inner = document.createElementNS(ns, "circle");
      inner.setAttribute("cx", endX); inner.setAttribute("cy", endY);
      inner.setAttribute("r", isGhost ? 1.7 : 2.8);
      inner.setAttribute("fill", color);
      inner.setAttribute("opacity", String(opacity * 0.65));
      inner.setAttribute("class", "outcome-marker outcome-" + verdict);
      if (aimNum != null) inner.dataset.aim = String(aimNum);
      targetGroup.appendChild(inner);
    }

    // Draw a ghost "resting cue ball" on committed shots so the endpoint reads
    // as "the white ball ends HERE", not just an abstract marker.
    if (!isGhost) {
      const endBall = document.createElementNS(ns, "circle");
      endBall.setAttribute("cx", endX);
      endBall.setAttribute("cy", endY);
      endBall.setAttribute("r", willPocket ? "5.5" : "7.2");
      endBall.setAttribute("opacity", String(opacity * (willPocket ? 0.35 : 0.9)));
      endBall.setAttribute("class", "outcome-end-ball outcome-end-ball-" + verdict);
      targetGroup.appendChild(endBall);

      // Flash the lower rail segment under the endpoint so the eye instantly
      // reads where the shot "lands" in table-space.
      const rail = document.createElementNS(ns, "line");
      const railHalf = willPocket ? 40 : 52;
      rail.setAttribute("x1", Math.max(72, endX - railHalf));
      rail.setAttribute("x2", Math.min(648, endX + railHalf));
      rail.setAttribute("y1", "323");
      rail.setAttribute("y2", "323");
      rail.setAttribute("opacity", String(opacity));
      rail.setAttribute("class", "outcome-rail-flash outcome-rail-flash-" + verdict);
      targetGroup.appendChild(rail);
    }

    if (showLabel) {
      // Bold one-word verdict label near the endpoint.
      const label = document.createElementNS(ns, "text");
      // Push label slightly inward so it doesn't get clipped at the rails.
      const labelDx = endX > 360 ? -14 : 14;
      const labelDy = endY > POOL.cueBallY ? 22 : -16;
      const labelX = endX + labelDx;
      const labelY = endY + labelDy;
      label.setAttribute("x", labelX);
      label.setAttribute("y", labelY);
      label.setAttribute("text-anchor", endX > 360 ? "end" : "start");
      label.setAttribute("class", "outcome-label outcome-label-" + verdict);
      label.textContent = willPocket ? "ENDS: DROP"
                        : verdict === "graze" ? "ENDS: GRAZE"
                        : "ENDS: MISS";
      targetGroup.appendChild(label);

      // Animated pointer from label to endpoint.
      const pointer = document.createElementNS(ns, "line");
      const pointerStartX = endX > 360 ? (labelX - 6) : (labelX + 6);
      const pointerStartY = labelY - 5;
      pointer.setAttribute("x1", pointerStartX);
      pointer.setAttribute("y1", pointerStartY);
      pointer.setAttribute("x2", endX);
      pointer.setAttribute("y2", endY);
      pointer.setAttribute("class", "outcome-pointer outcome-pointer-" + verdict);
      targetGroup.appendChild(pointer);

      // Tiny arrowhead at endpoint (manual triangle, no defs required).
      const vx = endX - pointerStartX;
      const vy = endY - pointerStartY;
      const mag = Math.max(1e-6, Math.hypot(vx, vy));
      const ux = vx / mag, uy = vy / mag;
      const px = -uy, py = ux;
      const tipX = endX, tipY = endY;
      const backX = endX - ux * 7;
      const backY = endY - uy * 7;
      const leftX = backX + px * 3.2;
      const leftY = backY + py * 3.2;
      const rightX = backX - px * 3.2;
      const rightY = backY - py * 3.2;
      const arrow = document.createElementNS(ns, "polygon");
      arrow.setAttribute("points", `${tipX},${tipY} ${leftX},${leftY} ${rightX},${rightY}`);
      arrow.setAttribute("fill", color);
      arrow.setAttribute("opacity", "0.95");
      arrow.setAttribute("class", "outcome-pointer outcome-pointer-" + verdict);
      targetGroup.appendChild(arrow);

      // Floating contextual chip.
      const chip = document.createElementNS(ns, "g");
      chip.setAttribute("class", "outcome-chip outcome-chip-" + verdict);
      const chipW = 122;
      const chipH = 20;
      const chipX = endX > 360 ? (labelX - chipW + 6) : (labelX - 6);
      const chipY = labelY - 42;
      const chipRect = document.createElementNS(ns, "rect");
      chipRect.setAttribute("x", chipX);
      chipRect.setAttribute("y", chipY);
      chipRect.setAttribute("width", String(chipW));
      chipRect.setAttribute("height", String(chipH));
      chipRect.setAttribute("rx", "6");
      chipRect.setAttribute("ry", "6");
      const chipText = document.createElementNS(ns, "text");
      chipText.setAttribute("x", chipX + 8);
      chipText.setAttribute("y", chipY + 13.5);
      chipText.textContent = "final resting spot";
      chip.appendChild(chipRect);
      chip.appendChild(chipText);
      targetGroup.appendChild(chip);
    }
  }

  function renderShotOnTable(pool, state, action, perfect, options = {}) {
    const { showPerfect = true, ood = false } = options;
    const {
      xs, aimGroup, perfectMarker, aimMarker, aimMarkerRing,
      cueBall, cueBallSpec, ballShadow, cueBallGlow,
    } = pool;

    const clamp = idx => xs[Math.min(Math.max(idx, 1), 8) - 1];
    const ballX    = clamp(state);
    const ballY    = POOL.cueBallY;
    const aimX     = clamp(action);
    const aimY     = POOL.aimContactY;
    const perfectX = clamp(perfect);

    // Cast shadow (placed slightly below the ball)
    ballShadow.setAttribute("cx", ballX);
    ballShadow.setAttribute("cy", ballY + 11);
    ballShadow.setAttribute("opacity", "0.55");

    // Soft halo behind the ball that breathes in CSS.
    if (cueBallGlow) {
      cueBallGlow.setAttribute("cx", ballX);
      cueBallGlow.setAttribute("cy", ballY);
      cueBallGlow.setAttribute("fill", ood ? "rgba(251,113,133,0.55)" : "rgba(255,255,255,0.55)");
    }

    // Cue ball
    cueBall.setAttribute("cx", ballX);
    cueBall.setAttribute("cy", ballY);
    cueBall.setAttribute("opacity", "1");
    if (ood) {
      cueBall.setAttribute("stroke", "#fb7185");
      cueBall.setAttribute("stroke-width", "3");
      cueBall.classList.add("ood-pulse");
    } else {
      cueBall.setAttribute("stroke", "rgba(0,0,0,0.55)");
      cueBall.setAttribute("stroke-width", "0.6");
      cueBall.classList.remove("ood-pulse");
    }

    // Specular highlight (offset top-left of the ball)
    cueBallSpec.setAttribute("cx", ballX - 4.2);
    cueBallSpec.setAttribute("cy", ballY - 4.6);
    cueBallSpec.setAttribute("opacity", "1");

    // Chosen aim marker on the top rail
    aimMarker.setAttribute("cx", aimX);
    aimMarker.setAttribute("cy", aimY);
    aimMarker.setAttribute("opacity", "1");
    aimMarkerRing.setAttribute("cx", aimX);
    aimMarkerRing.setAttribute("cy", aimY);
    aimMarkerRing.setAttribute("opacity", "0.8");

    // Ground-truth ghost ring
    if (showPerfect) {
      perfectMarker.setAttribute("cx", perfectX);
      perfectMarker.setAttribute("cy", aimY);
      perfectMarker.setAttribute("opacity", perfect === action ? "0.65" : "0.9");
    } else {
      perfectMarker.setAttribute("opacity", "0");
    }

    // Aim line(s) — engine's choice (solid-ish) and ground truth (green ghost).
    // The chosen aim line gets the .aim-line class so its dashes flow toward
    // the cushion (CSS animation), suggesting an active, live trajectory.
    aimGroup.innerHTML = "";
    const ns = "http://www.w3.org/2000/svg";
    const mkLine = (x2, y2, color, width, dash, opacity, alive) => {
      const ln = document.createElementNS(ns, "line");
      ln.setAttribute("x1", ballX); ln.setAttribute("y1", ballY - 12);
      ln.setAttribute("x2", x2);    ln.setAttribute("y2", y2);
      ln.setAttribute("stroke", color);
      ln.setAttribute("stroke-width", width);
      ln.setAttribute("stroke-dasharray", dash);
      ln.setAttribute("stroke-linecap", "round");
      ln.setAttribute("opacity", opacity);
      if (alive) ln.setAttribute("class", "aim-line");
      aimGroup.appendChild(ln);
    };
    if (showPerfect && perfect !== action) {
      mkLine(perfectX, aimY + 6, "#6ee7a7", "1.2", "2 4", "0.55", false);
      // Also show where the ground-truth aim would have landed — the
      // "what reality wanted" outcome. Faint, green, so it whispers.
      const truthOutcome = predictShot(state, perfect, perfect);
      drawOutcome(aimGroup, truthOutcome, { opacity: 0.45, isGhost: true });
    }
    mkLine(aimX, aimY + 6, ood ? "#fb7185" : "#f8e6c4", "1.6", "6 5", "0.92", true);

    // The big visual payoff — what *actually* happens on this shot. Reflection
    // line + color-coded endpoint marker + a one-word verdict label.
    const outcome = predictShot(state, action, perfect);
    drawOutcome(aimGroup, outcome, { opacity: 1, isGhost: false, showLabel: true });
  }

  // --- Story-event renderers ---

  let scrollTimer = null;
  let autoFollow = true;
  // If the user scrolls more than ~120px up from the bottom, stop auto-following.
  // When they come back near the bottom, resume.
  window.addEventListener("scroll", () => {
    const distance = document.body.scrollHeight - window.scrollY - window.innerHeight;
    autoFollow = distance < 200;
  }, { passive: true });

  function appendToStory(node) {
    if (node && !node.parentElement) story.appendChild(node);
    if (!autoFollow) return;
    if (scrollTimer) clearTimeout(scrollTimer);
    scrollTimer = setTimeout(() => {
      const target = document.body.scrollHeight - window.innerHeight - 100;
      window.scrollTo({ top: Math.max(0, target), behavior: "smooth" });
    }, 120);
  }

  function ensureActBody() {
    if (!actBody) {
      actBody = make("div", { class: "act-body" });
      story.appendChild(actBody);
    }
    return actBody;
  }

  function updateProgress(step) {
    $$("#progress li").forEach((li, i) => {
      li.classList.toggle("current", i === step);
      li.classList.toggle("done", i < step);
    });
  }

  function renderActStart(ev) {
    actBody = null;
    currentVizPanels = null;
    const card = make("section", { class: "act-card", id: "act-" + ev.id }, [
      make("div", { class: "act-label", text: ev.label }),
      make("h2", { class: "act-title", text: ev.title }),
      make("p", { class: "act-subtitle", text: ev.subtitle }),
      make("div", { class: "act-divider" }),
    ]);
    appendToStory(card);
    updateProgress(ev.progress);
    actBody = make("div", { class: "act-body" });
    story.appendChild(actBody);
  }

  function renderNarration(ev) {
    const body = ensureActBody();
    const node = make("div", { class: "narration" });
    ev.paragraphs.forEach(p => {
      const isBullet = /^[•▸\-]\s/.test(p) || /^\d\.\s/.test(p);
      const clean = isBullet ? p.replace(/^[•▸\-]\s/, "").replace(/^\d\.\s/, "") : p;
      node.appendChild(make("p", { class: isBullet ? "bullet" : "", text: clean }));
    });
    body.appendChild(node);
    appendToStory(make("div"));  // ensure scroll
  }

  function renderCallout(ev) {
    const body = ensureActBody();
    const co = make("div", { class: "callout" });
    ev.lines.forEach((l, i) => {
      co.appendChild(make("div", { class: "line" + (i === 0 ? "" : " dim"), text: l }));
    });
    body.appendChild(co);
    appendToStory(make("div"));
  }

  // The "river of thinking" callout — a thematic variant of the amber
  // callout, tinted cyan, with a slow flowing-current accent on the left
  // edge. Same content structure as a callout (header line + dim lines)
  // so editors can reuse the same shape.
  function renderRiverCallout(ev) {
    const body = ensureActBody();
    const co = make("div", { class: "callout callout-river" });
    // The decorative "current" track on the left edge — a thin gradient
    // that runs vertically and animates so the eye reads it as flowing
    // water carrying the metaphor.
    co.appendChild(make("div", { class: "river-current", "aria-hidden": "true" }));
    (ev.lines || []).forEach((l, i) => {
      co.appendChild(make("div", { class: "line" + (i === 0 ? "" : " dim"), text: l }));
    });
    body.appendChild(co);
    appendToStory(make("div"));
  }

  function ensureVizPanel(opts = { dual: true }) {
    if (currentVizPanels) return currentVizPanels;
    const body = ensureActBody();
    const viz = make("div", { class: "viz dual" });
    const poolPanel = make("div", { class: "panel pool-wrap" }, [
      make("h4", { text: "The world · pool table" }),
    ]);
    const pool = buildPoolSvg();
    poolPanel.appendChild(pool.svg);

    // ----- Narrative-mapped stat cards (the four moments of one shot) -----
    // collapse  → which attempt # (every shot collapses a wave of possible futures)
    // world     → the state of the table the engine is reading right now
    // bet       → the engine's confident guess about the geometry
    // reward    → reality answering back (the click, the drop, the silence)
    const stats = make("div", { class: "shot-stats" });

    const collapseCaption = make("div", { class: "stat-caption", text: "another shot, another future chosen" });
    const collapseCard = make("div", { class: "stat-card is-collapse" }, [
      make("div", { class: "stat-label" }, [
        make("span", { class: "stat-glyph", html: "✦" }),
        make("span", { text: "collapse" }),
      ]),
      make("div", { class: "stat-value", html: '<span class="num">—</span>' }),
      collapseCaption,
    ]);

    const worldCaption = make("div", { class: "stat-caption", text: "where reality is sitting right now" });
    const worldCard = make("div", { class: "stat-card is-world" }, [
      make("div", { class: "stat-label" }, [
        make("span", { class: "stat-glyph", html: "●" }),
        make("span", { text: "world read" }),
      ]),
      make("div", { class: "stat-value", html: '<span class="num">—</span><span class="unit">diamond</span>' }),
      worldCaption,
    ]);

    const betCaption = make("div", { class: "stat-caption", text: "the engine's belief, out loud" });
    const betCard = make("div", { class: "stat-card is-bet" }, [
      make("div", { class: "stat-label" }, [
        make("span", { class: "stat-glyph", html: "▸" }),
        make("span", { text: "engine's bet" }),
      ]),
      make("div", { class: "stat-value", html: '<span class="num">—</span><span class="unit">aim</span>' }),
      betCaption,
    ]);

    // Reward card can flip into "untested" mode during Act IV (OOD).
    // In that mode it stops pretending to be a reward and instead admits:
    // the engine never let reality answer.
    const rewardGauge = make("div", { class: "reward-gauge" });
    const rewardFill  = make("div", { class: "reward-gauge-fill" });
    rewardGauge.appendChild(rewardFill);
    const rewardLabelText = make("span", { text: "reality returned" });
    const rewardCaption = make("div", { class: "stat-caption", text: "the click, the drop, or the silence" });
    const rewardCard = make("div", { class: "stat-card is-reward" }, [
      make("div", { class: "stat-label" }, [
        make("span", { class: "stat-glyph", html: "◉" }),
        rewardLabelText,
      ]),
      make("div", { class: "stat-value", html: '<span class="num">—</span>' }),
      rewardGauge,
      rewardCaption,
    ]);

    stats.append(collapseCard, worldCard, betCard, rewardCard);
    // Mouse-tracked ambient highlight inside each card — feels like a soft
    // lamp moves with the reader's attention.
    [collapseCard, worldCard, betCard, rewardCard].forEach(card => {
      card.addEventListener("mousemove", e => {
        const rect = card.getBoundingClientRect();
        card.style.setProperty("--mx", ((e.clientX - rect.left) / rect.width  * 100) + "%");
        card.style.setProperty("--my", ((e.clientY - rect.top)  / rect.height * 100) + "%");
      });
    });
    poolPanel.appendChild(stats);

    const rpePanel = make("div", { class: "panel" }, [ make("h4", { text: "Reward Prediction Error" }) ]);
    const rpeMeter = make("div", { class: "rpe-meter" });
    const rpeFill = make("div", { class: "rpe-fill pos" });
    rpeMeter.appendChild(rpeFill);
    rpePanel.appendChild(rpeMeter);
    const rpeValEl = make("span", { text: "RPE +0.00" });
    const rpeLabel = make("div", { class: "rpe-label" }, [
      make("span", { text: "−1.0 stall" }),
      rpeValEl,
      make("span", { text: "+1.0 spike" }),
    ]);
    rpePanel.appendChild(rpeLabel);

    const qHostEl = make("div");
    const qPanel = make("div", { class: "panel" }, [
      make("h4", { text: "The internal map · Q-table" }),
      qHostEl,
    ]);

    viz.append(poolPanel, qPanel);
    body.appendChild(viz);
    body.appendChild(rpePanel);

    currentVizPanels = {
      pool, rpeFill, rpeVal: rpeValEl, qHost: qHostEl,
      stat: {
        collapseCard, worldCard, betCard, rewardCard,
        rewardFill,
        rewardLabelText,
        collapseCaption, worldCaption, betCaption, rewardCaption,
        epNum:    collapseCard.querySelector(".num"),
        stateNum: worldCard.querySelector(".num"),
        aimNum:   betCard.querySelector(".num"),
        rewardNum:rewardCard.querySelector(".num"),
        rewardUnit: rewardCard.querySelector(".stat-value"),
      },
    };

    // Carry the Q-table state across acts so heatmap progress isn't lost
    // when we start a new viz block in a fresh act body.
    if (qState && qState.snapshot) {
      qState = buildQTable(qState.snapshot);
      qHostEl.appendChild(qState.node);
      updateBestHighlight(qState);
    }
    return currentVizPanels;
  }

  function renderQTableInit(snapshot) {
    const panels = ensureVizPanel();
    panels.qHost.innerHTML = "";
    qState = buildQTable(snapshot);
    panels.qHost.appendChild(qState.node);
  }

  function renderPoolInit(_ev) {
    ensureVizPanel(); // creates pool if not present
  }

  // Tween a numeric text node from its current value to a target value,
  // making every reading "settle" rather than snap. Cancels any in-flight
  // tween on the same element to avoid double-running.
  function tweenNumber(el, to, opts = {}) {
    const { duration = 380, decimals = 0, prefix = "", suffix = "" } = opts;
    if (!el) return;
    const fromRaw = (el.dataset.tweenVal != null) ? parseFloat(el.dataset.tweenVal) : parseFloat(el.textContent);
    const from = Number.isFinite(fromRaw) ? fromRaw : to;
    if (el._tweenRaf) cancelAnimationFrame(el._tweenRaf);
    const start = performance.now();
    function fmt(v) {
      return prefix + (decimals > 0 ? v.toFixed(decimals) : String(Math.round(v))) + suffix;
    }
    function tick(now) {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3);
      const v = from + (to - from) * eased;
      el.textContent = fmt(v);
      if (t < 1) el._tweenRaf = requestAnimationFrame(tick);
      else { el.dataset.tweenVal = String(to); el._tweenRaf = null; }
    }
    el._tweenRaf = requestAnimationFrame(tick);
  }

  function renderShot(ev, landmark = false) {
    const panels = ensureVizPanel();
    renderShotOnTable(panels.pool, ev.state, ev.action, ev.perfect,
      { showPerfect: landmark || ev.ep <= 10 });

    const s = panels.stat;

    // A real shot has been fired — make sure the cards are out of
    // "no-shot / OOD-probe" mode and back to their normal shape.
    s.rewardCard.classList.remove("untested", "ood");
    s.worldCard.classList.remove("ood");
    s.betCard.classList.remove("ood");
    s.rewardLabelText.textContent = "reality returned";
    s.collapseCaption.textContent = "another shot, another future chosen";
    s.worldCaption.textContent    = "where reality is sitting right now";
    s.betCaption.textContent      = "the engine's belief, out loud";
    s.rewardCaption.textContent   = "the click, the drop, or the silence";

    // Tween numeric readings instead of snapping them.
    tweenNumber(s.epNum, ev.ep, { duration: 320, suffix: "/" + ev.total });
    tweenNumber(s.stateNum,  ev.state,  { duration: 280 });
    tweenNumber(s.aimNum,    ev.action, { duration: 280 });
    tweenNumber(s.rewardNum, ev.reward, { duration: 480, decimals: 2 });

    // reward gauge fill + color band (hit / near / miss) + sweep highlight
    const r = Math.max(0, Math.min(1, ev.reward));
    s.rewardFill.style.width = (r * 100).toFixed(1) + "%";
    s.rewardCard.classList.toggle("hit",  r >= 0.85);
    s.rewardCard.classList.toggle("near", r >= 0.45 && r < 0.85);
    s.rewardCard.classList.toggle("miss", r < 0.45);
    const gauge = s.rewardFill.parentElement;
    if (gauge) {
      gauge.classList.remove("sweep");
      void gauge.offsetWidth;
      gauge.classList.add("sweep");
    }

    // pulse the cards that just changed so the eye follows the story
    [s.collapseCard, s.worldCard, s.betCard, s.rewardCard].forEach(c => {
      c.classList.remove("flash");
      void c.offsetWidth;     // restart animation
      c.classList.add("flash");
    });

    const rpe = ev.rpe;
    const pct = Math.min(100, Math.abs(rpe) * 80);
    panels.rpeFill.style.width = pct + "%";
    panels.rpeFill.classList.toggle("pos", rpe >= 0);
    panels.rpeFill.classList.toggle("neg", rpe < 0);
    if (rpe < 0) {
      panels.rpeFill.style.left = `calc(50% - ${pct}%)`;
    } else {
      panels.rpeFill.style.left = "50%";
    }
    panels.rpeVal.textContent = `RPE ${rpe >= 0 ? "+" : ""}${rpe.toFixed(2)}`;

    if (qState && ev.q_sa != null) updateQCell(qState, ev.state, ev.action, ev.q_sa);
  }

  function renderShotLandmark(ev) {
    renderShot(ev, true);
    // Play the full trajectory animation so the bank shot is *visible* —
    // ball travels up, bounces off the cushion, and either drops into a
    // pocket (perfect aim) or rolls to a halt (drift).
    const panels = currentVizPanels;
    if (panels && panels.pool) {
      // fire-and-forget; the backend already holds ~2.4s after a landmark
      playShotAnimation(panels.pool, ev.state, ev.action, ev.reward).catch(() => {});
    }
    const body = ensureActBody();
    const row = make("div", { class: "shot-row" });
    const top = make("div", { style: { display: "flex", alignItems: "center", gap: "0.8rem", flexWrap: "wrap" } });
    top.appendChild(make("span", {
      class: "lm " + ev.tag,
      text: ev.tag === "spike" ? "Dopamine spike"
           : ev.tag === "stall" ? "Engine stall" : "Coherence",
    }));
    top.appendChild(make("span", {
      class: "",
      style: { fontFamily: "var(--mono)", fontSize: "0.85rem", color: "var(--dim)" },
      text: `shot #${ev.ep} · state ${ev.state} → aimed ${ev.action} (hidden perfect: ${ev.perfect})`,
    }));
    row.appendChild(top);
    row.appendChild(make("div", {
      style: { fontFamily: "var(--mono)", fontSize: "0.85rem", color: "var(--text)" },
      html: `predicted <b style="color:var(--cyan)">${ev.predicted.toFixed(2)}</b>` +
            ` · reality <b style="color:var(--amber)">${ev.reward.toFixed(2)}</b>` +
            ` · RPE <b style="color:${ev.rpe >= 0 ? "var(--green)" : "var(--red)"}">${ev.rpe >= 0 ? "+" : ""}${ev.rpe.toFixed(2)}</b>`,
    }));
    row.appendChild(make("div", {
      style: { fontFamily: "var(--serif)", fontStyle: "italic", color: "var(--dim)" },
      text: ev.meaning,
    }));
    body.appendChild(row);
  }

  function renderQReplace(snap) {
    if (!qState) renderQTableInit(snap);
    else replaceQTable(qState, snap);
  }

  function renderInstinct(ev) {
    const body = ensureActBody();
    let grid = body.querySelector(".instinct-grid");
    if (!grid) {
      grid = make("div", { class: "instinct-grid" });
      body.appendChild(grid);
    }
    const correct = ev.action === ev.perfect;
    grid.appendChild(make("div", { class: "instinct-card" }, [
      make("div", { class: "label", text: "state " + ev.state }),
      make("p", {
        class: "stmt",
        html: `aims at <b>diamond ${ev.action}</b>` + (correct ? "" : ` <span style="color:var(--rose)">(truth: ${ev.perfect})</span>`),
      }),
      make("div", { class: "conf", text: "confidence " + ev.confidence.toFixed(2) }),
    ]));
    // Also show on the pool table
    const panels = ensureVizPanel();
    renderShotOnTable(panels.pool, ev.state, ev.action, ev.perfect, { showPerfect: true });
  }

  function renderOOD(ev) {
    const body = ensureActBody();
    const panels = ensureVizPanel();
    renderShotOnTable(panels.pool, ev.state, ev.guess, ev.truth, { showPerfect: true, ood: true });

    // Drive the narrative cards with OOD context. CRITICAL: no shot was
    // actually fired here — the engine was *queried* about a state it has
    // never seen. So the reward card must NOT pretend reality answered.
    // Instead the cards admit what really happened:
    //   world read  → red, "never seen in training"
    //   engine bet  → red, "answers anyway"
    //   reality     → muted, "no shot fired — reality stayed silent"
    const s = panels.stat;
    s.stateNum.textContent = ev.state;
    s.aimNum.textContent   = ev.guess;
    s.rewardNum.textContent = "no shot";

    // Flip the world + bet cards to OOD red treatment with story-mapped captions.
    s.worldCard.classList.add("ood");
    s.betCard.classList.add("ood");
    s.worldCaption.textContent = "never seen in training";
    s.betCaption.textContent   = "answers anyway, at full confidence";

    // Flip the reward card to "untested" — neutral, not red.
    // Reality was never asked, so do not show a reward number at all.
    s.rewardCard.classList.remove("hit", "near", "miss", "ood");
    s.rewardCard.classList.add("untested");
    s.rewardFill.style.width = "0%";
    s.rewardLabelText.textContent = "reality untested";
    s.rewardCaption.textContent   = "the engine spoke before reality could answer";

    // Soften the collapse card caption — no NEW collapse happened, just a question.
    s.collapseCaption.textContent = "no new shot — only a question asked";

    [s.worldCard, s.betCard, s.rewardCard].forEach(c => {
      c.classList.remove("flash");
      void c.offsetWidth;
      c.classList.add("flash");
    });

    const card = make("div", { class: "ood-card" }, [
      make("span", { class: "stamp", text: "Hallucination" }),
      make("p", {
        class: "query",
        html: `<span style="color:var(--rose)">Object ball at diamond ${ev.state}</span>` +
              ` — never observed in training. Closest thing the engine remembers: state ${ev.nearest}.`,
      }),
      make("div", { class: "compare" }, [
        make("div", {}, [
          make("div", { class: "lbl", text: "engine's confident answer" }),
          make("div", { class: "val-engine", text: "aim " + ev.guess }),
          make("div", {
            style: { fontSize: "0.7rem", color: "var(--muted)", marginTop: "0.2rem" },
            text: "stated confidence " + ev.confidence.toFixed(2),
          }),
        ]),
        make("div", {}, [
          make("div", { class: "lbl", text: "ground-truth optimum" }),
          make("div", { class: "val-truth", text: "aim " + ev.truth }),
          make("div", {
            style: { fontSize: "0.7rem", color: "var(--muted)", marginTop: "0.2rem" },
            text: "the table actually rewards this aim",
          }),
        ]),
      ]),
      make("div", {
        class: "verdict",
        text: ev.off_by === 0
          ? "→ lucky: the closest memory happened to be correct."
          : `→ Hallucination: off by ${ev.off_by} diamonds, stated with full confidence.`,
      }),
    ]);
    body.appendChild(card);
  }

  function renderTwoEnginesInit(ev) {
    const body = ensureActBody();
    const grid = make("div", { class: "two-engines" });
    function buildPanel(letter, lawText, snap) {
      const panel = make("div", { class: "engine-panel " + letter });
      panel.appendChild(make("h5", { text: "Player " + letter }));
      panel.appendChild(make("div", { class: "law", text: lawText }));
      const qHost = make("div");
      const qObj = buildQTable(snap);
      qHost.appendChild(qObj.node);
      panel.appendChild(qHost);
      grid.appendChild(panel);
      return { panel, qObj };
    }
    const a = buildPanel("A", "grew up where perfect aim = state × 2", ev.a_q);
    const b = buildPanel("B", "grew up where perfect aim = 9 − state × 2", ev.b_q);
    body.appendChild(grid);
    twoEngines = { A: a, B: b, container: body };
  }

  function renderTwoEnginesTrain(ev) {
    if (!twoEngines) return;
    const obj = twoEngines[ev.engine];
    if (!obj) return;
    updateQCell(obj.qObj, ev.state, ev.action, ev.q_sa);
  }

  function renderTwoEnginesDone(ev) {
    if (!twoEngines) return;
    replaceQTable(twoEngines.A.qObj, ev.a_q);
    replaceQTable(twoEngines.B.qObj, ev.b_q);
  }

  function renderTwoEnginesAnswer(ev) {
    if (!twoEngines) return;
    let host = twoEngines.container.querySelector(".answers");
    if (!host) {
      host = make("div", { class: "answers" });
      host.appendChild(make("h4", {
        style: { fontFamily: "var(--mono)", fontSize: "0.75rem", letterSpacing: "0.2em",
                 textTransform: "uppercase", color: "var(--dim)", margin: "1.2rem 0 0.6rem" },
        text: "Ask both players the same question",
      }));
      twoEngines.container.appendChild(host);
    }
    const fmt = (label, action, truth, confidence, cls) => {
      const ok = action === truth;
      const markCls = ok ? "mark-ok" : "mark-near";
      const mark = ok ? "✓" : "≈";
      return make("span", { class: cls, html:
        `${label} <b>aim ${action}</b>  ` +
        `<b class="${markCls}">${mark}</b>  ` +
        `<span class="conf">${confidence.toFixed(2)}</span>`
      });
    };
    host.appendChild(make("div", { class: "answer-row" }, [
      make("span", { class: "q", text: `state ${ev.query} →` }),
      fmt("A:", ev.a_action, ev.a_truth, ev.a_confidence, "ea"),
      fmt("B:", ev.b_action, ev.b_truth, ev.b_confidence, "eb"),
    ]));
  }

  function renderEpilogueBullets(ev) {
    const body = ensureActBody();
    const ul = make("ol", { class: "epilogue-bullets" });
    ev.bullets.forEach(b => ul.appendChild(make("li", { text: b })));
    body.appendChild(ul);
  }

  function renderFinalMap(ev) {
    const body = ensureActBody();
    const card = make("div", { class: "final-map" });
    card.appendChild(make("h4", { text: "The finalized map of the original engine" }));
    ev.rows.forEach(r => {
      card.appendChild(make("div", { class: "row" }, [
        make("span", { text: `object ball at diamond ${r.state}` }),
        make("span", { class: "arrow", text: "→" }),
        make("span", {}, [
          make("span", { class: "target", text: "aim diamond " + r.action }),
          make("span", { class: "conf", text: "  (confidence " + r.confidence.toFixed(2) + ")" }),
        ]),
      ]));
    });
    body.appendChild(card);
    const closing = make("div", { class: "closing" });
    ev.closing.forEach(l => closing.appendChild(make("p", { text: l })));
    body.appendChild(closing);
  }

  // ---------- Interactive Possibilities panel ----------
  // The prologue's metaphor made literal: the agent's "Q-table" is the
  // wave of every possible future. Each shot collapses one of them into
  // reality. Here we hand the cue to the reader: pick a state, hover the
  // 8 possible aims (the wave), then click one to collapse it.
  function renderPossibilities(ev) {
    const body = ensureActBody();
    const Q = ev.engine_q;             // { state_str: { action_str: q_value } }
    const trained = ev.trained_states; // [1,2,3,4]
    const ood     = ev.ood_states;     // [5,6,7]
    const allStates = [...trained, ...ood];
    let currentState = ev.default_state || 3;
    let busy = false;

    const wrap = make("div", { class: "possibilities-wrap" }, [
      make("h3", { text: "Now you compile a future." }),
      make("p", {
        class: "lede",
        text: "Pick where the ball sits. Hover the eight ghost trajectories — that's the wave of futures the engine is holding in superposition. Click one to collapse the wave and watch the trajectory you chose become the only reality that ever happened.",
      }),
    ]);

    // State picker
    const controls = make("div", { class: "possibilities-controls" });
    controls.appendChild(make("span", { text: "object ball at diamond:" }));
    const picker = make("div", { class: "state-picker" });
    const pills = {};
    allStates.forEach(s => {
      const p = make("button", {
        class: "state-pill" + (ood.includes(s) ? " ood" : ""),
        text: String(s),
      });
      p.addEventListener("click", () => {
        if (busy) return;
        currentState = s;
        Object.values(pills).forEach(el => el.classList.remove("active"));
        p.classList.add("active");
        rebuildGhosts();
        setResult("", "");
      });
      pills[s] = p;
      picker.appendChild(p);
    });
    pills[currentState].classList.add("active");
    controls.appendChild(picker);

    const resetBtn = make("button", { class: "reset-btn", text: "↻ reset the wave" });
    resetBtn.addEventListener("click", () => {
      if (busy) return;
      rebuildGhosts();
      setResult("", "");
    });
    controls.appendChild(resetBtn);

    wrap.appendChild(controls);

    // Pool table
    const tableWrap = make("div");
    const pool = buildPoolSvg();
    tableWrap.appendChild(pool.svg);
    wrap.appendChild(tableWrap);

    // Result text + hint
    const result = make("div", { class: "possibilities-result", html: "&nbsp;" });
    wrap.appendChild(result);
    wrap.appendChild(make("div", {
      class: "possibilities-hint",
      text: "Tip: try a state inside the training set (1–4), then one outside it (5–7). Notice how the engine still picks confidently — and notice how reality answers.",
    }));

    body.appendChild(wrap);

    // Ground-truth and engine's policy. We expose the same "truth" rule
    // the rest of the demo uses for OOD: state*2 for trained states,
    // max(1, 9-state) for OOD states. Mirrors world.true_optimum.
    function trueOptimum(s) {
      return s <= 4 ? s * 2 : Math.max(1, 9 - s);
    }
    function engineBest(s) {
      // For states in the Q-table, pick best action. For OOD, fall back
      // to nearest trained state's best action.
      if (Q[String(s)]) {
        let bestA = 1, bestV = -Infinity;
        for (let a = 1; a <= 8; a++) {
          const v = Q[String(s)][String(a)];
          if (v > bestV) { bestV = v; bestA = a; }
        }
        return { action: bestA, confidence: bestV };
      }
      // OOD: snap to nearest trained state
      const nearest = trained.reduce((b, t) =>
        (!b || Math.abs(t - s) < Math.abs(b - s)) ? t : b, null);
      let bestA = 1, bestV = -Infinity;
      for (let a = 1; a <= 8; a++) {
        const v = Q[String(nearest)][String(a)];
        if (v > bestV) { bestV = v; bestA = a; }
      }
      return { action: bestA, confidence: bestV, nearest };
    }
    // Reward function matches PoolTable.reward in Python: 1 - 0.5*|action - perfect|
    function rewardOf(state, action) {
      const r = 1 - 0.5 * Math.abs(action - trueOptimum(state));
      return Math.max(0, Math.min(1, r));
    }

    function setResult(html, cls) {
      result.innerHTML = html || "&nbsp;";
      result.classList.remove("hit", "near", "miss");
      if (cls) result.classList.add(cls);
    }

    function rebuildGhosts() {
      // Position ball at currentState
      const ballX = POOL.diamondsX[Math.min(Math.max(currentState, 1), 8) - 1];
      const ballY = POOL.cueBallY;
      // Reset all visual layers
      pool.aimGroup.innerHTML = "";
      pool.perfectMarker.setAttribute("opacity", "0");
      pool.aimMarker.setAttribute("opacity", "0");
      pool.aimMarkerRing.setAttribute("opacity", "0");
      setBallPos(pool, ballX, ballY);
      pool.cueBall.setAttribute("opacity", "1");
      pool.cueBallSpec.setAttribute("opacity", "1");
      pool.ballShadow.setAttribute("opacity", "0.55");
      pool.cueBall.setAttribute("r", 13);
      pool.cueBallSpec.setAttribute("r", 5);
      pool.cueBall.classList.remove("ood-pulse");
      pool.cueBall.setAttribute("stroke", "rgba(0,0,0,0.55)");
      pool.cueBall.setAttribute("stroke-width", "0.6");
      if (ood.includes(currentState)) {
        pool.cueBall.setAttribute("stroke", "#fb7185");
        pool.cueBall.setAttribute("stroke-width", "2.5");
      }

      // Draw 8 ghost trajectories — the wave of possibilities
      const ns = "http://www.w3.org/2000/svg";
      const engineChoice = engineBest(currentState).action;
      for (let aim = 1; aim <= 8; aim++) {
        const aimX = POOL.diamondsX[aim - 1];
        const aimY = POOL.aimContactY;
        // Wide invisible hit-target so clicks/hovers reliably land on the
        // (otherwise very thin) ghost line.
        const hit = document.createElementNS(ns, "line");
        hit.setAttribute("x1", ballX); hit.setAttribute("y1", ballY - 12);
        hit.setAttribute("x2", aimX);  hit.setAttribute("y2", aimY + 6);
        hit.setAttribute("stroke", "transparent");
        hit.setAttribute("stroke-width", "16");
        hit.setAttribute("stroke-linecap", "round");
        hit.setAttribute("class", "ghost-aim ghost-hit");
        hit.dataset.aim = String(aim);
        hit.addEventListener("click", () => collapse(aim));
        hit.addEventListener("mouseenter", () => previewAim(aim));
        hit.addEventListener("mouseleave", () => previewAim(null));
        pool.aimGroup.appendChild(hit);
        // Visible ghost trajectory
        const ln = document.createElementNS(ns, "line");
        ln.setAttribute("x1", ballX); ln.setAttribute("y1", ballY - 12);
        ln.setAttribute("x2", aimX);  ln.setAttribute("y2", aimY + 6);
        ln.setAttribute("stroke", aim === engineChoice ? "#fb7185" : "#9b91b8");
        ln.setAttribute("stroke-width", aim === engineChoice ? "1.6" : "1.2");
        ln.setAttribute("stroke-dasharray", "3 4");
        ln.setAttribute("opacity", aim === engineChoice ? "0.65" : "0.35");
        ln.setAttribute("stroke-linecap", "round");
        ln.setAttribute("class", "ghost-aim ghost-visible");
        ln.dataset.aim = String(aim);
        ln.style.pointerEvents = "none";
        pool.aimGroup.appendChild(ln);
        // Endpoint dot at the rail (subtle target, also clickable)
        const dot = document.createElementNS(ns, "circle");
        dot.setAttribute("cx", aimX);
        dot.setAttribute("cy", aimY);
        dot.setAttribute("r", aim === engineChoice ? 6 : 4);
        dot.setAttribute("fill", aim === engineChoice ? "#fb7185" : "rgba(155,145,184,0.7)");
        dot.setAttribute("class", "ghost-aim ghost-dot");
        dot.dataset.aim = String(aim);
        dot.addEventListener("click", () => collapse(aim));
        dot.addEventListener("mouseenter", () => previewAim(aim));
        dot.addEventListener("mouseleave", () => previewAim(null));
        pool.aimGroup.appendChild(dot);

        // Outcome preview — for THIS ghost, render where the ball would
        // actually end up (drop / graze / miss) and color-code it. Faint by
        // default; the hovered/engine ghost's marker brightens via previewAim.
        const perfect = trueOptimum(currentState);
        const outcome = predictShot(currentState, aim, perfect);
        drawOutcome(pool.aimGroup, outcome, {
          opacity: aim === engineChoice ? 0.92 : 0.55,
          isGhost: true,
          aimNum: aim,
        });
      }

      // Hint at the engine's preferred aim for this state.
      const eb = engineBest(currentState);
      let hint = `<span style="color:var(--muted)">the engine would aim at</span> <b>${eb.action}</b>`;
      if (eb.nearest != null) hint += ` <span style="color:var(--red)">(borrowed from state ${eb.nearest} — you are out of distribution)</span>`;
      setResult(hint, "");
    }

    function previewAim(aim) {
      if (busy) return;
      // Brighten the hovered aim's line + dot; restore others.
      pool.svg.querySelectorAll(".ghost-visible").forEach(el => {
        const isHover = aim != null && el.dataset.aim === String(aim);
        const isEngine = el.getAttribute("stroke") === "#fb7185";
        el.setAttribute("opacity", isHover ? "0.95"
                                  : isEngine ? "0.65"
                                  : "0.35");
        el.setAttribute("stroke-width", isHover ? "2.2"
                                       : isEngine ? "1.6"
                                       : "1.2");
      });
      pool.svg.querySelectorAll(".ghost-dot").forEach(el => {
        const isHover = aim != null && el.dataset.aim === String(aim);
        el.setAttribute("r", isHover ? "7" : (el.getAttribute("fill") === "#fb7185" ? "6" : "4"));
      });
      // Brighten the hovered ghost's outcome markers + reflection line.
      const engineChoice = engineBest(currentState).action;
      pool.svg.querySelectorAll(".outcome-line, .outcome-marker, .outcome-end-ball").forEach(el => {
        const a = el.dataset.aim ? Number(el.dataset.aim) : null;
        if (a == null) return;
        const isHover  = aim != null && a === aim;
        const isEngine = a === engineChoice;
        if (aim == null) {
          el.setAttribute("opacity", isEngine ? "0.92" : "0.52");
        } else {
          el.setAttribute("opacity", isHover ? "1" : (isEngine ? "0.58" : "0.14"));
        }
      });
      // When hovering, peek at the would-be reward without firing.
      if (aim != null) {
        const r = rewardOf(currentState, aim);
        const verdict = r >= 0.85 ? "drop" : r >= 0.45 ? "graze" : "miss";
        setResult(
          `<span style="color:var(--muted)">if you commit to</span> <b>aim ${aim}</b> <span style="color:var(--muted)">→ reward</span> <b>${r.toFixed(2)}</b> <span style="color:var(--muted)">(ends: ${verdict})</span>`,
          ""
        );
      } else {
        // Reset to the engine hint when no aim is hovered.
        const eb = engineBest(currentState);
        let hint = `<span style="color:var(--muted)">the engine would aim at</span> <b>${eb.action}</b>`;
        if (eb.nearest != null) hint += ` <span style="color:var(--red)">(borrowed from state ${eb.nearest} — you are out of distribution)</span>`;
        setResult(hint, "");
      }
    }

    async function collapse(aim) {
      if (busy) return;
      busy = true;
      // Stagger the dismissal of the 7 unchosen possibilities so the wave
      // visibly *collapses* — futures nearest to the chosen one fade last,
      // futures farthest from it vanish first. Each emits a tiny puff of
      // particles as it dies.
      const ns = "http://www.w3.org/2000/svg";
      const aimXFn = i => POOL.diamondsX[i - 1];
      pool.svg.querySelectorAll(".ghost-aim").forEach(el => {
        el.classList.add("locked");
      });
      const dismissed = [];
      for (let a = 1; a <= 8; a++) {
        if (a === aim) continue;
        dismissed.push(a);
      }
      dismissed.sort((a, b) => Math.abs(b - aim) - Math.abs(a - aim));
      dismissed.forEach((a, i) => {
        const delay = i * 55;
        setTimeout(() => {
          pool.svg.querySelectorAll(`.ghost-aim[data-aim="${a}"]`).forEach(el => {
            el.style.transition = "opacity 0.35s ease";
            el.setAttribute("opacity", "0");
          });
          // small puff of particles where the ghost's dot was
          const px = aimXFn(a), py = POOL.aimContactY;
          for (let k = 0; k < 3; k++) {
            const dot = document.createElementNS(ns, "circle");
            const ang = -Math.PI/2 + (Math.random() - 0.5) * 0.9;
            const dist = 8 + Math.random() * 10;
            dot.setAttribute("cx", px);
            dot.setAttribute("cy", py);
            dot.setAttribute("r", 1.4 + Math.random() * 0.8);
            dot.setAttribute("fill", "rgba(180,170,210,0.7)");
            dot.setAttribute("class", "pocket-particle");
            pool.svg.appendChild(dot);
            requestAnimationFrame(() => {
              dot.style.transition = "cx 0.45s ease-out, cy 0.45s ease-out, opacity 0.45s ease-out";
              dot.setAttribute("cx", px + Math.cos(ang) * dist);
              dot.setAttribute("cy", py + Math.sin(ang) * dist);
              dot.setAttribute("opacity", "0");
            });
            setTimeout(() => dot.remove(), 520);
          }
        }, delay);
      });
      // Highlight the chosen ghost line during the collapse.
      pool.svg.querySelectorAll(`.ghost-visible[data-aim="${aim}"]`).forEach(el => {
        el.setAttribute("opacity", "0.95");
        el.setAttribute("stroke-width", "2.2");
      });
      pool.svg.querySelectorAll(`.ghost-dot[data-aim="${aim}"]`).forEach(el => {
        el.setAttribute("r", "7");
      });
      await sleep(dismissed.length * 55 + 120);
      const reward = rewardOf(currentState, aim);
      const perfect = trueOptimum(currentState);
      // Show ground-truth marker
      const perfectX = POOL.diamondsX[perfect - 1];
      pool.perfectMarker.setAttribute("cx", perfectX);
      pool.perfectMarker.setAttribute("cy", POOL.aimContactY);
      pool.perfectMarker.setAttribute("opacity", aim === perfect ? "0.65" : "0.9");
      // Run the trajectory
      await playShotAnimation(pool, currentState, aim, reward);
      // Verdict
      const isOOD = ood.includes(currentState);
      let cls, verdict, tail;
      if (reward >= 0.85)      { cls = "hit";  verdict = "drop";   tail = "the click of contact."; }
      else if (reward >= 0.45) { cls = "near"; verdict = "graze";  tail = "close — but the table didn't pay full."; }
      else                     { cls = "miss"; verdict = "miss"; tail = "no reward. The table said no."; }
      let line = `you aimed <b>${aim}</b> · the table's perfect aim was <b>${perfect}</b> · reward <b>${reward.toFixed(2)}</b> — <b class="verdict">${verdict}</b> · <span style="color:var(--muted)">${tail}</span>`;
      if (isOOD) {
        line = `<span style="color:var(--red)">state ${currentState} was never in the training set.</span> ` + line;
      }
      setResult(line, cls);
      busy = false;
    }

    rebuildGhosts();
  }

  // ---------- Prologue · two-player intro card ----------
  // Renders the central image of the essay: same loop, two players, one table.
  // Player A glows magenta, Player B glows cyan. The shared-loop strip at the
  // bottom names the algorithm both players will be running for the rest of
  // the demo. This card is the conceptual hook the next four acts pay off.
  function renderTwoPlayerIntro(ev) {
    const body = ensureActBody();
    const card = make("div", { class: "two-player-intro" });
    const grid = make("div", { class: "tp-grid" });

    const buildSide = (letter, data) => {
      const panel = make("div", { class: "tp-card " + letter }, [
        make("div", { class: "tp-name", text: data.name }),
        make("div", { class: "tp-bio", text: data.biography }),
        make("div", { class: "tp-tagline", text: data.tagline }),
      ]);
      return panel;
    };

    grid.appendChild(buildSide("A", ev.player_a));
    grid.appendChild(make("div", { class: "tp-vs", text: "vs" }));
    grid.appendChild(buildSide("B", ev.player_b));
    card.appendChild(grid);

    if (ev.shared) {
      card.appendChild(make("div", { class: "tp-shared", html:
        '<b>shared algorithm · </b>' + ev.shared
      }));
    }
    body.appendChild(card);
    appendToStory(make("div"));
  }

  // ---------- Act I · the RL legend card ----------
  // The conceptual map: each story word paired with its real RL term and
  // symbol, plus a concrete example pulled from this exact pool table.
  // This is the reader's reference card for the rest of the demo.
  function renderRLLegend(ev) {
    const body = ensureActBody();
    const card = make("div", { class: "rl-legend" });
    card.appendChild(make("h4", { text: ev.title || "The Loop" }));
    const rows = make("div", { class: "rl-rows" });
    (ev.rows || []).forEach(r => {
      const body2 = make("div", { class: "rl-body" }, [
        make("div", { class: "rl-term", text: r.rl_term }),
        make("div", { class: "rl-story", text: r.story }),
        r.concrete ? make("div", { class: "rl-concrete", text: "· " + r.concrete }) : null,
      ]);
      const row = make("div", { class: "rl-row" }, [
        make("div", { class: "rl-sym", text: r.symbol }),
        body2,
      ]);
      rows.appendChild(row);
    });
    card.appendChild(rows);
    body.appendChild(card);
    appendToStory(make("div"));

    // Once each row has finished sliding in, fire its single "scan-line"
    // sweep so the cyan light reads as: "row activated, row activated...".
    const rowEls = rows.querySelectorAll(".rl-row");
    rowEls.forEach((row, i) => {
      const baseDelay = 300 + i * 100; // matches CSS animation-delay
      setTimeout(() => {
        row.classList.add("scan");
        setTimeout(() => row.classList.remove("scan"), 1100);
      }, baseDelay + 250);
    });
  }

  // ---------- Act II · Bellman update reveal ----------
  // The "whole act of learning compresses to one line of math" moment.
  // We color-code each piece of the equation so the eye reads the story:
  //   cyan    = engine's current belief
  //   magenta = learning rate / how willing to be rewritten
  //   amber   = reality answering back
  //   red     = surprise (RPE), the only ingredient that ever changes Q
  function renderBellmanCard(ev) {
    const body = ensureActBody();
    const card = make("div", { class: "bellman-card" });
    card.appendChild(make("h4", { text: "The Bellman update · the entire algorithm, in one line" }));

    // We paint the equation with inline spans matching the color of each part.
    // Hand-colored so the equation reads like a sentence: the reader sees
    // "old belief gets nudged by learning rate, times surprise."
    //
    // Critically: .bp tokens are `display: inline-block` (so they can fade /
    // pulse / lift individually), which means they cannot themselves wrap.
    // To allow the RPE bracket to wrap on narrower viewports we DO NOT wrap
    // its content in a single .bp-rpe inline-block. Instead, the bracket
    // tokens [ and ] carry the .bp-rpe-edge class, and each inner token gets
    // .bp-rpe so the underline-highlight reads as one continuous bracket
    // across line breaks.
    const eq = make("div", { class: "bellman-eq" });
    eq.innerHTML =
      '<span class="bp bp-cyan">Q(s,a)</span> ' +
      '<span class="bp bp-dim">\u2190</span> ' +
      '<span class="bp bp-cyan">Q(s,a)</span> ' +
      '<span class="bp bp-dim">+</span> ' +
      '<span class="bp bp-magenta">\u03b1</span> ' +
      '<span class="bp bp-dim">\u00b7</span> ' +
      '<span class="bp bp-dim bp-rpe bp-rpe-edge">[</span> ' +
      '<span class="bp bp-amber bp-rpe">r</span> ' +
      '<span class="bp bp-dim bp-rpe">+</span> ' +
      '<span class="bp bp-dim bp-rpe">\u03b3</span> ' +
      '<span class="bp bp-dim bp-rpe">\u00b7</span> ' +
      '<span class="bp bp-dim bp-rpe">max</span> ' +
      '<span class="bp bp-cyan bp-rpe">Q(s\u2032,a\u2032)</span> ' +
      '<span class="bp bp-dim bp-rpe">\u2212</span> ' +
      '<span class="bp bp-cyan bp-rpe">Q(s,a)</span> ' +
      '<span class="bp bp-dim bp-rpe bp-rpe-edge">]</span>';
    card.appendChild(eq);

    if (ev.story) {
      card.appendChild(make("div", { class: "bellman-story", html:
        '<b>read it as · </b>' + ev.story
      }));
    }

    // The annotated parts list.
    if (ev.parts && ev.parts.length) {
      const parts = make("div", { class: "bellman-parts" });
      ev.parts.forEach(p => {
        const color = p.color || "dim";
        const partEl = make("div", { class: "bellman-part " + color }, [
          make("div", { class: "key", text: p.key }),
          make("div", { class: "lab", text: p.label }),
        ]);
        // Hover-link: highlight every equation token of the matching color
        // and dim the rest. Mouseout restores. Touch users get the same
        // effect on touchstart.
        const enter = () => eq.classList.add("hl-" + color);
        const leave = () => eq.classList.remove("hl-" + color);
        partEl.addEventListener("mouseenter", enter);
        partEl.addEventListener("mouseleave", leave);
        partEl.addEventListener("focus", enter);
        partEl.addEventListener("blur",  leave);
        partEl.tabIndex = 0;
        parts.appendChild(partEl);
      });
      card.appendChild(parts);
    }

    body.appendChild(card);
    appendToStory(make("div"));
  }

  function renderDone() {
    const body = ensureActBody();
    const footer = make("div", {
      style: {
        marginTop: "3rem",
        textAlign: "center",
        fontFamily: "var(--mono)",
        fontSize: "0.75rem",
        letterSpacing: "0.25em",
        textTransform: "uppercase",
        color: "var(--muted)",
      },
    });
    footer.appendChild(make("div", { text: "fin." , style: { marginBottom: "1rem", color: "var(--magenta)", fontSize: "0.9rem" } }));
    const restart = make("button", {
      text: "↻ run the play again",
      style: {
        background: "transparent",
        border: "1px solid var(--line-strong)",
        color: "var(--dim)",
        fontFamily: "var(--mono)",
        fontSize: "0.75rem",
        letterSpacing: "0.2em",
        textTransform: "uppercase",
        padding: "0.7rem 1.4rem",
        borderRadius: "999px",
        cursor: "pointer",
      },
    });
    restart.addEventListener("click", () => location.reload());
    footer.appendChild(restart);
    body.appendChild(footer);
  }

  // --- Event queue / spotlight gate ---
  // When a "successful shot" lands (high reward, ball drops), we want the
  // page to genuinely STOP for a beat so the eye can land on the moment.
  // playShotAnimation calls holdSpotlight(ms) at the start of the victory
  // beat; any SSE events arriving during that window are queued and flushed
  // when the spotlight releases. Effect: the rapid post-landmark non-landmark
  // shots no longer drown out the drop.
  const _eventQueue = [];
  let _spotlightUntil = 0;
  function holdSpotlight(ms) {
    const until = Date.now() + ms;
    if (until > _spotlightUntil) _spotlightUntil = until;
    setTimeout(_flushQueue, ms + 30);
  }
  function _flushQueue() {
    while (_eventQueue.length && Date.now() >= _spotlightUntil) {
      _dispatch(_eventQueue.shift());
    }
    if (_eventQueue.length) setTimeout(_flushQueue, 60);
  }

  // --- Dispatcher ---
  function handle(ev) {
    if (Date.now() < _spotlightUntil) { _eventQueue.push(ev); return; }
    _dispatch(ev);
  }
  function _dispatch(ev) {
    switch (ev.type) {
      case "act_start":          renderActStart(ev); break;
      case "narration":          renderNarration(ev); break;
      case "callout":            renderCallout(ev); break;
      case "river_callout":      renderRiverCallout(ev); break;
      case "two_player_intro":   renderTwoPlayerIntro(ev); break;
      case "rl_legend":          renderRLLegend(ev); break;
      case "bellman_card":       renderBellmanCard(ev); break;
      case "qtable_init":        renderQTableInit(ev); break;
      case "qtable_replace":     renderQReplace(ev); break;
      case "pooltable_init":     renderPoolInit(ev); break;
      case "shot":               renderShot(ev); break;
      case "shot_landmark":      renderShotLandmark(ev); break;
      case "instinct_demo":      renderInstinct(ev); break;
      case "ood_query":          renderOOD(ev); break;
      case "two_engines_init":   renderTwoEnginesInit(ev); break;
      case "two_engines_train":  renderTwoEnginesTrain(ev); break;
      case "two_engines_done":   renderTwoEnginesDone(ev); break;
      case "two_engines_answer": renderTwoEnginesAnswer(ev); break;
      case "epilogue_bullets":   renderEpilogueBullets(ev); break;
      case "final_map":          renderFinalMap(ev); break;
      case "possibilities_invite": renderPossibilities(ev); break;
      case "done":               renderDone(); break;
    }
  }

  function begin() {
    hero.classList.add("gone");
    stage.classList.add("live");
    const es = new EventSource("/api/story");
    es.onmessage = (e) => {
      try { handle(JSON.parse(e.data)); }
      catch (err) { console.error("bad event", err, e.data); }
    };
    es.onerror = () => {
      // Stream closed (likely normal end-of-story). Show a final hint.
      es.close();
    };
  }

  startBtn.addEventListener("click", begin);
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# FastAPI app.
# ---------------------------------------------------------------------------


app = FastAPI(title="Calling the Shot")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)


@app.get("/api/story")
async def story_endpoint() -> StreamingResponse:
    return StreamingResponse(
        story_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        pass


def main() -> None:
    import uvicorn
    port = int(os.environ.get("PORT", "8765"))
    url = f"http://127.0.0.1:{port}"
    print()
    print(f"  Calling the Shot — opening {url}")
    print(f"  (set NO_OPEN=1 to skip auto-launch, FAST_MODE=1 to skip pauses)")
    print()
    if not os.environ.get("NO_OPEN"):
        _open_browser(url)
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
