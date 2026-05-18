/* Running the Table - prediction theater + training cinematic.
 *
 *   Modes:
 *     - PREDICT: click a pocket -> 6-step loop animation.
 *     - CINEMATIC: replay recorded training episodes one by one.
 *
 *   Canvas rendering:
 *     - Real pool table layout with side pockets carved out of the long rails.
 *     - Cue stick pulls back along -angle then snaps forward at "move".
 *     - Friction-eased ball travel with motion trail.
 *     - Rail impact ripple at the bounce point.
 *     - Ball drops into pocket on success.
 */

(() => {
  const sessionId = crypto.randomUUID();

  // ----------------------------------------------------------------------
  // DOM
  // ----------------------------------------------------------------------

  const canvas = document.getElementById("table-canvas");
  const ctx = canvas.getContext("2d");
  const tableWrap = canvas.parentElement;
  const dpr = Math.max(1, window.devicePixelRatio || 1);

  const els = {
    phaseTitle: document.getElementById("phase-title"),
    narrator: document.getElementById("narrator"),
    narratorBlock: document.querySelector(".narrator-block"),
    verdictRegion: document.getElementById("verdict-region"),
    verdictMetrics: document.getElementById("verdict-metrics"),
    hint: document.getElementById("hint"),
    loopItems: Array.from(document.querySelectorAll(".loop li")),
    watchBtn: document.getElementById("watch-btn"),
    // dashboard
    memoryGrid: document.getElementById("memory-grid"),
    memoryCount: document.getElementById("memory-count"),
    memoryKebab: document.getElementById("memory-kebab"),
    memoryKebabMenu: document.getElementById("memory-kebab-menu"),
    beliefVariant: document.getElementById("belief-variant"),
    beliefCaption: document.getElementById("belief-caption"),
    beliefHistogram: document.getElementById("belief-histogram"),
    beliefBest: document.getElementById("belief-best"),
    beliefAim: document.getElementById("belief-aim"),
    beliefVisits: document.getElementById("belief-visits"),
    beliefStatus: document.getElementById("belief-status"),
    trainingBlock: document.getElementById("training-block"),
    cineOverlay: document.getElementById("cinematic"),
    cineStatus: document.getElementById("cine-status"),
    cineCaption: document.getElementById("cine-caption"),
    cineAngle: document.getElementById("cine-angle"),
    cineReward: document.getElementById("cine-reward"),
    cineExplore: document.getElementById("cine-explore"),
    cineWinrate: document.getElementById("cine-winrate"),
    cineFirstmake: document.getElementById("cine-firstmake"),
    cinePhase: document.getElementById("cine-phase"),
    cineRewardVal: document.getElementById("cine-reward-val"),
    cineWinrateVal: document.getElementById("cine-winrate-val"),
    cineConverge: document.getElementById("cine-converge"),
    epsilonFill: document.getElementById("epsilon-fill"),
    convergeFill: document.getElementById("converge-fill"),
    cineTries: document.getElementById("cine-tries"),
    cineMakes: document.getElementById("cine-makes"),
    cinePauseBtn: document.getElementById("cine-pause-btn"),
    cineExitBtn: document.getElementById("cine-exit-btn"),
    rewardSpark: document.getElementById("reward-spark"),
    winrateSpark: document.getElementById("winrate-spark"),
    cineProgressFill: document.getElementById("cine-progress-fill"),
    sanctum: document.getElementById("sanctum"),
    sanctumCounter: document.getElementById("sanctum-counter"),
    sanctumQuote: document.getElementById("sanctum-quote"),
    // verdict metrics (folded into narrator block - same DOM nodes
    // whether aiming or frozen)
    psMiss: document.getElementById("ps-miss"),
    psBounces: document.getElementById("ps-bounces"),
    psReward: document.getElementById("ps-reward"),
    psRpeValue: document.getElementById("ps-rpe-value"),
    psRpeFill: document.getElementById("ps-rpe-fill"),
    // frozen-state affordance tips (one-shot per session)
    // help / pool tips modal
    helpFab: document.getElementById("help-fab"),
    aimCue: document.getElementById("aim-cue"),
    aimTarget: document.getElementById("aim-target"),
    aimAngle: document.getElementById("aim-angle"),
    aimBanks: document.getElementById("aim-banks"),
    tipsModal: document.getElementById("tips-modal"),
    tipsClose: document.getElementById("tips-close"),
    // wipe agent memory (lives inside the Q-table kebab menu)
    wipeMemory: document.getElementById("wipe-memory"),
    // graceful-transition affordances
    cineEyebrow: document.getElementById("cine-eyebrow"),
    cineLockedIn: document.getElementById("cine-locked-in"),
    feltPulse: document.getElementById("felt-pulse"),
    stageToast: document.getElementById("stage-toast"),
    // wipe-memory custom confirmation modal (replaces window.confirm)
    wipeModal: document.getElementById("wipe-modal"),
    wipeClose: document.getElementById("wipe-close"),
    wipeCancel: document.getElementById("wipe-cancel"),
    wipeConfirm: document.getElementById("wipe-confirm"),
    rpeHistory: document.getElementById("rpe-history"),
    rpeHistoryBars: document.getElementById("rpe-history-bars"),
    // learning telemetry
    learnCoveragePill: document.getElementById("learning-coverage-pill"),
    learnCoverageVal: document.getElementById("learn-coverage-val"),
    learnCoverageFill: document.getElementById("learn-coverage-fill"),
    learnSpikes: document.getElementById("learn-spikes"),
    learnSharpest: document.getElementById("learn-sharpest"),
    learnNative: document.getElementById("learn-native"),
    learnConfidence: document.getElementById("learn-confidence"),
  };

  // The sanctum (Dr Strange) beat now reads a single static line.
  // No rotation - the visual sigil + counter carry the moment.
  const SANCTUM_LINE = "Running every shot it has never taken.";

  // ----------------------------------------------------------------------
  // state
  // ----------------------------------------------------------------------

  let cfg = null;
  let selectedPocket = "TR";
  let hoverPocket = null;
  let busy = false;

  let appMode = "predict"; // predict | cinematic | sanctum

  // Rolling window of the last 12 reward-prediction-error samples
  // rendered as the dopamine sparkline beneath the verdict region.
  const rpeHistory = []; // {value, made, fresh, isOOD}
  const RPE_HISTORY_CAP = 12;

  // The agent is ALWAYS the player. You are the coach. You aim with the
  // cue stick to demonstrate shots you want the agent to learn. The cue
  // tracks your mouse and the live trajectory shows what the shot will
  // do. Whatever pocket the trajectory terminates in becomes the variant
  // you'd be teaching.
  const aim = {
    angleDeg: 90,
    previewSim: null,
    targetPocket: null,
  };

  // sanctum (Dr Strange beat) state
  const sanctum = {
    active: false,
    targetPocket: null,
    ws: null,
    episodesSeen: 0,
    plannedTimesteps: 1000,
    completionResolver: null,
  };

  const PHASE_PLAN = [
    { step: "belief", duration: 900 },
    { step: "guess", duration: 1100 },
    { step: "move", duration: 1600 },
    { step: "reality", duration: 900 },
    { step: "surprise", duration: 1100 },
    { step: "sharper belief", duration: 1100 },
  ];

  // predict animation state
  const anim = {
    active: false,
    startedAt: 0,
    phase: "idle",
    prediction: null,
    ballPos: null,
    ballAlpha: 1,
    ballScale: 1,
    trail: [],
    bounceRipple: { active: false, startedAt: 0 },
    // After the shot animation completes we FREEZE the table on its
    // final state - ball resting where it landed, trail still visible,
    // post-shot overlay showing Replay / Rack / Teach next. Aim and cue
    // tracking are suspended until the user explicitly racks. This
    // makes every shot feel like it had a result instead of evaporating.
    frozen: false,
    freezeResult: null,
  };

  // cinematic state (live training stream)
  const cine = {
    enabled: false,
    paused: false,
    episodes: [],
    summary: {},
    status: "idle",
    idx: -1,
    accumulator: 0,
    perEpisodeMs: 90, // dynamic; speeds up as the model converges
    grooves: [], // {path, alpha, color, born}  - parallel futures glowing
    rewardHistory: [],
    winrateHistory: [],
    explorationHistory: [],
    lastFrameTime: performance.now(),
    ws: null,
    streaming: false,
    completionShown: false,
    q: null,           // current Q-table snapshot (Float32Array of length N)
    bestAngleDeg: null,
    strictMatchLearned: null,
    actualBouncesLearned: null,
    // Dr Strange mystical state
    particles: [],     // {path, t, speed, life, color}  - sparkles streaming along trajectories
    recentAngles: [],  // last N angles for cue ghost fan
    portalPhase: 0,    // accumulated phase for target pocket portal pulses
  };

  let lastResult = null;

  // ----------------------------------------------------------------------
  // canvas sizing / geometry
  // ----------------------------------------------------------------------

  function resizeCanvas() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.floor(rect.width * dpr);
    canvas.height = Math.floor(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    sizeSparkline(els.rewardSpark);
    sizeSparkline(els.winrateSpark);
  }

  function sizeSparkline(canv) {
    if (!canv) return;
    const r = canv.getBoundingClientRect();
    canv.width = Math.floor(r.width * dpr);
    canv.height = Math.floor(r.height * dpr);
  }

  function tableGeometry() {
    const rect = canvas.getBoundingClientRect();
    const margin = Math.min(rect.width, rect.height) * 0.06;
    const railThickness = Math.max(
      28,
      Math.min(rect.width, rect.height) * 0.05
    );

    const outerW = rect.width - margin * 2;
    const outerH = rect.height - margin * 2;
    const aspect = cfg.table.width / cfg.table.height;

    let tableH = outerH;
    let tableW = tableH * aspect;
    if (tableW > outerW) {
      tableW = outerW;
      tableH = tableW / aspect;
    }

    const outerX = (rect.width - tableW) / 2;
    const outerY = (rect.height - tableH) / 2;

    const feltX = outerX + railThickness;
    const feltY = outerY + railThickness;
    const feltW = tableW - railThickness * 2;
    const feltH = tableH - railThickness * 2;

    return {
      outerX,
      outerY,
      tableW,
      tableH,
      feltX,
      feltY,
      feltW,
      feltH,
      railThickness,
    };
  }

  function worldToScreen(x, y, geo) {
    const sx = geo.feltX + (x / cfg.table.width) * geo.feltW;
    const sy = geo.feltY + geo.feltH - (y / cfg.table.height) * geo.feltH;
    return { x: sx, y: sy };
  }

  function pocketScreenPositions(geo) {
    const out = {};
    for (const [id, p] of Object.entries(cfg.pockets)) {
      out[id] = worldToScreen(p.x, p.y, geo);
    }
    return out;
  }

  // ----------------------------------------------------------------------
  // drawing primitives
  // ----------------------------------------------------------------------

  function roundRect(x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  // ----------------------------------------------------------------------
  // pool table render
  // ----------------------------------------------------------------------

  function drawTable() {
    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);

    const geo = tableGeometry();
    const positions = pocketScreenPositions(geo);
    const pocketWellR = 22;
    const pocketLipR = 18;

    // ----- rail / frame
    roundRect(geo.outerX, geo.outerY, geo.tableW, geo.tableH, 26);
    const railGrad = ctx.createLinearGradient(
      geo.outerX,
      geo.outerY,
      geo.outerX,
      geo.outerY + geo.tableH
    );
    railGrad.addColorStop(0, "#2a2520");
    railGrad.addColorStop(1, "#171410");
    ctx.fillStyle = railGrad;
    ctx.fill();

    // soft rail bevel
    ctx.save();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(255, 230, 200, 0.05)";
    roundRect(geo.outerX, geo.outerY, geo.tableW, geo.tableH, 26);
    ctx.stroke();
    ctx.restore();

    // rail diamonds
    drawDiamonds(geo);
    // Numbered overlay - real diamond-system labels (1-7 on long rails,
    // skipping 4 at the side pocket; 1-3 on short rails). Drawn on top
    // of the inlaid diamonds, just inboard so the player can read them
    // the same way they would at a real table.
    drawDiamondNumberLabels(geo);
    // KITCHEN label on the bottom short rail (where the cue lives).
    drawKitchenLabel(geo);

    // ----- felt with circular pocket cutouts using clip()
    ctx.save();
    roundRect(geo.feltX, geo.feltY, geo.feltW, geo.feltH, 14);
    ctx.clip();

    const feltGrad = ctx.createRadialGradient(
      geo.feltX + geo.feltW / 2,
      geo.feltY + geo.feltH / 2,
      Math.min(geo.feltW, geo.feltH) * 0.1,
      geo.feltX + geo.feltW / 2,
      geo.feltY + geo.feltH / 2,
      Math.max(geo.feltW, geo.feltH) * 0.7
    );
    feltGrad.addColorStop(0, "#236d63");
    feltGrad.addColorStop(1, "#0d3933");
    ctx.fillStyle = feltGrad;
    ctx.fillRect(
      geo.feltX - 4,
      geo.feltY - 4,
      geo.feltW + 8,
      geo.feltH + 8
    );

    // felt grain
    ctx.save();
    ctx.globalAlpha = 0.045;
    ctx.fillStyle = "#000";
    for (let i = 0; i < 30; i += 1) {
      const y = geo.feltY + (i / 30) * geo.feltH;
      ctx.fillRect(geo.feltX, y, geo.feltW, 1);
    }
    ctx.restore();

    // foot spot marker
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(
      geo.feltX + geo.feltW / 2,
      geo.feltY + geo.feltH * 0.25,
      6,
      0,
      Math.PI * 2
    );
    ctx.stroke();

    // head string - the line two diamonds from the kitchen end. Used in
    // real pool for ball-in-hand restrictions; here it doubles as a
    // visual landmark for the trajectory readout.
    ctx.beginPath();
    const headY = geo.feltY + geo.feltH * 0.75;
    ctx.moveTo(geo.feltX, headY);
    ctx.lineTo(geo.feltX + geo.feltW, headY);
    ctx.stroke();
    // Tiny "HS" tick on the left rail edge instead of a big mid-felt label.
    // Players already read the dashed line; the abbreviation is enough.
    ctx.save();
    ctx.font = "700 8.5px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "rgba(247, 232, 190, 0.55)";
    ctx.fillText("HS", geo.feltX + 4, headY);
    ctx.restore();

    // cinematic-mode grooves draw INSIDE the felt clip
    if (appMode === "cinematic") {
      drawGrooves(geo);
    }

    ctx.restore(); // end felt clip

    // ----- pocket wells (carved into rails AND felt at the boundary)
    for (const [id, pos] of Object.entries(positions)) {
      drawPocketWell(pos, pocketWellR, pocketLipR, id);
    }

    // ----- prediction overlay (predict mode) or active replay episode
    if (appMode === "predict") {
      drawPredictionLayer(geo);
      // While the table is frozen on a post-shot state, suppress the
      // live aim ghost so the result stays clean and readable.
      if (!anim.active && !anim.frozen && aim.previewSim) drawAimPreview(geo);
      drawCueStickAndBall(geo);
    } else {
      drawCinematicLayer(geo);
    }

    return geo;
  }

  function drawDiamonds(geo) {
    // Real pool table sight diamonds. Long rails: 8 equal segments
    // between corner pockets => 7 sight points, the middle one being
    // the side pocket itself (skipped). Short rails: 4 equal segments
    // => 3 sights. These are the diamonds players use for the
    // "diamond system" mental math (and so anyone watching the demo
    // can read the trajectory the way they would on a real table).
    const diamondHalf = Math.max(4, geo.railThickness * 0.14);
    const railMid = geo.railThickness * 0.5;

    function diamondAt(cx, cy) {
      // Inlaid look: bright fill, gold highlight ring, dark inner core.
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(Math.PI / 4); // square rotated = diamond
      const s = diamondHalf * 2;
      // shadow
      ctx.fillStyle = "rgba(0,0,0,0.45)";
      ctx.fillRect(-diamondHalf + 1, -diamondHalf + 1, s, s);
      // inlay body
      const grad = ctx.createLinearGradient(-diamondHalf, -diamondHalf, diamondHalf, diamondHalf);
      grad.addColorStop(0, "#f6e8c4");
      grad.addColorStop(0.5, "#e8d49a");
      grad.addColorStop(1, "#b69356");
      ctx.fillStyle = grad;
      ctx.fillRect(-diamondHalf, -diamondHalf, s, s);
      // inner darker core for depth
      ctx.fillStyle = "rgba(70, 50, 20, 0.55)";
      ctx.fillRect(-diamondHalf * 0.45, -diamondHalf * 0.45, s * 0.45, s * 0.45);
      // bright edge
      ctx.strokeStyle = "rgba(255, 240, 200, 0.85)";
      ctx.lineWidth = 1;
      ctx.strokeRect(-diamondHalf, -diamondHalf, s, s);
      ctx.restore();
    }

    // Portrait table (100w x 200h): LEFT and RIGHT are the long rails,
    // TOP and BOTTOM are the short rails.
    // Long-rail diamonds (left & right): 7 sights, skip #4 (side pocket).
    for (let i = 1; i <= 7; i += 1) {
      if (i === 4) continue;
      const ty = geo.outerY + (i / 8) * geo.tableH;
      diamondAt(geo.outerX + railMid, ty);
      diamondAt(geo.outerX + geo.tableW - railMid, ty);
    }
    // Short-rail diamonds (top & bottom): 3 sights, evenly spaced.
    for (let i = 1; i <= 3; i += 1) {
      const tx = geo.outerX + (i / 4) * geo.tableW;
      diamondAt(tx, geo.outerY + railMid);
      diamondAt(tx, geo.outerY + geo.tableH - railMid);
    }
  }

  // Numbered overlay on the diamonds. Real diamond-system convention:
  // long rails read 1..7 (with #4 being the side pocket, skipped); short
  // rails read 1..3. Printed on the rail wood adjacent to each sight,
  // between the diamond inlay and the felt edge. Drawn before the felt
  // clip so the labels live ON the rail wood, not inside the felt
  // (where the felt gradient fill would cover them).
  function drawDiamondNumberLabels(geo) {
    ctx.save();
    ctx.font = "700 13px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const rt = geo.railThickness;
    // Place the number on the OUTBOARD side of the diamond inlay so the
    // two don't overlap. The diamond sits at the rail mid-line; numbers
    // sit ~0.2 of railThickness outboard of that.
    const numOffset = rt * 0.22;
    const drawNum = (x, y, n) => {
      // Tiny dark plate behind each number for contrast over the wood
      // grain. Reads cleanly against both dark and gold-flecked rail.
      const txt = String(n);
      const tw = ctx.measureText(txt).width;
      ctx.fillStyle = "rgba(7, 10, 14, 0.55)";
      roundRect(x - tw / 2 - 3, y - 8, tw + 6, 16, 3);
      ctx.fill();
      ctx.fillStyle = "rgba(247, 232, 190, 1)";
      ctx.fillText(txt, x, y);
    };

    for (let i = 1; i <= 7; i += 1) {
      if (i === 4) continue;
      const ty = geo.outerY + (i / 8) * geo.tableH;
      drawNum(geo.outerX + numOffset, ty, i);
      drawNum(geo.outerX + geo.tableW - numOffset, ty, i);
    }
    for (let i = 1; i <= 3; i += 1) {
      const tx = geo.outerX + (i / 4) * geo.tableW;
      drawNum(tx, geo.outerY + numOffset, i);
      drawNum(tx, geo.outerY + geo.tableH - numOffset, i);
    }
    ctx.restore();
  }

  // KITCHEN label on the bottom short rail (where the cue ball sits at
  // world y=30). Sits on the OUTBOARD half of the bottom rail wood -
  // below the row of diamond inlays + their numbered labels - so it
  // never collides with diamond #2 (which is dead-center on the rail).
  function drawKitchenLabel(geo) {
    ctx.save();
    ctx.font = "700 11px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    // Sit on the felt-side half of the bottom rail wood, BELOW the
    // diamond inlays + numbered labels (which now live on the outboard
    // half). Tracking is letter-spaced manually so the word reads as a
    // rail engraving rather than a tight word.
    const railY = geo.outerY + geo.tableH - geo.railThickness * 0.82;
    const txt = "KITCHEN";
    // Manual letter-spacing for legibility on the wood rail.
    const letters = txt.split("");
    const spacing = 11;
    const totalW = (letters.length - 1) * spacing;
    let x = geo.outerX + geo.tableW / 2 - totalW / 2;
    // Single backplate spanning the whole word.
    const plateW = totalW + 22;
    ctx.fillStyle = "rgba(7, 10, 14, 0.6)";
    roundRect(geo.outerX + geo.tableW / 2 - plateW / 2, railY - 9, plateW, 18, 4);
    ctx.fill();
    ctx.fillStyle = "rgba(247, 232, 190, 0.95)";
    for (const ch of letters) {
      ctx.fillText(ch, x, railY);
      x += spacing;
    }
    ctx.restore();
  }

  // Cue-ball departure pip: a small green diamond-system pip on the
  // RIGHT long rail at the cue's diamond coord. For cue at world
  // (50, 30) the coord is 30/25 = 1.2, rendered as "D1.2".
  function drawCueDeparturePip(geo) {
    if (!cfg || !cfg.fixed_cue) return;
    const cueY = cfg.fixed_cue.y;
    const railMid = geo.railThickness * 0.5;
    const rightRailX = geo.outerX + geo.tableW - railMid;
    const sy = worldToScreen(cfg.fixed_cue.x, cueY, geo).y;
    // Just the marker dot - the D1.2 text label now lives in the right
    // panel's Aim Readout block so the canvas stays uncluttered.
    ctx.save();
    ctx.shadowColor = "rgba(91, 225, 192, 0.55)";
    ctx.shadowBlur = 6;
    ctx.fillStyle = "rgba(91, 225, 192, 1)";
    ctx.beginPath();
    ctx.arc(rightRailX, sy, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  // For a screen-space point that lies on (or near) a rail, return the
  // diamond coordinate of that point. Long rails: coord = world-y / 25
  // (range 0..8). Short rails: coord = world-x / 25 (range 0..4).
  // Returns { rail: "L"|"R"|"T"|"B", coord, sx, sy } or null.
  function worldPointToDiamond(worldX, worldY, geo) {
    const W = cfg.table.width;
    const H = cfg.table.height;
    const tol = 1.5; // world-units of rail proximity
    const onLeft = worldX <= tol;
    const onRight = worldX >= W - tol;
    const onBottom = worldY <= tol;
    const onTop = worldY >= H - tol;
    if (!(onLeft || onRight || onBottom || onTop)) return null;
    let rail = null;
    let coord = 0;
    if (onLeft) { rail = "L"; coord = worldY / 25; }
    else if (onRight) { rail = "R"; coord = worldY / 25; }
    else if (onBottom) { rail = "B"; coord = worldX / 25; }
    else if (onTop) { rail = "T"; coord = worldX / 25; }
    const screen = worldToScreen(worldX, worldY, geo);
    return { rail, coord, sx: screen.x, sy: screen.y };
  }

  // Trajectory readout for the live aim preview. For every rail bounce
  // on the simulated path, print the diamond coord at the rail in a
  // small gold chip. Plus a single chip near the cue showing the
  // departure -> first-bounce arithmetic.
  function drawDiamondTrajectoryReadout(geo, sim) {
    if (!sim || !sim.path || sim.path.length < 3) return;
    const interior = sim.path.slice(1, sim.path.length - 1);
    let firstBounceCoord = null;
    ctx.save();
    ctx.font = "700 11px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const [wx, wy] of interior) {
      const d = worldPointToDiamond(wx, wy, geo);
      if (!d) continue;
      if (firstBounceCoord === null) firstBounceCoord = d.coord;
      const inset = geo.railThickness * 0.55;
      let lx = d.sx, ly = d.sy;
      if (d.rail === "L") lx = d.sx + inset;
      else if (d.rail === "R") lx = d.sx - inset;
      else if (d.rail === "T") ly = d.sy + inset;
      else if (d.rail === "B") ly = d.sy - inset;
      const txt = `D${d.coord.toFixed(1)}`;
      const padX = 5, padY = 3;
      const tw = ctx.measureText(txt).width;
      ctx.fillStyle = "rgba(7, 10, 14, 0.85)";
      ctx.strokeStyle = "rgba(255, 211, 123, 0.7)";
      ctx.lineWidth = 1;
      const cw = tw + padX * 2;
      const ch = 14 + padY * 2;
      const cx0 = lx - cw / 2, cy0 = ly - ch / 2;
      roundRect(cx0, cy0, cw, ch, 5);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "rgba(255, 222, 145, 1)";
      ctx.fillText(txt, lx, ly);
    }
    ctx.restore();

    // Arithmetic chip near the cue. Real diamond-system:
    //   bank target diamond = departure - aim diamond
    // We print the departure -> first-bounce mapping so the player can
    // see the math their brain is supposed to be doing.
    if (firstBounceCoord !== null && cfg && cfg.fixed_cue) {
      const cueScreen = worldToScreen(cfg.fixed_cue.x, cfg.fixed_cue.y, geo);
      const cueCoord = cfg.fixed_cue.y / 25;
      const txt = `DEP D${cueCoord.toFixed(1)} \u2192 BNC D${firstBounceCoord.toFixed(1)}`;
      ctx.save();
      ctx.font = "700 10.5px 'JetBrains Mono', monospace";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      const padX = 8, padY = 4;
      const tw = ctx.measureText(txt).width;
      const lx = cueScreen.x + 22;
      const ly = cueScreen.y + 44;
      ctx.fillStyle = "rgba(7, 10, 14, 0.9)";
      ctx.strokeStyle = "rgba(91, 225, 192, 0.65)";
      ctx.lineWidth = 1;
      roundRect(lx, ly - 9 - padY, tw + padX * 2, 18 + padY * 2, 6);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = "rgba(91, 225, 192, 1)";
      ctx.fillText(txt, lx + padX, ly);
      ctx.restore();
    }
  }

  function drawPocketWell(pos, wellR, lipR, id) {
    const isSelected = id === selectedPocket;
    const isHover = id === hoverPocket;
    const trained = isPocketTrained(id);

    const grad = ctx.createRadialGradient(
      pos.x,
      pos.y,
      lipR * 0.3,
      pos.x,
      pos.y,
      wellR + 4
    );
    grad.addColorStop(0, "#000");
    grad.addColorStop(0.6, "#040608");
    grad.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, wellR + 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.beginPath();
    ctx.arc(pos.x, pos.y, lipR, 0, Math.PI * 2);
    ctx.fillStyle = "#04060a";
    ctx.fill();

    ctx.lineWidth = isSelected ? 3 : isHover ? 2 : 1.5;
    ctx.strokeStyle = isSelected
      ? "#5be1c0"
      : trained
      ? "#5be1c0"
      : isHover
      ? "#9aa4b5"
      : "#2c3340";
    ctx.stroke();

    if (isSelected) {
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, lipR + 8, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(91, 225, 192, 0.35)";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }

    if (trained) {
      // small filled "learned" dot at the lip
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(91, 225, 192, 0.85)";
      ctx.fill();
    }

    ctx.font = "700 12px 'JetBrains Mono', monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const labelOffset = labelOffsetFor(id);
    const lx = pos.x + labelOffset.dx;
    const ly = pos.y + labelOffset.dy;
    // Dark plate under the pocket id so it reads against both the
    // black pocket disc AND the wood rail.
    const tw = ctx.measureText(id).width;
    ctx.fillStyle = "rgba(7, 10, 14, 0.7)";
    roundRect(lx - tw / 2 - 4, ly - 9, tw + 8, 18, 4);
    ctx.fill();
    ctx.fillStyle = isSelected
      ? "#d8fff5"
      : trained
      ? "#9bf0d6"
      : "#a8b2c2";
    ctx.fillText(id, lx, ly);
  }

  function isPocketTrained(id) {
    return Boolean(
      cfg &&
        cfg.model &&
        Array.isArray(cfg.model.trained_pockets) &&
        cfg.model.trained_pockets.includes(id)
    );
  }

  function labelOffsetFor(id) {
    // positions labels onto the rail nearest the pocket
    switch (id) {
      case "TL":
        return { dx: 18, dy: -16 };
      case "TR":
        return { dx: -18, dy: -16 };
      case "BL":
        return { dx: 18, dy: 22 };
      case "BR":
        return { dx: -18, dy: 22 };
      case "LM":
        return { dx: -22, dy: 4 };
      case "RM":
        return { dx: 22, dy: 4 };
      default:
        return { dx: 0, dy: -22 };
    }
  }

  // ----------------------------------------------------------------------
  // predict-mode rendering
  // ----------------------------------------------------------------------

  function drawCueStickAndBall(geo) {
    const cueHome = worldToScreen(cfg.fixed_cue.x, cfg.fixed_cue.y, geo);

    // home reticle
    ctx.save();
    ctx.setLineDash([3, 4]);
    ctx.lineWidth = 1;
    ctx.strokeStyle = "rgba(255,255,255,0.22)";
    ctx.beginPath();
    ctx.arc(cueHome.x, cueHome.y, 14, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();

    // motion trail
    drawTrail();

    // Cue stick: animated during a shot, follows the coach's aim
    // between shots, and HELD on the fired angle while the table is
    // frozen on a result so the user can read the angle directly off
    // the geometry instead of inferring it from the trajectory.
    if (anim.active && anim.prediction) {
      drawCueStick(cueHome);
    } else if (anim.frozen && anim.freezeResult) {
      drawFiredCueStick(cueHome, anim.freezeResult);
    } else if (!anim.active && !anim.frozen) {
      drawAimCueStick(cueHome);
    }

    // bounce ripple
    if (anim.bounceRipple.active) {
      drawBounceRipple(geo);
    }

    // ball
    const ballPos = anim.ballPos ?? cueHome;
    drawBall(ballPos.x, ballPos.y, 11 * anim.ballScale, anim.ballAlpha);

    // LOCKED badge - a compact chip so the eye registers it without
    // hunting for it. Sits up-and-right of the cue home reticle.
    // "CUE LOCKED" badge removed - the cue ball position is fixed and
    // self-evident; the topbar chip already says "Cue locked at (50, 30)".
    // Keeping it on the felt was just clutter next to the ball.

    // Departure-value pip on the right long rail. Drawn here (not in
    // drawDiamonds) so it sits at the same layer as the cue ball -
    // always visible, always pointing at the cue's diamond coord.
    drawCueDeparturePip(geo);
  }

  function drawCueStick(cueHome) {
    if (!anim.prediction) return;
    const angleDeg = anim.prediction.angle_deg;

    let pullback = 60;
    let opacity = 1;

    if (anim.phase === "belief") {
      pullback = 70 + Math.sin(performance.now() / 220) * 6;
      opacity = 0.7;
    } else if (anim.phase === "guess") {
      const totalMs = PHASE_PLAN[1].duration;
      const elapsedInGuess = phaseElapsed("guess");
      const t = Math.min(1, elapsedInGuess / totalMs);
      pullback = lerp(90, 30, easeOutQuad(t));
      opacity = 1;
    } else if (anim.phase === "move") {
      const elapsedInMove = phaseElapsed("move");
      if (elapsedInMove < 80) {
        pullback = 8;
        opacity = 0.9;
      } else {
        return;
      }
    } else {
      return;
    }
    renderAnatomicalCue(cueHome, angleDeg, pullback, opacity);
  }

  // Continuous cue rendering for manual aim mode. Floats just behind the
  // ball at a fixed pullback - no breath animation. The mantra is firm
  // commitment, not idle motion.
  function drawAimCueStick(cueHome) {
    renderAnatomicalCue(cueHome, aim.angleDeg, 28, 0.92);
  }

  // Held-position cue rendering for the frozen post-shot state. The
  // stick stays on the exact angle that was fired so the user can
  // read the launch angle directly off the cue geometry. Slightly
  // dimmed and at a deeper "follow-through" pullback so it visually
  // codes as "shot taken, line of fire preserved."
  function drawFiredCueStick(cueHome, result) {
    const angleDeg =
      typeof result.angle_deg === "number" ? result.angle_deg : aim.angleDeg;
    renderAnatomicalCue(cueHome, angleDeg, 18, 0.7);
  }

  // Draws a more anatomically correct pool cue:
  //   tip (leather, blue) -> ferrule (white) -> shaft (tapered maple) ->
  //   joint (silver ring) -> forearm (dark wood w/ inlay band) -> wrap
  //   (Irish linen / leather) -> butt sleeve (dark) -> butt cap (black).
  // angleDeg uses the backend convention (0 = +x, 90 = +y).
  function renderAnatomicalCue(cueHome, angleDeg, pullback, opacity) {
    const rad = (angleDeg * Math.PI) / 180;
    const shotDx = Math.cos(rad);
    const shotDy = -Math.sin(rad); // screen y is flipped
    // perpendicular for drawing tapered segments
    const perpX = -shotDy;
    const perpY = shotDx;

    const cueLength = 250;
    const tipGap = 12; // gap between ball and tip
    const tipX = cueHome.x - shotDx * (tipGap + pullback - 12);
    const tipY = cueHome.y - shotDy * (tipGap + pullback - 12);
    const buttX = tipX - shotDx * cueLength;
    const buttY = tipY - shotDy * cueLength;

    ctx.save();
    ctx.globalAlpha = opacity;

    // shadow under stick
    ctx.save();
    ctx.shadowColor = "rgba(0,0,0,0.55)";
    ctx.shadowBlur = 5;
    ctx.shadowOffsetX = 2;
    ctx.shadowOffsetY = 3;
    drawTaperedQuad(tipX, tipY, buttX, buttY, perpX, perpY, 2.2, 4.6, "rgba(0,0,0,0.0001)");
    ctx.restore();

    // Section boundaries as a fraction of cue length, from TIP (0) to BUTT (1)
    const fracTipEnd = 0.012;        // leather tip
    const fracFerruleEnd = 0.040;    // white ferrule
    const fracShaftEnd = 0.50;       // shaft (maple)
    const fracJointEnd = 0.525;      // silver joint ring
    const fracForearmEnd = 0.70;     // forearm (dark wood)
    const fracInlayEnd = 0.715;      // thin inlay band
    const fracWrapEnd = 0.88;        // grip wrap
    const fracButtEnd = 0.985;       // butt sleeve
                                     // last 1.5%: butt cap
    const widthAtTip = 2.2;
    const widthAtButt = 4.6;
    const widthAt = (f) => widthAtTip + (widthAtButt - widthAtTip) * f;

    function pointAt(f) {
      return { x: tipX + (buttX - tipX) * f, y: tipY + (buttY - tipY) * f };
    }

    function paintSection(f0, f1, fillOrGrad) {
      const w0 = widthAt(f0);
      const w1 = widthAt(f1);
      const p0 = pointAt(f0);
      const p1 = pointAt(f1);
      const ax = p0.x + perpX * w0, ay = p0.y + perpY * w0;
      const bx = p1.x + perpX * w1, by = p1.y + perpY * w1;
      const cx = p1.x - perpX * w1, cy = p1.y - perpY * w1;
      const dx2 = p0.x - perpX * w0, dy2 = p0.y - perpY * w0;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.lineTo(cx, cy);
      ctx.lineTo(dx2, dy2);
      ctx.closePath();
      ctx.fillStyle = fillOrGrad;
      ctx.fill();
    }

    // shaft gradient (maple -> slightly darker near joint)
    const shaftP0 = pointAt(fracFerruleEnd);
    const shaftP1 = pointAt(fracShaftEnd);
    const shaftGrad = ctx.createLinearGradient(shaftP0.x, shaftP0.y, shaftP1.x, shaftP1.y);
    shaftGrad.addColorStop(0, "#f0d9ad");
    shaftGrad.addColorStop(1, "#d8b27a");

    // forearm gradient (dark walnut)
    const fp0 = pointAt(fracJointEnd);
    const fp1 = pointAt(fracForearmEnd);
    const foreGrad = ctx.createLinearGradient(fp0.x, fp0.y, fp1.x, fp1.y);
    foreGrad.addColorStop(0, "#3a2110");
    foreGrad.addColorStop(0.5, "#5b3a1f");
    foreGrad.addColorStop(1, "#3a2110");

    // wrap gradient (Irish linen, subtle banding)
    const wp0 = pointAt(fracInlayEnd);
    const wp1 = pointAt(fracWrapEnd);
    const wrapGrad = ctx.createLinearGradient(wp0.x, wp0.y, wp1.x, wp1.y);
    wrapGrad.addColorStop(0, "#5e4630");
    wrapGrad.addColorStop(0.5, "#8a6a48");
    wrapGrad.addColorStop(1, "#5e4630");

    // butt sleeve (dark with inlay)
    const bp0 = pointAt(fracWrapEnd);
    const bp1 = pointAt(fracButtEnd);
    const buttGrad = ctx.createLinearGradient(bp0.x, bp0.y, bp1.x, bp1.y);
    buttGrad.addColorStop(0, "#2a160a");
    buttGrad.addColorStop(0.5, "#48281a");
    buttGrad.addColorStop(1, "#2a160a");

    // tip (leather, blue chalked)
    paintSection(0, fracTipEnd, "#3a4f72");
    // ferrule (white plastic)
    paintSection(fracTipEnd, fracFerruleEnd, "#f7eedf");
    // shaft (maple)
    paintSection(fracFerruleEnd, fracShaftEnd, shaftGrad);
    // joint ring (silver)
    paintSection(fracShaftEnd, fracJointEnd, "#d6d8db");
    // forearm
    paintSection(fracJointEnd, fracForearmEnd, foreGrad);
    // thin metallic inlay
    paintSection(fracForearmEnd, fracInlayEnd, "#c9c5b8");
    // wrap
    paintSection(fracInlayEnd, fracWrapEnd, wrapGrad);
    // butt sleeve
    paintSection(fracWrapEnd, fracButtEnd, buttGrad);
    // butt cap (black rubber)
    paintSection(fracButtEnd, 1, "#111");

    // subtle wrap stitching - thin perpendicular bands
    ctx.save();
    ctx.strokeStyle = "rgba(20,12,8,0.5)";
    ctx.lineWidth = 0.6;
    const steps = 14;
    for (let i = 0; i <= steps; i += 1) {
      const f = fracInlayEnd + (fracWrapEnd - fracInlayEnd) * (i / steps);
      const p = pointAt(f);
      const w = widthAt(f);
      ctx.beginPath();
      ctx.moveTo(p.x + perpX * w, p.y + perpY * w);
      ctx.lineTo(p.x - perpX * w, p.y - perpY * w);
      ctx.stroke();
    }
    ctx.restore();

    // shine highlight running down the upper edge of the shaft
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = 0.9;
    ctx.beginPath();
    const hp0 = pointAt(fracFerruleEnd);
    const hp1 = pointAt(fracShaftEnd);
    const w0 = widthAt(fracFerruleEnd) * 0.55;
    const w1 = widthAt(fracShaftEnd) * 0.55;
    ctx.moveTo(hp0.x + perpX * w0, hp0.y + perpY * w0);
    ctx.lineTo(hp1.x + perpX * w1, hp1.y + perpY * w1);
    ctx.stroke();
    ctx.restore();

    ctx.restore();
  }

  // helper: filled tapered quad used only for the shadow underlay
  function drawTaperedQuad(tipX, tipY, buttX, buttY, perpX, perpY, w0, w1, fill) {
    const ax = tipX + perpX * w0, ay = tipY + perpY * w0;
    const bx = buttX + perpX * w1, by = buttY + perpY * w1;
    const cx = buttX - perpX * w1, cy = buttY - perpY * w1;
    const dx = tipX - perpX * w0, dy = tipY - perpY * w0;
    ctx.fillStyle = fill;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.lineTo(cx, cy);
    ctx.lineTo(dx, dy);
    ctx.closePath();
    ctx.fill();
  }

  function drawTrail() {
    if (anim.trail.length < 2) return;
    for (let i = 1; i < anim.trail.length; i += 1) {
      const a = anim.trail[i - 1];
      const b = anim.trail[i];
      const t = i / anim.trail.length;
      ctx.beginPath();
      ctx.strokeStyle = `rgba(245, 255, 250, ${0.05 + 0.35 * t})`;
      ctx.lineWidth = 1.5 + 4 * t;
      ctx.lineCap = "round";
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
    }
  }

  function drawBounceRipple(geo) {
    const elapsed = performance.now() - anim.bounceRipple.startedAt;
    const total = 380;
    const t = Math.min(1, elapsed / total);
    if (t >= 1) {
      anim.bounceRipple.active = false;
      return;
    }
    const bounce = anim.bounceRipple.point;
    for (const offset of [0, 110]) {
      const local = Math.max(0, Math.min(1, (elapsed - offset) / (total - offset)));
      if (local <= 0) continue;
      const r = lerp(4, 22, local);
      ctx.beginPath();
      ctx.arc(bounce.x, bounce.y, r, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255, 255, 255, ${(1 - local) * 0.45})`;
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }
  }

  function drawBall(x, y, r, alpha = 1) {
    if (alpha <= 0) return;
    ctx.save();
    ctx.globalAlpha = alpha;
    // soft shadow
    ctx.beginPath();
    ctx.arc(x + 1.5, y + 3, r * 1.05, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(0,0,0,0.45)";
    ctx.fill();

    const g = ctx.createRadialGradient(x - r * 0.35, y - r * 0.35, 0.5, x, y, r);
    g.addColorStop(0, "#ffffff");
    g.addColorStop(0.6, "#e9eef6");
    g.addColorStop(1, "#a8b1c1");
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = g;
    ctx.fill();
    ctx.lineWidth = 0.8;
    ctx.strokeStyle = "rgba(0,0,0,0.45)";
    ctx.stroke();
    ctx.restore();
  }

  // Live trajectory preview while the player is aiming. Dashed ghost
  // line for the path, fixed-size dot at each rail bounce, bounce-count
  // badge near the cue, and the diamond-system readout printed on the
  // rail at every bounce so the trajectory reads like real billiards.
  function drawAimPreview(geo) {
    const sim = aim.previewSim;
    if (!sim || !sim.path || sim.path.length < 2) return;

    const pts = sim.path.map(([x, y]) => worldToScreen(x, y, geo));
    const color = sim.made ? "rgba(91, 225, 192," : "rgba(255, 211, 123,";

    // ghost trajectory line
    ctx.save();
    ctx.setLineDash([6, 6]);
    ctx.lineDashOffset = -(performance.now() / 60) % 12;
    ctx.strokeStyle = color + "0.75)";
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.moveTo(pts[0].x, pts[0].y);
    for (let i = 1; i < pts.length; i += 1) {
      ctx.lineTo(pts[i].x, pts[i].y);
    }
    ctx.stroke();
    ctx.restore();

    // bounce dots (every interior waypoint)
    for (let i = 1; i < pts.length - 1; i += 1) {
      const p = pts[i];
      ctx.save();
      ctx.fillStyle = color + "0.9)";
      ctx.beginPath();
      ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = color + "0.45)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 7, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }

    // Bounce-count + aim-angle chip near the cue. A single chip carries
    // both the shape of the shot (DIRECT / 1 BANK / etc.) and the angle
    // the cue is pointing - the two pieces of info the player needs to
    // decide if they like the line they're about to fire.
    const cueScreen = pts[0];
    const shape = sim.bounces === 0
      ? "DIRECT"
      : sim.bounces === 1
      ? "1 BANK"
      : sim.bounces === 2
      ? "2 BANK"
      : "3 BANK";
    const targetTxt = sim.made ? ` -> ${sim.pocket_hit}` : "";
    const angTxt = `${Math.round(((aim.angleDeg % 360) + 360) % 360)}°`;
    const chipTxt = `${shape}${targetTxt}  ${angTxt}`;
    ctx.save();
    ctx.font = "700 11px 'JetBrains Mono', monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const tw = ctx.measureText(chipTxt).width;
    const padX = 8, padY = 4;
    const cx0 = cueScreen.x - 32 - padX;
    const cy0 = cueScreen.y + 26;
    ctx.fillStyle = "rgba(7, 10, 14, 0.9)";
    ctx.strokeStyle = color + "0.7)";
    ctx.lineWidth = 1;
    roundRect(cx0, cy0 - 9 - padY, tw + padX * 2, 18 + padY * 2, 6);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = color + "1)";
    ctx.fillText(chipTxt, cx0 + padX, cy0);
    ctx.restore();

    // Diamond-system trajectory readout - print the diamond coord of
    // every bounce on the rail it landed on, plus the sum-arithmetic
    // chip near the cue.
    drawDiamondTrajectoryReadout(geo, sim);

    // Fixed-radius marker on the final point (no animated halo).
    const last = pts[pts.length - 1];
    ctx.save();
    ctx.strokeStyle = color + "0.75)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(last.x, last.y, 9, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
  }

  function drawPredictionLayer(geo) {
    if (!anim.prediction) return;
    const path = anim.prediction.predicted.path.map(([x, y]) =>
      worldToScreen(x, y, geo)
    );
    const isGhost = anim.phase === "belief" || anim.phase === "guess";

    const beliefColor = anim.prediction.is_trained_pocket
      ? "rgba(91, 225, 192, "
      : "rgba(255, 211, 123, ";

    // belief / committed line. When the table is frozen on the result,
    // the line gets a slightly heavier stroke so it reads as the
    // replay-target affordance.
    ctx.save();
    ctx.setLineDash([6, 6]);
    ctx.lineWidth = anim.frozen ? 2.6 : 2.0;
    ctx.strokeStyle = beliefColor + (isGhost ? 0.8 : (anim.frozen ? 0.65 : 0.4)) + ")";
    ctx.beginPath();
    ctx.moveTo(path[0].x, path[0].y);
    for (let i = 1; i < path.length; i += 1) {
      ctx.lineTo(path[i].x, path[i].y);
    }
    ctx.stroke();
    ctx.restore();

    // When frozen and the cursor is over the path, render a soft
    // replay glyph at the first bounce to telegraph the gesture.
    if (anim.frozen && anim.hoverPath && path.length >= 2) {
      const replayAt = path[Math.min(1, path.length - 1)];
      ctx.save();
      ctx.fillStyle = "rgba(91, 225, 192, 0.95)";
      ctx.font = "600 14px 'JetBrains Mono', monospace";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("\u21BB", replayAt.x, replayAt.y);
      ctx.strokeStyle = "rgba(91, 225, 192, 0.55)";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(replayAt.x, replayAt.y, 11, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }

  // ----------------------------------------------------------------------
  // cinematic-mode rendering
  // ----------------------------------------------------------------------

  function drawGrooves(geo) {
    // Render last N futures as glowing parallel timelines. Made shots
    // are vivid orange-gold (the futures where the agent wins). Misses
    // are dim cyan ghosts (the futures that fade). Together they look
    // like Strange's "14 million possibilities" playing out at once.
    const now = performance.now();
    for (const g of cine.grooves) {
      if (g.alpha < 0.04) continue;
      const pts = g.path.map(([x, y]) => worldToScreen(x, y, geo));
      if (pts.length < 2) continue;

      const age = (now - (g.born || now)) / 1000;
      // shimmer: subtle alpha modulation per groove so the bundle feels alive
      const shimmer = 0.85 + 0.15 * Math.sin(age * 3 + (g.born || 0) * 0.0001);
      const a = Math.min(0.85, g.alpha * shimmer);

      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      // Outer halo glow
      ctx.strokeStyle = g.color.replace("ALPHA", (a * 0.4).toFixed(3));
      ctx.lineWidth = 4.5;
      ctx.shadowColor = g.color.replace("ALPHA", (a * 0.9).toFixed(3));
      ctx.shadowBlur = g.made ? 14 : 6;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i += 1) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.stroke();

      // Crisp core line
      ctx.shadowBlur = 0;
      ctx.strokeStyle = g.color.replace("ALPHA", a.toFixed(3));
      ctx.lineWidth = g.made ? 1.8 : 1.2;
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i += 1) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.stroke();

      ctx.restore();
    }
  }

  function drawCinematicLayer(geo) {
    const cueHome = worldToScreen(cfg.fixed_cue.x, cfg.fixed_cue.y, geo);
    const ep = cine.episodes[cine.idx];
    const now = performance.now();
    const explore = ep ? (ep.exploration_rate || 0) : 1;
    const lockedIn =
      cine.status === "done" ||
      (cine.summary && cine.summary.status === "done");

    // ---- Layer 1: Target pocket portal (under everything) --------------
    drawPocketPortal(geo, now, 1 - explore);

    // ---- Layer 2: Big table mandala ------------------------------------
    drawTableMandala(geo, cueHome, now, 1 - explore);

    // ---- Layer 3: Parallel future trails (the grooves) -----------------
    // (already painted under the felt clip in drawTable; nothing to do here)

    // ---- Layer 4: Energy particles flowing along trajectories ----------
    advanceAndDrawParticles(geo);

    // ---- Layer 5: Active episode trace (subtle - flying balls + grooves carry the load) ---
    // At 30+ episodes/sec a heavy stroke would strobe. Keep this as a
    // quiet accent that ghosts on top of the groove library.
    if (ep && ep.path && ep.path.length >= 2) {
      const pts = ep.path.map(([x, y]) => worldToScreen(x, y, geo));
      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.shadowBlur = 6;
      ctx.shadowColor = ep.made
        ? "rgba(255, 220, 140, 0.7)"
        : "rgba(180, 220, 240, 0.55)";
      ctx.lineWidth = ep.made ? 2.2 : 1.5;
      ctx.strokeStyle = ep.made
        ? "rgba(255, 250, 220, 0.9)"
        : "rgba(220, 240, 250, 0.7)";
      ctx.beginPath();
      ctx.moveTo(pts[0].x, pts[0].y);
      for (let i = 1; i < pts.length; i += 1) ctx.lineTo(pts[i].x, pts[i].y);
      ctx.stroke();
      ctx.restore();
    }

    // ---- Layer 6: Q-halo and rotating sigils around cue ----------------
    drawQHalo(cueHome);

    // ---- Layer 7: Cue ghost fan (recent angles tried) ------------------
    drawCueGhostFan(cueHome, now);

    // ---- Layer 8: Main cue stick (locks in as exploration drops) -------
    if (ep) {
      const pullback = 30 + explore * 40;
      const opacity = Math.min(1, 0.45 + (1 - explore) * 0.55);
      if (lockedIn && cine.summary && cine.summary.final_angle !== null) {
        renderAnatomicalCue(cueHome, cine.summary.final_angle, 28, 0.98);
      } else {
        renderAnatomicalCue(cueHome, ep.angle_deg, pullback, opacity);
      }
    } else if (cine.anchorAngleDeg !== undefined && cine.anchorAngleDeg !== null) {
      // pre-first-episode: render a faint cue at the anchor angle so the
      // table doesn't feel empty while we wait for the first frame
      renderAnatomicalCue(cueHome, cine.anchorAngleDeg, 60, 0.35);
    }

    // ---- Layer 9: Eye of Agamotto cue ball -----------------------------
    drawEyeOfAgamotto(cueHome, now, 1 - explore);
  }

  // ---- Helpers for cinematic mode visuals ------------------------------

  function getActivePocketScreen(geo) {
    const pid = cine.targetPocket || cine.summary?.target_pocket;
    if (!pid || !cfg.pockets || !cfg.pockets[pid]) return null;
    const p = cfg.pockets[pid];
    return worldToScreen(p.x, p.y, geo);
  }

  function drawPocketPortal(geo, now, confidence) {
    // Pulsing portal radiating outward from the target pocket. As the
    // agent's confidence grows (exploration drops), the portal tightens
    // and burns brighter. This is the destination Strange is steering toward.
    const pp = getActivePocketScreen(geo);
    if (!pp) return;

    const t = (now % 2400) / 2400;
    const baseR = 24 + 18 * confidence;
    const maxR = 92 - 28 * confidence;

    ctx.save();
    ctx.translate(pp.x, pp.y);

    // Three staggered pulse rings
    for (let i = 0; i < 3; i += 1) {
      const phase = (t + i / 3) % 1;
      const r = baseR + phase * (maxR - baseR);
      const a = (1 - phase) * (0.28 + confidence * 0.45);
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(255, 198, 110, ${a.toFixed(3)})`;
      ctx.lineWidth = 1.5 + confidence * 1.5;
      ctx.shadowColor = `rgba(255, 178, 84, ${(a * 0.8).toFixed(3)})`;
      ctx.shadowBlur = 14;
      ctx.stroke();
    }

    // Inner radial glow
    const glowR = baseR * 1.4;
    const grad = ctx.createRadialGradient(0, 0, 0, 0, 0, glowR);
    grad.addColorStop(0, `rgba(255, 220, 140, ${(0.22 + 0.35 * confidence).toFixed(3)})`);
    grad.addColorStop(1, "rgba(255, 178, 84, 0)");
    ctx.fillStyle = grad;
    ctx.shadowBlur = 0;
    ctx.beginPath();
    ctx.arc(0, 0, glowR, 0, Math.PI * 2);
    ctx.fill();

    // Counter-rotating sigil ring with dashes
    ctx.rotate(now / 1800);
    ctx.strokeStyle = `rgba(255, 220, 150, ${(0.35 + 0.4 * confidence).toFixed(3)})`;
    ctx.lineWidth = 1.2;
    ctx.setLineDash([6, 9, 2, 5]);
    ctx.beginPath();
    ctx.arc(0, 0, baseR + 6, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.restore();
  }

  function drawTableMandala(geo, cueHome, now, confidence) {
    // A massive translucent sacred-geometry mandala drifting over the
    // felt, centered at the table center between cue and target pocket.
    // It rotates, scales with confidence, and overlays glyphs at compass
    // points like Strange's spell circles.
    const target = getActivePocketScreen(geo);
    let cx, cy;
    if (target) {
      cx = (cueHome.x + target.x) / 2;
      cy = (cueHome.y + target.y) / 2;
    } else {
      cx = geo.feltX + geo.feltW / 2;
      cy = geo.feltY + geo.feltH / 2;
    }

    const baseScale = Math.min(geo.feltW, geo.feltH) * 0.42;
    // Mandala tightens as confidence rises (less exploration = tighter spell)
    const scale = baseScale * (0.75 + 0.25 * confidence);
    const alpha = 0.18 + 0.18 * confidence;

    ctx.save();
    // clip to the felt rectangle so the mandala doesn't paint over rails / pocket wells
    ctx.beginPath();
    ctx.rect(geo.feltX, geo.feltY, geo.feltW, geo.feltH);
    ctx.clip();
    ctx.translate(cx, cy);
    ctx.globalAlpha = alpha;

    // --- Outermost ring (slow spin, dotted) -----
    ctx.save();
    ctx.rotate(now / 9000);
    ctx.strokeStyle = "rgba(255, 198, 110, 0.55)";
    ctx.lineWidth = 1.4;
    ctx.setLineDash([1, 7]);
    ctx.beginPath();
    ctx.arc(0, 0, scale, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    // 12 glyph squares around the outer ring
    for (let i = 0; i < 12; i += 1) {
      const a = (i / 12) * Math.PI * 2;
      const gx = Math.cos(a) * scale;
      const gy = Math.sin(a) * scale;
      ctx.save();
      ctx.translate(gx, gy);
      ctx.rotate(a);
      ctx.strokeStyle = "rgba(255, 220, 140, 0.85)";
      ctx.lineWidth = 1;
      ctx.strokeRect(-3, -3, 6, 6);
      ctx.restore();
    }
    ctx.restore();

    // --- Middle ring (counter spin, dashed) -----
    ctx.save();
    ctx.rotate(-now / 5500);
    ctx.strokeStyle = "rgba(255, 220, 140, 0.6)";
    ctx.lineWidth = 1.2;
    ctx.setLineDash([14, 10]);
    ctx.beginPath();
    ctx.arc(0, 0, scale * 0.72, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.restore();

    // --- Inscribed octagram (two squares rotated 45deg) -----
    ctx.save();
    ctx.rotate(now / 7000);
    ctx.strokeStyle = "rgba(255, 240, 200, 0.55)";
    ctx.lineWidth = 1;
    for (let sq = 0; sq < 2; sq += 1) {
      ctx.save();
      ctx.rotate(sq * Math.PI / 4);
      ctx.beginPath();
      const r = scale * 0.55;
      for (let k = 0; k < 5; k += 1) {
        const a = (k / 4) * Math.PI * 2;
        const x = Math.cos(a) * r;
        const y = Math.sin(a) * r;
        if (k === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }
    ctx.restore();

    // --- Inner sigil ring (fast spin) -----
    ctx.save();
    ctx.rotate(now / 2200);
    ctx.strokeStyle = "rgba(255, 240, 180, 0.7)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 5]);
    ctx.beginPath();
    ctx.arc(0, 0, scale * 0.34, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
    // Six small glyph triangles
    for (let i = 0; i < 6; i += 1) {
      const a = (i / 6) * Math.PI * 2;
      const r = scale * 0.34;
      const gx = Math.cos(a) * r;
      const gy = Math.sin(a) * r;
      ctx.save();
      ctx.translate(gx, gy);
      ctx.rotate(a + Math.PI / 2);
      ctx.beginPath();
      ctx.moveTo(0, -3);
      ctx.lineTo(3, 3);
      ctx.lineTo(-3, 3);
      ctx.closePath();
      ctx.strokeStyle = "rgba(255, 240, 200, 0.85)";
      ctx.stroke();
      ctx.restore();
    }
    ctx.restore();

    ctx.restore();
  }

  function advanceAndDrawParticles(geo) {
    if (!cine.particles.length) return;
    const dt = 1 / 60;
    const survivors = [];
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    for (const p of cine.particles) {
      p.t += p.speed * dt;
      if (p.t >= 1) continue; // off the end
      if (p.t < 0) {
        survivors.push(p);
        continue;
      }
      const pos = sampleAlongWorldPath(p.path, p.t);
      if (!pos) continue;
      const s = worldToScreen(pos.x, pos.y, geo);
      // Headlight at the front of each flying shot
      const headA = Math.min(1, (1 - p.t) * 1.05 + 0.05);
      const headR = p.made ? 4.2 : 3.4;

      // Multi-segment comet tail
      const tailSegs = 7;
      const tailStep = 0.032;
      ctx.shadowBlur = 0;
      for (let k = tailSegs; k >= 1; k -= 1) {
        const ta = Math.max(0, p.t - k * tailStep);
        const tb = Math.max(0, p.t - (k - 1) * tailStep);
        if (ta >= tb) continue;
        const pa = sampleAlongWorldPath(p.path, ta);
        const pb = sampleAlongWorldPath(p.path, tb);
        if (!pa || !pb) continue;
        const sa = worldToScreen(pa.x, pa.y, geo);
        const sb = worldToScreen(pb.x, pb.y, geo);
        const segA = (headA * (k / tailSegs)) * 0.55;
        ctx.strokeStyle = p.color.replace("ALPHA", segA.toFixed(3));
        ctx.lineWidth = (k / tailSegs) * (p.made ? 2.8 : 2.0);
        ctx.beginPath();
        ctx.moveTo(sa.x, sa.y);
        ctx.lineTo(sb.x, sb.y);
        ctx.stroke();
      }

      // Glowing head (outer halo)
      ctx.shadowColor = p.color.replace("ALPHA", (headA * 0.95).toFixed(3));
      ctx.shadowBlur = p.made ? 18 : 12;
      ctx.fillStyle = p.color.replace("ALPHA", (headA * 0.85).toFixed(3));
      ctx.beginPath();
      ctx.arc(s.x, s.y, headR, 0, Math.PI * 2);
      ctx.fill();

      // Hot white core
      ctx.shadowBlur = 4;
      ctx.fillStyle = `rgba(255, 250, 220, ${(headA * 0.95).toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(s.x, s.y, headR * 0.45, 0, Math.PI * 2);
      ctx.fill();

      survivors.push(p);
    }
    ctx.restore();
    cine.particles = survivors;
  }

  function sampleAlongWorldPath(pathWorld, t) {
    if (!pathWorld || pathWorld.length < 2) return null;
    // Compute cumulative segment lengths in world units
    let total = 0;
    const segLen = [];
    for (let i = 1; i < pathWorld.length; i += 1) {
      const dx = pathWorld[i][0] - pathWorld[i - 1][0];
      const dy = pathWorld[i][1] - pathWorld[i - 1][1];
      const L = Math.hypot(dx, dy);
      segLen.push(L);
      total += L;
    }
    if (total <= 0) return { x: pathWorld[0][0], y: pathWorld[0][1] };
    const target = Math.max(0, Math.min(1, t)) * total;
    let acc = 0;
    for (let i = 1; i < pathWorld.length; i += 1) {
      const L = segLen[i - 1];
      if (acc + L >= target) {
        const u = L > 0 ? (target - acc) / L : 0;
        return {
          x: pathWorld[i - 1][0] + (pathWorld[i][0] - pathWorld[i - 1][0]) * u,
          y: pathWorld[i - 1][1] + (pathWorld[i][1] - pathWorld[i - 1][1]) * u,
        };
      }
      acc += L;
    }
    const last = pathWorld[pathWorld.length - 1];
    return { x: last[0], y: last[1] };
  }

  function drawCueGhostFan(cueHome, now) {
    // Render recent angle attempts as faint ghost cue-shafts radiating
    // from the cue ball. Strange weighing futures in his hand.
    if (!cine.recentAngles.length) return;
    ctx.save();
    for (let i = 0; i < cine.recentAngles.length - 1; i += 1) {
      const entry = cine.recentAngles[i];
      const age = (now - entry.born) / 1000;
      const a = Math.max(0, 0.45 - age * 0.6);
      if (a < 0.04) continue;
      const rad = (entry.angle * Math.PI) / 180;
      const dx = Math.cos(rad);
      const dy = -Math.sin(rad);
      // Ghost shaft extending OUT from the cue ball along the angle of attack
      const tipX = cueHome.x + dx * 14;
      const tipY = cueHome.y + dy * 14;
      const buttX = cueHome.x - dx * 130;
      const buttY = cueHome.y - dy * 130;
      const grad = ctx.createLinearGradient(tipX, tipY, buttX, buttY);
      grad.addColorStop(0, `rgba(255, 220, 150, ${a.toFixed(3)})`);
      grad.addColorStop(1, "rgba(255, 178, 84, 0)");
      ctx.strokeStyle = grad;
      ctx.lineWidth = 2.2;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(buttX, buttY);
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawEyeOfAgamotto(cueHome, now, confidence) {
    // The cue ball during training is the Eye of Agamotto: an inner
    // rotating sigil, an amber halo, a bright iris that pulses with
    // the agent's confidence.
    const r = 11;

    // Outer warm halo
    ctx.save();
    const haloR = r + 8 + 3 * Math.sin(now / 400);
    const halo = ctx.createRadialGradient(
      cueHome.x, cueHome.y, r,
      cueHome.x, cueHome.y, haloR
    );
    halo.addColorStop(0, `rgba(255, 220, 140, ${(0.28 + confidence * 0.35).toFixed(3)})`);
    halo.addColorStop(1, "rgba(255, 178, 84, 0)");
    ctx.fillStyle = halo;
    ctx.beginPath();
    ctx.arc(cueHome.x, cueHome.y, haloR, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // The cue ball itself
    drawBall(cueHome.x, cueHome.y, r);

    // Iris on top - rotating amber sigil triangle
    ctx.save();
    ctx.translate(cueHome.x, cueHome.y);
    ctx.rotate(now / 1100);
    const irisR = 4 + 1.5 * Math.sin(now / 320);
    // amber outer ring
    ctx.strokeStyle = `rgba(255, 188, 90, ${(0.7 + confidence * 0.3).toFixed(3)})`;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(0, 0, irisR, 0, Math.PI * 2);
    ctx.stroke();
    // inner pupil
    ctx.fillStyle = `rgba(255, 240, 200, ${(0.65 + confidence * 0.35).toFixed(3)})`;
    ctx.beginPath();
    ctx.arc(0, 0, 1.6, 0, Math.PI * 2);
    ctx.fill();
    // three little rune marks around the iris
    for (let i = 0; i < 3; i += 1) {
      const a = (i / 3) * Math.PI * 2;
      ctx.save();
      ctx.rotate(a);
      ctx.strokeStyle = `rgba(255, 220, 150, ${(0.55).toFixed(3)})`;
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(irisR + 1.5, 0);
      ctx.lineTo(irisR + 3.5, 0);
      ctx.stroke();
      ctx.restore();
    }
    ctx.restore();

    // CUE label, smaller and more elegant
    ctx.save();
    ctx.font = "600 8.5px 'JetBrains Mono', monospace";
    ctx.fillStyle = "rgba(255, 220, 140, 0.7)";
    ctx.fillText("CUE", cueHome.x + 16, cueHome.y - 12);
    ctx.restore();
  }

  function drawQHalo(cueHome) {
    if (!cine.q || !cine.q.length) return;
    const q = cine.q;
    const n = q.length;

    // We want the halo to show POSITIVE spikes clearly.
    // If the agent has only found negative rewards (or 0), the halo should remain flat.
    let hi = 1;
    for (let i = 0; i < n; i += 1) {
      if (q[i] > hi) hi = q[i];
    }

    const baseR = 22;
    const maxLen = 110;
    const bestIdx = bestQIndex(q);

    ctx.save();
    for (let i = 0; i < n; i += 1) {
      const v = q[i];
      // Only positive Q values create a spike. Unvisited (0) or negative (-25) stay at baseR.
      const norm = v > 0 ? v / hi : 0;
      const len = baseR + Math.pow(norm, 0.6) * maxLen;
      const angleDeg = (i / n) * 360;
      const rad = (angleDeg * Math.PI) / 180;
      const dx = Math.cos(rad);
      const dy = -Math.sin(rad);
      const x0 = cueHome.x + dx * baseR;
      const y0 = cueHome.y + dy * baseR;
      const x1 = cueHome.x + dx * len;
      const y1 = cueHome.y + dy * len;

      // Heat ramp from dim sapphire (low Q, cold futures) to bright amber
      // (high Q, the right future). Strange's spell weighing possibilities.
      const a = 0.18 + norm * 0.6;
      const r = Math.round(90 + norm * 165);
      const g = Math.round(140 + norm * 80);
      const b = Math.round(200 - norm * 130);
      ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${a.toFixed(3)})`;
      ctx.lineWidth = i === bestIdx && norm > 0 ? 2.8 : 1.2;
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }

    // committed-angle spike - the bright golden ray pointing to the future Strange has chosen
    if (bestIdx >= 0) {
      const angleDeg = (bestIdx / n) * 360;
      const rad = (angleDeg * Math.PI) / 180;
      const dx = Math.cos(rad);
      const dy = -Math.sin(rad);
      const x1 = cueHome.x + dx * (baseR + maxLen + 12);
      const y1 = cueHome.y + dy * (baseR + maxLen + 12);
      ctx.strokeStyle = "rgba(255, 220, 140, 0.95)";
      ctx.lineWidth = 3;
      ctx.shadowColor = "rgba(255, 188, 90, 0.8)";
      ctx.shadowBlur = 12;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(cueHome.x + dx * baseR, cueHome.y + dy * baseR);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }
    ctx.restore();
  }

  function bestQIndex(q) {
    let bi = -1, bv = -Infinity;
    for (let i = 0; i < q.length; i += 1) {
      if (q[i] > bv) {
        bv = q[i];
        bi = i;
      }
    }
    return bi;
  }

  // ----------------------------------------------------------------------
  // animation loop
  // ----------------------------------------------------------------------

  let dashboardLastDraw = 0;
  function tick() {
    if (appMode === "predict") {
      tickPredict();
    } else {
      tickCinematic();
    }
    drawTable();
    // Dashboard side-cars repaint at ~20fps to stay smooth during
    // training without burning frames in the idle case.
    const now = performance.now();
    if (now - dashboardLastDraw > 50) {
      drawBeliefHistogram();
      // Refresh the memory grid + aim readout during cinematic so the
      // "WRITING" pulse stays applied to the active cell as training runs.
      if (appMode === "cinematic") {
        refreshMemoryGrid();
        refreshAimReadout();
      }
      dashboardLastDraw = now;
    }
    requestAnimationFrame(tick);
  }

  function tickPredict() {
    // While frozen on a post-shot, preserve ballPos + trail so the result
    // stays painted on the felt until the user explicitly racks.
    if (anim.frozen) return;
    if (!anim.active || !anim.prediction) {
      anim.ballPos = null;
      anim.trail = [];
      return;
    }

    const elapsed = performance.now() - anim.startedAt;
    let cursor = 0;
    let current = PHASE_PLAN[0].step;
    let phaseStart = 0;
    let phaseDuration = PHASE_PLAN[0].duration;
    for (const phase of PHASE_PLAN) {
      if (elapsed < cursor + phase.duration) {
        current = phase.step;
        phaseStart = cursor;
        phaseDuration = phase.duration;
        break;
      }
      cursor += phase.duration;
      current = phase.step;
      phaseStart = cursor - phase.duration;
      phaseDuration = phase.duration;
    }

    if (current !== anim.phase) {
      anim.phase = current;
      setLoopActive(current);
      setPhaseUI(current, anim.prediction);
      if (current === "move") {
        anim.ballScale = 1;
        anim.ballAlpha = 1;
        anim.trail = [];
      }
    }

    const geo = tableGeometry();
    const path = anim.prediction.predicted.path.map(([x, y]) =>
      worldToScreen(x, y, geo)
    );

    const finalPoint = path[path.length - 1];

    if (current === "move") {
      const moveT = Math.min(1, (elapsed - phaseStart) / phaseDuration);
      const eased = easeInOutQuad(moveT);
      anim.ballPos = pointAlongPath(path, eased);
      pushTrail(anim.ballPos);

      // fire a ripple as each new bounce point is crossed
      const passed = bouncesPassed(path, eased);
      if (passed > anim.lastBounceIdx) {
        anim.lastBounceIdx = passed;
        anim.bounceRipple = {
          active: true,
          startedAt: performance.now(),
          point: path[passed],
        };
      }
    } else if (current === "reality") {
      anim.ballPos = finalPoint;
      if (anim.prediction.predicted.made) {
        const t = Math.min(1, phaseElapsed("reality") / 600);
        anim.ballScale = lerp(1, 0.2, t);
        anim.ballAlpha = lerp(1, 0, t);
      }
    } else if (current === "surprise" || current === "sharper belief") {
      anim.ballPos = finalPoint;
      if (current === "sharper belief" && !anim.finalSnap) {
        anim.finalSnap = true;
        fireRPE(anim.prediction);
      }
    } else {
      anim.ballPos = path[0];
    }

    const totalDuration = PHASE_PLAN.reduce((a, b) => a + b.duration, 0);
    const isOOD =
      anim.prediction.decision_source === "agent_native_reflex" &&
      !anim.prediction.predicted.made;

    // Masterclass UX: as soon as the ball has visually settled in reality,
    // surface actions immediately (replay / teach next / rack / reset).
    // Waiting for the full 6-step tail felt broken after a visible drop.
    if (!anim.frozen && current === "reality" && phaseElapsed("reality") >= 620) {
      if (isOOD) {
        if (!anim.endHandled) {
          anim.endHandled = true;
          onPredictAnimationDone(anim.prediction);
        }
      } else {
        freezeShot(anim.prediction, finalPoint);
      }
    }

    // Failsafe in case timing ever skips past the reality gate.
    if (elapsed > totalDuration + 600 && !anim.endHandled) {
      anim.endHandled = true;
      onPredictAnimationDone(anim.prediction);
    }
    if (elapsed > totalDuration + 900 && !anim.frozen && !isOOD) {
      freezeShot(anim.prediction, finalPoint);
    }
  }

  function onPredictAnimationDone(result) {
    if (!result || appMode !== "predict") return;
    const isOODMiss =
      result.decision_source === "agent_native_reflex" &&
      result.predicted &&
      !result.predicted.made;
    if (isOODMiss) {
      triggerSanctum(result.pocket_id);
    }
  }

  function tickCinematic() {
    const now = performance.now();
    const dt = now - cine.lastFrameTime;
    cine.lastFrameTime = now;

    if (cine.paused) return;
    if (cine.idx >= cine.episodes.length - 1) {
      // caught up to the stream - wait for more or for completion
      if (!cine.streaming && cine.status === "done" && !cine.completionShown) {
        cine.completionShown = true;
        onCinematicEnd();
      }
      return;
    }

    cine.accumulator += dt;
    while (
      cine.accumulator >= cine.perEpisodeMs &&
      cine.idx < cine.episodes.length - 1
    ) {
      cine.accumulator -= cine.perEpisodeMs;
      advanceEpisode();
    }
  }

  function advanceEpisode() {
    cine.idx += 1;
    const ep = cine.episodes[cine.idx];
    if (!ep) return;

    // Made shots stay vivid orange-gold (Strange's "right future"); misses
    // are dim cyan ghosts (futures that fail). Both linger so the player
    // sees many parallel simulations playing out at once.
    const grooveColor = ep.made
      ? "rgba(255, 188, 90, ALPHA)"
      : "rgba(120, 200, 220, ALPHA)";
    cine.grooves.push({
      path: ep.path,
      alpha: ep.made ? 0.78 : 0.42,
      color: grooveColor,
      made: ep.made,
      born: performance.now(),
    });
    const cap = 160;
    while (cine.grooves.length > cap) cine.grooves.shift();
    // Slow decay so 100+ parallel futures stay visible at once
    for (let i = 0; i < cine.grooves.length; i += 1) {
      cine.grooves[i].alpha *= 0.992;
    }

    // Track recent angles for the cue ghost fan
    cine.recentAngles.push({ angle: ep.angle_deg, born: performance.now() });
    while (cine.recentAngles.length > 8) cine.recentAngles.shift();

    // Launch a "flying ghost shot" - a glowing ball that flies down the
    // path. With episodes firing at ~30 per second and each ball taking
    // ~1.2s to traverse, we get ~30+ balls in flight simultaneously,
    // which reads as "the agent is trying many futures right now".
    if (ep.path && ep.path.length >= 2) {
      cine.particles.push({
        path: ep.path,
        t: 0,
        speed: 0.7 + Math.random() * 0.3, // path-units per second
        life: 1.0,
        made: ep.made,
        color: ep.made
          ? "rgba(255, 220, 140, ALPHA)"
          : "rgba(180, 220, 240, ALPHA)",
      });
    }
    // Cap so we don't grow unbounded under heavy episode rates
    while (cine.particles.length > 80) cine.particles.shift();

    // Pacing arc: FAST when chaotic (early), slow-mo on the lock-in moment.
    // High exploration = many shots per second = the "try/fail/adjust" reel.
    // Low exploration = drag it out so the viewer feels the commitment.
    const explore = ep.exploration_rate ?? 0.5;
    cine.perEpisodeMs = lerp(14, 70, 1 - explore);

    // capture Q-table snapshot for the halo viz
    if (ep.q) {
      cine.q = ep.q;
      cine.bestAngleDeg = ep.best_angle_deg ?? null;
    }

    cine.rewardHistory.push(ep.reward);
    cine.winrateHistory.push(ep.made ? 1 : 0);
    cine.explorationHistory.push(ep.exploration_rate);
    if (cine.rewardHistory.length > 600) cine.rewardHistory.shift();
    if (cine.winrateHistory.length > 600) cine.winrateHistory.shift();
    if (cine.explorationHistory.length > 600) cine.explorationHistory.shift();

    updateCinematicHud(ep);
    updateSparklines();
  }

  function updateCinematicHud(ep) {
    const status = cine.summary;
    const totalTimesteps = status.total_timesteps || 0;
    const epDone = status.episodes_done || ep.episode_idx;
    els.cineStatus.textContent = totalTimesteps
      ? `Episode ${epDone} / ${totalTimesteps}`
      : `Episode ${epDone}`;
    els.cineAngle.textContent = `${ep.angle_deg.toFixed(1)} deg`;
    els.cineReward.textContent = (ep.reward >= 0 ? "+" : "") + ep.reward.toFixed(1);

    // ---- ε bar: visibly shrinks from 100% (chaos) to ~1% (greedy) -----
    const epsPct = Math.max(0, Math.min(100, ep.exploration_rate * 100));
    els.cineExplore.textContent = `${epsPct.toFixed(0)}%`;
    if (els.epsilonFill) {
      els.epsilonFill.style.width = `${epsPct.toFixed(1)}%`;
    }

    // ---- Convergence bar: how confident is the agent in its best angle ----
    // 0% = no positive Q anywhere (still failing), 100% = best Q saturated at
    // the max-reward ceiling (the agent has fully imprinted the demo).
    // We use top_Q normalized to the reward ceiling because, with the
    // ±20deg exploration window, mass concentration would stay low while
    // the agent is in fact converging strongly on the demo bucket.
    let topQ = 0;
    if (cine.q && cine.q.length) {
      for (let i = 0; i < cine.q.length; i += 1) {
        if (cine.q[i] > topQ) topQ = cine.q[i];
      }
    }
    const convergePct = Math.max(0, Math.min(100, topQ));
    if (els.cineConverge) {
      els.cineConverge.textContent = `${convergePct.toFixed(0)}%`;
    }
    if (els.convergeFill) {
      els.convergeFill.style.width = `${convergePct.toFixed(1)}%`;
    }

    // ---- Tries / Makes tally (cumulative across the whole run) -------
    // winrateHistory is capped at 600 so we keep separate counters.
    if (cine.totalTries === undefined) cine.totalTries = 0;
    if (cine.totalMakes === undefined) cine.totalMakes = 0;
    cine.totalTries += 1;
    if (ep.made) cine.totalMakes += 1;
    if (els.cineTries) els.cineTries.textContent = String(cine.totalTries);
    if (els.cineMakes) els.cineMakes.textContent = String(cine.totalMakes);

    const winrate = rollingMean(cine.winrateHistory, 40);
    const rewardMean = rollingMean(cine.rewardHistory, 40);
    els.cineWinrate.textContent = `${(winrate * 100).toFixed(0)}%`;
    els.cineWinrateVal.textContent = `${(winrate * 100).toFixed(0)}%`;
    els.cineRewardVal.textContent =
      (rewardMean >= 0 ? "+" : "") + rewardMean.toFixed(1);

    const firstMake = status.first_make_episode;
    els.cineFirstmake.textContent =
      firstMake !== null && firstMake !== undefined ? `#${firstMake}` : "—";

    // progress bar uses episodes_done / total_timesteps
    if (totalTimesteps && els.cineProgressFill) {
      const pct = Math.min(100, (epDone / totalTimesteps) * 100);
      els.cineProgressFill.style.width = `${pct.toFixed(1)}%`;
    }

    const converged = status.converged_at_episode;
    const target = cine.targetPocket || status.target_pocket || ep.target_pocket;
    const targetName = target ? pocketLabel(target) : "the chosen pocket";

    let phase = `Q[${targetName}] at zero`;
    let caption =
      `Q-row for ${targetName} is empty. Every angle equally plausible \u2014 the model has not been written yet.`;

    if (firstMake !== null && firstMake !== undefined && ep.episode_idx >= firstMake) {
      phase = `First make: ${targetName}`;
      caption = `First pocket at episode ${firstMake}. Prediction matched reality \u2014 dopamine just wrote a new entry.`;
    }
    if (ep.exploration_rate < 0.4) {
      phase = "Exploration cooling";
      caption = `Epsilon decaying. The agent leans on the spokes that paid off. Confidence is becoming embodied.`;
    }
    if (ep.exploration_rate < 0.1 && winrate > 0.5) {
      phase = "Locking in";
      caption = `One spoke pulling away. The shot that used to be a guess is becoming a reflex.`;
    }
    if (converged !== null && converged !== undefined && ep.episode_idx >= converged) {
      phase = `Converged on ${targetName}`;
      const a =
        cine.bestAngleDeg !== null && cine.bestAngleDeg !== undefined
          ? cine.bestAngleDeg
          : ep.angle_deg;
      caption = `\u226595% win rate. The agent\u2019s reflex for ${targetName} is now ${a.toFixed(1)}\u00B0. Expertise compressed into one row.`;
    }

    els.cinePhase.textContent = caption;
    els.cineCaption.textContent = caption;
    // Pipe the training caption through the main narrator too, so the
    // dashboard reads as a single voice instead of two duelling panels.
    if (els.narrator) els.narrator.textContent = caption;
    if (els.phaseTitle) els.phaseTitle.textContent = phase;
  }

  function updateSparklines() {
    drawSparkline(
      els.rewardSpark,
      cine.rewardHistory,
      { minY: -150, maxY: 110, lineColor: "#5be1c0", fillColor: "rgba(91,225,192,0.18)" }
    );
    drawSparkline(
      els.winrateSpark,
      runningWindowed(cine.winrateHistory, 40),
      { minY: 0, maxY: 1, lineColor: "#ffd37b", fillColor: "rgba(255,211,123,0.18)" }
    );
  }

  function drawSparkline(canv, data, { minY, maxY, lineColor, fillColor }) {
    if (!canv || !data.length) return;
    const c = canv.getContext("2d");
    const w = canv.width / dpr;
    const h = canv.height / dpr;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);

    // zero baseline
    const zeroY = h - ((0 - minY) / (maxY - minY)) * h;
    c.strokeStyle = "rgba(255,255,255,0.08)";
    c.lineWidth = 1;
    c.beginPath();
    c.moveTo(0, zeroY);
    c.lineTo(w, zeroY);
    c.stroke();

    // path
    const n = data.length;
    c.beginPath();
    for (let i = 0; i < n; i += 1) {
      const x = (i / (n - 1 || 1)) * w;
      const y = h - ((data[i] - minY) / (maxY - minY)) * h;
      if (i === 0) c.moveTo(x, y);
      else c.lineTo(x, y);
    }
    c.lineTo(w, h);
    c.lineTo(0, h);
    c.closePath();
    c.fillStyle = fillColor;
    c.fill();

    c.beginPath();
    for (let i = 0; i < n; i += 1) {
      const x = (i / (n - 1 || 1)) * w;
      const y = h - ((data[i] - minY) / (maxY - minY)) * h;
      if (i === 0) c.moveTo(x, y);
      else c.lineTo(x, y);
    }
    c.lineWidth = 1.6;
    c.strokeStyle = lineColor;
    c.stroke();

    // current marker
    const lastX = w;
    const lastY = h - ((data[n - 1] - minY) / (maxY - minY)) * h;
    c.beginPath();
    c.arc(lastX - 2, lastY, 2.2, 0, Math.PI * 2);
    c.fillStyle = lineColor;
    c.fill();
  }

  function rollingMean(arr, window) {
    if (!arr.length) return 0;
    const n = arr.length;
    const w = Math.min(window, n);
    let s = 0;
    for (let i = n - w; i < n; i += 1) s += arr[i];
    return s / w;
  }

  function runningWindowed(arr, window) {
    const out = [];
    let s = 0;
    for (let i = 0; i < arr.length; i += 1) {
      s += arr[i];
      if (i >= window) s -= arr[i - window];
      out.push(s / Math.min(i + 1, window));
    }
    return out;
  }

  // ----------------------------------------------------------------------
  // JS port of demo.py's evaluate_angle.
  // Kept structurally identical so manual-aim previews exactly match what
  // the backend would simulate when you actually fire the shot.
  // ----------------------------------------------------------------------

  const POCKET_RADIUS_FACTOR = 1; // cfg.pocket_radius drives it

  function rayToRail(x, y, dx, dy, w, h) {
    const candidates = [];
    if (dx > 1e-12) candidates.push(["right", (w - x) / dx]);
    else if (dx < -1e-12) candidates.push(["left", (0 - x) / dx]);
    if (dy > 1e-12) candidates.push(["top", (h - y) / dy]);
    else if (dy < -1e-12) candidates.push(["bottom", (0 - y) / dy]);
    const filtered = candidates.filter(([, t]) => t > 1e-9);
    if (!filtered.length) return null;
    filtered.sort((a, b) => a[1] - b[1]);
    const [rail, t] = filtered[0];
    return { rail, t, x: x + dx * t, y: y + dy * t };
  }

  function segPointDistance(sx, sy, ex, ey, px, py) {
    const vx = ex - sx, vy = ey - sy;
    const wx = px - sx, wy = py - sy;
    const segLenSq = vx * vx + vy * vy;
    if (segLenSq < 1e-12) return Math.hypot(px - sx, py - sy);
    const t = Math.max(0, Math.min(1, (wx * vx + wy * vy) / segLenSq));
    return Math.hypot(px - (sx + t * vx), py - (sy + t * vy));
  }

  // A pocket is a NOTCH in the cushion. Two complementary checks mirror
  // demo.py:
  //   1) Mid-flight disc crossing (direction-gated) - catches steep
  //      diagonal approaches into a pocket disc from interior.
  //   2) Rail-jaw arrival - a "rail bounce" point that lands inside a
  //      pocket jaw has no cushion to bounce off of, so it drops in.
  const POCKET_MOUTH_MIN_ANGLE_DEG = 6.0;
  const MOUTH_MIN_DIR = Math.sin((POCKET_MOUTH_MIN_ANGLE_DEG * Math.PI) / 180);

  function railJawPocket(rail, hx, hy) {
    if (!cfg) return null;
    const j = cfg.pocket_radius; // jaw half-width = pocket radius
    const W = cfg.table.width;
    const H = cfg.table.height;
    const mid = H / 2;
    if (rail === "left") {
      if (hy <= j) return "BL";
      if (hy >= H - j) return "TL";
      if (Math.abs(hy - mid) <= j) return "LM";
    } else if (rail === "right") {
      if (hy <= j) return "BR";
      if (hy >= H - j) return "TR";
      if (Math.abs(hy - mid) <= j) return "RM";
    } else if (rail === "bottom") {
      if (hx <= j) return "BL";
      if (hx >= W - j) return "BR";
    } else if (rail === "top") {
      if (hx <= j) return "TL";
      if (hx >= W - j) return "TR";
    }
    return null;
  }

  function middlePocketEntryOk(pid, dx) {
    const middles = cfg.middle_pocket_ids || ["LM", "RM"];
    if (!middles.includes(pid)) return true;
    if (pid === "RM") return dx > MOUTH_MIN_DIR;
    if (pid === "LM") return dx < -MOUTH_MIN_DIR;
    return true;
  }

  function cornerPocketEntryOk(pid, dx, dy) {
    const corners = cfg.corner_pocket_ids || ["TL", "TR", "BL", "BR"];
    if (!corners.includes(pid)) return true;
    if (pid === "TL") return dx < -MOUTH_MIN_DIR && dy > MOUTH_MIN_DIR;
    if (pid === "TR") return dx > MOUTH_MIN_DIR && dy > MOUTH_MIN_DIR;
    if (pid === "BL") return dx < -MOUTH_MIN_DIR && dy < -MOUTH_MIN_DIR;
    if (pid === "BR") return dx > MOUTH_MIN_DIR && dy < -MOUTH_MIN_DIR;
    return true;
  }

  function pocketEntryOk(pid, dx, dy) {
    return middlePocketEntryOk(pid, dx) && cornerPocketEntryOk(pid, dx, dy);
  }

  function firstPocketHitOnSegment(sx, sy, ex, ey) {
    const pockets = cfg.pockets;
    const r = cfg.pocket_radius * POCKET_RADIUS_FACTOR;
    const r2 = r * r;
    const vx = ex - sx, vy = ey - sy;
    const segLenSq = vx * vx + vy * vy;
    if (segLenSq < 1e-12) return null;

    const segLen = Math.sqrt(segLenSq);
    const dirx = vx / segLen;
    const diry = vy / segLen;

    let best = null;
    let bestT = Infinity;
    for (const pid in pockets) {
      // Direction gate first - skip pockets the ball isn't heading into.
      // Also stops a rail-bounce that occurs INSIDE a corner pocket disc
      // from being falsely treated as a drop on the outgoing segment.
      if (!pocketEntryOk(pid, dirx, diry)) continue;
      const p = pockets[pid];
      const wx = p.x - sx, wy = p.y - sy;
      const a = segLenSq;
      const b = -2 * (wx * vx + wy * vy);
      const c = wx * wx + wy * wy - r2;
      const disc = b * b - 4 * a * c;
      if (disc < 0) continue;
      const sqd = Math.sqrt(disc);
      let tEnter = (-b - sqd) / (2 * a);
      if (tEnter < 0) tEnter = 0;
      if (tEnter > 1) continue;
      const hx = sx + tEnter * vx;
      const hy = sy + tEnter * vy;
      if (tEnter < bestT) {
        bestT = tEnter;
        best = { pid, t: tEnter, x: hx, y: hy };
      }
    }
    return best;
  }

  // World-coords simulation. angleDeg follows the backend convention
  // (0 = +x, 90 = +y). Returns {made, pocket_hit, path, bounces, final}.
  //
  // In the new UX there's no "target pocket the player picked" - whichever
  // pocket the trajectory terminates in IS the target, full stop. If the
  // trajectory doesn't reach any pocket, pocket_hit is null.
  function localEvaluateAngle(angleDeg, maxBounces = 3) {
    if (!cfg) return null;
    const cue = cfg.fixed_cue;
    const W = cfg.table.width;
    const H = cfg.table.height;

    const rad = (angleDeg * Math.PI) / 180;
    let dx = Math.cos(rad);
    let dy = Math.sin(rad);

    let x = cue.x, y = cue.y;
    const path = [[x, y]];
    const rails = [];

    for (let i = 0; i <= maxBounces; i += 1) {
      const hit = rayToRail(x, y, dx, dy, W, H);
      if (!hit) break;

      // 1) Mid-flight disc crossing (direction-gated).
      let dropPid = null;
      const pocketHit = firstPocketHitOnSegment(x, y, hit.x, hit.y);
      if (pocketHit) dropPid = pocketHit.pid;

      // 2) Rail-jaw arrival - no cushion at the pocket mouth.
      if (!dropPid) dropPid = railJawPocket(hit.rail, hit.x, hit.y);

      if (dropPid) {
        const pCenter = cfg.pockets[dropPid];
        path.push([pCenter.x, pCenter.y]);
        return {
          made: true,
          pocket_hit: dropPid,
          path,
          bounces: rails.length,
          final: [pCenter.x, pCenter.y],
        };
      }

      path.push([hit.x, hit.y]);
      rails.push(hit.rail);
      if (i === maxBounces) break;
      if (hit.rail === "left" || hit.rail === "right") dx = -dx;
      else dy = -dy;
      x = hit.x;
      y = hit.y;
    }

    return {
      made: false,
      pocket_hit: null,
      path,
      bounces: rails.length,
      final: path[path.length - 1],
    };
  }

  // ----------------------------------------------------------------------
  // animation helpers
  // ----------------------------------------------------------------------

  function pushTrail(p) {
    if (!p) return;
    anim.trail.push({ x: p.x, y: p.y });
    if (anim.trail.length > 14) anim.trail.shift();
  }

  // For a variable-length path (cue + N rail bounces + final drop/stop),
  // return the index of the LAST bounce point whose normalized position
  // along the cumulative arc length is <= t. Used to fire ripples once
  // per bounce we pass.
  function bouncesPassed(path, t) {
    if (path.length < 3) return 0;
    const segLens = [];
    let total = 0;
    for (let i = 1; i < path.length; i += 1) {
      const d = dist(path[i - 1], path[i]);
      segLens.push(d);
      total += d;
    }
    const target = t * total;
    let acc = 0;
    let passed = 0;
    // every interior waypoint path[1..length-2] is a bounce point
    for (let i = 0; i < segLens.length - 1; i += 1) {
      acc += segLens[i];
      if (target >= acc - 0.5) passed = i + 1;
    }
    return passed;
  }

  function pointAlongPath(path, t) {
    if (path.length < 2) return path[0];
    const segLens = [];
    let total = 0;
    for (let i = 1; i < path.length; i += 1) {
      const d = dist(path[i - 1], path[i]);
      segLens.push(d);
      total += d;
    }
    if (total < 1e-6) return path[0];
    const target = t * total;
    let acc = 0;
    for (let i = 0; i < segLens.length; i += 1) {
      const next = acc + segLens[i];
      if (target <= next) {
        const k = (target - acc) / Math.max(segLens[i], 1e-6);
        return lerpPoint(path[i], path[i + 1], k);
      }
      acc = next;
    }
    return path[path.length - 1];
  }

  function lerpPoint(a, b, k) {
    return { x: a.x + (b.x - a.x) * k, y: a.y + (b.y - a.y) * k };
  }

  function dist(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function easeInOutQuad(t) {
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  }

  function easeOutQuad(t) {
    return 1 - (1 - t) * (1 - t);
  }

  function phaseElapsed(name) {
    if (!anim.active) return 0;
    const elapsed = performance.now() - anim.startedAt;
    let cursor = 0;
    for (const phase of PHASE_PLAN) {
      if (phase.step === name) return Math.max(0, elapsed - cursor);
      cursor += phase.duration;
    }
    return 0;
  }

  // ----------------------------------------------------------------------
  // narration / phase UI (predict mode)
  // ----------------------------------------------------------------------

  function setLoopActive(step) {
    let foundActive = false;
    els.loopItems.forEach((li) => {
      li.classList.remove("active", "done");
      const s = li.dataset.step;
      if (step && s === step) {
        li.classList.add("active");
        foundActive = true;
      } else if (step && !foundActive) {
        li.classList.add("done");
      }
    });
  }

  function setPhaseUI(step, result) {
    const line = result.narration[step] || "";
    els.phaseTitle.textContent = step;
    els.narrator.textContent = line;
  }

  function fireRPE(result) {
    // RPE now lives inside the post-shot verdict card. The card is hidden
    // during the shot animation, so this only paints when freeze appears.
    if (!els.psRpeFill || !els.psRpeValue) return;
    const reward = result.predicted.reward;
    const isMade = result.predicted.made;
    const magnitude = Math.min(1, Math.abs(reward) / 100);
    els.psRpeFill.style.width = (magnitude * 50).toFixed(1) + "%";
    if (isMade) {
      els.psRpeFill.classList.remove("negative");
      els.psRpeFill.style.transform = "translateX(0)";
    } else {
      els.psRpeFill.classList.add("negative");
      els.psRpeFill.style.transform = "translateX(-100%)";
    }
    els.psRpeValue.textContent = (isMade ? "+" : "") + reward.toFixed(1);
  }

  function updateMetrics(_result) {
    // Last-Shot dashboard block was retired; the verdict reads straight
    // from the result object inside the narrator card via
    // paintVerdictIntoRail. Kept as a hook for any future side-effects.
  }

  // ----------------------------------------------------------------------
  // Dashboard: Agent's Memory grid + Belief histogram + Native reflex
  // ----------------------------------------------------------------------

  // 6 pockets x 4 bank-count columns. Layout is laid out to physically
  // echo the table: corners top/bottom, side-middles in row 2.
  const MEMORY_ROW_ORDER = [
    ["TL", "TR"],
    ["LM", "RM"],
    ["BL", "BR"],
  ];
  const BANK_COLS = [0, 1, 2, 3]; // direct, 1-bank, 2-bank, 3-bank

  function buildMemoryGrid() {
    if (!els.memoryGrid || !cfg) return;
    const grid = els.memoryGrid;
    grid.innerHTML = "";
    // header row
    const header = document.createElement("div");
    header.className = "memory-header";
    header.appendChild(spanCls("memory-header-pocket", "pocket"));
    BANK_COLS.forEach((b) => {
      const label = b === 0 ? "direct" : `${b}-bank`;
      header.appendChild(spanCls("memory-header-bank", label));
    });
    grid.appendChild(header);

    MEMORY_ROW_ORDER.flat().forEach((pid) => {
      const row = document.createElement("div");
      row.className = "memory-row";
      const lbl = spanCls("memory-row-label", pocketLabel(pid));
      lbl.title = pid;
      row.appendChild(lbl);
      BANK_COLS.forEach((b) => {
        const cell = document.createElement("button");
        cell.className = "memory-cell";
        cell.dataset.pocket = pid;
        cell.dataset.bank = String(b);
        cell.type = "button";
        const angleEl = document.createElement("span");
        angleEl.className = "memory-cell-angle";
        angleEl.textContent = "—";
        const visitEl = document.createElement("span");
        visitEl.className = "memory-cell-visits";
        visitEl.textContent = "";
        cell.appendChild(angleEl);
        cell.appendChild(visitEl);
        cell.addEventListener("click", () => {
          // Click a learned cell to ask the agent to fire it. Untrained
          // cell click = same OOD beat as clicking the pocket.
          selectedPocket = pid;
          askAgentToCall(pid);
        });
        row.appendChild(cell);
      });
      grid.appendChild(row);
    });
  }

  function spanCls(cls, text) {
    const s = document.createElement("span");
    s.className = cls;
    s.textContent = text;
    return s;
  }

  function refreshMemoryGrid() {
    if (!els.memoryGrid || !cfg || !cfg.model) return;
    const variants = cfg.model.trained_variants || {};
    const angles = cfg.model.trained_variant_angles || {};
    const qRows = cfg.model.q_rows || {};
    const visits = cfg.model.visits || {};
    const nAngles = cfg.model.n_angles || 180;
    const aimedVid = aim.targetPocket && aim.previewSim
      ? variantId(aim.targetPocket, aim.previewSim.bounces)
      : null;

    // The variant the cinematic is currently training. We pulse its cell
    // so the user can SEE which row of the Q-table is being written right
    // now, mirroring the on-felt mandala/portal that's painting the table.
    const trainingVid =
      appMode === "cinematic" && cine.targetPocket
        ? variantId(cine.targetPocket, cine.targetBounces || 0)
        : null;

    let trainedCount = 0;
    const cells = els.memoryGrid.querySelectorAll(".memory-cell");
    cells.forEach((cell) => {
      const pid = cell.dataset.pocket;
      const b = Number(cell.dataset.bank);
      const vid = `${pid}:b${b}`;
      const isTrained = (variants[pid] || []).includes(b);
      const isAim = aimedVid === vid;
      const isTraining = trainingVid === vid;
      cell.classList.toggle("trained", isTrained);
      cell.classList.toggle("aiming", isAim);
      cell.classList.toggle("training-now", isTraining);
      const angleEl = cell.querySelector(".memory-cell-angle");
      const visitEl = cell.querySelector(".memory-cell-visits");
      if (isTrained) {
        trainedCount += 1;
        const ang = angles[pid] && angles[pid][b] != null ? angles[pid][b] : 0;
        angleEl.textContent = `${Math.round(ang)}°`;
        // confidence color: peak Q over total visits gives "how peaked" the row is
        const q = qRows[vid] || [];
        const peak = q.length ? Math.max(...q) : 0;
        const mean = q.length ? q.reduce((a, c) => a + c, 0) / q.length : 0;
        const sharpness = Math.max(0, Math.min(1, (peak - mean) / 60));
        cell.style.setProperty("--confidence", sharpness.toFixed(2));
        const v = visits[vid] || [];
        const totalV = v.reduce((a, c) => a + c, 0);
        visitEl.textContent = `${totalV}v`;
      } else {
        angleEl.textContent = "—";
        visitEl.textContent = "";
        cell.style.removeProperty("--confidence");
      }
    });

    if (els.memoryCount) {
      els.memoryCount.textContent = `${trainedCount}/${24} variants`;
    }

    // Native reflex - the first-trained variant gets a gold ring on
    // its cell inside the Q-table heatmap. No more separate block.
    const nativeVid = cfg.model.native_variant;
    cells.forEach((cell) => {
      const pid = cell.dataset.pocket;
      const b = Number(cell.dataset.bank);
      const vid = `${pid}:b${b}`;
      cell.classList.toggle("native", vid === nativeVid);
    });
  }

  function variantId(pid, bounces) {
    return `${pid}:b${bounces}`;
  }

  // Aggregate learning telemetry across the whole Q-table. The Q-grid
  // above shows WHERE the agent has learned; this block shows the
  // SHAPE of that learning - how many spikes, which row is sharpest,
  // average confidence, and the native reflex it falls back to when
  // it has never seen the shot.
  function refreshLearningTelemetry() {
    if (!cfg || !cfg.model) return;
    const variants = cfg.model.trained_variants || {};
    const angles = cfg.model.trained_variant_angles || {};
    const qRows = cfg.model.q_rows || {};
    const visits = cfg.model.visits || {};
    const nativeVid = cfg.model.native_variant;

    // Total trained variants out of the full 6 pockets x 4 bank-counts
    // = 24 grid. Coverage is the headline progress metric.
    let trainedCount = 0;
    let totalSpikes = 0;
    let sharpestVid = null;
    let sharpestSharpness = -1;
    let sharpestAngle = null;
    let confidenceSum = 0;
    let confidenceCount = 0;

    Object.keys(variants).forEach((pid) => {
      (variants[pid] || []).forEach((b) => {
        trainedCount += 1;
        const vid = `${pid}:b${b}`;
        const q = qRows[vid] || [];
        const v = visits[vid] || [];
        const vTotal = v.reduce((acc, c) => acc + c, 0);
        totalSpikes += vTotal;
        if (q.length) {
          const peak = Math.max(...q);
          const mean = q.reduce((a, c) => a + c, 0) / q.length;
          // sharpness = how far the peak juts above the mean of the row,
          // normalized into a 0..1 confidence proxy.
          const sharpness = Math.max(0, Math.min(1, (peak - mean) / 60));
          confidenceSum += sharpness;
          confidenceCount += 1;
          if (sharpness > sharpestSharpness) {
            sharpestSharpness = sharpness;
            sharpestVid = vid;
            sharpestAngle =
              angles[pid] && angles[pid][b] != null ? angles[pid][b] : null;
          }
        }
      });
    });

    const total = 24;
    const pct = total > 0 ? (trainedCount / total) * 100 : 0;
    const pctRounded = Math.round(pct);

    if (els.learnCoveragePill) {
      els.learnCoveragePill.textContent = `${pctRounded}%`;
    }
    if (els.learnCoverageVal) {
      els.learnCoverageVal.textContent = `${trainedCount} / ${total}`;
    }
    if (els.learnCoverageFill) {
      els.learnCoverageFill.style.width = `${pct.toFixed(1)}%`;
      els.learnCoverageFill.classList.toggle("advanced", trainedCount >= 12);
    }
    if (els.learnSpikes) {
      els.learnSpikes.textContent = totalSpikes
        ? totalSpikes.toLocaleString()
        : "0";
      els.learnSpikes.classList.toggle("gold", totalSpikes > 0);
    }
    if (els.learnSharpest) {
      if (sharpestVid) {
        const [spid, sbStr] = sharpestVid.split(":b");
        const sb = Number(sbStr);
        const bankBadge = sb === 0 ? "direct" : `${sb}-bank`;
        const ang =
          sharpestAngle !== null && sharpestAngle !== undefined
            ? `${Math.round(sharpestAngle)}\u00B0`
            : "\u2014";
        els.learnSharpest.textContent = `${pocketLabel(spid)} \u00B7 ${bankBadge} \u00B7 ${ang}`;
        els.learnSharpest.classList.add("accent");
      } else {
        els.learnSharpest.textContent = "\u2014";
        els.learnSharpest.classList.remove("accent");
      }
    }
    if (els.learnNative) {
      // native_variant may be stored either as a full variant id
      // ("TR:b0") or just the pocket id ("TR") depending on which
      // model snapshot we loaded. Handle both.
      let npid = null;
      let nb = null;
      if (nativeVid) {
        if (nativeVid.includes(":b")) {
          const parts = nativeVid.split(":b");
          npid = parts[0];
          nb = Number(parts[1]);
        } else {
          npid = nativeVid;
          // pick the lowest-bounces variant that's trained for this pocket
          const trainedBs = variants[npid] || [];
          nb = trainedBs.length ? Math.min(...trainedBs) : null;
        }
      }
      if (npid) {
        const bankBadge =
          nb === null || nb === undefined
            ? "any"
            : nb === 0
            ? "direct"
            : `${nb}-bank`;
        const nAng =
          nb !== null && nb !== undefined && angles[npid] && angles[npid][nb] != null
            ? `${Math.round(angles[npid][nb])}\u00B0`
            : "\u2014";
        els.learnNative.textContent = `${pocketLabel(npid)} \u00B7 ${bankBadge} \u00B7 ${nAng}`;
        els.learnNative.classList.add("gold");
      } else {
        els.learnNative.textContent = "none yet";
        els.learnNative.classList.remove("gold");
      }
    }
    if (els.learnConfidence) {
      if (confidenceCount > 0) {
        const avgPct = Math.round((confidenceSum / confidenceCount) * 100);
        els.learnConfidence.textContent = `${avgPct}%`;
        els.learnConfidence.classList.toggle("accent", avgPct >= 60);
        els.learnConfidence.classList.toggle("gold", avgPct >= 30 && avgPct < 60);
      } else {
        els.learnConfidence.textContent = "\u2014";
        els.learnConfidence.classList.remove("gold", "accent");
      }
    }
  }

  // Belief histogram: a horizontal angular bar chart of Q-values for the
  // currently-aimed variant. Updates LIVE during the cinematic via the
  // streaming Q-row in cine.q.
  function drawBeliefHistogram() {
    const canvas = els.beliefHistogram;
    if (!canvas || !cfg) return;
    // make sure canvas pixel-size matches its CSS size
    const dpr = window.devicePixelRatio || 1;
    const cssW = canvas.clientWidth || 320;
    const cssH = canvas.clientHeight || 120;
    if (canvas.width !== Math.floor(cssW * dpr)) {
      canvas.width = Math.floor(cssW * dpr);
      canvas.height = Math.floor(cssH * dpr);
    }
    const c = canvas.getContext("2d");
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, cssW, cssH);

    const nAngles = (cfg.model && cfg.model.n_angles) || 180;
    let q = null;
    let visits = null;
    let vid = null;

    // During the cinematic the streamed Q-row is the freshest source.
    if (appMode === "cinematic" && cine.q && cine.q.length === nAngles) {
      q = cine.q;
      vid = variantId(
        cine.targetPocket || "TR",
        cine.targetBounces || 0
      );
    } else if (aim.targetPocket && aim.previewSim) {
      vid = variantId(aim.targetPocket, aim.previewSim.bounces);
      q = (cfg.model && cfg.model.q_rows && cfg.model.q_rows[vid]) || null;
      visits = (cfg.model && cfg.model.visits && cfg.model.visits[vid]) || null;
    }

    // pill + caption
    if (els.beliefVariant) {
      if (vid) {
        const [pid, b] = vid.split(":b");
        const banks = Number(b);
        const bankBadge = banks === 0 ? "direct" : `${banks}-bank`;
        els.beliefVariant.textContent = `${pocketLabel(pid)} · ${bankBadge}`;
      } else {
        els.beliefVariant.textContent = "aim at a pocket";
      }
    }
    if (els.beliefStatus) {
      const isTrained = vid &&
        cfg.model &&
        cfg.model.q_rows &&
        cfg.model.q_rows[vid] &&
        cfg.model.q_rows[vid].some((x) => x !== 0);
      els.beliefStatus.textContent = isTrained ? "yes" : "no";
      els.beliefStatus.classList.toggle("status-yes", !!isTrained);
    }
    if (els.beliefBest) {
      if (q && q.length) {
        let bestI = 0, bestV = q[0];
        for (let i = 1; i < q.length; i += 1) if (q[i] > bestV) { bestV = q[i]; bestI = i; }
        const bestAngle = bestI * (360 / nAngles);
        els.beliefBest.textContent = `${bestAngle.toFixed(0)}° (Q=${bestV.toFixed(1)})`;
      } else {
        els.beliefBest.textContent = "—";
      }
    }
    if (els.beliefAim) {
      els.beliefAim.textContent = aim.previewSim
        ? `${aim.angleDeg.toFixed(0)}°`
        : "—";
    }
    if (els.beliefVisits) {
      const total = visits ? visits.reduce((a, c) => a + c, 0) : 0;
      els.beliefVisits.textContent = total ? `${total}` : "—";
    }
    if (els.beliefCaption) {
      if (!q || !q.some((x) => x !== 0)) {
        els.beliefCaption.innerHTML =
          "Untrained. Every angle equally plausible. <em>The model hasn\u2019t been written yet.</em>";
      } else if (appMode === "cinematic") {
        els.beliefCaption.innerHTML =
          "Watch one spike emerge. <em>The dopamine system is writing Q-values right now.</em>";
      } else {
        els.beliefCaption.innerHTML =
          "A spike formed at one angle. <em>The agent is no longer deciding \u2014 it is recognizing.</em>";
      }
    }

    // axes
    const padL = 28;
    const padR = 8;
    const padT = 10;
    const padB = 18;
    const W = cssW - padL - padR;
    const H = cssH - padT - padB;

    // background
    c.fillStyle = "rgba(20, 26, 36, 0.45)";
    c.fillRect(padL, padT, W, H);

    // grid lines at 0/90/180/270
    c.strokeStyle = "rgba(255,255,255,0.06)";
    c.lineWidth = 1;
    for (const deg of [0, 90, 180, 270]) {
      const x = padL + (deg / 360) * W;
      c.beginPath();
      c.moveTo(x, padT);
      c.lineTo(x, padT + H);
      c.stroke();
    }

    if (q && q.length) {
      // Find the peak to scale against, but never scale negative values as positive bars.
      let qMax = 1;
      for (let i = 0; i < q.length; i += 1) {
        if (q[i] > qMax) qMax = q[i];
      }
      const barW = W / q.length;

      // During the cinematic, track previous Q values so we can flash
      // bars whose Q just increased (the "writing" sparkle that ties this
      // chart to the on-felt update animation).
      const isTraining = appMode === "cinematic";
      const qPrev = isTraining && cine.qPrev && cine.qPrev.length === q.length
        ? cine.qPrev
        : null;
      const sparks = []; // {x, y, intensity}

      // Palette: cyan when at rest, amber/gold during training to mirror
      // the Dr Strange spell light on the felt.
      const baseRGB = isTraining ? "255, 188, 90" : "91, 225, 192";
      const peakRGB = isTraining ? "255, 230, 160" : "91, 225, 192";

      for (let i = 0; i < q.length; i += 1) {
        const deg = i * (360 / q.length);
        const x = padL + (deg / 360) * W;
        const v = q[i];

        // Only draw bars for positive Q values (learned makes).
        // Unvisited (0) and negative (misses) remain flat at the baseline.
        const norm = v > 0 ? v / qMax : 0;
        const h = norm * H;
        const yTop = padT + H - h;
        const isPeak = v > 0 && v === qMax;
        c.fillStyle = isPeak
          ? `rgba(${peakRGB}, 0.95)`
          : v > 0
          ? `rgba(${baseRGB}, ${0.25 + 0.55 * norm})`
          : "rgba(255, 107, 107, 0.4)";

        if (v < 0) {
          c.fillStyle = "rgba(255, 107, 107, 0.4)";
          c.fillRect(x, padT + H, Math.max(0.6, barW - 0.2), 2);
        } else {
          c.fillRect(x, yTop, Math.max(0.6, barW - 0.2), h);
        }

        // Flag updated bars for sparkle pass below
        if (qPrev && v > qPrev[i] + 0.5) {
          sparks.push({
            x: x + barW / 2,
            y: yTop,
            intensity: Math.min(1, (v - qPrev[i]) / 20),
          });
        }
      }

      // Sparkle pass: glowing dot + halo above each bar that grew this
      // frame. Reads as "this Q-value just got written to right now".
      if (sparks.length) {
        c.save();
        for (const sp of sparks) {
          const r = 2.5 + sp.intensity * 3;
          c.shadowColor = "rgba(255, 220, 140, 0.9)";
          c.shadowBlur = 10;
          c.fillStyle = `rgba(255, 240, 200, ${0.55 + sp.intensity * 0.45})`;
          c.beginPath();
          c.arc(sp.x, Math.max(padT, sp.y - 4), r, 0, Math.PI * 2);
          c.fill();
        }
        c.restore();
      }

      // Snapshot current Q for next-frame delta detection
      if (isTraining) {
        cine.qPrev = q.slice();
      }
    } else {
      c.fillStyle = "rgba(140, 154, 174, 0.6)";
      c.font = '11px "JetBrains Mono", monospace';
      c.textAlign = "center";
      c.textBaseline = "middle";
      c.fillText("Q-row is all zeros · untrained", padL + W / 2, padT + H / 2);
    }

    // user-aim marker
    if (aim.previewSim) {
      const aimX = padL + ((aim.angleDeg + 360) % 360) / 360 * W;
      c.strokeStyle = "rgba(255, 211, 123, 0.9)";
      c.lineWidth = 1.5;
      c.setLineDash([3, 3]);
      c.beginPath();
      c.moveTo(aimX, padT);
      c.lineTo(aimX, padT + H);
      c.stroke();
      c.setLineDash([]);
      c.fillStyle = "rgba(255, 211, 123, 0.95)";
      c.font = '9px "JetBrains Mono", monospace';
      c.textAlign = "center";
      c.textBaseline = "bottom";
      c.fillText("YOU", aimX, padT - 1);
    }

    // x-axis labels
    c.fillStyle = "rgba(140, 154, 174, 0.75)";
    c.font = '9px "JetBrains Mono", monospace';
    c.textAlign = "center";
    c.textBaseline = "top";
    for (const deg of [0, 90, 180, 270]) {
      const x = padL + (deg / 360) * W;
      c.fillText(`${deg}°`, x, padT + H + 3);
    }
    // y-axis label
    c.save();
    c.translate(8, padT + H / 2);
    c.rotate(-Math.PI / 2);
    c.fillText("Q", 0, 0);
    c.restore();
  }

  function refreshDashboard() {
    refreshMemoryGrid();
    refreshLearningTelemetry();
    drawBeliefHistogram();
    refreshAimReadout();
  }

  // Keep the right-panel "Aim Readout" in sync with the current aim.
  // CUE diamond coord is static (the cue is locked). TARGET / ANGLE /
  // BANKS update live as the mouse moves into a pocket lane. During
  // a frozen post-shot or the cinematic, we show locked values.
  function refreshAimReadout() {
    if (!els.aimCue) return;

    // CUE coord - locked, but shown live so users can see the convention.
    if (cfg && cfg.fixed_cue) {
      const cueDiamond = cfg.fixed_cue.y / 25;
      els.aimCue.textContent = `D${cueDiamond.toFixed(1)}`;
    }

    let target = null, angleDeg = null, bounces = null;
    let isLocked = false;

    if (appMode === "cinematic" && cine.targetPocket) {
      target = cine.targetPocket;
      angleDeg = cine.anchorAngleDeg;
      bounces = cine.targetBounces;
      isLocked = true;
    } else if (anim.frozen && anim.freezeResult) {
      target = anim.freezeResult.pocket_id || anim.freezeResult.predicted?.pocket_id;
      angleDeg = anim.freezeResult.angle_deg;
      bounces = anim.freezeResult.predicted?.bounces;
      isLocked = true;
    } else if (aim.targetPocket && aim.previewSim) {
      target = aim.targetPocket;
      angleDeg = aim.angleDeg;
      bounces = aim.previewSim.bounces;
    }

    if (els.aimTarget) {
      els.aimTarget.textContent = target ? pocketLabel(target) : "—";
      els.aimTarget.classList.toggle("active", !!target && !isLocked);
      els.aimTarget.classList.toggle("locked", isLocked);
    }
    if (els.aimAngle) {
      els.aimAngle.textContent =
        angleDeg !== null && angleDeg !== undefined
          ? `${Number(angleDeg).toFixed(0)}°`
          : "—";
      els.aimAngle.classList.toggle("active", angleDeg !== null && !isLocked);
      els.aimAngle.classList.toggle("locked", isLocked);
    }
    if (els.aimBanks) {
      els.aimBanks.textContent =
        bounces !== null && bounces !== undefined
          ? (bounces === 0 ? "direct" : `${bounces}`)
          : "—";
      els.aimBanks.classList.toggle("active", bounces !== null && !isLocked);
      els.aimBanks.classList.toggle("locked", isLocked);
    }
  }

  function sourceLabel(src) {
    switch (src) {
      case "agent_trained":
        return "Agent (trained on this pocket)";
      case "agent_native_reflex":
        return "Agent (native reflex)";
      case "agent":
        return "Agent (Q-table)";
      case "oracle":
        return "Geometric oracle";
      case "oracle_fallback":
        return "Oracle (no model)";
      default:
        return src;
    }
  }

  function pocketLabel(id) {
    if (cfg && cfg.pockets && cfg.pockets[id]) return cfg.pockets[id].label;
    return id;
  }

  // ----------------------------------------------------------------------
  // interaction (predict mode)
  // ----------------------------------------------------------------------

  function pickPocketAt(mx, my) {
    if (appMode !== "predict") return null;
    const geo = tableGeometry();
    const positions = pocketScreenPositions(geo);
    let best = null;
    let bestD = Infinity;
    for (const [id, p] of Object.entries(positions)) {
      const d = Math.hypot(mx - p.x, my - p.y);
      if (d < 28 && d < bestD) {
        bestD = d;
        best = id;
      }
    }
    return best;
  }

  function mouseToAngleFromCue(mx, my) {
    const geo = tableGeometry();
    const cueScreen = worldToScreen(cfg.fixed_cue.x, cfg.fixed_cue.y, geo);
    const dxs = mx - cueScreen.x;
    const dys = my - cueScreen.y;
    // screen y is inverted vs world y; flip
    const ang = Math.atan2(-dys, dxs) * (180 / Math.PI);
    return (ang + 360) % 360;
  }

  function refreshAim() {
    if (!cfg) {
      aim.previewSim = null;
      aim.targetPocket = null;
      refreshAimReadout();
      return;
    }
    const sim = localEvaluateAngle(aim.angleDeg, 3);
    aim.previewSim = sim;
    aim.targetPocket = sim && sim.pocket_hit ? sim.pocket_hit : null;
    if (aim.targetPocket) selectedPocket = aim.targetPocket;
    updateTeachButton();
    updateAimHint();
    refreshMemoryGrid();
    refreshAimReadout();
  }

  function handlePointerMove(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    // Always coaching mode. While the agent is idle, the cue tracks your
    // mouse and the trajectory previews live. While frozen on a result,
    // aim tracking pauses so the painted result stays put.
    if (!busy && !anim.active && !anim.frozen && appMode === "predict") {
      aim.angleDeg = mouseToAngleFromCue(mx, my);
      refreshAim();
    }
    const id = pickPocketAt(mx, my);
    hoverPocket = id;

    if (anim.frozen) {
      // While frozen, the gesture grammar collapsed to two regions:
      //   - over the painted path -> replay (pointer)
      //   - anywhere else         -> rack   (crosshair)
      // Pockets specifically read as "rack and re-aim" in this state so
      // the result the user just watched doesn't get scrubbed by an
      // accidental call-shot.
      const overPath = isOverPredictionPath(mx, my);
      anim.hoverPath = overPath;
      canvas.style.cursor = overPath ? "pointer" : "crosshair";
    } else if (id) {
      anim.hoverPath = false;
      canvas.style.cursor = "pointer";
    } else {
      anim.hoverPath = false;
      canvas.style.cursor = aim.targetPocket ? "crosshair" : "default";
    }
  }

  function handleClick(e) {
    if (busy || appMode !== "predict") return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (anim.frozen) {
      // Frozen-state gesture grammar:
      //   click pocket     -> ask agent to call that pocket from memory
      //   click trajectory -> replay the shot animation
      //   click felt       -> rack the cue, return to aim mode
      // Simplified frozen-state grammar: only replay or rack. Clicking
      // a pocket from a frozen state was previously a "test-recall"
      // shortcut, but it felt odd next to the painted shot result. Now
      // any click that isn't on the trajectory just racks the table
      // and returns to aim mode - where calling a pocket reads
      // naturally.
      if (isOverPredictionPath(mx, my)) {
        replayLastShot();
        return;
      }
      gracefulRack({ pulse: true, toast: false });
      return;
    }

    // Active aim mode:
    //   1) Click a pocket -> "Agent, call this from memory."
    //   2) Click the felt while a trajectory is locked on a pocket
    //      -> open the cinematic to teach this variant.
    const id = pickPocketAt(mx, my);
    if (id) {
      selectedPocket = id;
      askAgentToCall(id);
      return;
    }
    aim.angleDeg = mouseToAngleFromCue(mx, my);
    refreshAim();
    if (!aim.targetPocket) {
      els.hint.textContent =
        "Aim into a pocket. The trajectory has to reach a hole to teach a shot.";
      return;
    }
    enterCinematic();
  }

  // Per-pixel hit test against the painted trajectory line. Tolerance
  // is ~10 screen pixels perpendicular from any segment of the path.
  function isOverPredictionPath(mx, my) {
    if (!anim.frozen || !anim.prediction || !anim.prediction.predicted) return false;
    const path = anim.prediction.predicted.path;
    if (!path || path.length < 2) return false;
    const geo = tableGeometry();
    const pts = path.map(([x, y]) => worldToScreen(x, y, geo));
    const tolerance = 10;
    for (let i = 1; i < pts.length; i += 1) {
      const d = pointToSegmentDistance(mx, my, pts[i - 1].x, pts[i - 1].y, pts[i].x, pts[i].y);
      if (d <= tolerance) return true;
    }
    return false;
  }

  function pointToSegmentDistance(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1;
    const dy = y2 - y1;
    const lenSq = dx * dx + dy * dy;
    if (lenSq === 0) return Math.hypot(px - x1, py - y1);
    let t = ((px - x1) * dx + (py - y1) * dy) / lenSq;
    t = Math.max(0, Math.min(1, t));
    const cx = x1 + t * dx;
    const cy = y1 + t * dy;
    return Math.hypot(px - cx, py - cy);
  }

  async function askAgentToCall(pocketId) {
    if (!cfg) return;
    busy = true;
    els.hint.textContent = `Agent calling ${pocketLabel(pocketId)}...`;
    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, pocket_id: pocketId, mode: "agent" }),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      lastResult = data;
      updateMetrics(data);
      startShotAnimation(data);
      els.hint.textContent =
        "Aim a new shot to teach, or click another pocket to test the agent.";
    } catch (err) {
      els.narrator.textContent = `Prediction failed: ${err}`;
    } finally {
      busy = false;
    }
  }

  function updateTeachButton() {
    if (!els.watchBtn) return;
    if (appMode !== "predict") {
      els.watchBtn.disabled = true;
      return;
    }
    // Coming back from a cinematic - shed the training/committed vibe.
    els.watchBtn.classList.remove("training");
    if (!aim.targetPocket || !aim.previewSim) {
      els.watchBtn.disabled = true;
      els.watchBtn.classList.remove("committed");
      els.watchBtn.textContent = "Teach this shot";
      els.watchBtn.title =
        "Aim into a pocket. The trajectory has to reach a hole to teach a shot.";
      return;
    }
    const name = pocketLabel(aim.targetPocket);
    const banks = aim.previewSim.bounces;
    const bankBadge = banks === 0 ? "direct" : `${banks}-bank`;
    els.watchBtn.disabled = false;
    els.watchBtn.classList.remove("committed");
    els.watchBtn.textContent = `Teach: ${name} (${bankBadge})`;
    els.watchBtn.title = `Train the agent's Q-row for ${name} via ${bankBadge}.`;
  }

  function updateAimHint() {
    if (!els.hint) return;
    if (appMode !== "predict") return;
    if (aim.targetPocket) {
      const name = pocketLabel(aim.targetPocket);
      const bounces = aim.previewSim ? aim.previewSim.bounces : 0;
      const badge = bounces === 0 ? "direct" : `${bounces}-bank`;
      const isVariantKnown = isVariantTrained(aim.targetPocket, bounces);
      const suffix = isVariantKnown
        ? " — agent already knows this exact variant"
        : isPocketTrained(aim.targetPocket)
        ? " — agent knows a different bank count for this pocket"
        : "";
      els.hint.textContent = `${name} locked — ${badge}. Click "Teach this shot" to imprint it${suffix}.`;
    } else {
      els.hint.textContent =
        "Move the cue to aim. Trajectory picks the pocket. Click any pocket to test the agent's memory.";
    }
  }

  function isVariantTrained(pocketId, bounces) {
    if (!cfg || !cfg.model) return false;
    const arr = (cfg.model.trained_variants || {})[pocketId] || [];
    return arr.includes(bounces);
  }

  function startShotAnimation(result) {
    // Clear any prior freeze so the new shot can run cleanly
    if (anim.frozen) {
      tableWrap.classList.remove("frozen");
      hideFrozenTips();
      clearVerdictFromRail();
      if (els.narratorBlock) els.narratorBlock.classList.remove("made", "miss", "ood");
    }
    anim.frozen = false;
    anim.freezeResult = null;
    anim.active = true;
    anim.startedAt = performance.now();
    anim.phase = PHASE_PLAN[0].step;
    anim.prediction = result;
    anim.ballPos = null;
    anim.ballScale = 1;
    anim.ballAlpha = 1;
    anim.trail = [];
    anim.bounceRipple = { active: false, startedAt: 0 };
    anim.finalSnap = false;
    anim.endHandled = false;
    anim.lastBounceIdx = 0;
    setLoopActive(anim.phase);
    setPhaseUI(anim.phase, result);
  }

  // --- Sanctum (Dr Strange) scenario -----------------------------------

  async function triggerSanctum(pocketId) {
    if (sanctum.active) return;
    sanctum.active = true;
    sanctum.targetPocket = pocketId;
    sanctum.episodesSeen = 0;
    sanctum.plannedTimesteps = 1200;
    busy = true;
    appMode = "sanctum";
    tableWrap.classList.add("sanctum-mode");
    els.sanctum.classList.remove("hidden");
    els.sanctum.setAttribute("aria-hidden", "false");

    // Single, static line for the sanctum beat. No rotation - the
    // visual sigil + episode counter carry the moment.
    if (els.sanctumQuote) els.sanctumQuote.textContent = SANCTUM_LINE;

    if (els.hint) {
      els.hint.textContent = `The agent never trained ${pocketLabel(
        pocketId
      )}. Running the simulations forward...`;
    }

    try {
      await fetch("/train/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          target_pocket: pocketId,
          target_bounces: 0, // sanctum learns whatever works fastest
          timesteps: sanctum.plannedTimesteps,
          seed: (Date.now() & 0xffff) | 1,
          pace_ms: 2,
          reset_q: true,
        }),
      });
    } catch (err) {
      console.error("sanctum: failed to kick off training", err);
    }

    await runSanctumWebsocket();
    await sleep(420);
    // Sanctum trains the "any bounce count" variant (b0), so the
    // chained shot should pull from that exact variant.
    await chainTrainedShot(pocketId, 0);
    exitSanctum();
  }

  function runSanctumWebsocket() {
    return new Promise((resolve) => {
      const proto = window.location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${window.location.host}/ws/train/${sessionId}`);
      sanctum.ws = ws;
      sanctum.completionResolver = resolve;

      ws.onmessage = (event) => {
        let msg;
        try {
          msg = JSON.parse(event.data);
        } catch (_) {
          return;
        }
        if (msg.type === "episode") {
          sanctum.episodesSeen = msg.payload.episode_idx;
          updateSanctumCounter();
        } else if (msg.type === "status") {
          if (
            typeof msg.payload.episodes_done === "number" &&
            msg.payload.episodes_done > sanctum.episodesSeen
          ) {
            sanctum.episodesSeen = msg.payload.episodes_done;
            updateSanctumCounter();
          }
          if (msg.payload.status === "error") {
            console.error("sanctum: training error", msg.payload.error);
            finishSanctumStream();
          }
        } else if (msg.type === "complete") {
          if (msg.payload && msg.payload.model && cfg) {
            cfg.model = {
              ...(cfg.model || {}),
              ...msg.payload.model,
              loaded: msg.payload.model.trained_pockets.length > 0,
            };
          }
          finishSanctumStream();
        }
      };
      ws.onclose = finishSanctumStream;
      ws.onerror = finishSanctumStream;
    });
  }

  function finishSanctumStream() {
    if (sanctum.completionResolver) {
      const r = sanctum.completionResolver;
      sanctum.completionResolver = null;
      r();
    }
    if (sanctum.ws) {
      try {
        sanctum.ws.close();
      } catch (_) {}
      sanctum.ws = null;
    }
  }

  function updateSanctumCounter() {
    if (!els.sanctumCounter) return;
    const n = sanctum.episodesSeen;
    const total = sanctum.plannedTimesteps;
    els.sanctumCounter.textContent = `FUTURE ${n.toLocaleString()} / ${total.toLocaleString()}`;
  }

  async function chainTrainedShot(pocketId, targetBounces) {
    try {
      const body = { session_id: sessionId, pocket_id: pocketId, mode: "agent" };
      // Pin the exact variant the coach just taught so the auto-fire
      // demonstrates THAT shot, not a different bank count the agent
      // happens to remember from an earlier session.
      if (targetBounces !== undefined && targetBounces !== null) {
        body.target_bounces = targetBounces;
      }
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      lastResult = data;
      updateMetrics(data);

      // Mark this shot as the auto-fire from training so the freeze
      // step can throw a celebration overlay + delay the "teach another"
      // prompts until the user has SEEN the success.
      data._chainedFromTraining = true;
      startShotAnimation(data);

      // Hint area now reinforces "WATCH" during the auto-fire instead of
      // immediately jumping to "teach another pocket". The post-freeze
      // logic (paintVerdictIntoRail) will update the hint after the
      // shot resolves and the celebration plays.
      if (els.hint) {
        els.hint.textContent = data.predicted.made
          ? "Watch the agent fire the exact shot you taught it."
          : "Watching the agent's best attempt at the angle it found.";
      }
    } catch (err) {
      console.error("sanctum: chained predict failed", err);
    }
  }

  function exitSanctum() {
    if (!sanctum.active) return;
    sanctum.active = false;
    appMode = "predict";
    tableWrap.classList.remove("sanctum-mode");
    els.sanctum.classList.add("hidden");
    els.sanctum.setAttribute("aria-hidden", "true");
    busy = false;
    // refresh cfg so trained badges + dashboard reflect the new state
    refreshConfigButton();
  }

  function sleep(ms) {
    return new Promise((r) => window.setTimeout(r, ms));
  }

  // ----------------------------------------------------------------------
  // post-shot freeze: the moment lands, and the user picks what's next
  // ----------------------------------------------------------------------

  // Post-shot is no longer a banner. We freeze the table and paint the
  // verdict INTO the narrator block on the right rail. The table-wrap
  // gets a .frozen class so the cursor + tip overlays light up.
  function freezeShot(result, finalPoint) {
    anim.frozen = true;
    anim.freezeResult = result;
    anim.active = false;
    if (result.predicted && result.predicted.made) {
      anim.ballPos = finalPoint;
      anim.ballScale = 0;
      anim.ballAlpha = 0;
    } else {
      anim.ballPos = finalPoint;
      anim.ballScale = 1;
      anim.ballAlpha = 1;
    }
    tableWrap.classList.add("frozen");
    paintVerdictIntoRail(result);
    showFrozenTipsOnce(finalPoint);
  }

  function thawShot() {
    anim.frozen = false;
    anim.freezeResult = null;
    anim.ballPos = null;
    anim.ballScale = 1;
    anim.ballAlpha = 1;
    anim.trail = [];
    anim.bounceRipple = { active: false, startedAt: 0 };
    anim.hoverPath = false;
    tableWrap.classList.remove("frozen");
    hideFrozenTips();
    clearVerdictFromRail();
  }

  // Write the verdict directly into the narrator card on the right
  // rail. Same DOM nodes whether aiming or frozen, so the layout never
  // reflows - just the content + a class flag swap on .narrator-block.
  function paintVerdictIntoRail(result) {
    const pred = result.predicted || {};
    const pocketId = result.pocket_id || aim.targetPocket;
    const pocketName = pocketLabel(pocketId);
    const banks = pred.bounces ?? 0;
    const bankBadge = banks === 0 ? "direct" : `${banks}-bank`;
    const angle = typeof result.angle_deg === "number"
      ? `${result.angle_deg.toFixed(0)}\u00B0`
      : "\u2014";
    const made = !!pred.made;
    const isOOD = result.decision_source === "agent_native_reflex";

    if (els.narratorBlock) {
      els.narratorBlock.classList.remove("made", "miss", "ood");
      els.narratorBlock.classList.add(made ? "made" : (isOOD ? "ood" : "miss"));
    }
    if (els.phaseTitle) {
      const verdict = made
        ? (result._chainedFromTraining ? "REFLEX FIRED \u00B7 MADE IT" : "MADE IT")
        : isOOD
        ? "CONFIDENTLY MISSED"
        : "MISSED";
      els.phaseTitle.textContent = `${verdict} \u00B7 ${pocketName}`;
    }
    if (els.narrator) {
      let line;
      if (made) {
        if (result._chainedFromTraining) {
          line = `${bankBadge} \u00B7 ${angle}. Prediction matched reality. The Q-row is locked. Click the path to replay or anywhere on the felt to rack and aim again.`;
        } else {
          line = `${bankBadge} \u00B7 ${angle}. Q-row updated for the exact coached shot. Click the path to replay or the felt to rack.`;
        }
      } else {
        const missDist = typeof pred.min_distance === "number"
          ? `${pred.min_distance.toFixed(1)}u off`
          : "wide";
        if (isOOD) {
          line = `${bankBadge} \u00B7 ${angle} \u00B7 ${missDist}. Native reflex fired confidently into unseen terrain. Click the felt to rack and teach it.`;
        } else {
          line = `${bankBadge} \u00B7 ${angle} \u00B7 ${missDist}. The row updates from the angle you actually fired. Click the felt to rack.`;
        }
      }
      els.narrator.textContent = line;
    }

    // Update the bottom hint with the post-shot action prompt now that
    // the freeze has happened and the user has SEEN the verdict. We
    // intentionally don't set this earlier (during the shot animation)
    // because "teach another pocket" prompts on top of a still-firing
    // shot destroy the moment.
    if (els.hint) {
      if (made) {
        els.hint.textContent = "Click the path to replay \u00B7 click the felt to rack and aim again.";
      } else {
        els.hint.textContent = isOOD
          ? "Out-of-distribution. Click the felt to rack, then teach this shot to fill the Q-table."
          : "Close, but not the coached angle. Click the felt to rack and try teaching again.";
      }
    }

    // Wisdom callout reflects the verdict
    if (made) {
      setWisdomCallout("made");
    } else if (isOOD) {
      setWisdomCallout("ood");
    } else {
      setWisdomCallout("miss");
    }

    // Push a fresh sample to the RPE sparkline so the user can SEE
    // the dopamine spike from this exact shot relative to recent
    // history. This is the chemistry the manifesto is teaching about.
    pushRPESample(result);

    // Celebration moment for shots auto-fired from training. The user
    // just spent 30 seconds watching the agent learn - the reveal
    // needs to LAND. We swap the watch button + restore narrator
    // gestures only AFTER the verdict has had time to register.
    if (result._chainedFromTraining) {
      if (made) {
        celebrateLockInSuccess(result);
      } else {
        // Even on miss-from-training, make sure the watch button is
        // restored so the user isn't trapped in "Calling the shot..."
        if (els.watchBtn) {
          els.watchBtn.disabled = false;
          els.watchBtn.classList.remove("training");
          els.watchBtn.classList.add("committed");
          els.watchBtn.textContent = "Train another shot";
        }
      }
    }

    // Metrics row inside the narrator card.
    if (els.psMiss) {
      els.psMiss.textContent = typeof pred.min_distance === "number"
        ? `${pred.min_distance.toFixed(1)}u` : "\u2014";
    }
    if (els.psBounces) {
      els.psBounces.textContent = String(banks);
    }
    if (els.psReward) {
      const r = pred.reward;
      els.psReward.textContent = typeof r === "number"
        ? `${r >= 0 ? "+" : ""}${r.toFixed(0)}` : "\u2014";
    }
    if (els.psRpeValue && els.psRpeFill) {
      const reward = typeof pred.reward === "number" ? pred.reward : 0;
      const magnitude = Math.min(1, Math.abs(reward) / 100);
      els.psRpeFill.style.width = (magnitude * 50).toFixed(1) + "%";
      if (made) {
        els.psRpeFill.classList.remove("negative");
        els.psRpeFill.style.transform = "translateX(0)";
      } else {
        els.psRpeFill.classList.add("negative");
        els.psRpeFill.style.transform = "translateX(-100%)";
      }
      els.psRpeValue.textContent = (made ? "+" : "") + reward.toFixed(1);
    }
  }

  function clearVerdictFromRail() {
    if (els.narratorBlock) {
      els.narratorBlock.classList.remove("made", "miss", "ood", "celebration");
    }
    if (els.psMiss) els.psMiss.textContent = "\u2014";
    if (els.psBounces) els.psBounces.textContent = "\u2014";
    if (els.psReward) els.psReward.textContent = "\u2014";
    if (els.psRpeValue) els.psRpeValue.textContent = "\u2014";
    if (els.psRpeFill) {
      els.psRpeFill.style.width = "0%";
      els.psRpeFill.classList.remove("negative");
      els.psRpeFill.style.transform = "translateX(0)";
    }
    // Restore aim-state wisdom when verdict clears (i.e. user racked).
    if (appMode === "predict") {
      setWisdomCallout(aim.targetPocket ? "aim_locked" : "aiming");
    }
  }

  // The old on-felt frozen-state "tip pills" (click the felt / click
  // the path) are gone. The narrator-gestures pictograms in the right
  // rail show the same affordances continuously without painting text
  // on the table.
  function showFrozenTipsOnce(_finalPoint) {}
  function hideFrozenTips() {}

  // Replay the last shot. Triggered by clicking the painted trajectory.
  function replayLastShot() {
    if (!lastResult) return;
    hideFrozenTips();
    clearVerdictFromRail();
    if (els.narratorBlock) els.narratorBlock.classList.remove("made", "miss", "ood");
    anim.frozen = false;
    anim.freezeResult = null;
    tableWrap.classList.remove("frozen");
    startShotAnimation(lastResult);
  }

  function rackAgain() {
    anim.active = false;
    anim.frozen = false;
    anim.freezeResult = null;
    anim.phase = "idle";
    anim.prediction = null;
    anim.ballPos = null;
    anim.trail = [];
    anim.bounceRipple = { active: false, startedAt: 0 };
    anim.hoverPath = false;
    tableWrap.classList.remove("frozen");
    hideFrozenTips();
    clearVerdictFromRail();
    setLoopActive(null);
    lastResult = null;
    aim.previewSim = null;
    aim.targetPocket = null;
    updateTeachButton();
    if (els.phaseTitle) els.phaseTitle.textContent = "awaiting shot";
    if (els.narrator) {
      els.narrator.textContent =
        "Cue ball is locked at (50, 30). Move the cue to aim. Trajectory picks the pocket. Click Teach this shot to imprint the Q-row, or click any pocket to test memory.";
    }
    if (els.hint) els.hint.textContent = "Click any pocket to call the shot.";
    // Re-seed the default aim so the cue stick and trajectory preview
    // reappear instantly after a rack instead of requiring a mouse move.
    aim.angleDeg = defaultAimAngle();
    refreshAim();
  }

  // ----------------------------------------------------------------------
  // cinematic mode lifecycle
  // ----------------------------------------------------------------------

  async function enterCinematic() {
    if (cine.enabled) return;
    if (sanctum.active) return;
    // Pre-flight: if no valid aim, refuse the open in-place with a soft
    // toast instead of opening the cinematic and immediately bouncing
    // out of it. The previous flow flashed a captioned overlay for a
    // single frame which was jarring.
    if (!aim.targetPocket || !aim.previewSim) {
      showStageToast("Aim into a pocket first", {
        variant: "default",
        glyph: "✦",
        dwellMs: 2000,
      });
      return;
    }
    // Capture the aim NOW, before any awaits below. The fade-out
    // sleep keeps appMode in "predict" briefly, so a stray mousemove
    // could otherwise repaint aim.targetPocket out from under us.
    const targetPocket = aim.targetPocket;
    const targetBounces = aim.previewSim.bounces;
    const anchorAngleDeg = aim.angleDeg;
    cine.enabled = true;
    cine.paused = false;
    cine.episodes = [];
    cine.summary = {};
    cine.status = "starting";
    cine.idx = -1;
    cine.accumulator = 0;
    cine.perEpisodeMs = 90;
    cine.grooves = [];
    cine.particles = [];
    cine.recentAngles = [];
    cine.portalPhase = 0;
    cine.totalTries = 0;
    cine.totalMakes = 0;
    cine.qPrev = null;
    cine.rewardHistory = [];
    cine.winrateHistory = [];
    cine.explorationHistory = [];
    cine.lastFrameTime = performance.now();
    cine.streaming = true;
    cine.completionShown = false;
    cine.q = null;
    cine.bestAngleDeg = null;
    cine.targetPocket = null;
    cine.targetBounces = null;
    cine.lastTier = null;
    cine.strictMatchLearned = null;
    cine.actualBouncesLearned = null;

    // If we entered from a frozen post-shot state, clear the
    // verdict in the right rail so the cinematic narration takes over.
    if (anim.frozen) {
      tableWrap.classList.remove("frozen");
      hideFrozenTips();
      clearVerdictFromRail();
      if (els.narratorBlock) els.narratorBlock.classList.remove("made", "miss", "ood");
      anim.frozen = false;
      anim.freezeResult = null;
    }

    // Wisdom callout: hold the training-loop quote during the
    // cinematic so the right rail stays in voice.
    setWisdomCallout("training");

    appMode = "cinematic";
    tableWrap.classList.add("cinematic-mode");
    // Ensure the leaving class from any prior exit is cleared so the
    // enter animation plays cleanly.
    els.cineOverlay.classList.remove("leaving", "locked-in");
    els.cineOverlay.classList.remove("hidden");
    if (els.trainingBlock) els.trainingBlock.classList.remove("hidden");
    sizeSparkline(els.rewardSpark);
    sizeSparkline(els.winrateSpark);
    els.cinePauseBtn.textContent = "Pause";
    if (els.cineEyebrow) els.cineEyebrow.textContent = "Running simulations";
    els.cinePhase.textContent = "Spinning up Q-learning";
    els.cineStatus.textContent = "Connecting...";
    if (els.cineProgressFill) els.cineProgressFill.style.width = "0%";
    // Reset new live-training widgets to the "all chaos" starting state
    if (els.epsilonFill) els.epsilonFill.style.width = "100%";
    if (els.convergeFill) els.convergeFill.style.width = "0%";
    if (els.cineExplore) els.cineExplore.textContent = "100%";
    if (els.cineConverge) els.cineConverge.textContent = "0%";
    if (els.cineTries) els.cineTries.textContent = "0";
    if (els.cineMakes) els.cineMakes.textContent = "0";
    if (els.cineWinrate) els.cineWinrate.textContent = "0%";
    if (els.cineAngle) els.cineAngle.textContent = "—";
    if (els.cineReward) els.cineReward.textContent = "—";
    if (els.cineFirstmake) els.cineFirstmake.textContent = "—";
    els.watchBtn.disabled = true;
    els.watchBtn.classList.remove("committed");
    els.watchBtn.classList.add("training");
    els.watchBtn.textContent = "Training...";

    // Felt-level pulse from the cue ball so the eye knows where to look
    // first. Pairs with the cinematic overlay's fade-in.
    firePulse({ gold: false });

    // Intro caption beats: each one lands for a second, then crossfades
    // to the next. The cinematic opens with intent.
    playCinematicCaptionSequence([
      "Every future equally plausible. Spinning them up.",
      "Imprinting your demonstration as the prior.",
      "Collapsing parallel timelines around your angle.",
    ], 1100);
    // (targetPocket / targetBounces / anchorAngleDeg captured at top
    // of the function, before any awaits, so the aim can't drift.)
    cine.targetPocket = targetPocket;
    cine.targetBounces = targetBounces;
    cine.anchorAngleDeg = anchorAngleDeg;
    if (els.cinePhase) {
      const bankBadge =
        targetBounces === 0 ? "direct" : `${targetBounces}-bank`;
      els.cinePhase.textContent =
        `Teaching agent: ${pocketLabel(targetPocket)} (${bankBadge}) · demo ${anchorAngleDeg.toFixed(0)}°`;
    }

    try {
      const res = await fetch("/train/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          target_pocket: targetPocket,
          target_bounces: targetBounces,
          anchor_angle_deg: anchorAngleDeg,
          // 2-bank shots have a much narrower viable angle range than 1-bank,
          // and 3-bank narrower still. Scale episodes so the Q-row has
          // time to actually find the spike.
          timesteps:
            targetBounces === 0
              ? 2000
              : targetBounces === 1
              ? 2000
              : targetBounces === 2
              ? 4000
              : 6000,
          seed: 42,
          pace_ms: targetBounces >= 2 ? 6 : 12,
          reset_q: true,
        }),
      });
      if (!res.ok && res.status !== 409) {
        const txt = await res.text();
        els.cineCaption.textContent = `Training failed to start: ${txt}`;
      }
    } catch (err) {
      els.cineCaption.textContent = `Could not reach /train/start: ${err}`;
    }

    openTrainingSocket();
  }

  function openTrainingSocket() {
    if (cine.ws) {
      try {
        cine.ws.close();
      } catch (_) {}
      cine.ws = null;
    }
    const proto = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${window.location.host}/ws/train/${sessionId}`);
    cine.ws = ws;
    cine.streaming = true;

    ws.onmessage = (event) => {
      let msg;
      try {
        msg = JSON.parse(event.data);
      } catch (_) {
        return;
      }
      if (msg.type === "status") {
        cine.summary = { ...cine.summary, ...msg.payload };
        cine.status = msg.payload.status;
        if (msg.payload.target_pocket) {
          cine.targetPocket = msg.payload.target_pocket;
        }
        if (msg.payload.status === "error") {
          cancelCinematicCaptionSequence();
          els.cineCaption.textContent = `Training error: ${msg.payload.error}`;
        }
      } else if (msg.type === "episode") {
        cine.episodes.push(msg.payload);
        cine.summary.episodes_done = msg.payload.episode_idx;
        if (msg.payload.target_pocket) {
          cine.targetPocket = msg.payload.target_pocket;
        }
        // Legacy tier-change captions (kept for non-anchored flows).
        const tier = msg.payload.search_tier;
        const radius = msg.payload.search_radius_deg;
        if (typeof tier === "number" && tier !== cine.lastTier) {
          cine.lastTier = tier;
          if (els.cineCaption) {
            // Cancel the intro caption sequence so the tier-change line
            // doesn't get overwritten by a still-running scripted beat.
            cancelCinematicCaptionSequence();
            if (tier > 0 && typeof radius === "number") {
              els.cineCaption.textContent =
                `Your angle wasn’t reachable. Widening search to ±${radius.toFixed(0)}° around your demo and trying again.`;
            }
          }
        }
        // Track what bounce count actually got learned so the
        // auto-fire phase + post-shot banner can honestly reflect it.
        if (typeof msg.payload.actual_bounces_learned === "number") {
          cine.actualBouncesLearned = msg.payload.actual_bounces_learned;
        }
      } else if (msg.type === "complete") {
        cine.summary = { ...cine.summary, ...msg.payload };
        cine.status = msg.payload.status;
        cine.streaming = false;
        if (typeof msg.payload.strict_match_learned === "boolean") {
          cine.strictMatchLearned = msg.payload.strict_match_learned;
        }
        if (typeof msg.payload.actual_bounces_learned === "number") {
          cine.actualBouncesLearned = msg.payload.actual_bounces_learned;
        }
        if (msg.payload.model && msg.payload.model.trained_pockets) {
          cfg.model = {
            ...(cfg.model || {}),
            ...msg.payload.model,
            loaded: msg.payload.model.trained_pockets.length > 0,
          };
        }
      } else if (msg.type === "error") {
        els.cineCaption.textContent = `Stream error: ${msg.payload.message}`;
        cine.streaming = false;
      }
    };

    ws.onclose = () => {
      cine.streaming = false;
    };

    ws.onerror = () => {
      els.cineCaption.textContent =
        "WebSocket dropped. Training may still be running -- reopen Watch It Learn to reconnect.";
      cine.streaming = false;
    };
  }

  async function exitCinematic() {
    cine.enabled = false;
    cine.streaming = false;
    cancelCinematicCaptionSequence();
    if (cine.autoExitTimer) {
      window.clearTimeout(cine.autoExitTimer);
      cine.autoExitTimer = null;
    }
    if (cine.ws) {
      try {
        cine.ws.close();
      } catch (_) {}
      cine.ws = null;
    }
    const justTrainedPocket = cine.targetPocket;
    const strictOk =
      cine.strictMatchLearned === null ? true : Boolean(cine.strictMatchLearned);
    const converged = cine.status === "done" && strictOk;

    // Graceful fade-down of the overlay before we drop into predict
    // mode + chain the auto-fire. Without this the cinematic vanishes
    // on a frame-perfect class toggle which reads as "click-snap."
    els.cineOverlay.classList.add("leaving");
    await sleep(260);
    appMode = "predict";
    tableWrap.classList.remove("cinematic-mode");
    els.cineOverlay.classList.add("hidden");
    els.cineOverlay.classList.remove("leaving", "locked-in");
    if (els.trainingBlock) els.trainingBlock.classList.add("hidden");

    // When we're about to auto-fire the trained shot, DON'T rack to
    // idle state (which briefly flashes "awaiting shot" and kills the
    // magic). Instead, prep the table silently and go straight into
    // the handoff.
    if (converged && justTrainedPocket) {
      anim.active = false;
      anim.frozen = false;
      anim.freezeResult = null;
      anim.phase = "idle";
      anim.prediction = null;
      anim.ballPos = null;
      anim.trail = [];
      anim.bounceRipple = { active: false, startedAt: 0 };
      tableWrap.classList.remove("frozen");
      hideFrozenTips();
      clearVerdictFromRail();
      if (els.narratorBlock) els.narratorBlock.classList.remove("made", "miss", "ood");
      lastResult = null;

      // re-pull config so the dashboard reflects the newly-trained Q-row
      await refreshConfigButton();

      selectedPocket = justTrainedPocket;
      const bounceToReplay =
        typeof cine.actualBouncesLearned === "number"
          ? cine.actualBouncesLearned
          : cine.targetBounces;

      firePulse({ gold: true });
      // Dramatic "calling the shot" beat - the reflex is committed,
      // the table is loaded, the cue is rising. The user needs to FEEL
      // the moment before the ball moves. This phase title is shown
      // for ~1.4s before chainTrainedShot starts the animation.
      if (els.phaseTitle) els.phaseTitle.textContent = "CALLING THE SHOT";
      if (els.narrator) {
        const angleStr =
          cine.summary.final_angle !== null && cine.summary.final_angle !== undefined
            ? `${cine.summary.final_angle.toFixed(0)}\u00B0`
            : "the locked angle";
        els.narrator.textContent =
          `Corner pocket. ${pocketLabel(justTrainedPocket)}. ${angleStr}. Take the shot.`;
      }
      setWisdomCallout("calling");
      showStageToast("Calling the shot \u00B7 watch the reflex fire", {
        variant: "gold",
        glyph: "\u2726",
        dwellMs: 2200,
      });
      // Hold for the calling beat - the user needs to read the call
      // before the agent acts on it. This is the bridge between
      // intention and action that the whole text is about.
      await sleep(1400);
      await chainTrainedShot(justTrainedPocket, bounceToReplay);
    } else {
      rackAgain();
      await refreshConfigButton();
    }
  }

  function onCinematicEnd() {
    cine.paused = true;
    cancelCinematicCaptionSequence();
    els.cinePauseBtn.textContent = "Done";
    const strictOk =
      cine.strictMatchLearned === null ? true : Boolean(cine.strictMatchLearned);
    const finalAngle =
      cine.summary.final_angle !== null && cine.summary.final_angle !== undefined
        ? `${cine.summary.final_angle.toFixed(1)}°`
        : "its committed angle";
    if (els.cineEyebrow) {
      els.cineEyebrow.textContent = strictOk ? "Locked in" : "No strict match";
    }
    els.cineCaption.textContent = strictOk
      ? `Q-row locked at ${finalAngle}. The agent will now call the shot.`
      : "Your exact coached shot could not be imprinted from this cue lock and bounce target. No fallback route was learned.";
    if (els.cineProgressFill) els.cineProgressFill.style.width = "100%";
    els.cineOverlay.classList.add("locked-in");
    firePulse({ gold: strictOk });
    // Keep the watch button in "training" mode until the auto-fire
    // celebration plays - "Train another shot" appearing before the
    // success animation undercuts the moment.
    els.watchBtn.disabled = true;
    els.watchBtn.textContent = strictOk ? "Calling the shot..." : "No match found";

    // Hold the locked-in beat longer so the user actually feels the
    // commitment. The progress bar fills, the mandala tightens, the
    // golden spike pulses - this needs SPACE to land.
    if (cine.autoExitTimer) {
      window.clearTimeout(cine.autoExitTimer);
    }
    cine.autoExitTimer = window.setTimeout(() => {
      if (cine.enabled) exitCinematic();
    }, strictOk ? 2800 : 3200);
  }

  async function refreshConfigButton() {
    try {
      const res = await fetch(`/config?session_id=${sessionId}`);
      cfg = await res.json();
      updateTeachButton();
      refreshDashboard();
    } catch (_) {}
  }

  function updateHint() {
    updateAimHint();
  }

  function toggleCinematicPause() {
    cine.paused = !cine.paused;
    els.cinePauseBtn.textContent = cine.paused ? "Resume" : "Pause";
  }

  // ----------------------------------------------------------------------
  // bind & boot
  // ----------------------------------------------------------------------

  function bindUI() {
    canvas.addEventListener("mousemove", handlePointerMove);
    canvas.addEventListener("click", handleClick);
    window.addEventListener("resize", () => resizeCanvas());

    if (els.watchBtn) {
      els.watchBtn.addEventListener("click", enterCinematic);
    }
    if (els.cinePauseBtn) {
      els.cinePauseBtn.addEventListener("click", toggleCinematicPause);
    }
    if (els.cineExitBtn) {
      els.cineExitBtn.addEventListener("click", exitCinematic);
    }

    // ---- help FAB + tips modal ----
    if (els.helpFab) {
      els.helpFab.addEventListener("click", openTipsModal);
    }
    if (els.tipsClose) {
      els.tipsClose.addEventListener("click", closeTipsModal);
    }
    if (els.tipsModal) {
      els.tipsModal.addEventListener("click", (e) => {
        if (e.target === els.tipsModal) closeTipsModal();
      });
    }
    window.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        if (els.tipsModal && !els.tipsModal.classList.contains("hidden")) {
          closeTipsModal();
        }
        if (els.memoryKebabMenu && !els.memoryKebabMenu.classList.contains("hidden")) {
          closeMemoryKebab();
        }
      }
    });

    // ---- Q-table kebab menu (wipe-memory action lives here now) ----
    if (els.memoryKebab) {
      els.memoryKebab.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleMemoryKebab();
      });
    }
    if (els.wipeMemory) {
      els.wipeMemory.addEventListener("click", () => {
        closeMemoryKebab();
        wipeAgentMemory();
      });
    }
    // Outside click closes the kebab menu.
    document.addEventListener("click", (e) => {
      if (!els.memoryKebabMenu || els.memoryKebabMenu.classList.contains("hidden")) return;
      if (
        e.target === els.memoryKebab ||
        els.memoryKebab.contains(e.target) ||
        els.memoryKebabMenu.contains(e.target)
      ) return;
      closeMemoryKebab();
    });
  }

  function toggleMemoryKebab() {
    if (!els.memoryKebabMenu || !els.memoryKebab) return;
    const isOpen = !els.memoryKebabMenu.classList.contains("hidden");
    if (isOpen) closeMemoryKebab();
    else openMemoryKebab();
  }
  function openMemoryKebab() {
    if (!els.memoryKebabMenu || !els.memoryKebab) return;
    els.memoryKebabMenu.classList.remove("hidden");
    els.memoryKebabMenu.setAttribute("aria-hidden", "false");
    els.memoryKebab.setAttribute("aria-expanded", "true");
  }
  function closeMemoryKebab() {
    if (!els.memoryKebabMenu || !els.memoryKebab) return;
    els.memoryKebabMenu.classList.add("hidden");
    els.memoryKebabMenu.setAttribute("aria-hidden", "true");
    els.memoryKebab.setAttribute("aria-expanded", "false");
  }

  function openTipsModal() {
    if (!els.tipsModal) return;
    els.tipsModal.classList.remove("hidden");
    els.tipsModal.setAttribute("aria-hidden", "false");
  }
  function closeTipsModal() {
    if (!els.tipsModal) return;
    els.tipsModal.classList.add("hidden");
    els.tipsModal.setAttribute("aria-hidden", "true");
  }

  async function wipeAgentMemory() {
    if (cine.enabled || sanctum.active) {
      // Training in flight - show a soft refusal instead of swallowing
      // the click silently so the user understands why nothing happened.
      showStageToast("Wait for training to finish", {
        variant: "danger",
        glyph: "⌫",
        dwellMs: 1800,
      });
      return;
    }
    const ok = await openWipeModal();
    if (!ok) return;
    if (els.wipeMemory) {
      els.wipeMemory.disabled = true;
    }
    try {
      // Beat 1: the visible memory grid dissolves column by column so
      // the user *sees* the forgetting before the backend confirms it.
      const sweep = playMemoryWipeSweep();
      // Beat 2: the network call runs in parallel with the sweep so the
      // animation never feels gated on latency. We wait for both.
      const fetchP = fetch("/memory/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      }).then(
        async (res) => {
          if (!res.ok) throw new Error(await res.text());
          return res.json();
        }
      );
      const [, data] = await Promise.all([sweep, fetchP]);
      if (data.model && cfg) cfg.model = data.model;
      // Beat 3: thaw any frozen shot, return to aim, restamp the dashboard.
      gracefulRack({ pulse: false, toast: false });
      refreshDashboard();
      aim.angleDeg = defaultAimAngle();
      refreshAim();
      if (els.narrator) {
        els.narrator.textContent =
          "Memory wiped. The agent has zero priors \u2014 every shot will fire its native reflex first. Aim and teach to fill the Q-table back up.";
      }
      if (els.hint) {
        els.hint.textContent =
          "Cold-start. Aim into a pocket and click Teach to give the agent its first Q-row.";
      }
      showStageToast("Memory cleared \u00B7 blank slate", {
        variant: "danger",
        glyph: "\u2327",
        dwellMs: 2400,
      });
    } catch (err) {
      showStageToast("Could not wipe memory", {
        variant: "danger",
        glyph: "!",
        dwellMs: 2400,
      });
      console.error("wipe failed:", err);
    } finally {
      if (els.wipeMemory) {
        els.wipeMemory.disabled = false;
      }
    }
  }

  // Wipe-memory uses this to thaw any frozen verdict + rack the cue.
  // Pulse + toast are configurable so a wipe doesn't double-up on
  // pulses already happening elsewhere.
  function gracefulRack(opts) {
    const wantPulse = !opts || opts.pulse !== false;
    const wantToast = !opts || opts.toast !== false;
    rackAgain();
    if (wantPulse) firePulse({ gold: false });
    if (wantToast) {
      showStageToast("Rack ready \u00B7 aim the next shot", {
        variant: "default",
        glyph: "\u25C7",
        dwellMs: 1800,
      });
    }
  }

  // ----------------------------------------------------------------------
  // graceful-transition primitives: stage toast, felt pulse, the
  // wipe-memory custom modal, frozen-state affordance tips. Used by
  // Teach / Rack / Replay so nothing in this flow ever feels like a
  // button press snapping a value.
  // ----------------------------------------------------------------------

  let stageToastTimer = null;
  function showStageToast(text, opts) {
    if (!els.stageToast) return;
    const variant = (opts && opts.variant) || "default"; // default | gold | danger
    const dwellMs = (opts && opts.dwellMs) || 2200;
    const glyph = (opts && opts.glyph) || "◇";
    const glyphEl = els.stageToast.querySelector(".stage-toast-glyph");
    const textEl = els.stageToast.querySelector(".stage-toast-text");
    if (glyphEl) glyphEl.textContent = glyph;
    if (textEl) textEl.textContent = text;
    els.stageToast.classList.remove("gold", "danger");
    if (variant === "gold") els.stageToast.classList.add("gold");
    if (variant === "danger") els.stageToast.classList.add("danger");
    // Force animation restart by toggling hidden + reflow.
    els.stageToast.classList.add("hidden");
    void els.stageToast.offsetWidth; // reflow
    els.stageToast.classList.remove("hidden");
    if (stageToastTimer) window.clearTimeout(stageToastTimer);
    stageToastTimer = window.setTimeout(() => {
      els.stageToast.classList.add("hidden");
      stageToastTimer = null;
    }, dwellMs);
  }

  function firePulse(opts) {
    if (!els.feltPulse) return;
    const gold = opts && opts.gold;
    els.feltPulse.classList.remove("firing", "gold");
    if (gold) els.feltPulse.classList.add("gold");
    void els.feltPulse.offsetWidth; // reflow to restart anim
    els.feltPulse.classList.add("firing");
  }

  // No-op: wisdom callouts were removed (felt cluttered/MySpace-y).
  // Kept as a stub so existing call sites don't have to be deleted
  // one by one. RPE history below remains as the lone storytelling
  // element on the right rail.
  function setWisdomCallout(_state) {}

  // Build the 12-slot RPE history bar grid. Called once when the
  // RPE region renders for the first time and every time the list
  // changes.
  function renderRPEHistory() {
    if (!els.rpeHistoryBars) return;
    els.rpeHistoryBars.innerHTML = "";
    for (let i = 0; i < RPE_HISTORY_CAP; i += 1) {
      const cell = document.createElement("div");
      cell.className = "rpe-bar-cell";
      const sample = rpeHistory[i];
      if (sample) {
        const fill = document.createElement("div");
        const mag = Math.max(0.06, Math.min(1, Math.abs(sample.value) / 100));
        const heightPct = (mag * 100 * 0.46).toFixed(1) + "%";
        fill.className = "rpe-bar-fill " + (sample.made ? "up" : "down");
        fill.style.height = heightPct;
        if (sample.fresh) cell.classList.add("fresh");
        cell.appendChild(fill);
        if (sample.fresh) {
          // Decay the fresh flag so subsequent renders don't re-pulse
          // a bar that's already settled in place.
          window.setTimeout(() => {
            sample.fresh = false;
          }, 1400);
        }
      }
      els.rpeHistoryBars.appendChild(cell);
    }
  }

  // Push a new sample to the RPE history sparkline. Called from the
  // freeze step once a shot resolves. Positive RPE = made better than
  // expected (gold spike up). Negative = missed worse than expected
  // (crimson spike down).
  function pushRPESample(result) {
    if (!result || !result.predicted) return;
    const reward = typeof result.predicted.reward === "number"
      ? result.predicted.reward
      : 0;
    const made = !!result.predicted.made;
    const isOOD = result.decision_source === "agent_native_reflex";
    rpeHistory.unshift({ value: reward, made, fresh: true, isOOD });
    if (rpeHistory.length > RPE_HISTORY_CAP) rpeHistory.length = RPE_HISTORY_CAP;
    renderRPEHistory();
  }

  // Big "the agent called it and made it" moment that fires after a
  // training session locks in and the auto-fired shot drops. The user
  // just spent 30 seconds watching the agent learn - the success
  // deserves a real spike, not a quiet badge swap. This holds the
  // narrator celebration class for 3s, plays the gold pulse, and
  // re-arms the topbar watch button only AFTER the moment has landed.
  function celebrateLockInSuccess(result) {
    if (els.narratorBlock) {
      els.narratorBlock.classList.add("celebration");
    }
    setWisdomCallout("celebration");
    firePulse({ gold: true });
    const angle = typeof result.angle_deg === "number"
      ? `${result.angle_deg.toFixed(0)}\u00B0`
      : "the locked angle";
    showStageToast(
      `Prediction matched reality \u00B7 ${angle} \u00B7 the loop closed`,
      { variant: "gold", glyph: "\u2726", dwellMs: 3200 }
    );
    // Hold the watch button at "Reflex fired" for the celebration
    // window before letting the user start a fresh training cycle.
    if (els.watchBtn) {
      els.watchBtn.disabled = true;
      els.watchBtn.classList.remove("training");
      els.watchBtn.classList.add("committed");
      els.watchBtn.textContent = "Reflex fired \u2726";
    }
    window.setTimeout(() => {
      if (els.narratorBlock) {
        els.narratorBlock.classList.remove("celebration");
      }
      if (els.watchBtn) {
        els.watchBtn.disabled = false;
        els.watchBtn.textContent = "Train another shot";
      }
    }, 3200);
  }

  // Sequenced caption beats for the cinematic. Each beat lands for
  // ~beatMs before crossfading to the next. The promise resolves once
  // the final beat has settled.
  let cineCaptionSequenceToken = 0;
  async function playCinematicCaptionSequence(beats, beatMs = 900) {
    if (!els.cineCaption) return;
    const myToken = ++cineCaptionSequenceToken;
    for (let i = 0; i < beats.length; i += 1) {
      if (myToken !== cineCaptionSequenceToken) return;
      if (i === 0) {
        // first beat lands without a fade-out
        els.cineCaption.textContent = beats[i];
      } else {
        els.cineCaption.classList.add("cine-caption-fade");
        await sleep(220);
        if (myToken !== cineCaptionSequenceToken) return;
        els.cineCaption.textContent = beats[i];
        els.cineCaption.classList.remove("cine-caption-fade");
      }
      if (i < beats.length - 1) {
        await sleep(beatMs);
      }
    }
  }
  function cancelCinematicCaptionSequence() {
    cineCaptionSequenceToken += 1;
    if (els.cineCaption) {
      els.cineCaption.classList.remove("cine-caption-fade");
    }
  }

  // ----- wipe-memory custom confirmation modal -------------------------

  let wipeModalPending = false;
  function openWipeModal() {
    if (!els.wipeModal || wipeModalPending) return Promise.resolve(false);
    wipeModalPending = true;
    return new Promise((resolve) => {
      const close = (ok) => {
        if (!wipeModalPending) return;
        wipeModalPending = false;
        els.wipeModal.classList.add("hidden");
        els.wipeModal.setAttribute("aria-hidden", "true");
        document.removeEventListener("keydown", onKey);
        if (els.wipeCancel) els.wipeCancel.onclick = null;
        if (els.wipeConfirm) els.wipeConfirm.onclick = null;
        if (els.wipeClose) els.wipeClose.onclick = null;
        els.wipeModal.onclick = null;
        resolve(!!ok);
      };
      const onKey = (e) => {
        if (e.key === "Escape") close(false);
        else if (e.key === "Enter") close(true);
      };
      if (els.wipeCancel) els.wipeCancel.onclick = () => close(false);
      if (els.wipeClose) els.wipeClose.onclick = () => close(false);
      if (els.wipeConfirm) els.wipeConfirm.onclick = () => close(true);
      els.wipeModal.onclick = (e) => {
        if (e.target === els.wipeModal) close(false);
      };
      document.addEventListener("keydown", onKey);
      els.wipeModal.classList.remove("hidden");
      els.wipeModal.setAttribute("aria-hidden", "false");
      // give the modal a beat to mount before focusing the cancel
      // button so Escape/Enter behave intuitively.
      window.setTimeout(() => {
        if (els.wipeCancel) els.wipeCancel.focus();
      }, 60);
    });
  }

  // Sweep the Q-table grid: cells light up red and fade out one column
  // at a time, then the grid resets. Sells "memory dissolving" before
  // the dashboard repaints empty.
  async function playMemoryWipeSweep() {
    if (!els.memoryGrid) return;
    const cells = Array.from(els.memoryGrid.querySelectorAll(".memory-cell.trained"));
    if (cells.length === 0) return;
    // Group cells by column (bank count) so the wipe travels left-to-right.
    const byCol = new Map();
    cells.forEach((c) => {
      const b = c.dataset.bank || "0";
      if (!byCol.has(b)) byCol.set(b, []);
      byCol.get(b).push(c);
    });
    const cols = Array.from(byCol.keys()).sort();
    els.memoryGrid.classList.add("wiping");
    for (const col of cols) {
      for (const cell of byCol.get(col)) {
        cell.classList.add("wiping");
      }
      await sleep(140);
    }
    await sleep(260);
    cells.forEach((c) => c.classList.remove("wiping"));
    els.memoryGrid.classList.remove("wiping");
  }

  async function boot() {
    const res = await fetch(`/config?session_id=${sessionId}`);
    cfg = await res.json();
    resizeCanvas();
    bindUI();
    buildMemoryGrid();
    refreshDashboard();
    renderRPEHistory();
    setWisdomCallout("idle");
    aim.angleDeg = defaultAimAngle();
    refreshAim();
    requestAnimationFrame(tick);
  }

  function defaultAimAngle() {
    if (!cfg) return 90;
    const cue = cfg.fixed_cue;
    const tr = cfg.pockets.TR;
    const ang = Math.atan2(tr.y - cue.y, tr.x - cue.x) * (180 / Math.PI);
    return (ang + 360) % 360;
  }

  boot().catch((err) => {
    document.getElementById("narrator").textContent = `Boot failed: ${err}`;
  });
})();
