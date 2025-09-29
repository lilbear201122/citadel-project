import React, { useState } from "react";
import { motion } from "framer-motion";

/**
 * Elo Chip-Based Scoring | Demo (Version 1, fixed sizes)
 * Progression & field sizes: 256 (population) → 128 (Cup participants per run) ×3 → 32 (Regionals) → 12 (our region to Worlds; total field 48 incl. external regions)
 * Fees & payouts: Fee = current chips / z. Cup pays fixed y only to the top half; Regionals/Worlds pay fixed y to all participants.
 * Parameters (V1 defaults): z_cup=8, y_cup=50; z_reg=4, y_reg=100; z_world=3, y_world=400. Initial chips x0=100.
 * Performance model used in every event: performance = skill + noise, where noise is approximately normal (sum of three U[-0.5, 0.5]).
 *
 * New (S15 TPC format) simulation:
 * - TPC (Pro League) uses the 32 players from the most recent 32-player Regionals.
 * - 3× TPC rounds (32 players). Pros do NOT play Cups.
 * - 2× S15 Cups (128 players each) drawn from the remaining 224 non-pro players.
 * - S15 Regionals (48 players) = 32 pros + top 16 by chips among non-pros after the 2 Cups.
 */

const DEFAULT_PARAMS = Object.freeze({
  x0: 100,
  z_cup: 8,
  y_cup: 50,
  z_reg: 4,
  y_reg: 100,
  z_world: 3,
  y_world: 400,
  // S15: TPC sits between Cup and Regionals (slightly lower than legacy Regionals)
  z_tpc: 5,      // a bit higher than z_reg → smaller fees than Regionals
  y_tpc: 80,     // between y_cup and y_reg
  // Percentage decay applied to all players when pressing "Decay"
  decay_pct: 10, // default 10%
});

const FIXED_SIZES = Object.freeze({
  population: 256,
  cupSlots: 128,
  reg: 32,
  world: 48,
  ourWorldQuota: 12,
});

// ────────────────────────────────────────────────────────────────────────────────
// Utilities & Core Algorithms
// ────────────────────────────────────────────────────────────────────────────────

function createPlayers(n: number, x0: number) {
  return Array.from({ length: n }, (_, id) => ({ id, skill: normalLike(), chips: x0 }));
}

function normalLike() {
  // Approximately normal: sum of three U(-0.5, 0.5)
  return (Math.random() - 0.5) + (Math.random() - 0.5) + (Math.random() - 0.5);
}

function choiceIndexes(n: number, k: number) {
  // Sample k unique indexes from 0..n-1
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, k);
}

function choiceFromList<T>(list: T[], k: number) {
  const a = [...list];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a.slice(0, k);
}

function rankBy(values: number[]) {
  const order = values.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v).map((x) => x.i);
  const ranks = new Array(values.length);
  order.forEach((pid, k) => (ranks[pid] = k + 1));
  return { order, ranks } as { order: number[]; ranks: number[] };
}

// ────────────────────────────────────────────────────────────────────────────────
// Monotone-by-pairs weight builder (keeps each halving-band total; enforces global non-increasing)
// ────────────────────────────────────────────────────────────────────────────────
function buildMonotoneByPairs(
  N: number,
  top1: number,
  top2: number,
  bands: { start: number; end: number; total: number }[],
) {
  const w = new Array(N).fill(0) as number[];
  if (N >= 1) w[0] = top1;
  if (N >= 2) w[1] = top2;
  let prevTail = N >= 2 ? w[1] : Infinity; // cap for next band's head per-person

  for (const b of bands) {
    const s = Math.max(1, b.start);
    const e = Math.min(b.end, N);
    let m = e - s + 1;
    if (m <= 0) continue;
    if (m % 2 !== 0) { // enforce even length inside a band (defensive; our bands are even)
      m -= 1;
    }
    const p = Math.floor(m / 2); // number of pairs inside band
    const S = b.total;           // band total mass (preserved)
    if (p <= 0) continue;

    // Unconstrained head (per-person) if using proportional ramp T_k ∝ k
    const unconstrainedHeadPP = S / (p + 1);
    const headPP = Math.min(prevTail, unconstrainedHeadPP); // enforce cross-band monotonicity
    const headPair = headPP * 2;

    if (p === 1) {
      // Single pair in this band → split the whole band equally
      const per = S / 2;
      const r1 = e - 1; // better rank
      const r2 = e;     // worse rank
      w[r1 - 1] = per;
      w[r2 - 1] = per;
      prevTail = per;   // tail = rank e
      continue;
    }

    // Solve linear ramp of pair totals: T_k = A + B*k (k=1..p), with
    //   (1) Sum_k T_k = S
    //   (2) T_p = headPair
    const denom = (p * (1 - p)) / 2; // negative
    const B = (S - p * headPair) / denom; // ≥ 0 when headPP ≤ unconstrained
    const A = headPair - B * p;

    // If numeric issues, fall back to uniform within the band
    const safeA = A < -1e-12 ? 0 : A;
    const safeB = B < -1e-12 ? 0 : B;

    for (let k = 1; k <= p; k++) {
      const T_k = safeA + safeB * k;
      const per = T_k / 2;
      const r1 = e - (k - 1) * 2 - 1; // better rank in the pair
      const r2 = r1 + 1;              // worse rank in the pair
      w[r1 - 1] = per;
      w[r2 - 1] = per;
    }
    prevTail = w[e - 1]; // tail per-person (rank e)
  }
  return w;
}

// CUP: keep top two fixed, then halving bands with preserved totals and pair-wise ramp
function weightsCup(N: number) {
  return buildMonotoneByPairs(N, 0.15, 0.09, [
    { start: 3,  end: 4,   total: 0.12 }, // 3–4: total 0.12
    { start: 5,  end: 8,   total: 0.16 }, // 5–8:  0.16
    { start: 9,  end: 16,  total: 0.16 }, // 9–16: 0.16
    { start: 17, end: 32,  total: 0.16 }, // 17–32:0.16
    { start: 33, end: 64,  total: 0.16 }, // 33–64:0.16 (65+ remain 0)
  ]);
}

// REGIONALS: 32 players
function weightsReg(N: number) {
  return buildMonotoneByPairs(N, 0.20, 0.12, [
    { start: 3,  end: 4,   total: 0.16 }, // 3–4:  0.16
    { start: 5,  end: 8,   total: 0.20 }, // 5–8:  0.20
    { start: 9,  end: 16,  total: 0.32 }, // 9–16: 0.32
  ]);
}

// WORLDS: 48 players
function weightsWorld(N: number) {
  // Excel BandTotals → Worlds (48): top1=0.16, top2=0.10, 3–4=0.14, 5–8=0.20, 9–16=0.24, 17–32=0.16, 33–48=0
  return buildMonotoneByPairs(N, 0.16, 0.10, [
    { start: 3,  end: 4,   total: 0.14 },
    { start: 5,  end: 8,   total: 0.20 },
    { start: 9,  end: 16,  total: 0.24 },
    { start: 17, end: 32,  total: 0.16 },
  ]);
}

// S15 Regionals (48 players) — using the same band structure as Worlds (48)
function weightsReg48(N: number) {
  // S15 Regionals (48) — Excel BandTotals (S15 Regional 48):
  // top1=0.15, top2=0.09, 3–4=0.14, 5–8=0.20, 9–16=0.25, 17–32=0.17, 33–48=0
  return buildMonotoneByPairs(N, 0.15, 0.09, [
    { start: 3,  end: 4,   total: 0.14 },
    { start: 5,  end: 8,   total: 0.20 },
    { start: 9,  end: 16,  total: 0.25 },
    { start: 17, end: 32,  total: 0.17 },
  ]);
}

// TPC weights (32 players) — close to legacy Regionals, slightly lower tier, keep same weight curve
function weightsTPC(N: number) {
  // S15 TPC (32) — Excel BandTotals: top1=0.18, top2=0.11, 3–4=0.16, 5–8=0.19, 9–16=0.36, 17–32=0
  return buildMonotoneByPairs(N, 0.18, 0.11, [
    { start: 3,  end: 4,   total: 0.16 },
    { start: 5,  end: 8,   total: 0.19 },
    { start: 9,  end: 16,  total: 0.36 },
  ]);
}

function format2(num: number) { return Math.round(num * 100) / 100; }
function median(arr: number[]){ const a=[...arr].sort((x,y)=>x-y); const m=Math.floor(a.length/2); return a.length%2? a[m] : (a[m-1]+a[m])/2; }
function clampInt(v: number, min: number, max: number) { return Math.max(min, Math.min(max, Math.round((v as any)||0))); }
function delay(ms: number){ return new Promise(res=>setTimeout(res, ms)); }

// Single-event settlement
function tierRun({ playersIdx, players, z, y, weights, giveYToTopHalf }:{
  playersIdx: number[];
  players: { id: number; skill: number; chips: number }[];
  z: number;
  y: number;
  weights: number[];
  giveYToTopHalf: boolean;
}) {
  const N = playersIdx.length;
  const perf = playersIdx.map((pi) => players[pi].skill + normalLike());
  const { order, ranks } = rankBy(perf);
  const fees = playersIdx.map((pi) => players[pi].chips / z);
  const pool = fees.reduce((a, b) => a + b, 0);

  const payouts = new Array(N).fill(0) as number[];
  const sharePool = new Array(N).fill(0) as number[];
  const fixedY = new Array(N).fill(0) as number[];

  for (let r = 1; r <= N; r++) {
    const idxInParticipants = ranks.findIndex((rr) => rr === r);
    const share = (weights[r - 1] || 0) * pool;
    const topHalf = r <= Math.ceil(N / 2);
    const fixed = (giveYToTopHalf ? (topHalf ? y : 0) : y);
    sharePool[idxInParticipants] = share;
    fixedY[idxInParticipants] = fixed;
    payouts[idxInParticipants] = share + fixed;
  }

  const deltas = payouts.map((p, j) => p - fees[j]);
  playersIdx.forEach((pi, j) => { players[pi].chips += deltas[j]; });

  return { order, ranks, deltas, pool, fees, payouts, sharePool, fixedY };
}

// ────────────────────────────────────────────────────────────────────────────────
export default function EloChipsDemo() {
  const [sizes] = useState({ ...FIXED_SIZES });
  const [params, setParams] = useState({ ...DEFAULT_PARAMS });
  const [players, setPlayers] = useState(() => createPlayers(FIXED_SIZES.population, DEFAULT_PARAMS.x0));
  const [stage, setStage] = useState("idle");
  const [history, setHistory] = useState<any[]>([]);

  // Reset all state
  const reset = () => {
    setPlayers(createPlayers(sizes.population, params.x0));
    setStage("idle");
    setHistory([]);
  };

  // Decay all players' chips by (1 - decay_pct/100)
  const decayAll = () => {
    const pct = Math.max(0, Math.min(100, Number(params.decay_pct) || 0));
    const factor = 1 - pct / 100;
    setPlayers(prev => prev.map(p => ({ ...p, chips: Math.max(0, p.chips * factor) }))));
    setStage("decay");
    setHistory(h => [...h, { type: "decay", factor, pct }]);
  };

  // One Cup run (128 participants, any population)
  const runCup = (cupIndex: number) => {
    const n = sizes.cupSlots;
    const idxs = choiceIndexes(players.length, n);
    const w = weightsCup(n);
    const giveYToTopHalf = true; // Cup: y paid only to top half
    const next = players.map((p) => ({ ...p }));
    const res = tierRun({ playersIdx: idxs, players: next, z: params.z_cup, y: params.y_cup, weights: w, giveYToTopHalf });

    // Full details (sorted by rank)
    const rows = idxs.map((pi, j) => ({
      name: `P${pi+1}`,
      playerId: pi,
      rank: res.ranks[j],
      fee: res.fees[j],
      y: res.fixedY[j],
      share: res.sharePool[j],
      payout: res.payouts[j],
      delta: res.deltas[j],
      newChips: next[pi].chips,
    })).sort((a,b)=>a.rank-b.rank);

    setPlayers(next);
    setStage(`cup${cupIndex}`);
    setHistory((h) => [...h, { type: "cup", cupIndex, idxs, rows, ...res }]);
  };

  // Cup phase: run three Cup rounds in sequence
  const runCupsPhase = async () => {
    runCup(1); await delay(300);
    runCup(2); await delay(300);
    runCup(3);
  };

  // Take top k by chips (global)
  const topByChips = (k: number) => (
    players.map((p, i) => ({ i, chips: p.chips }))
      .sort((a, b) => b.chips - a.chips)
      .slice(0, k)
      .map((x) => x.i)
  );

  // Helper: get last Regionals (32) indices as pros; fallback to current top-32 if not found
  const getProIdxsFromLastRegional32 = () => {
    const lastReg32 = [...history].reverse().find((e) => e.type === "regional" && e.idxs?.length === 32);
    if (lastReg32) return [...(lastReg32 as any).idxs];
    return topByChips(sizes.reg);
  };

  // TPC round (32 players = previous Regionals' 32)
  const runTPC = (tpcRound: number) => {
    const proIdxs = getProIdxsFromLastRegional32();
    const w = weightsTPC(proIdxs.length);
    const next = players.map((p) => ({ ...p }));
    const res = tierRun({ playersIdx: proIdxs, players: next, z: params.z_tpc, y: params.y_tpc, weights: w, giveYToTopHalf: false });

    const rows = proIdxs.map((pi, j) => ({
      name: `P${pi+1}`,
      playerId: pi,
      rank: res.ranks[j],
      fee: res.fees[j],
      y: res.fixedY[j],
      share: res.sharePool[j],
      payout: res.payouts[j],
      delta: res.deltas[j],
      newChips: next[pi].chips,
    })).sort((a,b)=>a.rank-b.rank);

    setPlayers(next);
    setStage(`tpc${tpcRound}`);
    setHistory(h => [...h, { type: "tpc", tpcRound, idxs: proIdxs, rows, ...res }]);
  };

  const runTPCPhase = async () => {
    runTPC(1); await delay(300);
    runTPC(2); await delay(300);
    runTPC(3);
  };

  // S15 Cups (2×, 128 players each) — EXCLUDING pros
  const runS15CupsPhase = async () => {
    const proIdxs = getProIdxsFromLastRegional32();
    const proSet = new Set(proIdxs);
    const nonProIdxs = players.map((_, i) => i).filter(i => !proSet.has(i)); // 224
    const w = weightsCup(128);
    const giveYToTopHalf = true;

    const runOne = (round: number) => {
      const idxs = choiceFromList(nonProIdxs, 128);
      const next = players.map((p) => ({ ...p }));
      const res = tierRun({ playersIdx: idxs, players: next, z: params.z_cup, y: params.y_cup, weights: w, giveYToTopHalf });
      const rows = idxs.map((pi, j) => ({
        name: `P${pi+1}`,
        playerId: pi,
        rank: res.ranks[j],
        fee: res.fees[j],
        y: res.fixedY[j],
        share: res.sharePool[j],
        payout: res.payouts[j],
        delta: res.deltas[j],
        newChips: next[pi].chips,
      })).sort((a,b)=>a.rank-b.rank);
      setPlayers(next);
      setStage(`s15_cup${round}`);
      setHistory(h => [...h, { type: "s15_cup", round, idxs, rows, ...res, note: "pros excluded" }]);
    };

    runOne(1); await delay(300);
    runOne(2);
  };

  // Regionals (32 players) — legacy flow
  const runRegional = () => {
    const idxs = topByChips(sizes.reg);
    const w = weightsReg(idxs.length);
    const next = players.map((p) => ({ ...p }));
    const res = tierRun({ playersIdx: idxs, players: next, z: params.z_reg, y: params.y_reg, weights: w, giveYToTopHalf: false });

    const rows = idxs.map((pi, j) => ({
      name: `P${pi+1}`,
      playerId: pi,
      rank: res.ranks[j],
      fee: res.fees[j],
      y: res.fixedY[j],
      share: res.sharePool[j],
      payout: res.payouts[j],
      delta: res.deltas[j],
      newChips: next[pi].chips,
    })).sort((a,b)=>a.rank-b.rank);

    setPlayers(next);
    setStage("regional");
    setHistory((h) => [...h, { type: "regional", idxs, rows, ...res }]);
  };

  // Worlds (our 12 + fill with external regions to 48)
  const runWorlds = () => {
    const lastReg = [...history].reverse().find((e) => e.type === "regional");
    if (!lastReg) return;
    const { idxs: regIdxs, ranks: regRanks } = lastReg as any;
    const wReg = weightsReg(regIdxs.length);
    const regPoints = regRanks.map((r: number) => (wReg[r - 1] || 0) * 100);
    const regOrder = regPoints.map((v: number, j: number) => ({ j, v })).sort((a, b) => b.v - a.v).map((o) => o.j);
    const quota = sizes.ourWorldQuota; // 12
    const ourLocal = regOrder.slice(0, quota).map((j) => regIdxs[j]);

    const extN = Math.max(0, sizes.world - ourLocal.length);
    const medSkill = median(ourLocal.map((pi: number) => players[pi].skill));
    const medChips = median(ourLocal.map((pi: number) => players[pi].chips));

    const allIdxs: number[] = [...ourLocal];
    const allSkill: number[] = [...ourLocal.map((pi: number) => players[pi].skill)];
    const allChips: number[] = [...ourLocal.map((pi: number) => players[pi].chips)];
    for (let i = 0; i < extN; i++) {
      allIdxs.push(-1);
      allSkill.push(medSkill + normalLike() * 0.8);
      allChips.push(medChips);
    }
    const N = allIdxs.length; // 48
    const w = weightsWorld(N);

    const fakePlayers = allIdxs.map((id, k) => ({ id, skill: allSkill[k], chips: allChips[k] }));
    const { order: wOrder, ranks: wRanks, deltas, pool, fees, payouts, sharePool, fixedY } = tierRun({
      playersIdx: fakePlayers.map((_, k) => k),
      players: fakePlayers,
      z: params.z_world,
      y: params.y_world,
      weights: w,
      giveYToTopHalf: false,
    });

    const next = players.map((p) => ({ ...p }));
    fakePlayers.forEach((fp, k) => {
      const gid = allIdxs[k];
      if (gid >= 0) next[gid].chips = fp.chips;
    });

    const rows = wOrder.map((k, pos) => {
      const gid = allIdxs[k];
      const isExt = gid < 0;
      const name = isExt ? `EXT${pos+1}` : `P${gid+1}`;
      return {
        name,
        playerId: gid,
        rank: pos+1,
        fee: fees[k],
        y: fixedY[k],
        share: sharePool[k],
        payout: payouts[k],
        delta: deltas[k],
        newChips: isExt ? fakePlayers[k].chips : next[gid]?.chips,
      };
    });

    setPlayers(next);
    setStage("worlds");
    setHistory((h) => [...h, { type: "worlds", ourLocal, allIdxs, order: wOrder, ranks: wRanks, deltas, pool, rows }]);
  };

  // S15 Regionals (48) = 32 pros + top-16 non-pros after S15 Cups
  const runS15Regional48 = () => {
    const proIdxs = getProIdxsFromLastRegional32();
    const proSet = new Set(proIdxs);
    const nonProIdxs = players.map((_, i) => i).filter(i => !proSet.has(i));
    // Pick top-16 by chips among NON-PRO population (after whatever Cups have run)
    const nonProTop16 = nonProIdxs
      .map(i => ({ i, chips: players[i].chips }))
      .sort((a,b)=>b.chips - a.chips)
      .slice(0, 16)
      .map(x => x.i);

    const idxs = [...proIdxs, ...nonProTop16]; // 48 total
    const w = weightsReg48(idxs.length);
    const next = players.map((p) => ({ ...p }));
    const res = tierRun({ playersIdx: idxs, players: next, z: params.z_reg, y: params.y_reg, weights: w, giveYToTopHalf: false });

    const rows = idxs.map((pi, j) => ({
      name: `P${pi+1}`,
      playerId: pi,
      rank: res.ranks[j],
      fee: res.fees[j],
      y: res.fixedY[j],
      share: res.sharePool[j],
      payout: res.payouts[j],
      delta: res.deltas[j],
      newChips: next[pi].chips,
    })).sort((a,b)=>a.rank-b.rank);

    setPlayers(next);
    setStage("s15_regional");
    setHistory(h => [...h, { type: "s15_regional", proIdxs, cupTop16: nonProTop16, idxs, rows, ...res }]);
  };

  // One-click S15 flow: 3×TPC → 2×Cups → S15 Regionals (48)
  const runS15Auto = async () => {
    await runTPCPhase();
    await delay(400);
    await runS15CupsPhase();
    await delay(400);
    runS15Regional48();
  };

  const autoPlay = async () => {
    reset();
    await delay(250);
    await runCupsPhase();
    await delay(500);
    runRegional();
    await delay(600);
    runWorlds();
  };

  return (
    <div className="w-full min-h-screen bg-slate-950 text-slate-100 px-6 py-8">
      <div className="max-w-6xl mx-auto">
        <header className="mb-6">
          <h1 className="text-2xl md:text-3xl font-bold">Elo Chip-Based Scoring | Demo (Version 1)</h1>
          <p className="text-slate-300 mt-2">
            Progression: 256 → 128 → 32 → 12 → 48. Fee = current chips / z.
            Cup pays fixed y to the top half only; Regionals/Worlds pay fixed y to all.
          </p>
        </header>

        <Controls
          params={params}
          setParams={setParams}
          sizes={sizes}
          reset={reset}
          autoPlay={autoPlay}
          runCupsPhase={runCupsPhase}
          runRegional={runRegional}
          runWorlds={runWorlds}
          decayAll={decayAll}
          // S15
          runTPCPhase={runTPCPhase}
          runS15CupsPhase={runS15CupsPhase}
          runS15Regional48={runS15Regional48}
          runS15Auto={runS15Auto}
        />

        <div className="grid md:grid-cols-3 gap-6 mt-6">
          <PlayersPanel title="Population / Current Chips (sorted by chips)" players={players} highlightCount={sizes.reg} />
          <PoolsPanel history={history} params={params} />
          <NotesPanel params={params} sizes={sizes} />
        </div>

        {/* Order: Matrix first, then Timeline */}
        <ChangesMatrix history={history} playersState={players} />
        <Timeline history={history} />
        <TestsPanel />
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────────
// UI Blocks
// ────────────────────────────────────────────────────────────────────────────────

function Controls({ params, setParams, sizes, reset, autoPlay, runCupsPhase, runRegional, runWorlds, decayAll, runTPCPhase, runS15CupsPhase, runS15Regional48, runS15Auto }:{
  params: any;
  setParams: React.Dispatch<React.SetStateAction<any>>;
  sizes: any;
  reset: () => void;
  autoPlay: () => Promise<void>;
  runCupsPhase: () => Promise<void>;
  runRegional: () => void;
  runWorlds: () => void;
  decayAll: () => void;
  runTPCPhase: () => Promise<void>;
  runS15CupsPhase: () => Promise<void>;
  runS15Regional48: () => void;
  runS15Auto: () => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  return (
    <div className="bg-slate-900/60 rounded-2xl p-4 flex flex-col gap-3 border border-slate-800">
      <div className="flex flex-wrap gap-3 items-end">
        <NumberField label="z_cup" value={params.z_cup} onChange={(v)=>setParams((p:any)=>({ ...p, z_cup: clampInt(v,2,20) }))} step={1} />
        <NumberField label="y_cup (top half only)" value={params.y_cup} onChange={(v)=>setParams((p:any)=>({ ...p, y_cup: clampInt(v,0,2000) }))} step={10} />
        <NumberField label="z_reg" value={params.z_reg} onChange={(v)=>setParams((p:any)=>({ ...p, z_reg: clampInt(v,2,20) }))} step={1} />
        <NumberField label="y_reg (all players)" value={params.y_reg} onChange={(v)=>setParams((p:any)=>({ ...p, y_reg: clampInt(v,0,3000) }))} step={10} />
        <NumberField label="z_world" value={params.z_world} onChange={(v)=>setParams((p:any)=>({ ...p, z_world: clampInt(v,2,20) }))} step={1} />
        <NumberField label="y_world (all players)" value={params.y_world} onChange={(v)=>setParams((p:any)=>({ ...p, y_world: clampInt(v,0,10000) }))} step={50} />
        {/* S15 TPC params */}
        <NumberField label="z_tpc" value={params.z_tpc} onChange={(v)=>setParams((p:any)=>({ ...p, z_tpc: clampInt(v,2,20) }))} step={1} />
        <NumberField label="y_tpc (all players)" value={params.y_tpc} onChange={(v)=>setParams((p:any)=>({ ...p, y_tpc: clampInt(v,0,3000) }))} step={10} />
        {/* Decay percentage input */}
        <NumberField label="decay %" value={params.decay_pct} onChange={(v)=>setParams((p:any)=>({ ...p, decay_pct: clampInt(v,0,100) }))} step={1} />
      </div>

      {/* Legacy flow controls */}
      <div className="flex flex-wrap gap-2 pt-2">
        <button className="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 border border-slate-700" onClick={reset}>Reset</button>
        <button className="px-3 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500" onClick={runCupsPhase}>Cup</button>
        <button className="px-3 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500" onClick={runRegional}>Regionals</button>
        <button className="px-3 py-2 rounded-xl bg-yellow-600 hover:bg-yellow-500" onClick={runWorlds}>Worlds</button>
        {/* Decay button */}
        <button className="px-3 py-2 rounded-xl bg-rose-600 hover:bg-rose-500" onClick={decayAll}>Decay</button>
        <button
          className="px-3 py-2 rounded-xl bg-fuchsia-600 hover:bg-fuchsia-500"
          onClick={async()=>{
            if(busy) return;
            setBusy(true);
            await autoPlay();
            setBusy(false);
          }}
        >
          Auto Play
        </button>
      </div>

      {/* S15 TPC-format controls */}
      <div className="flex flex-wrap gap-2 pt-3 border-t border-slate-800 mt-2">
        <div className="text-xs uppercase tracking-wide text-slate-400 mr-2 pt-2">S15 (TPC format)</div>
        <button className="px-3 py-2 rounded-xl bg-blue-600 hover:bg-blue-500" onClick={runTPCPhase}>TPC ×3</button>
        <button className="px-3 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500" onClick={runS15CupsPhase}>S15 Cups ×2</button>
        <button className="px-3 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500" onClick={runS15Regional48}>S15 Regionals (48)</button>
        <button
          className="px-3 py-2 rounded-xl bg-fuchsia-600 hover:bg-fuchsia-500"
          onClick={async()=>{
            if(busy) return;
            setBusy(true);
            await runS15Auto();
            setBusy(false);
          }}
        >
          S15 Auto
        </button>
      </div>
    </div>
  );
}

function PlayersPanel({ title, players, highlightCount }:{ title: string; players: any[]; highlightCount: number; }) {
  const sorted = [...players].sort((a,b)=>b.chips-a.chips);
  return (
    <div className="bg-slate-900/60 rounded-2xl p-4 border border-slate-800">
      <h3 className="font-semibold mb-3">{title}</h3>
      <div className="grid grid-cols-8 gap-2">
        {sorted.map((p, idx) => (
          <motion.div
            key={p.id}
            layout
            className={`p-2 rounded-xl text-center border text-xs select-none ${idx<highlightCount?"border-emerald-500/60 bg-emerald-500/10":"border-slate-700 bg-slate-800/60"}`}
          >
            <div className="font-mono">P{p.id+1}</div>
            <div className="opacity-80">{format2(p.chips)}</div>
          </motion.div>
        ))}
      </div>
      <p className="text-xs mt-2 text-slate-400">
        Green frame: current top {highlightCount} by chips (advances to Regionals).
      </p>
    </div>
  );
}

function PoolsPanel({ history, params }:{ history: any[]; params: any; }) {
  const last = history[history.length-1];
  const label =
    last?.type === "cup" ? `Cup ${last.cupIndex}` :
    last?.type === "regional" ? "Regionals" :
    last?.type === "worlds" ? "Worlds" :
    last?.type === "decay" ? `Decay (-${params.decay_pct}%)` :
    last?.type === "tpc" ? `TPC ${last.tpcRound}` :
    last?.type === "s15_cup" ? `S15 Cup ${last.round}` :
    last?.type === "s15_regional" ? "S15 Regionals (48)" :
    "Not started";
  const pool = last?.pool ?? 0;
  return (
    <div className="bg-slate-900/60 rounded-2xl p-4 border border-slate-800">
      <h3 className="font-semibold mb-3">Latest Prize Pool & Fixed Rewards</h3>
      <div className="text-sm mb-2">Stage: <span className="font-semibold">{label}</span></div>
      <div className="h-3 w-full bg-slate-800 rounded-full overflow-hidden mb-2">
        <motion.div className="h-full bg-indigo-500" initial={{width:0}} animate={{width: Math.min(100, Math.sqrt(pool)/10*100)+"%"}} transition={{type:"spring"}} />
      </div>
      <div className="text-xs text-slate-300 space-y-1">
        <div>Pool (from fees): <span className="font-mono">{format2(pool)}</span></div>
        <div>Cup fixed y: <span className="font-mono">{params.y_cup}</span> (top half only)</div>
        <div>Reg fixed y: <span className="font-mono">{params.y_reg}</span> (all players)</div>
        <div>World fixed y: <span className="font-mono">{params.y_world}</span> (all players)</div>
        <div>TPC fixed y: <span className="font-mono">{params.y_tpc}</span> (all players)</div>
        <div className="opacity-70">
          Fee = current chips / z; z: Cup {params.z_cup}, Reg {params.z_reg}, World {params.z_world}, TPC {params.z_tpc}
        </div>
      </div>
    </div>
  );
}

function NotesPanel({ params, sizes }:{ params: any; sizes: any; }) {
  return (
    <div className="bg-slate-900/60 rounded-2xl p-4 border border-slate-800">
      <h3 className="font-semibold mb-3">Rules Overview</h3>
      <ul className="text-sm space-y-2 list-disc pl-5">
        <li>Three Cups: randomly draw {sizes.cupSlots} participants each time; <span className="underline">only the top half</span> receive the fixed reward y.</li>
        <li>Fee = current chips / z; all fees form the prize pool, which is distributed by <span className="underline">halving-band weights</span>.</li>
        <li>After three Cups, advance the top {sizes.reg} by chips to Regionals; <span className="underline">all players</span> receive fixed y.</li>
        <li>Worlds: select {sizes.ourWorldQuota} from our Regionals by weight points; fill to {sizes.world} with external entrants.</li>
        <li className="italic">S15 (TPC format): use last Regionals' 32 as pros for 3×TPC; pros skip Cups; 2× S15 Cups on non-pros; S15 Regionals (48) = 32 pros + top-16 non-pros by chips.</li>
      </ul>
      <p className="text-xs text-slate-400 mt-2">This is a simplified advancement flow; the chip distribution remains close to the real outcome.</p>
    </div>
  );
}

function Timeline({ history }:{ history: any[]; }) {
  const [collapsed, setCollapsed] = useState<{[k: number]: boolean}>({}); // key: event index → true=show only top 16
  return (
    <div className="mt-8 bg-slate-900/60 rounded-2xl p-4 border border-slate-800">
      <h3 className="font-semibold mb-3">Event Timeline</h3>
      <div className="space-y-3">
        {history.length===0 && <div className="text-slate-400 text-sm">No events yet. Click Cup/Regionals/Worlds or the S15 buttons above, or use “Auto Play”.</div>}
        {history.map((ev, idx) => {
          const label = ev.type === 'cup' ? `Cup ${ev.cupIndex}`
                       : ev.type === 'regional' ? 'Regionals'
                       : ev.type === 'worlds' ? 'Worlds'
                       : ev.type === 'tpc' ? `TPC ${ev.tpcRound}`
                       : ev.type === 's15_cup' ? `S15 Cup ${ev.round}`
                       : ev.type === 's15_regional' ? 'S15 Regionals (48)'
                       : '—';
          const rows = ev.rows || [];
          const showTop16 = !!collapsed[idx];
          const viewRows = showTop16 ? rows.slice(0,16) : rows;
          return (
            <motion.div key={idx} layout className="p-3 rounded-xl bg-slate-800/60 border border-slate-700">
              <div className="flex items-center justify-between">
                <div className="text-sm font-semibold">{label}</div>
                <div className="flex items-center gap-2">
                  <div className="text-xs text-slate-300">Pool: {format2(ev.pool)} &nbsp; Participants: {rows.length}</div>
                  <button
                    className="text-xs px-2 py-1 rounded-lg border border-slate-600 bg-slate-700/50 hover:bg-slate-700"
                    onClick={()=>setCollapsed(s=>({ ...s, [idx]: !s[idx] }))}
                  >
                    {showTop16? 'Show all' : 'Show top 16 only'}
                  </button>
                </div>
              </div>
              {rows.length>0 && (
                <div className="mt-3 overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead className="text-slate-400">
                      <tr>
                        <th className="text-left p-1">Rank</th>
                        <th className="text-left p-1">Player</th>
                        <th className="text-right p-1">Fee (chips/z)</th>
                        <th className="text-right p-1">Fixed y</th>
                        <th className="text-right p-1">Pool Share</th>
                        <th className="text-right p-1">Total Payout</th>
                        <th className="text-right p-1">Δ (Net Change)</th>
                        <th className="text-right p-1">Post-Event Chips</th>
                      </tr>
                    </thead>
                    <tbody>
                      {viewRows.map((row: any) => (
                        <tr key={row.rank} className="border-t border-slate-700/60">
                          <td className="p-1">#{row.rank}</td>
                          <td className="p-1">{row.name}</td>
                          <td className="p-1 text-right">{format2(row.fee)}</td>
                          <td className="p-1 text-right">{format2(row.y)}</td>
                          <td className="p-1 text-right">{format2(row.share)}</td>
                          <td className="p-1 text-right">{format2(row.payout)}</td>
                          <td className={`p-1 text-right ${row.delta>=0?"text-emerald-400":"text-rose-400"}`}>{format2(row.delta)}</td>
                          <td className="p-1 text-right">{format2(row.newChips)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div className="text-[10px] text-slate-400 mt-1">
                    Note: Δ = Total Payout − Fee = Fixed y + Pool Share − Fee.
                  </div>
                </div>
              )}
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

function NumberField({ label, value, onChange, step=1 }:{ label: string; value: number; onChange: (v:number)=>void; step?: number; }) {
  return (
    <label className="text-sm">
      <div className="mb-1 opacity-80">{label}</div>
      <input
        type="number"
        className="px-3 py-2 rounded-xl bg-slate-800 border border-slate-700 w-36"
        value={value}
        step={step}
        onChange={(e)=>onChange(parseFloat(e.target.value))}
      />
    </label>
  );
}

function ReadOnlyField({ label, value }:{ label: string; value: React.ReactNode; }) {
  return (
    <label className="text-sm">
      <div className="mb-1 opacity-80">{label}</div>
      <div className="px-3 py-2 rounded-xl bg-slate-800 border border-slate-700 w-36 opacity-80">{value}</div>
    </label>
  );
}

// Extra block: show each player's chip changes per event (matrix)
function ChangesMatrix({ history, totalPlayers=256, playersState }:{ history:any[]; totalPlayers?: number; playersState: any[]; }){
  const labels = history.map((ev)=> ev.type==='cup'? `Cup ${ev.cupIndex}`
    : ev.type==='regional'? 'Regional'
    : ev.type==='worlds'? 'Worlds'
    : ev.type==='tpc'? `TPC ${ev.tpcRound}`
    : ev.type==='s15_cup'? `S15 Cup ${ev.round}`
    : ev.type==='s15_regional'? 'S15 Regional (48)'
    : '—');
  const rows = Array.from({length: totalPlayers}, (_,pid)=>{
    const deltas = history.map((ev)=>{
      if(ev.type==='worlds'){
        const k = ev.allIdxs ? ev.allIdxs.findIndex((gid:number)=>gid===pid) : -1;
        return k>=0 ? ev.deltas[k] : null;
      } else {
        const k = ev.idxs ? ev.idxs.findIndex((i:number)=>i===pid) : -1;
        return k>=0 ? ev.deltas[k] : null;
      }
    });
    const finalChips = playersState?.[pid]?.chips ?? null;
    return { pid, deltas, finalChips };
  }).sort((a,b)=>(b.finalChips??-Infinity)-(a.finalChips??-Infinity));
  return (
    <div className="mt-8 bg-slate-900/60 rounded-2xl p-4 border border-slate-800 overflow-x-auto">
      <h3 className="font-semibold mb-3">All Players’ Δ by Event (Matrix)</h3>
      <table className="text-xs min-w-full">
        <thead className="text-slate-400">
          <tr>
            <th className="text-left p-1">Player</th>
            {labels.map((lb,i)=>(<th key={i} className="text-right p-1 w-24">{lb}</th>))}
            <th className="text-right p-1 w-28">Final Chips</th>
          </tr>
        </thead>
        <tbody>
          {rows.map(r=> (
            <tr key={r.pid} className="border-t border-slate-700/60">
              <td className="p-1">P{r.pid+1}</td>
              {r.deltas.map((d: number|null, j:number)=> (
                <td key={j} className={`p-1 text-right ${d==null?"text-slate-600":""}`}>
                  {d==null? "—" : <span className={d>=0?"text-emerald-400":"text-rose-400"}>{format2(d)}</span>}
                </td>
              ))}
              <td className="p-1 text-right font-mono">{r.finalChips==null?"—":format2(r.finalChips)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="text-[10px] text-slate-400 mt-1">
        “—” means did not participate; values are net change Δ for that event.
        Rows are sorted by Final Chips (descending).
      </div>
    </div>
  );
}

// ────────────────────────────────────────────────────────────────────────────────
// Tiny assertion helpers
function assertEq(a: any, b: any, msg: string){ const ok = a===b; return { ok, msg, detail: ok? "" : `${a} !== ${b}` }; }
function assertTrue(cond: any, msg: string){ const ok = !!cond; return { ok, msg, detail: ok? "" : `Condition false` }; }
function assertNear(a: number, b: number, eps: number, msg: string){ const ok = Math.abs(a-b) <= eps; return { ok, msg, detail: ok? "" : `${a} !~= ${b} (eps=${eps})` }; }

/* Self-tests (simple test cases) */
function TestsPanel(){
  const [result, setResult] = useState<any>(null);

  // helpers for band totals & monotonic checks
  const sum = (arr:number[]) => arr.reduce((a,b)=>a+b,0);
  const rangeSum = (arr:number[], s:number, e:number) => arr.slice(s-1, e).reduce((a,b)=>a+b,0);
  const pairsEqual = (arr:number[], s:number, e:number, eps=1e-9) => {
    for(let r=s; r<=e; r+=2){ if(Math.abs(arr[r-1]-arr[r])>eps) return false; }
    return true;
  };
  const nonIncreasing = (arr:number[], eps=1e-9) => {
    for(let i=1;i<arr.length;i++){ if(arr[i-1] + eps < arr[i]) return false; } return true;
  };

  const run = () => {
    const out: {ok:boolean; msg:string; detail?:string}[] = [];
    // Test 1: fixed sizes
    out.push(assertEq(FIXED_SIZES.population, 256, "Population = 256"));
    out.push(assertEq(FIXED_SIZES.cupSlots, 128, "Cup = 128"));
    out.push(assertEq(FIXED_SIZES.reg, 32, "Reg = 32"));
    out.push(assertEq(FIXED_SIZES.ourWorldQuota, 12, "Our region Worlds = 12"));
    out.push(assertEq(FIXED_SIZES.world, 48, "Worlds = 48"));

    // Test 2: state vs constants are different references (avoid direct mutation)
    const params2 = { ...DEFAULT_PARAMS } as any;
    out.push(assertTrue(params2 !== DEFAULT_PARAMS, "params is a copy, not same reference"));
    const sizes2 = { ...FIXED_SIZES } as any;
    out.push(assertTrue(sizes2 !== FIXED_SIZES, "sizes is a copy, not same reference"));

    // Test 3: createPlayers length & init values
    const ps = createPlayers(10, 100);
    out.push(assertEq(ps.length, 10, "createPlayers length = 10"));
    out.push(assertTrue(ps.every(p=>p.chips===100), "createPlayers initial chips = 100"));

    // Test 4: weights sum (128/32/48 designed to be 1.0)
    out.push(assertNear(sum(weightsCup(128)), 1.0, 1e-9, "Cup weights sum = 1"));
    out.push(assertNear(sum(weightsReg(32)), 1.0, 1e-9, "Reg weights sum = 1"));
    out.push(assertNear(sum(weightsWorld(48)), 1.0, 1e-9, "World weights sum = 1"));
    // New: S15-related helpers should also sum to 1
    out.push(assertNear(sum(weightsTPC(32)), 1.0, 1e-9, "TPC weights sum = 1"));
    out.push(assertNear(sum(weightsReg48(48)), 1.0, 1e-9, "S15 Regional(48) weights sum = 1"));

    // Test 5: tierRun conservation & output lengths
    const testTier = (N:number, giveHalf:boolean, y:number) => {
      const pl = createPlayers(N, 100);
      const idxs = Array.from({length:N}, (_,i)=>i);
      const w = weightsCup(N);
      const res = tierRun({ playersIdx: idxs, players: pl as any, z: 10, y, weights: w, giveYToTopHalf: giveHalf });
      const eligible = giveHalf ? Math.ceil(N/2) : N;
      const sumDelta = (res.deltas as number[]).reduce((a,b)=>a+b,0);
      out.push(assertNear(sumDelta, y*eligible, 1e-6, `ΣΔ = y*eligible (N=${N}, half=${giveHalf})`));
      out.push(assertEq(res.deltas.length, N, `deltas length = N (${N})`));
      out.push(assertEq(res.fixedY.length, N, `fixedY length = N (${N})`));
      out.push(assertEq(res.sharePool.length, N, `sharePool length = N (${N})`));
      const takeY = res.fixedY.filter((v:number)=>v>0).length;
      out.push(assertEq(takeY, eligible, `#receiving fixed y = eligible (${eligible})`));
    };
    testTier(8, true, 50);   // Cup-like: top half receives y
    testTier(8, false, 100); // Reg/World-like: all receive y

    // Test 6: rankBy produces unique ranks 1..N
    const vals = [3,1,2];
    const r = rankBy(vals).ranks; // 0→1st, 2→2nd, 1→3rd
    out.push(assertEq(r[0], 1, "rankBy: index0 should be 1st"));
    out.push(assertEq(r[2], 2, "rankBy: index2 should be 2nd"));
    out.push(assertEq(r[1], 3, "rankBy: index1 should be 3rd"));

    // Test 7: band totals & pairwise monotonicity are respected for each weight set
    const check = (label:string, w:number[], expect:{[k:string]:number}) => {
      const eps = 1e-9;
      const okNI = nonIncreasing(w, eps);
      out.push(assertTrue(okNI, `${label}: globally non-increasing`));
      // pairs and sums by bands
      if(expect["3_4"]!=null){
        out.push(assertTrue(pairsEqual(w,3,4,eps), `${label}: pairs equal 3–4`));
        out.push(assertNear(rangeSum(w,3,4), expect["3_4"], 1e-9, `${label}: Σ(3–4)`));
      }
      if(expect["5_8"]!=null){
        out.push(assertTrue(pairsEqual(w,5,8,eps), `${label}: pairs equal 5–8`));
        out.push(assertNear(rangeSum(w,5,8), expect["5_8"], 1e-9, `${label}: Σ(5–8)`));
      }
      if(expect["9_16"]!=null){
        out.push(assertTrue(pairsEqual(w,9,16,eps), `${label}: pairs equal 9–16`));
        out.push(assertNear(rangeSum(w,9,16), expect["9_16"], 1e-9, `${label}: Σ(9–16)`));
      }
      if(expect["17_32"]!=null && w.length>=32){
        out.push(assertTrue(pairsEqual(w,17,32,eps), `${label}: pairs equal 17–32`));
        out.push(assertNear(rangeSum(w,17,32), expect["17_32"], 1e-9, `${label}: Σ(17–32)`));
      }
      if(expect["33_64"]!=null && w.length>=64){
        out.push(assertTrue(pairsEqual(w,33,64,eps), `${label}: pairs equal 33–64`));
        out.push(assertNear(rangeSum(w,33,64), expect["33_64"], 1e-9, `${label}: Σ(33–64)`));
      }
      if(expect["top1"]!=null) out.push(assertNear(w[0], expect["top1"], 1e-12, `${label}: top1`));
      if(expect["top2"]!=null) out.push(assertNear(w[1], expect["top2"], 1e-12, `${label}: top2`));
      out.push(assertNear(sum(w), 1, 1e-12, `${label}: Σ(all)=1`));
    };

    // Cup (128)
    const wCup = weightsCup(128);
    check("Cup(128)", wCup, { top1:0.15, top2:0.09, "3_4":0.12, "5_8":0.16, "9_16":0.16, "17_32":0.16, "33_64":0.16 });

    // Regionals (32)
    const wReg = weightsReg(32);
    check("Reg(32)", wReg, { top1:0.20, top2:0.12, "3_4":0.16, "5_8":0.20, "9_16":0.32 });

    // Worlds (48)
    const wWorld = weightsWorld(48);
    check("World(48)", wWorld, { top1:0.16, top2:0.10, "3_4":0.14, "5_8":0.20, "9_16":0.24, "17_32":0.16 });

    // S15 Regionals (48)
    const wReg48 = weightsReg48(48);
    check("S15 Reg(48)", wReg48, { top1:0.15, top2:0.09, "3_4":0.14, "5_8":0.20, "9_16":0.25, "17_32":0.17 });

    // TPC (32)
    const wTPC = weightsTPC(32);
    check("TPC(32)", wTPC, { top1:0.18, top2:0.11, "3_4":0.16, "5_8":0.19, "9_16":0.36 });

    setResult(out);
  };

  return (
    <div className="mt-8 bg-slate-900/60 rounded-2xl p-4 border border-slate-800">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold mb-3">Self-tests</h3>
        <button className="px-3 py-2 rounded-xl bg-teal-600 hover:bg-teal-500 text-sm" onClick={run}>Run tests</button>
      </div>
      {result && (
        <div className="text-xs space-y-1">
          {result.map((r:any, i:number)=> (
            <div key={i} className={`p-2 rounded-lg border ${r.ok?"border-emerald-600/40 bg-emerald-500/10 text-emerald-300":"border-rose-600/40 bg-rose-500/10 text-rose-300"}`}>
              <span className="font-semibold">{r.ok? "PASS" : "FAIL"}:</span> {r.msg} {r.detail? `— ${r.detail}`:""}
            </div>
          ))}
        </div>
      )}
      {!result && <div className="text-slate-400 text-xs">Click “Run tests” to verify band totals, monotonicity, and accounting.</div>}
    </div>
  );
}
