# NeuroSwarm v2 — Implementation Breakdown

Goal: turn the 8 PRD milestones into small, independently testable steps.
Rule of thumb for each step: **build it, see it, THEN move on.** Don't stack invisible logic.

---

## Phase 1 — Static World & Collision (PRD Milestone 1)
**Why first:** everything else (sensors, navigation, difficulty scaling) depends on obstacles existing and collision working. Build this with a *hardcoded* map first — no randomness yet.

1.1 Create `environment/obstacle.py` — a simple circular `Obstacle(x, y, radius)` class, no physics yet.
1.2 Hardcode 5–10 obstacles in a fixed layout and render them (dark gray, per PRD).
1.3 Create `physics/collision_solver.py` — detect fish-vs-obstacle circle collision.
1.4 Implement "stop completely" collision first (simplest correct behavior).
1.5 Replace with "slide along surface" — project velocity onto the tangent of the obstacle surface.
1.6 Test: drive one fish manually (arrow keys) into rocks from multiple angles — confirm sliding feels smooth, not jittery.
1.7 Confirm predator (rule-based, Prototype 1 version) also collides/slides correctly.

**Definition of done:** 100+ fish + 1 predator moving through a fixed rock layout, sliding smoothly, no tunneling through obstacles at high speed.

---

## Phase 2 — Procedural Map Generator (Milestone 2)
**Depends on:** Phase 1 collision working, since you need obstacles to validate against.

2.1 Create `environment/map_generator.py` with parameters from the PRD (width, height, obstacle count, size, min gap width, max density).
2.2 Start with pure random placement, no validation — just get obstacles appearing differently each run.
2.3 Add overlap prevention (don't spawn obstacles on top of each other).
2.4 Add minimum-gap-width validation — this is the trickiest part. Simplest approach: after placing an obstacle, flood-fill or raycast-check that key paths aren't fully blocked. Don't over-engineer — a basic "distance to nearest obstacle at spawn points" check is fine for now.
2.5 Wire up the `G` key shortcut to regenerate the map live.
2.6 Test: generate 20+ maps in a row, visually confirm none of them fully trap all fish/predator spawn points.

**Definition of done:** pressing `G` produces a new valid map every time, never a fully-blocked one.

---

## Phase 3 — Environmental Sensors (Milestone 3)
**Depends on:** Phase 1 (obstacles exist) — does NOT depend on Phase 2, so you can build this against your fixed test map to isolate bugs.

3.1 Implement raycasting in `physics/raycasting.py` — cast a ray from fish in a direction, return distance to nearest obstacle hit.
3.2 Add the 3 directional distance sensors: forward, left, right obstacle distance.
3.3 Add obstacle angle sensors: nearest obstacle angle, relative direction.
3.4 Add density/open-space sensors: obstacle density, nearest gap direction, distance to safe area, local open space.
3.5 Add visibility sensor/logic: does a ray from fish to predator (or vice versa) get blocked by an obstacle?
3.6 Build debug visualization for sensor rays (`S` key toggle per PRD) — this is essential for verifying sensors are correct before trusting them in training.
3.7 Test: manually place a fish behind a rock relative to the predator, confirm visibility sensor correctly reports "hidden."

**Definition of done:** all 22 inputs are computing and visually verified via debug rays — you should be able to *see* that a sensor value makes sense before you ever hand it to a neural network.

**Tip:** log/print one fish's full sensor vector each frame during testing. Silent sensor bugs are the hardest thing to debug once training starts.

---

## Phase 4 — Predator Pathfinding Upgrade (Milestone 4)
**Depends on:** Phase 1 (collision) + Phase 3 (raycasting reused for "direct path clear?" check).

4.1 Implement "direct path clear?" check using raycasting from predator to target fish.
4.2 If clear → predator moves straight at fish (existing Prototype 1 behavior).
4.3 If blocked → implement basic alternative routing. Simplest viable version: sample a handful of angles around the direct line, pick the first one with a clear-enough path, steer there.
4.4 Test: place predator and fish on opposite sides of a wall/rock cluster — confirm predator routes around it instead of pushing into the obstacle.

**Definition of done:** predator no longer gets stuck pressing into rocks; it visibly reroutes.

---

## Phase 5 — Neural Network & Fitness Upgrade (Milestone 5)
**Depends on:** Phase 3 (sensors must all be correct and stable first) — this is why sensor validation in Phase 3 matters so much.

5.1 Expand network architecture: 22 → 64 → 32 → 16 → 3, per PRD.
5.2 Wire the new 22-length sensor vector into the network input.
5.3 Add new fitness rewards (navigation success, narrow gap passage, obstacle-avoidance, visibility awareness) — add them **one at a time**, not all at once, so you can see which one is driving which behavior.
5.4 Add new penalties (obstacle collision, getting trapped, repeated collision) — same incremental approach.
5.5 Adjust mutation parameters as needed for the larger network (PRD flags this explicitly — larger genome may need smaller mutation rate or larger population).
5.6 Run a short evolution test (20–30 generations) on the *fixed* map from Phase 1 first, before introducing procedural maps — isolates "is the network learning" from "is the map too hard/unfair."
5.7 Once stable on fixed map, switch to procedural maps (Phase 2) and re-run.

**Definition of done:** fish show measurable improvement in survival/navigation over generations on both a fixed map and procedurally generated ones.

---

## Phase 6 — Difficulty Scaling (Milestone 6)
**Depends on:** Phase 5 (evolution loop working) + Phase 2 (map generator).

6.1 Add a generation counter check that increases obstacle count/size every 20 generations, per PRD's example table.
6.2 Make the scaling curve configurable (see Phase 8 config system) rather than hardcoded.
6.3 Test: run 60+ generations, confirm difficulty visibly ramps and fish fitness trend still makes sense against a harder world.

**Definition of done:** world visibly gets harder over generations without breaking the simulation.

---

## Phase 7 — Performance: Spatial Partitioning (Milestone 7)
**Depends on:** everything above working correctly first — this is an optimization pass, not new behavior. Do this once correctness is proven, so you're not debugging performance and logic bugs simultaneously.

7.1 Implement `physics/spatial_grid.py` — uniform grid (PRD recommends this over quadtree for simplicity).
7.2 Replace O(n²) neighbor checks (fish-fish, fish-obstacle) with grid-based lookups.
7.3 Benchmark before/after with 150 fish + 20 obstacles — confirm you're hitting the 60 FPS (render) / 300–500 FPS (training) targets from the PRD.
7.4 Re-run a short evolution test to confirm behavior is unchanged after the optimization (a correctness regression check, not just a speed check).

**Definition of done:** performance targets hit, and simulation behavior is provably identical to pre-optimization.

---

## Phase 8 — Config, Persistence & Analytics (Milestone 8 + "Technical Improvements")
**Can be done in parallel with Phases 5–7**, ideally started early since later phases benefit from it.

8.1 Config system: move tunable params (population size, mutation rate, obstacle density, sensor radius, difficulty curve) into a YAML/JSON file.
8.2 Deterministic mode: seed-based reproducibility for debugging.
8.3 Genome persistence: auto-save top genomes every N generations.
8.4 Simulation recorder: save per-generation stats, optional replay.
8.5 Analytics dashboard: survival avg, collision count, obstacle utilization, predator success rate, heatmaps.
8.6 Profiling tools: identify bottlenecks (useful right before Phase 7).

**Definition of done:** you can change any tunable parameter without touching code, reproduce a run by seed, and see generation-over-generation stats in a dashboard.

---

## Suggested Build Order (dependency-respecting)

```
Phase 1 (Obstacles + Collision)
      ↓
Phase 2 (Procedural Maps)  ─────┐
      ↓                          │
Phase 3 (Sensors)                │
      ↓                          │
Phase 4 (Predator Pathfinding)   │
      ↓                          │
Phase 5 (NN + Fitness) ←─────────┘
      ↓
Phase 6 (Difficulty Scaling)
      ↓
Phase 7 (Spatial Optimization)

Phase 8 (Config/Persistence/Analytics) — start early, run alongside 3–7
```

This roughly matches the PRD's own day estimates (3+4+4+3+4+2+5+3 = 28 days) but sequences them so each phase produces something you can visually verify before the next one builds on it — same "vertical slice" philosophy your Prototype 1 PRD used.

## A Few Watch-Outs Going In
- **Sensor correctness is the highest-leverage bug source.** A silently wrong sensor won't crash anything — it'll just quietly cap how well fish can ever learn. Spend real time on Phase 3.7's debug visualization.
- **Don't tune fitness weights and difficulty scaling at the same time.** If evolution looks bad, you want to know whether it's the rewards or the world getting harder — change one variable at a time.
- **Keep the fixed test map from Phase 1 around permanently** as a regression-testing tool, even after procedural generation works. It's your fastest way to isolate "is this a map problem or a network problem."
