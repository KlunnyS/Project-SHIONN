> Historical record. Use [the current navigation guide](../../NAVIGATION.md) for operational instructions.

> Superseded for construction by [the Puzzle Maker specification](../../GOAL_CHOICE_CHAMBER_SPEC.md) and [SVG layout](../../diagrams/goal_choice_v1_layout.svg). Custom colored symbols below are historical proposal details, not current asset requirements.

# Two-waypoint navigation chamber

Design proposal, 2026-11-14. Build and hand-verify before collecting data.
Use a new map name after checking existing VMF/BSP names; do not overwrite test1
or reuse frozen evaluation maps. Follow `CHAMBER_AUTHORING.md`.

## Layout

A flat enclosed room, 1536 units wide × 1792 long × 512 high. Coordinates below
are proposed design coordinates, not measurements of an installed map. X increases
right; Y increases from spawn toward the goals. Floor surface Z=0. Keep floor
solid and nonportalable, with no pits, stairs, cubes, buttons or required jumps.
Use even lighting and equally salient distinct landmarks above both goal areas.

```text
                       NORTH / +Y
             interior X = -768 .. +768
          +--------------------------------+
          |             EXIT               | Y=1664
          |                                |
          |    [ A ]              [ B ]    | Y=1280
          |     ^                  ^       |
          |     |                  |       |
          |     |    optional      |       |
          |     |    obstacle      |       | Y=768
          |     |                  |       |
          |       \              /         |
          |          \        /            |
          |             S                  | Y=256
          +--------------------------------+
                     SOUTH / Y=0
```

- Spawn S: player standing at X=0, Y=256; face +Y (Source yaw 90 degrees),
  pitch/roll zero. Use an authored spawn entity and verify eye position and
  collision height in-game; do not assume a player origin of Z=0 is safe.
- A region: X=-512..-256, Y=1152..1408, floor Z=0.
- B region: X=256..512, Y=1152..1408, floor Z=0.
- Place floor markings centered at (-384,1280) and (384,1280). Their image
  bearings from S are approximately ±20.6 degrees; verify both fit in the actual
  policy camera view. Mark A with a blue circular motif and B with an amber
  triangular motif; use both shape and color, matched size/brightness. Labels
  identify locations for collection; they are not a hard-coded action mapping.
- Use a single final doorway centered near (0,1664). It is an ordinary chamber
  exit, separate from the two waypoint objectives. Neither waypoint requires
  visiting the other, and neither should be on the mandatory exit path.
- Keep the first variant empty. Markings should not collide with or stop the
  player. Verify adequate clearance between targets, walls and doorway.

## Smallest useful variants

1. **Choice/open:** no obstacle. Same S and orientation, request either A or B.
   First collection is exactly one human trajectory to each region. Stop and
   review before expanding. Reaching the same final exit is not the paired task.
2. **Detour:** after the open pair passes, add one opaque pillar with footprint
   X=-128..128, Y=640..896, height 256. Both target landmarks must remain visible
   from S. Use one destination and demonstrate safe left/right pillar detours.
   The pillar is not directly on either ideal A/B line; register a separate
   centered rear waypoint if testing a truly blocked direct route. Do not claim
   the pillar forces a detour to A/B without checking actual paths.
3. **Recovery:** register a start near a pillar face, looking into it, with an
   already identified destination. Demonstrate retreat/sideways movement and
   resume toward that same goal. A start that never observed an invisible goal
   cannot test memory of that goal; provide a visible initialization segment.

Create visually and geometrically varied counterparts later: swap landmark
appearance, move goal locations and alter starts/obstacle arrangements. Entire
start/layout families stay in one split. Mirroring a familiar view alone is
insufficient evidence of unfamiliar-layout generalization.

## Instrumentation and validity

Keep the existing shared chamber-ready/failure/exit events for the final exit.
Do not wire both waypoint regions to the existing global SignalGoalReached():
it would score arrival at the wrong target as success. Initially record waypoint
arrival through verified player-position telemetry and declared region bounds,
requiring 0.5 seconds inside the requested region. If that telemetry path is not
validated, collection may be reviewed visually as development-only, but objective
paired evaluation is blocked. A future destination-tagged event interface can
replace this; it has not been implemented in this design.

Reload for each attempt, verify spawn pose/stationary state and fresh events,
perform input preflight, then restore/verify pose because preflight turns the
camera. Register map/VMF/BSP hashes, requested goal, start and destination bounds,
checkpoint/control settings and capture validity. Use the two-waypoint procedure
and pending development paths; do not automatically approve training membership.

## Authoring acceptance checklist

- [ ] Both markers visible at S at deployed resolution/FOV, equally recognizable.
- [ ] Both goals independently reachable with walking/turning only.
- [ ] Shared starts match measured position, orientation and object state.
- [ ] Distinct early trajectories to A versus B; no forced single corridor.
- [ ] Wrong-waypoint entry cannot emit a requested-goal success.
- [ ] Exit success, failure and ready events work independently of waypoint review.
- [ ] No inherited triggers from a copied chamber intersect spawn or the routes.
- [ ] Camera/input and recording checks pass before collecting the first pair.

This proposal requires user building/compilation and supervised validation. No map
has been generated, installed, launched or assigned training/evaluation membership.
