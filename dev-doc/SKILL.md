---
name: darktable-review
description: Run a read-only, adversarial review of darktable changes. Use when reviewing a darktable commit, branch, pull request, working tree, or selected source paths for pixelpipe/cache defects, CPU/OpenCL divergence, image-format edge cases, arithmetic and colour errors, memory safety, serialization leaks, GTK event or worker-thread lifetime bugs, races, and module lifecycle regressions.
---

# Darktable Adversarial Review

## Review Contract

Perform the source analysis in the current session. Do not invoke another reviewer, wrapper, or background agent unless the user explicitly requests it.

Default to an independent, source-first pass. Do not open prior findings or review artifacts unless the user requests comparison, validation, deduplication, or continuation. Commit messages, tests, specifications, and project documentation are part of the source record. If prior findings are already known, disclose that in Coverage.

Keep the reviewed code read-only. Honor the requested base, range, file list, and focus, while following changed behavior through callers, callees, sibling CPU/OpenCL implementations, pipe variants, and framework code as needed.

Review introduced behavior. Treat all callers as part of the introduced surface when a shared helper changes observable state, ownership, identity, blocking, failure, or return semantics. Separate useful pre-existing defects from verdict-affecting findings.

## Establish the Scope

For a named base, verify it and review `git diff <base>...HEAD`. For an implicit working-tree review inspect:

```bash
git status --short --untracked-files=all
git diff --shortstat --cached
git diff --shortstat
git branch --show-current
```

Treat untracked files as reviewable. Read the relevant commits individually as well as the aggregate diff. Verify author claims and new invariant comments against the implementation and sibling paths.

### Designing vs. Hacking

The official guide warns against "saturday-afternoon projects that lack polish, disregard ergonomics and got their inner colour science wrong." Before diving into code review, assess whether the change follows a design process:
- Was the problem understood before coding? Are there traces of research, sketches, or specification?
- Does the solution match a real user need, or is it a half-baked "toy"?
- Is the approach consistent with how darktable's architecture is intended to work?

Flag findings where the implementation shows signs of hacking rather than design: missing edge cases, hardcoded values where parameters belong, ignored colour science, or UI that contradicts darktable's interaction model.

Read applicable `AGENTS.md` files. Use darktable's documentation to establish contracts, especially:

- `dev-doc/pixelpipe_architecture.md`: pipe variants, hashes, cache invalidation, ROI, processing order
- `dev-doc/IOP_Module_API.md`: `params_t`, `data_t`, lifecycle, tiling, and `commit_params()`
- `dev-doc/GUI.md`: GUI/worker boundaries and widget lifecycle
- `dev-doc/introspection.md`: parameter serialization and migration
- `dev-doc/maths.md`: matrix convention, white points, and colour helpers
- `src/iop/useless.c`: reference IOP boilerplate implementation
- `src/iop/CMakeLists.txt`: IOP registration in the build
- `tools/iop_dependencies.py`: IOP pixelpipe priority ordering

Beyond the project's own docs, the [official Developer's Guide](https://github.com/darktable-org/darktable/wiki/Developer%27s-guide) covers:
- Color management resources (Fairchild, ACES, handprint, Filmlight, etc.)
- Writing efficient SIMD/vectorization/OpenMP code
- Coding style conventions
- Module structure and boilerplate

Search nearby implementations and history for the intended invariant. A suspicious line is not a finding until reachability, state transition, and impact are established.

## Review Strategy

State the change's apparent aim, then choose the smallest relevant execution matrix capable of disproving it. Common darktable dimensions are:

- CPU, direct OpenCL, CPU/OpenCL tiling, and CPU fallback after OpenCL rejection or failure
- full/canvas, preview, preview2, thumbnail, export, CLI, and transient pipes
- cold/warm cache, cache disabled, blend-cache hit/miss, cancellation, and restart
- Bayer, X-Trans, 4-Bayer, sRaw/linear, monochrome, and ordinary RGB
- one-, three-, and four-channel storage; integer and float input
- tiny, odd, border-touching, cropped, rotated, zoomed, and very large images
- fresh import, history load, reset, undo/redo, style/preset application, and pasted history
- GUI attached, detached, or not realized; in-memory library; optional OpenCL/OpenMP/loader absent
- allocation, upload, kernel, readback, cancellation, and partial-initialization failures

Treat removed or relocated invalidation, release, copy-back, state reset, migration, and guard operations as high-signal lines. Map every old case to its replacement rather than assuming either that the deletion is wrong or that responsibility moved safely.

Trace state through the complete path. Compare against the closest correct sibling. Check every early return and every place results become visible or cacheable.

## General C Review Under darktable's Architecture

Ordinary C review — bounds, allocation size, NULL, integer conversion, string handling, ownership and lifetime — always applies. Concentrate it where darktable's architecture makes those classes both likely and consequential, because the deciding facts live several layers away from the edited line:

- Buffer extents come from ROI, pipe geometry, tiling, and vector/channel layout, not from the local expression.
- Cache entries are reference-counted and borrowed; correctness of a free or a dereference depends on the acquisition protocol.
- `params_t` bytes are persisted and hashed, so uninitialized or stale bytes escape the process.
- GLib/GTK containers, model wrappers, and widget helpers transfer ownership implicitly, so the free that matters is usually not the one in the diff.
- OpenCL objects have distinct ownership on success, fallback, cancellation, and error paths.
- Module lifecycle and persistent metadata give freed or stale state a second life across pipes, sessions, and files.

Ownership transfer deserves an explicit lifetime line — source construction, consumer(s), free — for each allocation. Any free preceding a consumer is a defect even if the consumer duplicates. When a container is created with destroy notifiers (`g_hash_table_new_full(..., g_free, g_free)` and equivalents), the inserting code must own what it inserts: either the source relinquishes ownership (`g_slist_free()` on the container alone) or the insert duplicates. Verify both sides, and check that one allocation is not handed to two owning containers.

## High-Value Bug Hunts

### Pixelpipe, Cache, Hashes, and Raster Masks

Treat cached output as coupled state: pixels, identity/hash, ROI, format and channels, colourspace descriptor, ownership, host/device validity, and importance. Mutation, failure, and publication must keep the whole tuple coherent.

- Prove every output-affecting dependency is represented by the hash or by an invalidation/lifetime invariant. Also reject identity fields that vary by installation or invocation but cannot affect output; they create permanent misses and stale on-disk artifacts.
- When a hash helper gains modes such as excluding a module, omitting ROI, or omitting blending, derive the identity for each mode. Confirm producers and consumers use the same mode and the same pipeline-node identity; do not confuse node position, IOP order, history position, and module-list index.
- An auxiliary cache inherits none of the pixelpipe cache's rules. Check all invalidation entry points, the global cache-disable mode, transient display modes excluded from the key, focus/topology changes, invalid-hash sentinels, failure paths, and host/device storage across tiled and non-tiled pipes.
- If a narrow cache deliberately omits state, prove that state cannot change while an entry is reusable. In particular, modules that process drawn forms internally (`IOP_FLAGS_NO_MASKS`) are not protected by a blend identity that omits mask geometry.
- In-place transformation of a cache-backed input requires either a new identity or invalidation, even if the format/colour descriptor is updated. Check reuse by a different downstream topology and CPU/OpenCL asymmetry.
- A cache line reserved before processing must not become valid after processing, upload, kernel, or readback failure. Trace whether fallback has an authoritative host input and whether an old run can publish after cancellation or invalidation.
- Treat one-shot hints and restart flags as consumed state. Identify who clears them, which side effect each requests, and whether a restart re-reads flags set during the preceding run before publishing.
- Verify the placement and polarity of shutdown/cancellation tests at every new call site. They belong at the entry of recursive processing and before heavy work, not between a fast cache lookup and its return, and their sense is easy to invert when the result is returned directly.
- Use the intended pipe predicate. Main canvas, navigation preview, all screen pipes, export, and transient pipes are different sets; verify the chosen predicate against the behavior or flag being implemented.
- If readiness depends on pipe-global state completed only after all modules commit, re-evaluate it at execution time. A decision made in `commit_params()` can be stale before the first tile runs.
- For raster masks, trace source module and mask identity, source/target ROI, coordinate transforms, intermediate dimensions, per-pipe registration, and ownership. Verify invalidation on history load, undo/redo, retargeting, disabling, and first use.

### Image and Mipmap Cache Handles

- Pair darktable cache acquisition and release on every path, including unchanged-result exits and failed lookups. Check NULL before copying or dereferencing an entry.
- Match read/write acquisition to the corresponding release and persistence semantics. A struct copied from a cache entry does not make its pointer fields owned; borrowed data cannot outlive release.
- Audit helpers reachable with `NO_IMGID`, during startup/shutdown, or without a conventional GUI image.
- For cached derived fields whose sentinel means "unknown", every mutator must recompute or restore the sentinel, and every reader must honor it. Emit change signals only when the semantic value changed.

### CPU, OpenCL, and Tiling

Treat `process()`, `process_cl()`, kernels, and tiling callbacks as implementations of one observable contract.

- Compare algorithm branches, guards, defaults, channels, boundary handling, colour transforms, metadata/descriptor updates, and side effects. Exercise values that select each implementation branch.
- Verify OpenCL and tiling readiness match supported parameter and image variants. Unsupported OpenCL cases must request the caller's CPU-fallback path (`DT_OPENCL_PROCESS_CL`); operational failures must not masquerade as unsupported modes. Do not flag a module that deliberately has no OpenCL implementation for a mode.
- Track authoritative host/device data through upload, in-place operations, readback, failure, and fallback. Incomplete output must be invalidated rather than published or cached.
- Match `cl_mem` ownership across direct, tiled, fallback, cancellation, and early-return paths. Check that pre-tiling full-image allocation does not defeat tiling.
- For spatial filters, derive overlap from the maximum footprint and compare CPU/OpenCL border behavior. A tile boundary is not necessarily an image boundary; clamping or reflecting there can create seams.
- `process()` may run once per tile. Updates to pipe-global descriptors or coefficients must be idempotent rather than accumulate per invocation.
- For manual OpenMP partitioning, derive work ranges from loop iterations, not worker identity. Thread numbers are suitable only for indexing correctly sized per-thread scratch and can change under nesting or scheduling.
- For OpenCL API findings, cite the relevant rule and show the device/path that reaches it. Formula differences alone are insufficient; establish an output, failure, or fallback difference.

### Official Vectorization and Performance Guidelines

The official developer's guide specifies how darktable code should be written for auto-vectorization and OpenMP efficiency. When reviewing `process()` loops, check for:

- **Base-pointer + index addressing**, not implicit pointer increments. The address must depend only on loop counters, not on the previous iteration, to enable SIMD and parallelization:
  ```c
  // good — index-only addressing
  float *const restrict image = (float *)in;
  for(size_t k = 0; k < height * width; k++)
    image[k] = whatever;

  // bad — pointer carry prevents vectorization
  float *pixel = (float *)in;
  for(size_t i = 0; i < height; i++)
    for(size_t j = 0; j < width; j++)
      { *pixel = whatever; pixel++; }
  ```
- **Flat indexing** (`for(size_t k = 0; k < ch * width * height; k += ch)`) preferred over nested width/height/channel loops.
- **No struct arguments inside loops** — unpack struct members into local scalars/arrays before the loop. Compilers cannot vectorize structures, only `float`/`int` arrays.
- **`restrict`** on all image/pixel pointers to eliminate aliasing. `*out` must never alias `*in`.
- **`const`** on input pointers and loop-invariant values to prevent false sharing in parallel regions.
- **`#pragma omp simd`** + `#pragma omp declare simd` on inner helpers; use `reduction` clauses for accumulators.
- **`collapse(2)`** on nested width/height loops so OpenMP splits iterations evenly.
- **Alignment**: arrays on 64-byte boundaries, pixels on 16-byte boundaries (`DT_ALIGNED_PIXEL`).
- **No type casts** in hot paths.
- **Branches**: use ternary expressions (`(x > 0) ? x : -x`) that compile to SIMD mask instructions, not `if/else` that break vectorization.

### Coding Style

Review formatting against the official style, enforced by `tools/beautify_style.sh` (via `clang-format`):

- American English spelling, especially for user-visible strings
- Spaces, not tabs; `shiftwidth=2`
- No trailing whitespace
- Braces `{` and `}` on their own lines
- Line length ≤ 90 characters
- Function parameters each on their own line
- Complex boolean operators (`||` / `&&`) at line start, one per line
- SQL formatting preserved as multi-line string literals

Style violations alone are not findings (per Finding Bar), but consistent disregard for project conventions is a signal of insufficient polish. When reviewing new IOPs, confirm they follow the module template at `src/iop/useless.c`.

### ROI, Buffer Layout, Image Classes, and Colour

- Size and fill a temporary buffer from the same ROI and scale domain. Do not mix `piece->buf_in` dimensions with a zoomed `roi_in`, or cache transform scales in shared `piece->data` when GUI and processing paths can use different ROIs concurrently.
- Never infer one pipe's geometry by scaling another pipe's dimensions. Modules round and transform independently; use each pipe's actual dimensions and the distortion transforms for cross-pipe coordinates.
- Derive allocation size from channel count, stride, vector width, neighborhood, rounding, and the final accessed index. Darktable's four-lane helpers can touch a fourth element of nominally three-component LUT, mask, accumulator, or device storage; provide and initialize padding when the actual access requires it.
- Check downscaled buffers and odd dimensions against the access pattern, not nominal integer division. Test exact vector multiples as well as remainders.
- For small images and thumbnails, derive reflection and overlap bounds when the filter radius approaches an axis. Do not assume left and right edge regions remain disjoint.
- Do not equate raw flags, `filters == 0`, "linear", or one classifier with a fixed buffer layout. Trace actual channel/format descriptors for 4-Bayer, sRaw/computational raw, monochrome, Bayer, X-Trans, and ordinary raster input.
- Treat sensor active-area and crop metadata as coordinate claims, not automatically as the processing boundary. Confirm the boundary used by raw spatial filters excludes non-image sensor regions for the relevant loader and camera class.
- Track declared and actual colourspace through every boundary. Check white point, chromatic adaptation, transposed matrix convention, profile absence, gamut handling, fourth-channel preservation, and CPU/OpenCL parity.
- The official guide's color-management references cover the underlying science. Consult these to distinguish correct colour transforms from naive or incorrect ones:
  - Fairchild, *Color Appearance Models*: colour appearance and chromatic adaptation
  - ACEScentral: industry-standard colour encoding and interchange
  - handprint.com: comprehensive colour science resource
  - Filmlight white papers: standard colour spaces and scene-referred workflow
  - VES *Cinematic Color*: colour management for visual effects pipelines
- For NaN/negative containment changes, find every producer variant and the earliest operation that spreads invalid values. Compare CPU/OpenCL placement and prove downstream behavior; do not impose a universal clamp on image data whose valid domain permits negatives.
- Before converting float geometry to an integer allocation or loop bound, validate the final derived value as finite and representable. Trace subsequent arithmetic for overflow using reachable parameter and image values.

### Params, Defaults, History, and Compatibility

- Keep GUI state, `self->params`, and pipe-local `piece->data` distinct. Processing uses deterministic data committed for that pipe; GUI state may disappear or change concurrently.
- `params_t` is persisted as bytes. Layout changes require the correct version and migration, no pointers, and deterministic initialization of every serialized byte. Fixed-size text fields in serialized or hashed structs must clear their unused tail: use `dt_strlcpy_to_fixed()` (`src/common/utility.h`), since `g_strlcpy`, `snprintf`, and `strncpy` all leave bytes past the terminator unchanged. This applies to `legacy_params()` destinations too.
- Verify `legacy_params()` against the old consumer's actual unit, range, and semantics, not the field name. Check untouched fields whose meaning changed, removed fields now derived elsewhere, and chained migrations.
- Sanitize image-dependent enum values at every foreign-params entry point: history, styles, paste, presets, and defaults. Ensure processing params and GUI choices remain consistent when the stored value is invalid for the current image class.
- Image-class predicates must agree across default enablement, `reload_defaults()`, preset auto-apply and `FOR_*` filters, module groups, and processing guards. Compare fresh import with reset and history/style application on edge image classes.
- Use `reload_defaults()` for per-image defaults. When modules communicate through pipe-global state, account for reverse-order default loading and forward-order parameter commits.
- An output-affecting algorithm rewrite can change old edits without changing `params_t`. Require an explicit compatibility decision and, when compatibility is intended, gate it on persisted edit state rather than live GUI/configuration state.
- Pair custom pipe/global initialization with complete cleanup of sub-allocations and OpenCL resources. Check duplication, disabled/skipped modules, focus, blending, raster masks, picker state, and QAP reparenting when touched.

### Persistence, Loaders, and Metadata

Treat database, history, preset/style, XMP/EXIF, image-header, cache-file, and exported metadata bytes as externally observable.

- Compatibility includes persisted identifier collections such as style module order and workspaces. Normalize missing, removed, renamed, duplicated, newly introduced, and multi-instance entries at a shared read boundary used by editing, preview, export, and CLI.
- Derived on-disk artifacts such as compiled kernels or model/backend results need semantic identity: source content/version, schema, producer/runtime, backend/device, relevant configuration, and resolved shape. Invalidate dependent artifacts before publishing replacement source state.
- Image loading uses a result-code protocol. Trace recognized/unsupported, wrong-loader, corrupt, and fatal results through fallback dispatch. Where container and EXIF/XMP provide the same property, define precedence and whether a transform was already applied.
- Never derive machine behavior from localized or human-readable text. Decisions must read the numeric or enum metadata value, not a library's pretty-printed interpretation of it, and identifiers reaching the database, filenames, presets, or shortcuts must be stored untranslated with translation applied only at display time. Distinguish absent from present-and-false.
- Metadata policy follows darktable identities and hierarchy. Preserve exact image-version identity through export and variable expansion; check hierarchical tag policy and compare GUI, batch, CLI, duplicate-image, and sidecar paths.
- Validate stored enum/range values before processing. Check metadata relocation/removal, type preservation, truncation, locale-independent conversion, and database prepare/bind/step/finalize behavior.
- Exiv2 XMP operations share namespace state across workers. Verify darktable's process-wide synchronization covers every decode/encode path, including import, export, and sidecar work.

### Jobs, Delayed Work, and Derived Views

- A long-running job must snapshot every output-affecting setting before queueing. Workers and per-item callbacks must not reread live widgets, configuration, selection, active preset, or mutable module data. Establish deep-copy ownership and immutability.
- Delayed work must carry explicit target identity. Distinguish active image, selection, hovered image, job-origin image, module instance, view, and image version; capture a generation when results can become stale and validate it before publication.
- A database or pixelpipe mutation does not automatically refresh thumbnails, overlays, tooltips, collection queries, labels, or cached GUI models. Enumerate consumers and verify the signal requests the needed action: repaint, recompute, database reload, query rebuild, or pipe restart.
- A multi-output backend succeeds only if every promised output has valid shape, allocation, conversion, and copy. Runtime shape must drive allocation, metadata, cache identity, and cleanup; partial results must not be cached, published, or reported as success.
- Trace queued callbacks through GUI destruction. Payloads need explicit ownership and target identity; callbacks must tolerate detached modules/views and must not retain GUI-owned state across teardown.
- For stop/cancellation state machines, enumerate requests while idle, during reset/restart, and across compare-and-swap failures. Prove whether a dropped request merely wastes work or can publish stale state or touch destroyed data.

### GTK Events and Module GUI Lifecycle

- GTK belongs on the main thread. Pixelpipe and background jobs must communicate through owned messages whose callbacks validate that the target module/view generation still exists.
- Darktable module widgets move between module panels, QAP, popovers, and hidden states. Trace focus, signal/controller connections, parent ownership, stored widget pointers, and teardown/rebuild order across reparenting.
- Model filters, sorters, completions, and views remain observers while referenced. Detach/destroy observing wrappers before bulk store mutation; release creator references after GTK takes its own, or repeated rebuilds retain stale observer graphs.
- For canvas events, verify smooth and discrete paths separately, including accumulator success, x/y dominance, modifiers, event consumption, and handoff among masks, module handlers, pan/zoom, and parent widgets.
- Keep widget, oriented-image, sensor, ROI, zoom, and mask coordinates distinct. Validate transient zero-sized allocations and pipe changes during expose/drawing rather than assuming steady-state geometry.
- Preserve history semantics: programmatic GUI synchronization adds no history, one user action adds exactly one entry, and picker/histogram updates target the correct module, pipe, and generation.

### Supported Absence and Platform Variants

- Trace touched code with GUI absent/detached/not realized, `darktable-cli`, `--library :memory:`, no attached develop pipe, and optional OpenCL/OpenMP/loader support compiled out. Guard absence at the helper that owns the assumption, and keep partial state and cleanup coherent.
- Lazy platform GUI services need the headless guard at their initialization site. Persisted window geometry must be validated against the current displays; check the relevant Wayland/X11 behavior rather than assuming one backend's coordinate or window-management contract.

## Common False Positives to Eliminate

Before reporting, actively disprove the claim:

- Search all callers; cleanup, reset, invalidation, synchronization, or fallback may occur one level above.
- Prove the disputed image class, pipe, mode, enum, and failure state pass caller guards.
- Prove ownership and lifetime before proposing release or alleging use-after-free.
- Prove same-identity/different-output reuse before requiring invalidation; prove observability before rejecting a key field as redundant.
- Check aliasing before demanding a second write: updating a descriptor through a pointer may already update the cache's stored copy.
- Distinguish authoritative GPU-only data from stale host storage and verify actual fallback behavior.
- Account for pipe, history, cache, and busy mutexes plus atomic reset protocols before reporting a race. Give a concrete interleaving.
- For queued-work lifetime claims, inspect entry guards before lock acquisition and state rechecks after queueing; the callback may abort cleanly after teardown. Show that teardown passes those guards and reaches the disputed dereference.
- Bound arithmetic and invalid-value impact through callers, compiler helpers, clamps, and final conversion. Give representative values or derive the accessed index.
- Distinguish wrong pixels, wrong persisted/exported data, crash, hang, leak, needless recomputation, stale UI, diagnostics, and unreachable code. Severity follows demonstrated impact and reachability.

## Categories Outside the Default Hunt

Do not spend review effort on generic authentication, authorization, sessions, tenant isolation, CSRF, web routing, distributed systems, deployment rollback, secret management, or remote-service checklists unless the scoped change introduces such a boundary. These exclusions must not suppress local equivalents that do matter: GUI blocking, cancellable background work, external-library failure, file import/export, database persistence, and metadata serialization.

## Finding Bar

Report only material, defensible issues. Skip style, speculative cleanup, one-off robustness advice, and semantic differences without demonstrated impact.

- `high`: demonstrated memory corruption or persistent data loss, silent wrong export/persistence, or a crash/hang in an ordinary workflow
- `medium`: deterministic wrong rendering/cache state in a restricted configuration, a narrow crash, recoverable hang, materially accumulating leak, or concrete bounded race
- `low`: reproducible localized UI, diagnostic, or performance regression without wrong persisted output or realistic escalation

Reachability overrides the defect-class label: corruption-class impact reachable only through a narrow interleaving, or a pre-existing race the change merely widens, is `medium`. Record suspicious but unproven or unreachable cases in Coverage rather than inventing a low-severity finding.

For each finding:

1. Name the exact changed location.
2. State the violated darktable invariant.
3. Trace a reachable trigger through callers and state transitions.
4. Explain concrete impact and affected pipe, device, image, GUI, or persistence variants.
5. Cite the sibling path, base behavior, project contract, or external specification establishing the error.
6. Recommend the smallest correction class without editing the code.

For concurrency, provide an interleaving. For arithmetic, provide representative values. For bounds, derive the index. For OpenCL rules, identify the specification requirement. State uncertainty and validation limits.

## Output

Return findings first, ordered by severity:

```markdown
**Findings**
- `high|medium|low` [path:line]: Title
  Trigger, violated invariant, execution path, impact, and minimal recommendation.

**Verdict**
`needs-attention` or `approve`: terse ship/no-ship assessment.

**Coverage**
Reviewed scope and relevant pipe, device, image-format, failure, persistence, or lifecycle variants.
Evidence provenance: `independent/source-only` or `artifact-assisted`; identify prior material and when it was consulted.
Untested or unresolved areas and central high-risk paths checked without a material issue.
```

Use exact paths and line numbers. Keep no-issue coverage terse. Do not describe static analysis as proof of correctness or inflate performance/diagnostic issues into corruption or security claims.
