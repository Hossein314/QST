# Quick Select

A behavioural clone of Photoshop's Quick Selection Tool, written from scratch in
Python. You paint over an object with a brush, the selection floods out to
similar pixels and stops at image edges, and you refine it by adding and
subtracting strokes.

No Adobe code, assets or APIs are used. Everything is implemented from published
research — the papers are listed at the bottom.

```
python run.py                 # start empty, then Ctrl+O
python run.py photo.jpg       # open an image straight away
```

There is a second application in this repository built on the same engine: a
COCO instance-segmentation annotator for electronic boards.

```
python annotate.py /path/to/images
```

It uses a stricter selection model in which user brush marks are permanent
constraints and the mask is a pure function of them. See **[ANNOTATOR.md](ANNOTATOR.md)** for that tool, its JSON output format, and the speed/quality
parameter table.

---

## 1. What makes it feel like the real thing

Getting graph-cut segmentation to work is easy. Getting it to *feel* like Quick
Selection is the interesting part, and it comes down to three behaviours.

**The solve is local.** Painting never re-segments the image. Each brush step
optimises a bounded region around the bristles, which is the core idea of
*Paint Selection* (Liu, Sun & Shum, SIGGRAPH 2009). A stroke in the top-left
corner cannot change what is selected in the bottom-right. That is what makes
the tool predictable, and incidentally what makes it fast.

**Background is inferred, never asked for.** You only paint what you want. The
engine treats the ring just outside the current reach as background: it both
bounds the flood and supplies the negative colour samples. Alt-painting adds
explicit background seeds on top of that.

**Growth is monotone within a gesture.** In Add mode the local result is
*unioned* into the selection. The selection never flickers backwards while you
drag — which is the single biggest difference between a tool that feels solid
and one that feels twitchy.

---

## 2. The algorithm, step by step

### 2.1 State

Three arrays, all at the *interactive* resolution (the image downscaled so its
longest side is 640 px):

| array | meaning |
| --- | --- |
| `trimap` | `FOREGROUND` where you painted, `BACKGROUND` where you Alt-painted, `UNKNOWN` elsewhere |
| `selection` | the current binary selection |
| `full_alpha` | the full-resolution soft alpha, produced on mouse release |

### 2.2 Energy

Every solve minimises the standard interactive-segmentation functional
(Boykov & Jolly 2001; GrabCut 2004):

```
E(a) = λ · Σ_p D_p(a_p)  +  Σ_(p,q)∈N  w(p,q) · [a_p ≠ a_q]

D_p(FG) = −log P(I_p | foreground model)
D_p(BG) = −log P(I_p | background model)
w(p,q)  = γ · exp(−β ‖I_p − I_q‖²) / dist(p,q)
```

`β = 1 / (2·E[‖I_p − I_q‖²])` is the adaptive normaliser from GrabCut. It makes
the contrast term scale-free, so the same `γ` works on a flat studio shot and on
a busy street scene.

The pairwise term is submodular, so one s-t min-cut gives the exact global
minimum. It is solved with Boykov–Kolmogorov via PyMaxflow.

**Terminal convention.** Source = foreground, sink = background. Cutting a
pixel's source edge assigns it to the background, so the source capacity is the
*cost of labelling it background*, `D_p(BG)`. PyMaxflow's `get_grid_segments`
returns `True` for the sink side, hence `foreground = ~segments`.

### 2.3 One brush step

```
stamp brush → update trimap → choose region → infer background band
            → fit colour models → build graph → min-cut → clean up → merge
```

**Choosing the region.** Start with a reach of about three brush radii. Solve.
If the resulting foreground presses against the edge of that region, the region
clearly does not contain the whole object, so grow it (×2.2) and solve again, up
to three times. Painting inside a large flat object therefore leaps out to its
real edges; painting on a small one converges on the first try and pays nothing
extra. Regions above ~22k pixels are solved at half resolution, which cuts the
node count by four — the boundary is half a pixel coarser, which nothing
downstream can see, because mouse-release re-cuts it at full resolution anyway.

**Inferring background.** Pixels beyond `reach + band` are pinned to background.
Note what is *not* done: the reach boundary itself is not a hard wall. Min-cut
prefers short boundaries, so a wall of background right next to the bristles
would make the cheapest cut the one that hugs the brush, and the selection would
never grow at all. Instead the wall sits a full band further out and the result
is clipped to the reach afterwards.

**Trusting the background model.** The negative colour samples come from a ring
just outside the allowed region. When the brush is deep inside a large object,
that ring is *still object* — a model fitted to it would be indistinguishable
from the foreground model, and the data term would carry no information at all.
The engine detects this by scoring the negative samples under the *foreground*
model: if they look like foreground, the background model is discarded and the
cost of labelling a pixel background becomes a constant (`fg_bias`). The cut
then reduces to a plain likelihood threshold with edge stopping — an
edge-stopped flood fill, which is exactly the right fallback.

This one check is the difference between a tool that expands to fill an object
and one that selects nothing but the bristles.

**Colour models.** Five-component full-covariance Gaussian mixtures in CIE Lab,
fitted with k-means++ seeding plus four EM iterations on at most 4000 samples.
They are refit on *every* brush step, so the sample cap is the single most
important latency knob.

A smoothed 3-D histogram is available as an alternative
(`EngineConfig.model = "hist"`) and is 2–4x faster per step. It needs two
things to work at all: the counts are blurred with a separable `[1,2,1]` pass
along each axis, because a few thousand samples in a 16³ grid is otherwise far
too sparse to discriminate anything; and its log-probabilities are normalised
against the model's own peak rather than against a uniform cube, since a
histogram bin is 20+ Lab units wide and its average density badly understates a
tight cluster's peak. With both in place it is calibrated within about 2% of the
GMM's cost scale, so the same `fg_bias` and `bg_trust_threshold` apply to
either. The GMM stays the default because it is the better-understood choice on
real photographs; if you want the speed, the histogram is genuinely usable.

**Subtracting.** The same machinery with the roles swapped: the stroke seeds
"remove", the untouched part of the selection seeds "keep", and the flood may
only eat into pixels that are currently selected and were not explicitly painted
as foreground. Subtracting never triggers region expansion — one careless stroke
should not remove far more than the brush covered.

### 2.4 On mouse release

1. Up-sample the interactive mask to full resolution.
2. **Narrow-band re-cut at full resolution.** Erode and dilate the mask to get a
   band around the boundary; pin the interior and far exterior; re-solve only
   the band. The reduction is exact, not approximate: an edge from a band pixel
   to a *fixed* neighbour cannot be cut on that side, so it collapses into a
   terminal edge of the same capacity. A 1080p image has millions of pixels but
   only a few hundred thousand near any boundary, which is what makes
   full-resolution refinement affordable at all.
   Because the interior is pinned, this can only move the boundary — it can
   never delete something you painted. That guarantee is why it is safe to run
   automatically.
3. **Auto-Enhance** (optional) runs a second, narrower band pass with the edge
   term tripled and the colour models dropped, so the boundary is pulled onto
   image gradients.
4. **Edge-aware alpha** via the guided filter (He, Sun & Tang 2010), which is
   the O(N) approximation to the closed-form matting Laplacian of Levin et al.
   (2008) — same local-linear-model assumption, solved with box filters instead
   of a sparse linear system.
5. **Refine Edge**: smooth, then contract/expand, then feather.

---

## 3. Setup

Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

or as a package:

```bash
pip install -e .
```

On Windows, if `PyMaxflow` has no wheel for your Python version you will need
the Microsoft C++ Build Tools, or an older Python (3.11 has wheels for
everything here).

---

## 4. Using it

### Tools and modes

| control | action |
| --- | --- |
| drag | paint the selection |
| `Shift` + drag | add to the selection |
| `Alt` + drag | subtract from the selection |
| `[` / `]` | brush smaller / larger |
| `{` / `}` | brush softer / harder |
| middle-drag, or `Space` + drag | pan |
| wheel | zoom at the cursor |
| `Ctrl` `0` / `Ctrl` `1` | fit on screen / 100% |
| `Ctrl` `Z` / `Ctrl` `Shift` `Z` | undo / redo (one step per stroke) |
| `Ctrl` `D` | deselect |
| `Ctrl` `Shift` `I` | invert selection |
| `Ctrl` `R` | re-run refinement with the current Refine Edge settings |
| `Ctrl` `S` | save the mask as PNG |
| `Ctrl` `Shift` `S` | save an RGBA cut-out |

The mode dropdown switches between New / Add / Subtract; after the first stroke
of a New selection it flips to Add, as Photoshop does.

**Auto-Enhance** costs roughly 100 ms on release and is worth turning on for
hair, fur and soft edges.

**Sample All Layers** controls which pixels the *selection* is computed from.
The canvas always shows the composite. Place extra layers with
`Ctrl` `Shift` `O`.

### The options that matter

Everything tunable lives in `quickselect/config.py` with a comment explaining
what it changes. The ones worth touching:

| setting | effect |
| --- | --- |
| `local_reach_factor` | how far the flood runs ahead of the cursor. Larger leaps further; smaller is more controlled |
| `gamma_smooth` | edge/contrast strength. Higher = smoother boundaries, slower solves |
| `lambda_data` | colour-likelihood strength relative to the edge term |
| `interactive_max_dim` | the during-drag resolution. The main speed/precision trade-off |
| `fg_bias` | how tolerant the flood is when the background model is untrustworthy |

---

## 5. Comparing against Photoshop

There is deliberately no scoring framework here. This is a deterministic
algorithm with nothing to train, so it is evaluated by looking at it.

The workflow:

1. Select an object in this app, `Ctrl` `S`, save into `predictions/`.
2. Select the same object in Photoshop, export the mask as PNG into
   `ground_truth/` under the *same filename*.
3. Generate diff images:

```bash
python tools/diff_masks.py predictions/ ground_truth/ diffs/ --images samples/
```

Orange is "only in yours", blue is "only in the reference", neutral is
agreement, and `--images` puts the source photo underneath so you can see what
the disagreement is sitting on.

4. Or compare interactively:

```bash
python tools/compare_viewer.py predictions/ ground_truth/ --images samples/
```

Yours on the left, the reference on the right, and a middle panel with a slider
that cross-fades between them. A boundary that has drifted by a few pixels is
invisible in side-by-side panels and obvious in a fade. Press `D` in the viewer
to switch the middle panel to the colour-coded difference; left/right arrows
step through pairs.

Files are paired by filename stem, and a trailing `_mask` is tolerated on either
side.

The same thing is available as a one-line function:

```python
from quickselect import mask_diff
diff = mask_diff("predictions/cat.png", "ground_truth/cat.png", image="samples/cat.jpg")
```

---

## 6. Using the engine without the UI

The engine has no Qt dependency.

```python
from quickselect import QuickSelectEngine, LayerStack, SelectionMode, load_image, save_mask

engine = QuickSelectEngine(LayerStack.from_image(load_image("cat.jpg")))
engine.tool.brush.diameter = 60
engine.tool.auto_enhance = True

engine.paint_polyline([(320, 240), (340, 250), (380, 265)], SelectionMode.NEW)
engine.paint_polyline([(500, 300), (520, 310)], SelectionMode.ADD)
engine.paint_polyline([(210, 180)], SelectionMode.SUBTRACT)

save_mask("cat_mask.png", engine.selection_alpha())
```

`paint_polyline` is the headless equivalent of a mouse drag — it stamps along
the path with the configured spacing, exactly as the canvas does.

---

## 7. Performance

Measured on a 1280×720 image, CPython 3.11, single core, no GPU:

| phase | GMM | histogram |
| --- | --- | --- |
| mouse-down (includes adaptive expansion) | ~80 ms | ~35 ms |
| each subsequent brush step | ~20–35 ms | ~8–14 ms |
| mouse-release refinement at full resolution | ~230 ms | ~230 ms |
| Auto-Enhance (additional) | ~100 ms | ~100 ms |

The canvas runs at 60 fps regardless, because the engine lives on its own thread
and the command queue coalesces: if the mouse produced ten move events while the
engine was busy, the worker drains all ten, feeds them to the engine in order so
stamp spacing stays correct, and emits one repaint. Selection update rate is
decoupled from frame rate — the brush cursor never lags.

On larger images the interactive resolution is capped, so the during-drag cost
is roughly constant; only the release refinement grows, and it grows with
boundary length rather than image area.

---

## 8. Assumptions and known limits

Stated plainly, since the brief asked for them:

* **Matting is approximated.** Full closed-form matting (Levin et al.) solves a
  sparse linear system over the unknown band. The guided filter makes the same
  local-linear assumption and is O(N), so that is what ships. For wispy hair
  against a busy background a true matting solve would be better.
* **Layers are simplified.** Source-over alpha only, no blend modes, no groups.
  Enough for "Sample All Layers" to mean something; not a layer engine.
* **Colour only.** No texture descriptors. Two regions that differ in texture
  but match in colour, with no gradient between them, will merge — and they will
  merge in Photoshop too.
* **Similar colours across a low-contrast edge will leak.** This is inherent to
  the energy, not a bug: if there is no edge and no colour difference, nothing
  in the model separates the regions. The answer is the same as in Photoshop —
  Alt-paint the part you do not want.
* **`intersect` mode is declared but not implemented.** `SelectionMode.INTERSECT`
  exists in the enum and is not wired up.
* **Two selection models coexist.** This app uses the Paint Selection model
  (bounded local flood, monotone growth within a gesture, selection as primary
  state). The annotator uses the constraint model (marks are permanent, mask is
  derived). They share the graph cut, the colour models and the canvas, but not
  the state machine. The constraint model is the better one; this app keeps the
  original because its feel is the point.
* **Single-threaded solves.** PyMaxflow is not parallel and there is no
  CUDA/OpenCL path. The obvious next speed-up is a parallel or GPU max-flow.
* **Undo granularity is one step per stroke**, stored as bit-packed snapshots
  (~300 kB per step at 1080p), capped at 60 steps.

---

## 9. Layout

```
quickselect/
  constraints.py  persistent user marks; the source of truth for the annotator
  segmenter.py    segment(image, constraints, params) -> mask, and its caches
  profile.py      per-stage timing hooks
  config.py       every tunable constant, with comments
  imagedata.py    layer stack, resolution pyramid, cached n-link weights
  brush.py        stamp kernels, spacing, stroke rasterisation
  colormodel.py   GMM (from scratch) and histogram colour models
  graphcut.py     graph construction, grid and narrow-band min-cut
  engine.py       the state machine: gestures in, selections out
  refine.py       guided filter, Refine Edge, contour extraction
  history.py      undo/redo
  diff.py         mask comparison images
  cli.py          batch diff command line
  io_utils.py     image loading/saving (Unicode-path safe)
  ui/
    app.py        entry point and theme
    mainwindow.py window, options bar, layers dock, shortcuts
    canvas.py     zoom/pan, brush cursor, overlay, marching ants
    worker.py     engine thread and coalescing command queue
    viewer.py     side-by-side comparison window
  annotator/
    polygon.py    mask -> COCO polygons
    dataset_io.py COCO JSON load/save, folder scanning
    session.py    per-image annotation state
    worker.py     segmentation thread
    app.py        the annotation window
tools/
  diff_masks.py          batch diff CLI
  compare_viewer.py      comparison viewer
  profile_segmentation.py  per-stage timing report
run.py               launch the selection app
annotate.py          launch the annotation app
ground_truth/        put Photoshop-exported masks here
predictions/         save your masks here
samples/             source images (optional, used as diff backdrops)
```

---

## 10. References

The implementation follows these; none of them is Adobe's.

* Y. Boykov, M.-P. Jolly. *Interactive Graph Cuts for Optimal Boundary & Region
  Segmentation of Objects in N-D Images.* ICCV 2001.
* Y. Li, J. Sun, C.-K. Tang, H.-Y. Shum. *Lazy Snapping.* SIGGRAPH 2004.
* C. Rother, V. Kolmogorov, A. Blake. *GrabCut: Interactive Foreground
  Extraction using Iterated Graph Cuts.* SIGGRAPH 2004.
* J. Liu, J. Sun, H.-Y. Shum. *Paint Selection.* SIGGRAPH 2009.
* Y. Boykov, V. Kolmogorov. *An Experimental Comparison of Min-Cut/Max-Flow
  Algorithms for Energy Minimization in Vision.* PAMI 2004.
* K. He, J. Sun, X. Tang. *Guided Image Filtering.* ECCV 2010.
* A. Levin, D. Lischinski, Y. Weiss. *A Closed-Form Solution to Natural Image
  Matting.* PAMI 2008.
