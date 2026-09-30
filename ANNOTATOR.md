# Board Annotator

Instance-segmentation annotation for electronic boards, built on the quick
selection engine. You paint an object, commit it, move to the next one; the
tool writes COCO polygons.

It has two modes, switched with `Tab`: **Annotate Mode** creates new instances
with the brush, **Edit Mode** lets you point at the instances already on the
image and delete or relabel the ones that are wrong. The number keys `0`-`9`
set the class in both modes.

```bash
pip install -r requirements.txt
python annotate.py /path/to/images
```

The folder is scanned for images (sorted by filename) and `annotations.json` is
read from and written to that same folder.

---

## 1. Workflow

1. **Open a folder.** `Ctrl+O`, or pass it on the command line. Existing
   annotations are loaded, so you can stop and continue later.
2. **Paint the object.** Drag to add. `Alt`-drag marks background. The
   selection floods to similar pixels and stops at edges.
3. **Commit** with `Enter`. The instance is frozen, its polygons are extracted,
   and the marks reset for the next object. The pixels you just committed
   become background marks, so the next instance cannot re-grab the board you
   already did.
4. **Repeat** for every board in the image.
5. **Move on** with `→`. Annotations for the current image are saved before the
   next one loads.

To fix something that is already annotated, press `Tab` for Edit Mode — see
[section 2](#2-edit-mode). The current mode is shown at the left of the status
bar, in square brackets.

### Keys

| key | action |
| --- | --- |
| `Tab` | switch between Annotate Mode and Edit Mode |
| drag | paint foreground |
| `Alt` + drag | paint background |
| `Shift` + drag | force foreground regardless of the mode dropdown |
| `Enter` | commit the current instance |
| `Ctrl` `Backspace` | delete the last committed instance |
| `Ctrl` `K` | clear the current instance's marks (keeps committed ones) |
| `Ctrl` `Z` / `Ctrl` `Shift` `Z` | undo / redo, within the current mode |
| click | *Edit Mode:* select the instance under the pointer |
| `Delete` | *Edit Mode:* delete the selected instance |
| `Esc` | *Edit Mode:* deselect |
| `0`-`9` | set the class: of the selected instance in Edit Mode, of the instance you commit next in Annotate Mode |
| `←` `→` | previous / next image (auto-saves) |
| `Home` / `End` | first / last image |
| `[` `]` | brush smaller / larger |
| `C` | show/hide constraint marks |
| `V` | show/hide committed instances |
| wheel | zoom at cursor |
| middle-drag, or `Space` + drag | pan |
| `Ctrl` `0` / `Ctrl` `1` | fit / 100% |
| `Ctrl` `S` | save everything now |

### What the colours mean

| colour | meaning |
| --- | --- |
| amber fill | instances already committed on this image |
| blue fill + marching ants | the instance you are working on |
| green | your foreground marks |
| red | your background marks |

Committed instances are pinned as background internally, but that is not drawn
in red — only marks you painted yourself are, or the image would disappear
under a wash of red as you worked.

Edit Mode replaces all of these with one colour per existing instance.

---

## 2. Edit Mode

`Tab` switches between the two modes; the status bar reads `[ANNOTATE]` or
`[EDIT]`. The modes are kept apart on purpose:

* In Annotate Mode the canvas paints. Nothing can select or delete an existing
  instance, and `Delete` does nothing at all.
* In Edit Mode the canvas cannot start a stroke. Clicking selects; it never
  creates an annotation, and no click reaches the segmentation engine.

Switching modes does not touch the annotations, and the brush marks for an
instance in progress survive the round trip.

### Seeing the instances

Every instance already on the image is drawn in the colour of its class (see
[Telling the classes apart](#telling-the-classes-apart)), filled and outlined
according to what the pointer is doing:

| state | fill | border |
| --- | --- | --- |
| normal | very light, the image stays readable | thin, 1.2 px |
| hovered | stronger | 1.8 px |
| selected | strongest | 2.6 px, over a white halo |

The fill is kept deliberately low because the reason to open Edit Mode is to
look at what is under the annotation. The border does the work of showing where
the instance actually ends, and it is drawn with a cosmetic pen, so it stays
one line wide at every zoom level.

### Selecting

Move the pointer over an instance to highlight it, click to select it. The
status bar then shows what it is: `selected #42  class 4: electronic_board`,
using the annotation's own id and category, not the tool's defaults.

Where instances overlap, **the smallest one under the pointer wins**. A small
instance drawn on top of a large one would otherwise be impossible to pick,
and the large one is still reachable everywhere the small one is not. Clicking
empty space, or pressing `Esc`, clears the selection.

### Telling the classes apart

**Colour means class, not instance.** Every instance of the same
`category_id` is drawn in the same hue, so how the image is labelled is
readable at a glance and a mislabelled object is the one wrong colour in a
group. Instances of the same class are still separated by their own borders
and by hover.

Each instance also carries a small chip at the centre of its bounding box with
its class id on it, drawn at a fixed size so it stays legible at any zoom. The
status bar spells out the selected instance in full:
`selected #42  class 4: electronic_board`.

### Setting the class

The number keys `0`-`9` assign a class. What they act on depends on the mode:

* **Edit Mode** — the selected instance is relabelled immediately, and the
  file is written straight away. With nothing selected the key does nothing
  but print a reminder. The colour and the chip change as you press it, so the
  result is visible without looking at the status bar.
* **Annotate Mode** — the key sets the class the *next* commit will use. Press
  it at any time while painting, before `Enter`; the active class is shown in
  the status bar, tinted in that class's colour.

Relabelling keeps the instance's geometry exactly as it was: only
`category_id` changes in the file, the polygons are not re-derived. A class
the file has never seen is added to `categories` as `class_<id>` when you
first use it, so the output stays a valid COCO file; rename it there if you
want a meaningful name. `Ctrl+Z` undoes a relabel like any other edit, and
pressing the same digit twice is one edit, not two.

Only classes `0`-`9` are reachable from the keyboard. For a class id above 9,
start the tool with `--category-id`.

### Deleting

`Delete` removes the selected instance and nothing else:

* it is dropped from the in-memory instance list for this image,
* the view updates immediately,
* and `annotations.json` is rewritten straight away, through the same atomic
  save the rest of the tool uses.

Every other annotation is left exactly as it was, down to its id and any keys
this tool does not understand (see [section 4](#4-output-format)). With nothing
selected, or in Annotate Mode, `Delete` only prints a note in the status bar.
The image file is never touched.

### Undo and redo

`Ctrl+Z` / `Ctrl+Shift+Z` undo and redo *within the current mode*. In Edit Mode
they walk a list of the edits you made — deletions and class changes — undoing
each one: a deleted instance goes back to the position it held, as the same
object with the same annotation id, and a relabelled one gets its old class
back. In Annotate Mode they keep working on the constraint marks exactly as
before.

The two histories are deliberately separate: one stack would mean `Ctrl+Z`
sometimes redrawing a brush mark and sometimes resurrecting an annotation, with
no way to tell in advance which you were about to get. The edit history holds
64 steps and is cleared when you move to another image, because by then the
change has already been written to disk.

Deleting an instance re-seeds the remaining ones as background and therefore
starts the in-progress instance over, exactly as `Ctrl+Backspace` always has.
Finish and commit what you are painting before editing.

---

## 3. Reference marks are permanent

This is the part worth understanding, because it changes how the tool behaves
compared to most paint-select tools.

There is one authoritative array per image, `constraints`, with `+1` where you
painted foreground, `-1` where you painted background, and `0` everywhere else.
The binary mask is **derived** from it:

```python
mask = segment(image, constraints, params)
```

Nothing in the pipeline writes to `constraints` except your brush. Every pass
reads it, honours `±1` as infinite-capacity terminal edges in the graph cut,
and stamps it onto the result again at the end. So:

* A pixel you marked background stays background through every later
  operation, however far away you paint next, and however much the colour
  statistics change around it.
* The only way to change a mark is to paint the opposite brush over it. That is
  deliberate — you are overriding yourself, explicitly.
* Undo and redo operate on `constraints`, not on the mask. Undoing a stroke
  restores exactly the marks that existed before it and the mask is recomputed
  from those.
* "Clear constraints" (`Ctrl+K`) is separate from undo: it drops every mark for
  the instance in progress in one action.

The previous design kept the selection as primary state with the marks as a
secondary hint, and the two could disagree — whichever ran last won. That is
the bug this fixes, and it is fixed structurally rather than patched.

### One consequence worth knowing

Colour statistics accumulate as you paint, and an incremental model cannot
un-see a sample. After an undo or a clear, the accumulated model no longer
matches the marks, so it is discarded and rebuilt from whatever marks remain.
That makes undo slightly more expensive than a normal stroke, and exactly
correct.

---

## 4. Output format

One `annotations.json` per folder, standard COCO:

```json
{
  "images": [
    {"id": 1, "file_name": "board_001.jpg", "width": 1920, "height": 1080}
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 4,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 227356.0,
      "bbox": [150.0, 140.0, 369.0, 279.0],
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 4, "name": "electronic_board", "supercategory": "object"}
  ]
}
```

* `segmentation` is a list of polygons, each a flat `[x1, y1, x2, y2, ...]`
  list in **absolute pixel coordinates**. One entry per external contour.
* `area` is the shoelace area of the simplified polygons, not the pixel count
  of the mask — the polygon is what ends up in the file, so the area should
  describe the polygon. A pixel count would disagree with anything that
  re-rasterises the annotation.
* `bbox` is `[x, y, width, height]`.
* No masks are written as PNG. Polygons only.

Image ids are assigned on first sight and kept, so reopening a folder restores
what you left and adding new images does not renumber the old ones. Saves are
atomic — written to a temporary file and renamed — so a crash mid-save leaves
the previous good file intact rather than a truncated one.

`--category-id` and `--category-name` override the defaults if you need a
different class.

### What a round trip preserves

An annotation loaded from the file and not edited is written back **as it came
in**: its id, its `category_id`, its polygons, and any extra keys the file
carried that this tool has no opinion about. Relabelling one changes its
`category_id` and nothing else — the original polygons, area and bbox are kept
verbatim, because only the label changed. Only instances you paint yourself,
or edit, have their `segmentation`, `area` and `bbox` regenerated from the
mask. Deleting one instance therefore leaves the others byte-for-byte as they
were, and a multi-class file keeps its other classes and its `categories`
list.

### Holes

Contours are extracted with `RETR_EXTERNAL` by default: outer boundaries only,
interior holes filled. This is deliberate. Standard COCO polygon segmentation
has no hole semantics, and a hole polygon sitting in the same list is rendered
as *additional filled area* by pycocotools, which silently corrupts the mask.

The **Keep holes** checkbox switches to `RETR_CCOMP`, which emits holes as
separate polygons. Only turn it on if your training code handles them.

---

## 5. Speed and quality

Measured on a 1920×1080 image, CPython 3.11, one core, no GPU, painting a
30-stamp drag. `python tools/profile_segmentation.py IMAGE --compare`
reproduces this on your own machine and images.

| setting | median | p95 | max |
| --- | --- | --- | --- |
| **default** (384 px, histogram, local + warm start) | **5.8 ms** | **21.5 ms** | 61.3 ms |
| `work_max_dim=320` | 4.1 ms | 13.7 ms | 43.4 ms |
| `work_max_dim=512` | 9.8 ms | 71.4 ms | 85.1 ms |
| GMM colour model | 5.9 ms | 66.1 ms | 73.5 ms |
| 4-neighbourhood | 6.5 ms | 20.3 ms | 28.8 ms |
| warm start disabled | 27.9 ms | 43.6 ms | 51.6 ms |
| local ROI disabled | 5.3 ms | 21.9 ms | 42.1 ms |

The `max` column is almost always the first pass, which pays the one-off
per-image setup: the working image, its Lab features, and the neighbour edge
weights. Those are computed once and reused by every later stroke.

A typical stage breakdown at the default settings:

```
color_model   2.99 ms   25%
maxflow       2.11 ms   18%
graph_build   1.93 ms   16%
postprocess   1.54 ms   13%
upsample      1.45 ms   12%
constraints   0.46 ms    4%
prepare       0.21 ms    2%
```

### The parameters that matter

All of them live in `SegmentParams` in `quickselect/segmenter.py`, each with a
comment explaining what it does.

| parameter | default | effect |
| --- | --- | --- |
| `work_max_dim` | `384` | **The main speed knob.** Cost scales with its square. 320 is noticeably faster and, on the test images, indistinguishable in IoU; 512 triples p95 for no measurable gain. Raise it only if you are annotating objects with fine detail relative to the frame. |
| `model` | `"hist"` | `hist` is exactly incremental (raw bin counts accumulate) and 2–4× faster. `gmm` is the GrabCut model; slightly better on smoothly shaded objects, and it dominates p95 because a refit costs ~28 ms. |
| `model_refresh_ratio` | `0.20` | Rebuild the colour models once the sample set grows by this fraction. Higher = fewer rebuilds = more warm-started solves = faster, at the cost of the models lagging the newest strokes within one drag. |
| `warm_start_max_change` | `1.01` | `>1` means "always warm-start when the graph structure is unchanged". Disabling it costs 5× on the median. Labels are identical either way — verified, not assumed. |
| `neighborhood` | `8` | 4 halves the edge count and tightens p95, at the cost of blockier diagonal boundaries. |
| `local_margin` | `40` | Full-resolution margin around the new stroke and the selection boundary when rebuilding locally. |
| `local_enabled` | `True` | With warm start on, the global path is competitive because its ROI never churns. Local wins on very large images; both are within noise at 1080p. |
| `require_seed_connectivity` | `True` | Keep only components containing a foreground mark. The main runaway guard: a same-coloured board elsewhere in the frame is never selected unless you paint it. |
| `gamma_smooth` | `20.0` | Edge/contrast strength. Higher = smoother boundaries and **markedly slower** max-flow; the smoothness/data balance, not the node count, was the original bottleneck. |
| `lambda_data` | `3.0` | Colour-likelihood strength relative to the edge term. |
| `fg_bias` | `7.0` | How tolerant the flood is when the background model carries no information. |
| `polygon_epsilon` (UI) | `0.0015` | `approxPolyDP` epsilon as a fraction of each contour's perimeter. On a convoluted outline: 0.0005 → 63 points at IoU 0.998; 0.0015 → ~33 at 0.996; 0.005 → 16 at 0.977. |

**Recommended defaults**: leave everything as shipped. If the tool feels slow
on your hardware, drop `work_max_dim` to 320 first — it is the only change that
buys real time without a visible quality cost. If your boards have subtle
shading and the selection keeps stopping short, try `model="gmm"` and accept
the worse p95.

---

## 6. Module layout

Exactly the split you asked for:

| module | contains |
| --- | --- |
| `quickselect/constraints.py` | the constraint matrix, brush application, patch-diff undo/redo |
| `quickselect/segmenter.py` | `segment(image, constraints, params) -> mask`, the caches, the graph cut |
| `quickselect/profile.py` | timing hooks used throughout |
| `quickselect/annotator/polygon.py` | mask → polygons, simplification, COCO serialisation |
| `quickselect/annotator/dataset_io.py` | COCO load/save, folder scanning |
| `quickselect/annotator/session.py` | per-image state: constraints, mask, instances |
| `quickselect/annotator/editing.py` | modes, instance hit-testing, selection, edit history |
| `quickselect/annotator/instance_view.py` | the Edit Mode palette and overlay compositing |
| `quickselect/annotator/worker.py` | the segmentation thread and its coalescing queue |
| `quickselect/annotator/app.py` | the PySide6 window |
| `tools/profile_segmentation.py` | the stage-timing CLI |

`quickselect/profile.py` shadows the standard library's `profile` only for code
inside the package that writes `import profile`. Nothing does — all imports here
are relative — and under Python 3's absolute-import rules an unrelated
`import profile` elsewhere still finds the stdlib module.

The segmentation core has no Qt dependency, so the whole workflow can be driven
from a script:

```python
from quickselect.annotator.session import AnnotationSession
from quickselect.constraints import POSITIVE, NEGATIVE

session = AnnotationSession(image)
session.brush.diameter = 48
session.begin_stroke(400, 300, POSITIVE)
session.continue_stroke(500, 350)
session.end_stroke()
session.commit_instance()
entries = session.to_annotations(image_id=1, next_id_fn=lambda: 1)
```

Edit Mode is scriptable for the same reason -- `editing.py` has no Qt import
either:

```python
from quickselect.annotator.editing import InstanceEditor

editor = InstanceEditor(session)
editor.select(editor.hit_test(x=820, y=430))
editor.delete_selected()
editor.undo()
```

---

## 7. Known limits

* **Same colour across a weak edge still leaks.** Two touching boards of
  identical colour will select together; alt-painting one clears it, but the
  boundary between them ends up wherever the weak seam is, which may cost a few
  percent of the board you are keeping. This is inherent to the energy, not a
  bug — with no edge and no colour difference there is nothing to separate on.
* **Committed instances are pinned as background** for later instances on the
  same image. Overlapping instances therefore need the overlap painted back in
  explicitly. Set `session.seed_committed_as_background = False` to allow free
  overlap.
* **Undo granularity is one stroke.** Committing an instance is not undoable
  through `Ctrl+Z`; use "Delete last instance".
* **RLE segmentations are not editable.** Loading a COCO file that uses RLE
  masks skips those entries rather than converting them. They are skipped on
  load, so they are also not written back -- do not edit a folder whose file
  mixes RLE with polygons.
* **Edit Mode deletes and relabels, it does not reshape.** Changing an
  instance's outline is not implemented; to fix geometry, delete the instance
  and paint it again.
* **The keyboard reaches classes 0-9 only.** Higher category ids have to come
  from `--category-id` or from the file.
* **Editing discards the instance in progress.** Deleting an instance re-seeds
  the constraint map, which starts the object you were painting over.
* **Single-threaded solves.** PyMaxflow is not parallel and there is no
  CUDA/OpenCL path.
