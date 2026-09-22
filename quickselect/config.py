"""Tunable parameters for the Quick Selection engine.

Every constant that shapes the *feel* of the tool lives here so it can be
tweaked without touching algorithm code.  Defaults were chosen to imitate the
responsiveness of Photoshop's Quick Selection Tool: a local, bounded flood that
stops hard at image edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Trimap label constants (uint8 so the trimap is a cheap image-sized array)
# --------------------------------------------------------------------------- #
UNKNOWN = 0
FOREGROUND = 1
BACKGROUND = 2

# Practical "infinity" for terminal-edge capacities.  Must dominate any sum of
# neighbour weights incident on a single node, otherwise a hard seed could be
# cut away.  Kept finite so max-flow arithmetic stays well conditioned.
HARD_SEED_CAPACITY = 1.0e4


@dataclass
class EngineConfig:
    """Configuration for :class:`quickselect.engine.QuickSelectEngine`."""

    # ---- multi-resolution -------------------------------------------------
    #: Longest side of the image used for the *interactive* (during-drag) solve.
    #: 640 keeps a full local solve comfortably inside one 16 ms frame budget.
    interactive_max_dim: int = 640
    #: Longest side used for the global solve performed on mouse release.
    #: ``0`` means "full resolution" (slow on very large images).
    refine_max_dim: int = 1600

    # ---- graph ------------------------------------------------------------
    #: 4 or 8 connected pixel graph.  8 gives visibly smoother diagonal edges.
    neighborhood: int = 8
    #: Strength of the pairwise (edge/contrast) term.  GrabCut calls this gamma.
    gamma_smooth: float = 20.0
    #: Strength of the unary (colour likelihood) term.
    lambda_data: float = 3.0
    #: Extra multiplier applied to n-links during the Auto-Enhance narrow-band
    #: solve, so the boundary is pulled onto strong gradients.
    auto_enhance_edge_gain: float = 3.0

    # ---- colour model -----------------------------------------------------
    #: 'gmm' (Gaussian mixture, GrabCut-style) or 'hist' (fast 3-D histogram).
    model: str = "gmm"
    #: Colour space the models and n-link contrasts are computed in.
    color_space: str = "lab"  # 'lab' | 'rgb' | 'hsv'
    gmm_components: int = 5
    gmm_iterations: int = 4
    #: Cap on pixels fed to EM; the models are refit on every stroke step so
    #: this is the single most important interactive-performance knob.
    gmm_max_samples: int = 4000
    #: Histogram bins per channel when ``model == 'hist'``.
    hist_bins: int = 16

    # ---- locality (the "Paint Selection" local optimisation) --------------
    #: Radius of the region the flood may expand into per stroke step,
    #: expressed in brush radii.  Larger = the selection leaps further ahead of
    #: the cursor; smaller = more controlled, more brush strokes needed.
    local_reach_factor: float = 3.0
    #: Floor on that radius, in interactive-scale pixels.
    local_reach_min: int = 18
    #: Ceiling, so one huge brush cannot turn a local solve into a global one.
    local_reach_max: int = 160
    #: Extra padding added around the solved ROI (interactive-scale pixels).
    roi_padding: int = 6
    #: Width of the ring used to harvest background colour samples, in units of
    #: the local reach radius.
    bg_band_factor: float = 0.6
    #: How many times the local region may grow when the result presses against
    #: its frontier (the adaptive local region of Paint Selection).
    max_expand_steps: int = 3
    #: Growth factor per expansion attempt.
    expand_factor: float = 2.2
    #: Fraction of the frontier the result must touch before we grow.
    expand_pressure: float = 0.05
    #: Hard ceiling on the interactive solve, in interactive-scale pixels.
    #: Guards the frame budget on pathological (near-uniform) images.
    max_solve_pixels: int = 200_000
    #: Above this ROI area the local solve drops to the half-resolution level.
    #: Only the adaptive-expansion steps are normally this big.
    coarse_solve_threshold: int = 22_000

    # ---- background trust -------------------------------------------------
    #: Mean ``-log P(negative samples | foreground model)`` above which the
    #: inferred background is considered informative.  Below it the background
    #: model is discarded, because the brush is deep inside a large object and
    #: the "background" ring is still object.
    bg_trust_threshold: float = 3.0
    #: Cost of labelling a pixel background when the background model is not
    #: trusted.  Acts as a plain threshold on the foreground likelihood, which
    #: turns the cut into an edge-stopped flood fill.
    fg_bias: float = 7.0
    #: Floor applied to the background cost even when the model *is* trusted,
    #: so the shrinking bias of min-cut never fully wins.
    bg_nll_floor: float = 0.6

    # ---- refinement -------------------------------------------------------
    #: Guided-filter radius (full-resolution pixels) used for edge-aware
    #: feathering of the up-sampled mask.
    guided_radius: int = 8
    guided_eps: float = 1e-4
    #: Half-width of the narrow band re-solved by Auto-Enhance, in full-res px.
    auto_enhance_band: int = 12
    #: Minimum area (in interactive-scale pixels) of a connected component kept
    #: when cleaning up the result.  Kills salt-and-pepper speckle.
    min_component_area: int = 24

    # ---- history ----------------------------------------------------------
    max_history: int = 60

    def reach_radius(self, brush_radius_px: float) -> int:
        """Local expansion radius, in interactive-scale pixels."""
        r = self.local_reach_factor * max(brush_radius_px, 1.0)
        return int(round(min(max(r, self.local_reach_min), self.local_reach_max)))


@dataclass
class BrushConfig:
    """Brush geometry, in *full-resolution image* pixels."""

    diameter: float = 40.0
    #: 0.0 = fully soft falloff, 1.0 = hard edged.
    hardness: float = 0.85
    #: Distance between successive stamps, as a fraction of the diameter.
    spacing: float = 0.20

    min_diameter: float = 1.0
    max_diameter: float = 2500.0

    @property
    def radius(self) -> float:
        return self.diameter * 0.5

    def scaled(self, factor: float) -> "BrushConfig":
        return BrushConfig(
            diameter=max(1.0, self.diameter * factor),
            hardness=self.hardness,
            spacing=self.spacing,
        )

    def step_size(self, factor: float = 1.0) -> float:
        return max(1.0, self.diameter * factor * self.spacing)


@dataclass
class RefineEdgeConfig:
    """Post-processing applied to the final mask ("Refine Edge" in Photoshop)."""

    #: Gaussian feather radius in full-resolution pixels (0 = off).
    feather: float = 0.0
    #: Morphological smoothing radius (0 = off).
    smooth: int = 0
    #: Positive expands the selection, negative contracts it (full-res pixels).
    shift_edge: int = 0
    #: Apply the edge-aware guided filter to the alpha.
    edge_aware: bool = True

    def is_identity(self) -> bool:
        return (
            self.feather <= 0.0
            and self.smooth <= 0
            and self.shift_edge == 0
            and not self.edge_aware
        )


@dataclass
class ToolState:
    """Everything the UI owns that the engine needs to know about."""

    brush: BrushConfig = field(default_factory=BrushConfig)
    refine: RefineEdgeConfig = field(default_factory=RefineEdgeConfig)
    auto_enhance: bool = False
    sample_all_layers: bool = True
