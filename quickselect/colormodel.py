"""Foreground / background colour models.

Two interchangeable models are provided:

``GaussianMixtureModel``
    A from-scratch full-covariance GMM fitted with k-means++ initialisation and
    a handful of EM iterations.  This is the model used by GrabCut (Rother et
    al. 2004) and it is what we use by default.

``HistogramModel``
    A coarse 3-D colour histogram with Laplace smoothing.  Roughly 10x faster
    to fit and evaluate, at the cost of blockier likelihoods.  Useful on very
    large images or slow machines.

Both expose the same tiny interface::

    model = Model.fit(samples, ...)
    nll   = model.negative_log_likelihood(pixels)   # cost of *this* label

so the graph builder can stay agnostic.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

# Cost handed to a pixel a model has never seen anything like.  Bounded so a
# single outlier cannot dominate the whole energy.
MAX_NLL = 24.0
_MIN_PROB = float(np.exp(-MAX_NLL))


def _subsample(samples: np.ndarray, limit: int, rng: np.random.Generator) -> np.ndarray:
    if samples.shape[0] <= limit:
        return samples
    idx = rng.choice(samples.shape[0], size=limit, replace=False)
    return samples[idx]


# --------------------------------------------------------------------------- #
class HistogramModel:
    """Smoothed 3-D histogram over the feature space."""

    def __init__(self, logprob: np.ndarray, bins: int, lo: np.ndarray, scale: np.ndarray):
        self._logprob = logprob  # (bins, bins, bins)
        self._bins = bins
        self._lo = lo
        self._scale = scale

    #: Feature channels are all scaled to 0..255, so one fixed bin width works
    #: for every colour space and lets counts from different calls be summed.
    CHANNEL_RANGE = 256.0

    @classmethod
    def bin_index(cls, samples: np.ndarray, bins: int) -> np.ndarray:
        """Flat bin indices for ``samples``; shared by fit and incremental add."""
        idx = np.clip(
            (np.asarray(samples, np.float32).reshape(-1, 3)
             * (bins / cls.CHANNEL_RANGE)).astype(np.int32),
            0,
            bins - 1,
        )
        return (idx[:, 0] * bins + idx[:, 1]) * bins + idx[:, 2]

    @classmethod
    def from_counts(cls, counts: np.ndarray, bins: int) -> Optional["HistogramModel"]:
        """Build from raw bin counts -- the entry point for incremental use.

        Counts accumulate as the user paints, so a new stroke costs one
        ``bincount`` over its own pixels rather than a refit over everything.
        """
        counts = np.asarray(counts, dtype=np.float64).reshape(bins, bins, bins)
        if counts.sum() <= 0:
            return None
        return cls._finalize(counts, bins)

    @classmethod
    def fit(
        cls,
        samples: np.ndarray,
        bins: int = 16,
        **_: object,
    ) -> "HistogramModel":
        flat = cls.bin_index(samples, bins)
        counts = np.bincount(flat, minlength=bins ** 3).astype(np.float64)
        counts = counts.reshape(bins, bins, bins)
        return cls._finalize(counts, bins)

    @classmethod
    def _finalize(cls, counts: np.ndarray, bins: int) -> "HistogramModel":
        lo = np.zeros(3, np.float32)
        scale = np.full(3, bins / cls.CHANNEL_RANGE, np.float32)

        # Blur the histogram before normalising.  A few thousand samples in a
        # bins^3 grid is very sparse, and an unsmoothed histogram gives almost
        # every pixel a near-identical likelihood -- the data term then carries
        # no information and the cut collapses to whatever the edge term wants.
        # A separable [1,2,1] pass along each axis fixes that cheaply.  Done
        # with shifted adds rather than apply_along_axis, which would make 768
        # Python-level convolve calls for a 16^3 grid.
        for axis in range(3):
            up = np.roll(counts, 1, axis=axis)
            dn = np.roll(counts, -1, axis=axis)
            lead = [slice(None)] * 3
            lead[axis] = 0
            up[tuple(lead)] = 0.0
            trail = [slice(None)] * 3
            trail[axis] = -1
            dn[tuple(trail)] = 0.0
            counts = 0.25 * up + 0.5 * counts + 0.25 * dn

        # A tiny pseudo-count, rather than Laplace's 0.5, so a colour the model
        # has genuinely never seen stays expensive instead of being smoothed
        # into the middle of the range.
        counts += 1e-6
        prob = counts / counts.sum()
        logprob = np.log(prob).astype(np.float32)
        # Normalise against the model's own peak so that "cost 0" means the
        # best-matching colour this model knows.  An absolute normalisation
        # against a uniform cube (as used for the GMM, whose density is a real
        # density) does not transfer: a histogram bin is 20+ Lab units wide, so
        # its average density badly understates a tight cluster's peak, and
        # every plausible colour would clip to zero cost.
        return cls(logprob - float(logprob.max()), bins, lo, scale)

    def negative_log_likelihood(self, pixels: np.ndarray) -> np.ndarray:
        px = np.asarray(pixels, dtype=np.float32).reshape(-1, 3)
        idx = np.clip(((px - self._lo) * self._scale).astype(np.int32), 0, self._bins - 1)
        # ``_logprob`` is already peak-normalised, so -lp is the cost directly.
        nll = -self._logprob[idx[:, 0], idx[:, 1], idx[:, 2]]
        return np.clip(nll, 0.0, MAX_NLL).astype(np.float32)


# --------------------------------------------------------------------------- #
class GaussianMixtureModel:
    """Full-covariance GMM, fitted with k-means++ seeding plus short EM.

    Implemented directly (rather than via scikit-learn) so the number of EM
    iterations, the covariance floor and the sample cap can be tuned for
    interactive latency -- the model is refit on *every* stroke step.
    """

    def __init__(self, weights: np.ndarray, means: np.ndarray, covs: np.ndarray):
        self.weights = weights                 # (K,)
        self.means = means                     # (K, 3)
        self.covs = covs                       # (K, 3, 3)
        self._inv = np.linalg.inv(covs)        # (K, 3, 3)
        sign, logdet = np.linalg.slogdet(covs)
        self._log_norm = -0.5 * (3.0 * np.log(2.0 * np.pi) + logdet)
        self._log_w = np.log(np.maximum(weights, 1e-12))

    # -- fitting ----------------------------------------------------------- #
    @staticmethod
    def _kmeanspp(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        n = x.shape[0]
        centers = np.empty((k, x.shape[1]), dtype=np.float64)
        centers[0] = x[rng.integers(n)]
        d2 = ((x - centers[0]) ** 2).sum(1)
        for i in range(1, k):
            total = d2.sum()
            if not np.isfinite(total) or total <= 1e-12:
                centers[i] = x[rng.integers(n)]
            else:
                centers[i] = x[rng.choice(n, p=d2 / total)]
            d2 = np.minimum(d2, ((x - centers[i]) ** 2).sum(1))
        return centers

    @classmethod
    def fit(
        cls,
        samples: np.ndarray,
        n_components: int = 5,
        iterations: int = 4,
        max_samples: int = 4000,
        reg: float = 4.0,
        seed: int = 0,
        **_: object,
    ) -> "GaussianMixtureModel":
        rng = np.random.default_rng(seed)
        x = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
        x = _subsample(x, max_samples, rng)
        n = x.shape[0]
        k = max(1, min(n_components, n))
        eye = np.eye(3) * reg

        if n < 2:
            mean = x.reshape(1, 3) if n else np.zeros((1, 3))
            return cls(np.ones(1), mean, eye[None] * 4.0)

        centers = cls._kmeanspp(x, k, rng)
        # A few Lloyd iterations give EM a sane starting point cheaply.
        labels = np.zeros(n, dtype=np.int64)
        x_sq = np.einsum("ni,ni->n", x, x)
        for _it in range(3):
            # ||x - c||^2 expanded so the distance matrix comes from a matmul.
            d = x_sq[:, None] - 2.0 * (x @ centers.T) + np.einsum(
                "ki,ki->k", centers, centers
            )[None, :]
            labels = d.argmin(1)
            for j in range(k):
                m = labels == j
                if m.any():
                    centers[j] = x[m].mean(0)

        weights = np.empty(k)
        means = np.empty((k, 3))
        covs = np.empty((k, 3, 3))
        for j in range(k):
            m = labels == j
            cnt = int(m.sum())
            if cnt < 4:
                weights[j] = max(cnt, 1) / n
                means[j] = centers[j]
                covs[j] = eye * 4.0
            else:
                pts = x[m]
                weights[j] = cnt / n
                means[j] = pts.mean(0)
                covs[j] = np.cov(pts, rowvar=False) + eye
        weights /= weights.sum()
        model = cls(weights, means, covs)

        for _it in range(max(0, iterations)):
            resp = model._responsibilities(x)             # (n, k)
            nk = resp.sum(0) + 1e-9
            weights = nk / n
            means = (resp.T @ x) / nk[:, None]
            for j in range(k):
                d = x - means[j]
                covs[j] = (resp[:, j, None] * d).T @ d / nk[j] + eye
            weights = np.maximum(weights, 1e-6)
            weights /= weights.sum()
            model = cls(weights, means, covs)
        return model

    # -- evaluation -------------------------------------------------------- #
    def _log_component_pdf(self, x: np.ndarray) -> np.ndarray:
        """(n, K) log N(x | mu_k, Sigma_k).

        Looping over the (at most five) components and using an ``(n, 3) @
        (3, 3)`` matmul is markedly faster than one ``nki,kij,nkj`` einsum:
        the matmul goes through BLAS, the einsum does not.  This is the single
        hottest routine during a drag, so it is worth the loop.
        """
        n = x.shape[0]
        k = self.means.shape[0]
        out = np.empty((n, k), dtype=np.float64)
        for j in range(k):
            d = x - self.means[j]
            m_dist = np.einsum("ni,ni->n", d @ self._inv[j], d)
            out[:, j] = self._log_norm[j] - 0.5 * m_dist
        return out

    def _responsibilities(self, x: np.ndarray) -> np.ndarray:
        log_joint = self._log_component_pdf(x) + self._log_w[None, :]
        mx = log_joint.max(1, keepdims=True)
        p = np.exp(log_joint - mx)
        return p / (p.sum(1, keepdims=True) + 1e-300)

    def log_likelihood(self, pixels: np.ndarray) -> np.ndarray:
        x = np.asarray(pixels, dtype=np.float64).reshape(-1, 3)
        log_joint = self._log_component_pdf(x) + self._log_w[None, :]
        mx = log_joint.max(1, keepdims=True)
        return (mx[:, 0] + np.log(np.exp(log_joint - mx).sum(1) + 1e-300))

    def negative_log_likelihood(self, pixels: np.ndarray) -> np.ndarray:
        ll = self.log_likelihood(pixels)
        # Densities in a 0..255 space are tiny; shift by the density of a
        # uniform distribution over the cube so 0 means "perfectly typical".
        nll = -(ll + 3.0 * np.log(255.0))
        return np.clip(nll, 0.0, MAX_NLL).astype(np.float32)


# --------------------------------------------------------------------------- #
def build_model(samples: np.ndarray, cfg, seed: int = 0):
    """Factory honouring :attr:`EngineConfig.model`."""
    if cfg.model == "hist":
        return HistogramModel.fit(samples, bins=cfg.hist_bins)
    return GaussianMixtureModel.fit(
        samples,
        n_components=cfg.gmm_components,
        iterations=cfg.gmm_iterations,
        max_samples=cfg.gmm_max_samples,
        seed=seed,
    )
