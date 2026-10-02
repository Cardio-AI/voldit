import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance


class WADDiv:
    """
    Wasserstein Distance-based Generative Diversity (WAD-Div)

    Computes the 1D Wasserstein distance between:
        - Observed kNN distance distribution
        - A chosen reference distribution

    Supports:
        - Raw WAD-Div
        - Normalized WAD-Div anchored to a maximally diverse dataset

    Notes
    -----
    Normalization requires calling `fit_max_reference()` first.
    The same metric configuration (k, reference type, etc.)
    must be used for fitting and evaluation.
    """

    def __init__(self, distance_metric: str = "euclidean", saturation: float = 0.99):
        """
        Parameters
        ----------
        distance_metric : str
            Distance metric used for pairwise feature distances.
            Passed to scipy.spatial.distance.cdist.

        saturation : float
            Target saturation level t in normalization.
            A dataset with W = W_max will obtain W_norm ≈ t.
            Typical value: 0.99
        """
        self.distance_metric = distance_metric
        self.saturation = saturation
        self.W_max = None
        self._max_config = None  # stores configuration used for W_max

    # ============================================================
    # Public API
    # ============================================================

    def fit_max_reference(
        self,
        features: np.ndarray,
        k: int = 3,
        reference: str = "zero",
        ref_features: np.ndarray = None,
        ref_sample_size: int = 10000,
        ref_percentile: float = 95.0,
        ref_prob: float = 0.95,
        random_state: int = None,
    ):
        """
        Fit maximal diversity anchor W_max.

        Typically computed on a real dataset.

        IMPORTANT:
        The same parameters must be used later when calling
        `compute(normalize=True)`.
        """

        result = self.compute(
            features=features,
            k=k,
            reference=reference,
            ref_features=ref_features,
            ref_sample_size=ref_sample_size,
            ref_percentile=ref_percentile,
            ref_prob=ref_prob,
            normalize=False,
            random_state=random_state,
        )

        self.W_max = result["wad_div"]

        # Store configuration for safety
        self._max_config = dict(
            k=k,
            reference=reference,
            ref_percentile=ref_percentile,
            ref_prob=ref_prob,
            ref_sample_size=ref_sample_size,
        )

        return self.W_max

    def compute(
        self,
        features: np.ndarray,
        k: int = 3,
        reference: str = "zero",
        ref_features: np.ndarray = None,
        ref_sample_size: int = 10000,
        ref_percentile: float = 95.0,
        ref_prob: float = 0.95,
        random_state: int = None,
        normalize: bool = False,
        return_distributions: bool = False,
        normalization_method: str = "linear",
    ):
        """
        Compute WAD-Div score.

        Parameters
        ----------
        features : np.ndarray (N x D)
            Feature embeddings of dataset to evaluate.

        k : int
            Number of nearest neighbors used to construct
            the local distance distribution.

        reference : {"zero", "exponential", "empirical"}
            Type of reference distribution.

        ref_features : np.ndarray
            Required if reference="empirical".
            Feature embeddings of reference dataset.

        ref_sample_size : int
            Sample size for synthetic reference distributions
            (zero or exponential).

        ref_percentile : float
            Percentile of observed distances used to fit
            exponential reference.

        ref_prob : float
            Target probability P(X <= x_ref) for exponential reference.

        random_state : int
            Optional random seed for reproducibility.

        normalize : bool
            If True, returns normalized WAD-Div in [0,1).

        return_distributions : bool
            If True, returns raw observed and reference
            distance distributions.

        normalization_method : str
            Choose between "linear" or "paper"
        """

        if random_state is not None:
            np.random.seed(random_state)

        obs_dist = self._compute_knn_distribution(features, k)

        if reference == "zero":
            ref_dist = self._zero_reference(ref_sample_size)

        elif reference == "exponential":
            ref_dist = self._exponential_reference(
                obs_dist,
                ref_sample_size,
                ref_percentile,
                ref_prob,
            )

        elif reference == "empirical":
            if ref_features is None:
                raise ValueError("ref_features required for empirical reference.")
            ref_dist = self._compute_knn_distribution(ref_features, k)

        else:
            raise ValueError(f"Unknown reference type: {reference}")

        W = float(wasserstein_distance(obs_dist, ref_dist))

        # ---------------- Normalization ----------------
        if normalize:
            if self.W_max is None:
                raise RuntimeError(
                    "W_max not set. Call fit_max_reference() first."
                )

            # Safety check
            if self._max_config is not None:
                if k != self._max_config["k"] or reference != self._max_config["reference"]:
                    raise ValueError(
                        "Normalization configuration mismatch with fitted W_max."
                    )

            # -------------------------------
            # Choose normalization method
            # -------------------------------
            # "linear": robust, W_norm = W / W_max
            # "paper": original formula from the BVM26 paper
            method = normalization_method

            if method == "linear":
                # Collapse → 0, real/reference → ~1
                W_norm = np.clip(W / self.W_max, 0.0, 1.0)

            elif method == "paper":
                # Original paper formula:
                # W_norm = W / (W + s), s = W_max * (1-t)/t
                # NOTE: Assumes W_max was computed over multiple datasets.
                # For single synthetic dataset, this can inflate W_norm.
                t = self.saturation
                s = self.W_max * (1 - t) / t
                W_norm = W / (W + s) if W > 0 else 0.0

            else:
                raise ValueError(f"Unknown normalization method: {method}")

        else:
            W_norm = None

        output = {
            "wad_div": W,
            "wad_div_norm": W_norm,
            "k": k,
            "reference": reference,
            "n_obs": len(obs_dist),
        }

        if return_distributions:
            output["obs_dist"] = obs_dist
            output["ref_dist"] = ref_dist

        return output

    # ============================================================
    # Internal Methods
    # ============================================================

    def _compute_knn_distribution(self, features: np.ndarray, k: int):

        if features.ndim != 2:
            raise ValueError("Features must be of shape (N, D)")

        if k >= features.shape[0]:
            raise ValueError("k must be smaller than number of samples.")

        dists = cdist(features, features, metric=self.distance_metric)
        np.fill_diagonal(dists, np.inf)

        topk = np.sort(dists, axis=1)[:, :k]
        return topk.flatten()

    def _zero_reference(self, size: int):
        return np.zeros(size)

    def _exponential_reference(
        self,
        obs_dist: np.ndarray,
        ref_sample_size: int,
        ref_percentile: float,
        ref_prob: float,
    ):

        x_ref = float(np.percentile(obs_dist, ref_percentile))

        if x_ref <= 0 or ref_prob <= 0:
            return np.zeros(ref_sample_size)

        lam = -np.log(1.0 - ref_prob) / max(x_ref, 1e-12)
        lam = np.clip(lam, 1e-12, 1e6)

        return np.random.exponential(scale=1.0 / lam, size=ref_sample_size)
