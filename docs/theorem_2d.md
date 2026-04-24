# 2D Sort-Match: Why It Doesn't Exist, and What We Do Instead

## 1. The 1D Result (Review)

**Theorem (1D Sort-Match Optimality).**
For real-valued sequences $\{p_i\}_{i=1}^n$ and $\{t_j\}_{j=1}^n$ and any
convex cost $c: \mathbb{R} \to \mathbb{R}_{\geq 0}$, the assignment that
minimizes $\sum_i c(p_{\sigma(i)} - t_i)$ is the one that pairs the sorted
sequences element-wise: sort both, match $i$-th smallest to $i$-th smallest.

This reduces optimal transport from $O(n^3)$ Hungarian to $O(n \log n)$ sorting.

## 2. Why No Exact 2D Analog Exists

**Claim.** For $d \geq 2$, sorting both point sets independently and matching
element-wise does *not* in general yield the optimal assignment.

**Counterexample (3 points in 2D).**

Consider predictions $P = \{(0, 0), (1, 0), (0.5, 0.87)\}$ (equilateral
triangle with side length 1) and targets $T = \{(0.5, 0), (0, 0.87), (1, 0.87)\}$.

- Sorting by $x$-coordinate and matching: cost $\approx 1.73$
- Sorting by $y$-coordinate and matching: cost $\approx 1.73$
- Optimal Hungarian matching: cost $\approx 0.87$

The optimal matching pairs $(0,0) \leftrightarrow (0.5, 0)$,
$(1, 0) \leftrightarrow (1, 0.87)$, $(0.5, 0.87) \leftrightarrow (0, 0.87)$.
No single sorting axis recovers this.

**Why?** In 1D, the optimal matching must be order-preserving (any crossing
increases cost for convex costs). In 2D, there is no canonical total order,
so the crossing argument fails.

## 3. Sliced-Wasserstein as a Principled Approximation

Since exact sort-match fails in 2D, we use the **sliced-Wasserstein distance**:

$$\text{SW}_p(\mu, \nu) = \left( \int_{S^{d-1}} W_p^p(\text{proj}_\theta \mu, \text{proj}_\theta \nu) \, d\sigma(\theta) \right)^{1/p}$$

where $\text{proj}_\theta$ projects onto direction $\theta$ and $W_p$ is the
1D Wasserstein distance (computed exactly via sorting).

**Properties:**
1. $\text{SW}$ is a valid metric on probability measures.
2. $\text{SW}$ converges to $W$ as the number of slices $\to \infty$ (for $d=2$).
3. Each slice uses the exact 1D sort-match, so gradient computation is trivial.
4. Computational cost: $O(K \cdot n \log n)$ for $K$ slices, vs $O(n^3)$ for
   exact 2D Hungarian.

## 4. Physics-Informed Direction Sampling (XRD-Biased)

Standard sliced-Wasserstein uses uniform direction sampling on $S^1$.
For XRD, we argue that **biased sampling toward the 2$\theta$ axis**
should converge faster.

**Physical argument:**
- Peak positions (2$\theta$) are determined exactly by the crystal lattice
  via Bragg's law: $n\lambda = 2d \sin\theta$.
- Peak intensities depend on many additional factors: atomic scattering
  factors, Debye-Waller thermal factors, texture, preferred orientation,
  particle size effects.
- Therefore, two XRD patterns from similar structures will have similar
  positions but potentially different intensities.
- Matching along the 2$\theta$ axis carries more discriminative information.

**Implementation:**
We sample directions from a von Mises distribution centered on the 2$\theta$
axis (angle 0 in our 2D space):

$$\theta \sim \text{vonMises}(0, \kappa)$$

where $\kappa$ controls concentration. At $\kappa = 0$ we recover uniform
sampling; at $\kappa \to \infty$ we recover the 1D position-only case.

## 5. Numerical Evidence

We compare approximation error of the sliced-Wasserstein distance (relative
to exact 2D Hungarian) for uniform vs xrd-biased sampling at different
numbers of slices.

See `tests/test_2d.py` for the full numerical verification.

**Expected behavior:**
- Both converge to the true 2D Wasserstein as $K$ increases.
- XRD-biased should converge faster when the "signal" is concentrated
  along the 2$\theta$ axis (which it is for XRD data).
- If the signal is isotropic (both dimensions equally important),
  uniform sampling should be comparable or better.
