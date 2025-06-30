# Example Applications of the Buckingham–Pi Theorem

This repository collects a variety of worked examples from the literature, demonstrating how to apply the Buckingham–Pi theorem in Python.

## Motivation

Dimensional analysis is a foundational tool in the physical sciences and engineering: by identifying combinations of variables that are truly dimensionless, we collapse complex phenomena onto universal scaling laws, reduce the experimental burden, and reveal the underlying structure of physical models. At its core lies the Buckingham π theorem, which tells us that any physically meaningful equation can be rewritten in terms of a smaller set of independent, dimensionless groups (the "π-groups") whose exponents satisfy the homogeneity condition $A \cdot E = 0$.

Historically, two main approaches have been used:

* **Enumerative (combinatorial) methods**
  - Systematically generate every integer-exponent nullspace vector  
  - Enumerate subsets of these π-groups to discover minimal complete sets  
  - Manually inspect each candidate for physical interpretability  

* **Continuous-exponent (optimization) methods**
  - Recognize that real-valued exponent vectors also satisfy $A \cdot E = 0$  
  - Use log-domain regression or evolutionary algorithms to fit continuous powers that maximize data collapse ($R^2$) under variance constraints  
  - Allow subtle, non-integer scalings that often yield superior empirical fits  

Originally, continuous exponents were extracted by hand from experimental log-plots—researchers would sweep one candidate π-group, hold others fixed, and iteratively tweak exponents until curves overlapped. Modern work has automated this tedious loop by embedding the nullspace basis into an evolutionary optimizer (e.g., GA, CMA-ES, BiteOpt). By parameterizing
$E = N_s \cdot C$ (where $N_s$ is the orthonormal nullspace of the dimension matrix $A$ and $C \in \mathbb{R}^{k \times m}$ contains the continuous weights), one can pose a single smooth optimization over $C$ to discover the best $m$ π-groups in one go.

**BuckinghamExamples** brings these ideas together in an open-source, Python-based toolkit. It:

* Enumerates all classic integer π-groups for canonical fluid-mechanics and heat-transfer problems  
* Optimizes continuous exponents for maximal data collapse under cross-validation  
* Automates the switch between enumeration and optimization, letting you explore $m = 1, 2, \ldots, k$ nullspace dimensions without manual algebra  

Whether you prefer the pedagogical clarity of integer π-groups or the empirical accuracy of continuous scalings, 
this repository provides a turnkey platform for rigorous, reproducible dimensional analysis.

## Examples

### From BuckinghamPy

Examples from Saad et al. (2021) [BuckinghamPy: A Python software for dimensional analysis](https://www.sciencedirect.com/science/article/pii/S2352711021001291)  

* [examples.ipynb](https://github.com/saadgroup/BuckinghamPy/blob/master/examples.ipynb)  
* Centrifugal Pump example in the README: [Centrifugal Pump](https://github.com/saadgroup/BuckinghamPy/blob/master/README.md)

### Hydrogen Knudsen Compressor

From Xie & Qian (2023) [Dimensional-Analysis of hydrogen Knudsen compressor](https://www.sciencedirect.com/science/article/abs/pii/S0360319923023030)  

* [Dimensional-Analysis repository](https://github.com/xqb-python/Dimensional-Analysis)  
* See their script: [Hydrogen Knudsen Compressor Code](https://github.com/xqb-python/Dimensional-Analysis/blob/main/%E4%B8%AD%E5%BF%83%E5%9E%82%E7%9B%B4%E7%BA%BF%E4%B8%8A%E7%9A%84%E9%80%9F%E5%BA%A6%E5%88%86%E5%B8%83/%E6%9C%80%E5%A4%A7%E6%BB%91%E7%A7%BB%E9%80%9F%E5%BA%A6.py)

### Optimierung von Stahlgießprozessen

From Lichtenberg (2021) "Optimierung von Stahlgießprozessen anhand eines Wassermodells mit begleitender Strömungssimulation"  

* [Dissertation repository](https://repo.bibliothek.uni-halle.de/handle/1981185920/87855)

### Several More Problems

#### Forced Convection over a Cylinder
* Variables: `['Nu', 'Re', 'Pr', 'L/D', 'k', 'rho', 'c_p', 'mu']`
* Dimension matrix:
  ```
  [   0,  0,  0,   0,   1,    1,   0,    1]   # M
  [   0,  0,  0,   0,   1,   -3,   2,   -1]   # L
  [   0,  0,  0,   0,  -3,    0,  -2,   -1]   # T
  [   0,  0,  0,   0,  -1,    0,  -1,    0]   # Θ
  ```
* Dimensions: `['M','L','T','Θ']`
* Target: `Nu`

#### Natural Convection from a Horizontal Plate
* Variables: `['Nu', 'Gr', 'Pr', 'L', 'beta', 'DeltaT', 'rho', 'mu', 'k', 'g']`
* Dimension matrix:
  ```
  [   0,  0,  0,   0,   0,   0,   1,    1,   1,   0]  # M
  [   0,  0,  0,   1,   0,   0,  -3,   -1,   1,   1]  # L
  [   0,  0,  0,   0,   0,   0,   0,   -1,  -3,  -2]  # T
  [   0,  0,  0,   0,  -1,   1,   0,    0,  -1,   0]  # Θ
  ```
* Dimensions: `['M','L','T','Θ']`
* Target: `Nu`

#### Packed-Bed Pressure Drop (Ergun)
* Variables: `['DeltaP','rho','mu','U','D_p','epsilon','L']`
* Dimension matrix:
  ```
  [    1,   1,  1,  0,  0,  0, 0]  # M
  [   -1,  -3, -1,  1,  1,  0, 1]  # L
  [   -2,   0, -1, -1,  0,  0, 0]  # T
  ```
* Dimensions: `['M','L','T']`
* Target: `DeltaP`

#### Stirred-Tank Mixing (Power Number)
* Variables: `['P','rho','N','D','mu','sigma']`
* Dimension matrix:
  ```
  [   1,   1, 0, 0,  1,     1]  # M
  [   2,  -3, 0, 1, -1,     0]  # L
  [  -3,   0, -1,0, -1,    -2]  # T
  [   0,   0, 0, 0,  0,     0]  # Θ
  ```
* Dimensions: `['M','L','T','Θ']`
* Target: `P`

#### Rayleigh–Bénard Convection
* Variables: `['Nu','Ra','Pr','H','k','rho','c_p','mu','g','beta','DeltaT']`
* Dimension matrix:
  ```
  [   0,  0, 0, 0,   1,    1,    0,    1,   0,   0,  0]  # M
  [   0,  0, 0, 1,   1,   -3,    2,   -1,   1,   0,  0]  # L
  [   0,  0, 0, 0,  -3,    0,   -2,   -1,  -2,   0,  0]  # T
  [   0,  0, 0, 0,  -1,    0,   -1,    0,   0,  -1,  1]  # Θ
  ```
* Dimensions: `['M','L','T','Θ']`
* Target: `Nu`

#### Drop Dynamics in Shear (Ca, Re)
* Variables: `['d','mu_c','mu_d','sigma','G','rho_c','rho_d']`
* Dimension matrix:
  ```
  [  0,   1,   1,     1,   0,     1,     1]  # M
  [  1,  -1,  -1,     0,   0,    -3,    -3]  # L
  [  0,  -1,  -1,    -2,  -1,     0,     0]  # T
  ```
* Dimensions: `['M','L','T']`
* Target: `d`

#### MHD Duct Flow
* Variables: `['Ha','Re','sigma_e','B','L','rho','mu']`
* Dimension matrix:
  ```
  [   0,  0, -1,  1,  0,  1, 1]  # M
  [   0,  0, -3,  0,  1, -3,-1]  # L
  [   0,  0,  3, -2,  0,  0,-1]  # T
  [   0,  0,  2, -1,  0,  0, 0]  # I
  ```
* Dimensions: `['M','L','T','I']`
* Target: `Ha`

#### Aeroacoustic Dipole Radiation
* Variables: `['p_prime','rho','U','L','c','omega','l']`
* Dimension matrix:
  ```
  [    1,  1, 0,  0,  0,   0, 0]  # M
  [   -1, -3, 1,  1,  1,   0, 1]  # L
  [   -2,  0, -1, 0, -1,  -1, 0]  # T
  ```
* Dimensions: `['M','L','T']`
* Target: `p_prime`

#### Laminar Forced-Convection over a Cylinder
* Variables: `['h','D','k','U','mu','rho','c_p']`
* Dimension matrix:
  ```
  [   1,  0,  1,  0,   1,   1,   0]   # M
  [  -2,  1,  1,  1,  -1,  -3,   2]   # L
  [  -3,  0, -3, -1,  -1,   0,  -2]   # T
  [  -1,  0, -1,  0,   0,   0,  -1]   # Θ
  ```
* Dimensions: `['M','L','T','Θ']`
* Target: `h`

## Python Scripts

Three complementary approaches are provided:

* [apply_buckinghampy.py](https://github.com/dietmarwo/BuckinghamExamples/blob/master/apply_buckinghampy.py)  
  Uses [BuckinghamPy](https://github.com/saadgroup/BuckinghamPy) to enumerate all valid π-groups.

* [rank_pi_groups.py](https://github.com/dietmarwo/BuckinghamExamples/blob/master/rank_pi_groups.py)  
  Uses NumPy and scikit-learn to:
  1. Find all repeating-variable sets
  2. Compute their π-groups
  3. Score each set by the predictive $R^2$ adding a penalty for bad coefficient-of-variation over your experimental range

* [optimize_pi_groups.py](https://github.com/dietmarwo/BuckinghamExamples/blob/master/optimize_pi_groups.py)  
  1. Uses [fcmaes](https://github.com/dietmarwo/fast-cma-es) to apply the [biteopt](https://github.com/avaneev/biteopt) evolutionary algorithm to vary the pi group exponents, allowing continuous values
  2. Score each set of pi group exponents by the predictive $R^2$ adding a penalty for bad coefficient-of-variation over your experimental range

### Customization

1. **Add your own examples** to the `examples` dict (variable names + dimension matrix)
2. **Plug in your real data** (in place of the random sampling) to get π-group rankings tailored to your experiment

## Further Comparison

Compare with:  
[Hydrogen Knudsen Compressor Code](https://github.com/xqb-python/Dimensional-Analysis/blob/main/%E4%B8%AD%E5%BF%83%E5%9E%82%E7%9B%B4%E7%BA%BF%E4%B8%8A%E7%9A%84%E9%80%9F%E5%BA%A6%E5%88%86%E5%B8%83/%E6%9C%80%E5%A4%A7%E6%BB%91%E7%A7%BB%E9%80%9F%E5%BA%A6.py), which uses genetic optimization. GA doesn't work well for this application, especially if you allow continuous exponents. [ijche2014](https://www.ijche.com/article_10200_e5d7175834c141c6c71c4fe626ec5cb4.pdf) applies CMA-ES, but is focused on a specific problem.

## Continuous-Exponent π-Group Pipeline

Typically, π-group determination necessitates an initial discrete, combinatorial decision: selecting which variables belong to which π-group (or determining whether to incorporate an additional group). The continuous-exponent pipeline described below circumvents explicit variable assignment to π-groups, unlike subset-enumeration approaches. Instead, it leverages the following principles:

### 1. Dimensional Homogeneity and Nullspace Equivalence

Any exponent vector $E \in \mathbb{R}^n$ that renders $\prod_{i=1}^N x_i^{E_i}$ dimensionless must satisfy $A \cdot E = 0$.

Computing $N_s = \text{null}(A)$ yields an orthonormal basis of that nullspace (with shape $N \times k$), enabling any valid $E$ to be expressed as $E = N_s \cdot c$ for some coefficient vector $c \in \mathbb{R}^k$.

### 2. Construction of m π-Groups

To construct m π-groups, select a $k \times m$ matrix $C$ whose columns represent the $c$ vectors for each π-group. Compute $E = N_s \cdot C$ (an $N \times m$ matrix) and construct the π-features: $\Pi_{:,j} = \exp(\log(X) \cdot E_{:,j})$.

These π-terms are guaranteed to be dimensionless by construction.

### 3. Continuous Optimization Framework

The approach formulates a single continuous optimization problem over the entries of $C \in \mathbb{R}^{k \times m}$, maximizing $R^2$ (adjusted for cross-validation penalty). The evolutionary optimizer explores $\mathbb{R}^{k \cdot m}$, implicitly investigating all possible combinations of the k basis vectors into m groups.

This eliminates the need for discrete enumeration of variable subsets—the continuous weights in $C$ determine each variable's exponent coefficient.

Since $E = N_s \cdot C$ enforces $A \cdot E = 0$, every continuous trial $C$ produces a valid π-group configuration. The combinatorial challenge of "which variables belong to π₁ versus π₂" is resolved implicitly through the optimizer's identification of optimal continuous weights, rather than through manual subset enumeration.

> **Note:** The optimization library employed ([fcmaes](https://github.com/dietmarwo/fast-cma-es)) supports CMA-ES and various other established algorithms. Substituting the current implementation requires only a single line change. BiteOpt was selected for its superior flexibility:
> 
> * Maintains success statistics (tracking which mutation scales generate actual improvements)
> * Dynamically re-weights proposal distributions based on performance statistics  
> * Automatically balances exploration versus exploitation as the optimization landscape evolves

## Application: Laminar Forced-Convection over a Cylinder

Once we have optimized continuous π-groups, we can directly use them to design and collapse experimental or simulation data. As an illustration, recall the case of forced convection over a cylinder, where our independent variables are

$$\{D,\ k,\ U,\ \mu,\ \rho,\ c_p\}$$

and the dependent response is the Nusselt number $Nu$.

### Single-π sweep ($m=1$)

The optimizer found one dominant π-group

$$\pi_1 = D^{0.5171}\,k^{-0.1687}\,U^{0.5171}\,\mu^{-0.3485}\,\rho^{0.5171}\,c_p^{0.1687}$$

with empirical fit $R^2=0.982$. In practice, you can:

1. Compute $\pi_1$ for each dataset:
   $$\pi_1^{(i)} = D_i^{0.5171}\,k_i^{-0.1687}\,\dots\,c_{p,i}^{0.1687}$$

2. Plot $\log(Nu)$ vs. $\log(\pi_1)$. If the collapse is good, all curves collapse to
   $$\log(Nu) \approx a + b\,\log(\pi_1),$$
   allowing a two-parameter fit $(a,b)$.

3. For quick "single-knob" experiments, sweep $\pi_1\in[0.1,10]$ on a log scale, holding all other physical parameters fixed. This lets you explore the physics along the dominant scaling direction.

### Two-π design ($m=2$)

For a lossless reduction to two knobs, we include a second π-group. The optimizer yields:

$$\begin{align}
\pi_1 &= D^{0.5222}\,k^{-0.1731}\,U^{0.5222}\,\mu^{-0.3492}\,\rho^{0.5222}\,c_p^{0.1731},\\
\pi_2 &= D^{0.0009}\,k^{-0.5777}\,U^{0.0009}\,\mu^{0.5767}\,\rho^{0.0009}\,c_p^{0.5777},
\end{align}$$

with perfect collapse $R^2=1.00$. To design a two-knob sweep:

1. Select four pivot conditions $\{D_0,k_0,U_0,\mu_0\}$ (or any four independent variables) and hold them fixed.

2. Vary $(\pi_1,\pi_2)$ over a 2D grid in log-space, computing the corresponding physical parameters by solving the linear system
   
   $$\log(D, k, U, \mu, \rho, c_p)^T = N_s \cdot C \cdot (\log \pi_1, \log \pi_2)^T$$
   
   while keeping the pivots constant.
   
3. Measure $Nu$ for each $(\pi_1,\pi_2)$ combination and verify that
   $\log(Nu)$ collapses onto a smooth surface in the $(\log\pi_1,\log\pi_2)$ plane.

In this way the continuous-exponent optimizer not only finds the optimal collapse, but also directly prescribes the experimental "knobs" (π-coordinates) needed for systematic studies.

### Choosing and Using Pivot Variables in Practice

In principle any four of the six variables $\{D,k,U,\mu,\rho,c_p\}$ will work as pivots, so long as the corresponding $4\times 2$ sub-matrix of the exponent matrix

$$E = \begin{bmatrix}
e_{D,1} & e_{D,2} \\
e_{k,1} & e_{k,2} \\
e_{U,1} & e_{U,2} \\
e_{\mu,1} & e_{\mu,2} \\
e_{\rho,1} & e_{\rho,2} \\
e_{c_p,1}& e_{c_p,2}
\end{bmatrix}$$

is full rank (i.e., those four rows are linearly independent).

1. **Select four pivot variables** whose exponent rows are linearly independent.  
   *Example:* $D, k, U, \mu$.

2. **Clamp them** at fixed values in your experiment or simulation:  
   $$D = D_0,\quad k = k_0,\quad U = U_0,\quad \mu = \mu_0$$

3. **Solve for the remaining two variables** $\rho$ and $c_p$ by inverting the following $4\times 4$ system in the log-domain:

   $$\begin{bmatrix}
   e_{D,1} & e_{D,2} \\
   e_{k,1} & e_{k,2} \\
   e_{U,1} & e_{U,2} \\
   e_{\mu,1} & e_{\mu,2}
   \end{bmatrix}
   \begin{bmatrix}\log\pi_1\\\log\pi_2\end{bmatrix}
   +
   \begin{bmatrix}0\\0\\0\\0\end{bmatrix}
   =
   \begin{bmatrix}
   \log D_0\\
   \log k_0\\
   \log U_0\\
   \log \mu_0
   \end{bmatrix}
   +
   \begin{bmatrix}
   e_{\rho,1} & e_{\rho,2}\\
   e_{c_p,1} & e_{c_p,2}
   \end{bmatrix}
   \begin{bmatrix}\log\rho\\ \log c_p\end{bmatrix}$$

   Invert this to recover $\log\rho$ and $\log c_p$ as functions of your chosen $(\pi_1,\pi_2)$.

---

### Does the Pivot Choice Matter?

- **Mathematically**  
  Any full-rank choice spans the same 2D complement of the nullspace—your collapse in $(\pi_1,\pi_2)$ is unchanged.

- **Experimentally**  
  Pick those variables you can hold most rigidly (e.g., geometry or material constants), so your π-sweep is simpler.

- **Numerically**  
  Avoid nearly collinear exponent rows, which lead to ill-conditioned inversions.

> **Rule of thumb:**  
> "Pivot on the parameters you can clamp most rigidly—let the continuous optimizer solve for the rest."


## Citing

```bibtex
@misc{buckpiexams2025,
  author       = {Dietmar Wolz},
  title        = {Example Applications of the Buckingham-Pi Theorem},
  year         = {2025},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {Available at \url{https://github.com/dietmarwo/BuckinghamExamples}},
}
```