# XPBD: Position-Based Simulation of Compliant Constrained Dynamics

> Miles Macklin, Matthias Müller, Nuttapong Chentanez.
> *Motion in Games (MiG) 2016.* NVIDIA.
> PDF: [`Macklin et al. - 2016 - XPBD position-based simulation of compliant constrained dynamics.pdf`](./Macklin%20et%20al.%20-%202016%20-%20XPBD%20position-based%20simulation%20of%20compliant%20constrained%20dynamics.pdf)

## TL;DR

PBD's constraint stiffness depends on the iteration count and timestep — more iterations or smaller `Δt`
makes everything stiffer, and there's no way to decouple material properties from solver parameters. XPBD
fixes this by introducing a **compliance parameter** `α` (inverse stiffness) and a **total Lagrange multiplier**
`λ` per constraint. The solver now converges to a well-defined implicit-Euler solution whose stiffness is
controlled solely by `α`, independent of iteration count or timestep. The cost: one extra scalar stored per
constraint, and trivial modifications to the PBD projection formula. Bonus: `λ` gives you **constraint force
estimates** for free — useful for haptics, breakable joints, and force-feedback devices.

## The problem

- **PBD's stiffness is an artifact of the solver.** Raising iteration count to stiffen one object inadvertently
  stiffens all objects. Relative stiffness between stretch and bending on the same mesh is non-linearly
  coupled to iteration count — re-tuning is required every time solver parameters change.
- **No force concept.** PBD operates purely on positions; there is no well-defined constraint force, making
  it unsuitable for haptics, breakable joints, or coupling to force-based subsystems.
- **No correspondence to constitutive models.** PBD's `k ∈ [0,1]` multiplier has no physical unit and
  cannot represent real material parameters (Young's modulus, Lamé constants).

XPBD targets all three: **physically meaningful stiffness**, **force estimates**, and **decoupling from solver
parameters**, while preserving PBD's simplicity and robustness.

## Method

### 1. From energy potentials to compliant constraints

Start with Newton's equations of motion subject to an elastic energy potential `U(x)`:

```
M ẍ = −∇Uᵀ(x)
```

Express `U` in terms of constraint functions `C(x) = [C₁, C₂, …, Cₘ]ᵀ` and a compliance matrix `α`
(block-diagonal, inverse stiffness):

```
U(x) = ½ C(x)ᵀ α⁻¹ C(x)
```

The elastic force decomposes into direction (`∇Cᵀ`) and magnitude (Lagrange multiplier `λ`):

```
λ = −α̃⁻¹ C(x),    where α̃ = α / Δt²
```

This gives the discrete constrained equations of motion:

```
M(x^{n+1} − x̃) − ∇C(x^{n+1})ᵀ λ^{n+1} = 0       (g)
C(x^{n+1}) + α̃ λ^{n+1} = 0                          (h)
```

where `x̃ = xⁿ + Δt vⁿ` is the predicted (inertial) position. The compliance `α̃` **regularizes** the constraint:
it limits the constraint force so the system behaves as if attached to an elastic potential with stiffness
`1/α`.

### 2. Solving via quasi-Newton iteration

Linearize `(g, h)` as a Newton subproblem:

```
[ K        −∇Cᵀ(xᵢ) ] [ Δx ]     [ g(xᵢ, λᵢ) ]
[ ∇C(xᵢ)   α̃        ] [ Δλ ]  = −[ h(xᵢ, λᵢ) ]
```

Two simplifications make this practical:

1. **Approximate `K ≈ M`** — drops constraint Hessians, introduces only `O(Δt²)` local error, does not
   change the fixed-point solution (quasi-Newton).
2. **Assume `g(xᵢ, λᵢ) = 0`** — exact at the first iteration when initialized with `x₀ = x̃`, `λ₀ = 0`;
   remains small when constraint gradients change slowly.

Take the Schur complement with respect to `M`:

```
[ ∇C(xᵢ) M⁻¹ ∇C(xᵢ)ᵀ + α̃ ] Δλ = −C(xᵢ) − α̃ λᵢ
```

Position update:

```
Δx = M⁻¹ ∇C(xᵢ)ᵀ Δλ
```

### 3. The Gauss–Seidel update — the core formula

For a single constraint `j`, solve directly:

```
         −Cⱼ(xᵢ) − α̃ⱼ λᵢⱼ
Δλⱼ = ─────────────────────────────
       ∇Cⱼ M⁻¹ ∇Cⱼᵀ + α̃ⱼ
```

Then update:

```
Δx  = M⁻¹ ∇Cⱼᵀ Δλⱼ
λᵢ₊₁ = λᵢ + Δλ
xᵢ₊₁ = xᵢ + Δx
```

This is the **entire XPBD engine**. Compare to PBD's scaling factor `s`:

- When `α = 0` (infinite stiffness): `Δλⱼ` reduces exactly to PBD's `sⱼ` with `k = 1`.
- When `α > 0`: the compliance term regularizes the denominator and introduces the `−α̃ⱼ λᵢⱼ` memory
  in the numerator — the constraint "remembers" how much force it has already applied.

### 4. The simulation loop

```
(1)  predict position  x̃ ← xⁿ + Δt vⁿ + Δt² M⁻¹ f_ext(xⁿ)

(3)  initialize solve  x₀ ← x̃
(4)  initialize multipliers  λ₀ ← 0

(5)  repeat solverIterations:
(6)      for all constraints j:
(7)          compute Δλⱼ                         # Eq (18)
(8)          compute Δx = M⁻¹ ∇Cⱼᵀ Δλⱼ          # Eq (17)
(9)          λ ← λ + Δλ
(10)         x ← x + Δx
(11)     end
(12) end

(15) update positions  x^{n+1} ← xᵢ
(16) update velocities v^{n+1} ← (x^{n+1} − xⁿ) / Δt
```

Identical to PBD with the addition of lines 4 (initialize `λ`), 7 (compute `Δλ` with compliance), and 9
(accumulate `λ`). One extra scalar per constraint.

### 5. Damping

Model additional dissipation via a Rayleigh potential:

```
D(x, v) = ½ Ċ(x)ᵀ β Ċ(x)
```

where `β` is a damping stiffness matrix (not inverse). Combine elastic and damping multipliers into one
equation:

```
         −Cⱼ(xᵢ) − α̃ⱼ λᵢⱼ − γⱼ ∇Cⱼ (xᵢ − xⁿ)
Δλⱼ = ──────────────────────────────────────────────────
         (1 + γⱼ) ∇Cⱼ M⁻¹ ∇Cⱼᵀ + α̃ⱼ
```

where `γⱼ = α̃ⱼ β̃ⱼ / Δt`. The damping force acts along the constraint gradient direction and requires no
additional storage beyond the combined `λ`.

### 6. FEM example — cantilever beam

Traditional FEM reformulates naturally in the compliant constraint framework. For a triangular element
with linear isotropic material:

```
C_tri(x) = ε_tri = [εₓ, εᵧ, εₓᵧ]ᵀ     (strain tensor in Voigt notation)
```

The compliance matrix is the inverse stiffness matrix in terms of Lamé parameters:

```
         ⎡ λ+2μ    λ     0  ⎤⁻¹
α_tri =  ⎢   λ   λ+2μ   0  ⎥
         ⎣   0      0    2μ ⎦
```

This correctly couples strains to model Poisson's effect — a key advantage over Strain-Based Dynamics,
which treats strain directions independently.

## Results

- **Simple harmonic oscillator**: XPBD matches the analytic solution closely regardless of iteration count;
  PBD's period and damping shift non-linearly with iterations.
- **Chain** (20 particles, `α = 10⁻⁸`): constraint force error vs. Newton reference is 6% at 50 iters, 2% at
  100, 0.5% at 1000.
- **Cantilever beam** (St. Venant–Kirchhoff FEM, `E = 10⁵`, `ν = 0.3`): 20 XPBD iterations are
  visually indistinguishable from the Newton reference.
- **Cloth** (64×64 grid, 24k distance constraints): XPBD behavior is qualitatively unchanged across 20–160
  iterations; PBD becomes progressively stiffer.
- **Performance overhead**: < 2% additional cost per iteration over PBD (one scalar multiply-add per
  constraint).

| Iterations | PBD (ms/step) | XPBD (ms/step) |
|---|---|---|
| 20 | 0.95 | 0.97 |
| 40 | 1.75 | 1.78 |
| 80 | 3.25 | 3.34 |
| 160 | 5.61 | 5.65 |

GPU results (NVIDIA GTX 1070) with Jacobi-style iteration for 3D models.

## Limitations

- **In the limit `α = 0`, XPBD = PBD with `k = 1`.** It still requires the same number of iterations to
  converge for infinitely stiff constraints — compliance decouples stiffness from iterations, but does not
  accelerate convergence.
- **Low iteration counts that terminate before convergence** introduce artificial compliance (the material
  appears softer than specified).
- **Only an approximation of implicit Euler.** The quasi-Newton assumptions (`K ≈ M`, `g ≈ 0`) introduce
  small error; traditional methods may be preferable when accuracy guarantees are needed.
- **No warm-starting** (yet) — `λ` is reset to zero each step. Temporal coherence from previous-frame
  multipliers is noted as future work.

## Relationship to PBD and follow-ups

- **PBD** (Müller et al. 2007): XPBD is a strict superset. Setting `α = 0` in XPBD recovers PBD with
  `k = 1`. The `Δλ` formula reduces to PBD's scaling factor `s`. XPBD adds physical meaning (energy,
  force) to what was previously a purely geometric projection.
- **Projective Dynamics** (Bouaziz et al. 2014): A global solver that pre-factors the system matrix — faster
  convergence but expensive re-factorization under topology changes. XPBD uses local Gauss–Seidel
  iterations, trading convergence rate for simplicity and dynamic topology support.
- **Small Steps PBD** (Macklin et al. 2019): Exploits substepping (many small `Δt` steps with few
  iterations each) to achieve stiffness. Complementary to XPBD — substeps improve convergence while
  compliance decouples material from solver.
