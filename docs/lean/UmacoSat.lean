import Mathlib

/-!
# MACO (UMACO's SAT solver): theorem obligations for a P = NP claim

The algorithm in `examples/macov8no-3-25-02-2025.py` is, per iteration:

1. **Construct** `K` assignments from a *product distribution*: for each variable `v`
   independently, `P(x_v = 1) = τ(v,1)^α / (τ(v,0)^α + τ(v,1)^α)` (times a noise factor).
   Clause structure never enters this step (`beta` is passed to the kernel and unused).
2. **Local search** on the top 20 % of ants: at most `F = 20` flips, each flip a random
   variable of a random unsatisfied clause, accepted if weighted score does not drop,
   else with Metropolis probability.
3. **Evaluate** weighted clause satisfaction `Q_a`.
4. **Clause weights** update from coverage counts.
5. **Pheromone**: evaporate, then every ant `a` deposits `α · Q_a^{3/2}` on *its own*
   polarity at *every* variable.
6. Stop when unweighted satisfaction is 100 %, or after `T_max` iterations.

A claim that this decides SAT in polynomial time decomposes into three obligations:

* **O1 (runtime)**: runs in time polynomial in `n + m`.  *Proved below; in fact linear,
  because the budget is a constant.  So "polynomial scaling" is vacuous.*
* **O2 (soundness)**: if it reports 100 %, the formula is satisfiable.  *Proved below.*
* **O3 (completeness)**: on every satisfiable formula it reaches 100 % within budget with
  probability bounded below.  *This is the whole content.  It is stated as a definition
  (`Complete`), never asserted.  We prove what it is equivalent to (`decision_rule_correct_iff`)
  and prove structural lemmas that bound the mechanisms available to achieve it.*

Nothing in this file uses `sorry` or an added axiom.
-/

namespace UmacoSat

/-! ## 1. CNF semantics -/

structure Lit where
  var : ℕ
  pos : Bool
deriving DecidableEq, Repr

abbrev Clause := List Lit
abbrev CNF := List Clause
abbrev Assignment := ℕ → Bool

def Lit.eval (a : Assignment) (l : Lit) : Bool :=
  if l.pos then a l.var else !(a l.var)

def Clause.sat (a : Assignment) (c : Clause) : Bool :=
  c.any (Lit.eval a)

/-- The unweighted "clauses satisfied" count used by the harness's `[SOLUTION ANALYSIS]`
line and the `abs(1 - best_q) < 1e-9` exit test. -/
def CNF.satCount (a : Assignment) (φ : CNF) : ℕ :=
  (φ.filter (Clause.sat a)).length

def CNF.Satisfies (a : Assignment) (φ : CNF) : Prop :=
  ∀ c ∈ φ, Clause.sat a c = true

def CNF.Satisfiable (φ : CNF) : Prop :=
  ∃ a : Assignment, φ.Satisfies a

/-! ## 2. O2: soundness of the 100 % check -/

/-- If the recount says every clause is satisfied, the formula is satisfiable.
This is the only direction the repository's data can ever certify. -/
theorem soundness (a : Assignment) (φ : CNF) (h : φ.satCount a = φ.length) :
    φ.Satisfiable := by
  refine ⟨a, fun c hc => ?_⟩
  have hall := List.length_filter_eq_length_iff.mp h
  simpa using hall c hc

/-! ## 3. O1: runtime of a fixed-budget run -/

/-- The constants the solver is run with. In the benchmarks: `T ≤ 5000`, `K ≤ 3072`,
`F = 20`, `k = 3` (or `4`). -/
structure Budget where
  T : ℕ   -- max iterations
  K : ℕ   -- ants
  F : ℕ   -- max flips per selected ant
  k : ℕ   -- max clause width

/-- Operation count per run: each iteration constructs `K·n` bits, evaluates `K·m·k`
literals, and each of at most `F` flips per ant rescans all clauses twice. -/
def cost (b : Budget) (n m : ℕ) : ℕ :=
  b.T * b.K * (n + m * b.k * (1 + 2 * b.F))

/-- With the budget fixed, the run is **linear** in the input size `n + m`.
Consequently a polynomial fit to wall-clock time carries no information about the
complexity of *deciding* SAT: any fixed-budget procedure has this property. -/
theorem cost_linear (b : Budget) (n m : ℕ) :
    cost b n m ≤ (b.T * b.K * (1 + b.k * (1 + 2 * b.F))) * (n + m) := by
  unfold cost
  have h1 : n + m * b.k * (1 + 2 * b.F) ≤ (1 + b.k * (1 + 2 * b.F)) * (n + m) := by
    nlinarith [Nat.zero_le n, Nat.zero_le m, Nat.zero_le (b.k * (1 + 2 * b.F))]
  calc b.T * b.K * (n + m * b.k * (1 + 2 * b.F))
      ≤ b.T * b.K * ((1 + b.k * (1 + 2 * b.F)) * (n + m)) :=
        Nat.mul_le_mul_left _ h1
    _ = (b.T * b.K * (1 + b.k * (1 + 2 * b.F))) * (n + m) := by ring

/-! ## 4. The decision rule, and what it needs -/

/-- An abstract randomized solver with a finite seed space `Fin N`.
`run φ s = some a` means "reached 100 %", and `sound` records O2: the returned
assignment really satisfies `φ` (this is what the Python-level recount checks). -/
structure Solver (N : ℕ) where
  run : CNF → Fin N → Option Assignment
  sound : ∀ φ s a, run φ s = some a → φ.Satisfies a

variable {N : ℕ}

/-- Number of seeds on which the solver reaches 100 %. -/
def successes (S : Solver N) (φ : CNF) : ℕ :=
  (Finset.univ.filter (fun s : Fin N => (S.run φ s).isSome)).card

/-- The paper's decision rule (§5.2): answer SAT iff 100 % was reached. -/
def decideSat (S : Solver N) (φ : CNF) (s : Fin N) : Bool :=
  (S.run φ s).isSome

/-- The rule never errs on an unsatisfiable formula (this is just soundness). -/
theorem decide_unsat_correct (S : Solver N) (φ : CNF) (hφ : ¬ φ.Satisfiable) (s : Fin N) :
    decideSat S φ s = false := by
  unfold decideSat
  cases h : S.run φ s with
  | none => rfl
  | some a => exact absurd ⟨a, S.sound φ s a h⟩ hφ

/-- On a satisfiable formula, the rule errs on exactly the seeds that do not succeed. -/
theorem decide_sat_errors (S : Solver N) (φ : CNF) :
    (Finset.univ.filter (fun s : Fin N => decideSat S φ s = false)).card
      = N - successes S φ := by
  unfold successes decideSat
  have : (Finset.univ.filter (fun s : Fin N => (S.run φ s).isSome = false))
       = (Finset.univ.filter (fun s : Fin N => (S.run φ s).isSome = true))ᶜ := by
    ext s; simp
  rw [this, Finset.card_compl, Fintype.card_fin]

/-- **O3, as a definition.** Completeness with success fraction `δ`: on every satisfiable
formula, at least a `δ` fraction of seeds reach 100 %.  This is the obligation that a
P = NP (precisely: SAT ∈ RP, hence NP = RP) claim from MACO would have to discharge, and it
is the one obligation the repository contains no argument for. -/
def Complete (S : Solver N) (δ : ℝ) : Prop :=
  ∀ φ : CNF, φ.Satisfiable → δ * N ≤ (successes S φ : ℝ)

/-- The decision rule is error-free on every formula and every seed **iff** the solver is
complete with `δ = 1`.  So "MACO's failure to reach 100 % is itself the correct answer"
(paper §5.2) is not an observation about UNSAT instances; it is exactly the unproven
completeness claim about SAT instances. -/
theorem decision_rule_correct_iff (S : Solver N) :
    (∀ φ : CNF, ∀ s : Fin N, decideSat S φ s = true ↔ φ.Satisfiable)
      ↔ Complete S 1 := by
  constructor
  · intro h φ hφ
    simp only [successes]
    have : (Finset.univ.filter (fun s : Fin N => (S.run φ s).isSome)) = Finset.univ := by
      ext s
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, iff_true]
      exact (h φ s).mpr hφ
    rw [this]
    simp
  · intro h φ s
    constructor
    · intro hs
      unfold decideSat at hs
      cases hr : S.run φ s with
      | none => simp [hr] at hs
      | some a => exact ⟨a, S.sound φ s a hr⟩
    · intro hφ
      have hφ' := h φ hφ
      unfold successes at hφ'
      simp only [one_mul] at hφ'
      have hcard : (Finset.univ.filter (fun s : Fin N => (S.run φ s).isSome)).card = N := by
        have hle := Finset.card_le_univ (Finset.univ.filter (fun s : Fin N => (S.run φ s).isSome))
        simp only [Fintype.card_fin] at hle
        exact_mod_cast le_antisymm hle (by exact_mod_cast hφ')
      have hcard' : (Finset.univ.filter (fun s : Fin N => (S.run φ s).isSome)).card
          = Fintype.card (Fin N) := by rw [Fintype.card_fin]; exact hcard
      have hfull := Finset.eq_univ_of_card _ hcard'
      have hs : s ∈ Finset.univ.filter (fun s : Fin N => (S.run φ s).isSome) := by
        rw [hfull]; exact Finset.mem_univ s
      simpa [decideSat] using (Finset.mem_filter.mp hs).2

/-! ## 5. Structural lemmas bounding how O3 could be achieved

The run can reach 100 % only through (a) construction sampling an assignment that is
(b) within `F` flips of a satisfying one.  We bound both mechanisms. -/

/-! ### 5a. Construction is a product distribution -/

/-- For a product distribution with per-variable marginal `p v` of the *correct* polarity
of a target assignment, the probability of constructing that assignment exactly is
`∏ p v`.  If at least `far.card` variables have marginal at most `1 - δ`, this is at
most `(1 - δ) ^ far.card`. -/
theorem prod_marginals_le {ι : Type*} (s far : Finset ι) (p : ι → ℝ) (δ : ℝ)
    (hp : ∀ i ∈ s, 0 ≤ p i ∧ p i ≤ 1)
    (hfar : far ⊆ s) (hfarp : ∀ i ∈ far, p i ≤ 1 - δ) :
    ∏ i ∈ s, p i ≤ (1 - δ) ^ far.card := by
  classical
  have hsplit := Finset.prod_sdiff (f := p) hfar
  have hrest_le : ∏ i ∈ s \ far, p i ≤ 1 :=
    Finset.prod_le_one (fun i hi => (hp i (Finset.mem_sdiff.mp hi).1).1)
      (fun i hi => (hp i (Finset.mem_sdiff.mp hi).1).2)
  have hrest_nn : 0 ≤ ∏ i ∈ s \ far, p i :=
    Finset.prod_nonneg (fun i hi => (hp i (Finset.mem_sdiff.mp hi).1).1)
  have hfar_nn : 0 ≤ ∏ i ∈ far, p i :=
    Finset.prod_nonneg (fun i hi => (hp i (hfar hi)).1)
  have hfar_le : ∏ i ∈ far, p i ≤ ∏ i ∈ far, (1 - δ) :=
    Finset.prod_le_prod (fun i hi => (hp i (hfar hi)).1) hfarp
  rw [Finset.prod_const] at hfar_le
  calc ∏ i ∈ s, p i = (∏ i ∈ s \ far, p i) * ∏ i ∈ far, p i := hsplit.symm
    _ ≤ 1 * ∏ i ∈ far, p i := by gcongr
    _ = ∏ i ∈ far, p i := one_mul _
    _ ≤ (1 - δ) ^ far.card := hfar_le

/-- `(1 - δ)^L ≤ exp(-δ L)`. -/
theorem one_sub_pow_le_exp (δ : ℝ) (hδ1 : δ ≤ 1) (L : ℕ) :
    (1 - δ) ^ L ≤ Real.exp (-(δ * L)) := by
  have h1 : 1 - δ ≤ Real.exp (-δ) := by
    have := Real.add_one_le_exp (-δ); linarith
  have h2 : (1 - δ) ^ L ≤ (Real.exp (-δ)) ^ L :=
    pow_le_pow_left₀ (by linarith) h1 L
  rw [← Real.exp_nat_mul] at h2
  calc (1 - δ) ^ L ≤ Real.exp (L * -δ) := h2
    _ = Real.exp (-(δ * L)) := by ring_nf

/-- Union bound over the `K·T` constructions of a run: the probability that *any*
construction hits the target is at most `K·T·(1-δ)^L`. If a run is to succeed with
probability at least `1/2` through construction alone, then
`L ≤ ln(2KT)/δ`: all but `O(log(KT)/δ)` marginals must already be within `δ` of the
solution.  With `K·T ≤ 3072·5000`, that is at most `⌈17/δ⌉` "undecided" variables
per hit, independent of `n`. -/
theorem far_vars_bound (K T L : ℕ) (δ : ℝ) (hδ0 : 0 < δ) (hδ1 : δ ≤ 1)
    (hKT : 0 < (K * T : ℝ))
    (hsucc : (1 : ℝ) / 2 ≤ (K * T : ℝ) * (1 - δ) ^ L) :
    (L : ℝ) ≤ Real.log (2 * (K * T : ℝ)) / δ := by
  have hexp := one_sub_pow_le_exp δ hδ1 L
  have h1 : (1 : ℝ) / 2 ≤ (K * T : ℝ) * Real.exp (-(δ * L)) :=
    le_trans hsucc (by gcongr)
  have h2 : Real.exp (-(δ * L)) ≥ 1 / (2 * (K * T : ℝ)) := by
    rw [ge_iff_le, div_le_iff₀ (by positivity)]
    linarith [mul_comm (K * T : ℝ) (Real.exp (-(δ * L)))]
  have h3 : Real.log (1 / (2 * (K * T : ℝ))) ≤ -(δ * L) := by
    have := Real.log_le_log (by positivity) h2
    simpa [Real.log_exp] using this
  rw [Real.log_div (by norm_num) (by positivity), Real.log_one, zero_sub] at h3
  rw [le_div_iff₀ hδ0]
  linarith

/-! ### 5b. Pheromone deposit carries a bounded fitness signal -/

/-- Every ant deposits `w(Q_a) = α·Q_a^{3/2}` on the polarity *it chose*.  So the deposit
on polarity `b` at variable `v` is a sum over the ants that chose `b`.  If all ants have
quality in `[q_min, q_max]`, the deposit is sandwiched between `w_min · count` and
`w_max · count`. -/
theorem deposit_bounds {ι : Type*} (A : Finset ι) (w : ι → ℝ) (wmin wmax : ℝ)
    (h : ∀ a ∈ A, wmin ≤ w a ∧ w a ≤ wmax) :
    (A.card : ℝ) * wmin ≤ ∑ a ∈ A, w a ∧ ∑ a ∈ A, w a ≤ (A.card : ℝ) * wmax := by
  constructor
  · have := Finset.card_nsmul_le_sum A w wmin (fun a ha => (h a ha).1)
    simpa [nsmul_eq_mul] using this
  · have := Finset.sum_le_card_nsmul A w wmax (fun a ha => (h a ha).2)
    simpa [nsmul_eq_mul] using this

/-- Consequently the deposited *ratio* between the two polarities at a variable differs
from the plain *frequency* ratio (how many ants chose each polarity) by a factor of at
most `w_max / w_min`.  For `w = α Q^{3/2}` and the observed hard-run band
`Q ∈ [0.93, 0.96]`, that factor is `(0.96/0.93)^{1.5} ≈ 1.05`.  The pheromone update
is therefore, to within 5 %, a *self-reinforcement of the current marginals*, not a
fitness-driven move toward a satisfying assignment. -/
theorem deposit_ratio_bound {ι : Type*} (A₁ A₀ : Finset ι) (w : ι → ℝ) (wmin wmax : ℝ)
    (hwmin : 0 < wmin)
    (h₁ : ∀ a ∈ A₁, wmin ≤ w a ∧ w a ≤ wmax) (h₀ : ∀ a ∈ A₀, wmin ≤ w a ∧ w a ≤ wmax)
    (hA₀ : 0 < A₀.card) :
    (∑ a ∈ A₁, w a) / (∑ a ∈ A₀, w a)
      ≤ (wmax / wmin) * ((A₁.card : ℝ) / A₀.card) := by
  obtain ⟨_, hup⟩ := deposit_bounds A₁ w wmin wmax h₁
  obtain ⟨hlo, _⟩ := deposit_bounds A₀ w wmin wmax h₀
  have hA₀' : (0 : ℝ) < A₀.card := by exact_mod_cast hA₀
  have hden : 0 < (A₀.card : ℝ) * wmin := by positivity
  have hnum_nn : 0 ≤ (A₁.card : ℝ) * wmax := by
    have : 0 ≤ ∑ a ∈ A₁, w a :=
      Finset.sum_nonneg (fun a ha => le_trans hwmin.le (h₁ a ha).1)
    linarith
  calc (∑ a ∈ A₁, w a) / (∑ a ∈ A₀, w a)
      ≤ ((A₁.card : ℝ) * wmax) / ((A₀.card : ℝ) * wmin) :=
        div_le_div₀ hnum_nn hup hden hlo
    _ = (wmax / wmin) * ((A₁.card : ℝ) / A₀.card) := by
        field_simp

/-! ### 5c. Local search moves at most `F` coordinates -/

def flipAt (a : Assignment) (v : ℕ) : Assignment :=
  fun u => if u = v then !(a u) else a u

/-- Apply a sequence of flips (the local-search kernel's accepted moves, in order). -/
def applyFlips (a : Assignment) : List ℕ → Assignment
  | [] => a
  | v :: vs => applyFlips (flipAt a v) vs

theorem applyFlips_diff_mem (fs : List ℕ) :
    ∀ (a : Assignment) (u : ℕ), applyFlips a fs u ≠ a u → u ∈ fs := by
  induction fs with
  | nil => intro a u h; exact absurd rfl h
  | cons v vs ih =>
    intro a u h
    by_cases hu : u ∈ vs
    · exact List.mem_cons_of_mem v hu
    · have h1 : applyFlips (flipAt a v) vs u = flipAt a v u := by
        by_contra hne
        exact hu (ih (flipAt a v) u hne)
      simp only [applyFlips] at h
      rw [h1] at h
      unfold flipAt at h
      by_cases huv : u = v
      · exact huv ▸ List.mem_cons_self
      · simp [huv] at h

/-- Hamming distance from the constructed assignment after at most `F` accepted flips
is at most `F`.  Combined with 5a: a run can only succeed on an instance whose satisfying
assignments lie within Hamming radius `F = 20` of something the product sampler draws. -/
theorem hamming_after_flips (a : Assignment) (fs : List ℕ) (n : ℕ) :
    ((Finset.range n).filter (fun u => applyFlips a fs u ≠ a u)).card ≤ fs.length := by
  calc ((Finset.range n).filter (fun u => applyFlips a fs u ≠ a u)).card
      ≤ fs.toFinset.card := by
        apply Finset.card_le_card
        intro u hu
        rw [Finset.mem_filter] at hu
        exact List.mem_toFinset.mpr (applyFlips_diff_mem fs a u hu.2)
    _ ≤ fs.length := List.toFinset_card_le fs

/-! ## 6. Summary of what is and is not established

* `soundness`, `decide_unsat_correct` : O2, and the half of the decision rule that holds.
* `cost_linear`                       : O1, vacuously — a fixed budget is linear time.
* `decision_rule_correct_iff`         : the paper's UNSAT-by-failure rule is *equivalent*
                                        to `Complete S 1`, i.e. to O3.
* `prod_marginals_le`, `far_vars_bound`: for construction to hit a solution, the pheromone
                                        marginals must already be within `δ` of it on all
                                        but `ln(2KT)/δ` variables.
* `deposit_ratio_bound`               : per iteration, fitness can tilt those marginals by
                                        at most a `(q_max/q_min)^{3/2}` factor beyond
                                        self-reinforcement.
* `hamming_after_flips`               : local search extends reach by Hamming radius `F`.

O3 is therefore equivalent to: *the pheromone Markov chain, under a per-step fitness
signal of at most `(q_max/q_min)^{3/2}`, concentrates all but `O(log KT / δ)` marginals to
within `δ` of a satisfying assignment (or within Hamming radius `F` of one) on every
satisfiable 3-CNF, within `T` steps.*  No argument for that exists in the repository, the
paper, or the ACO runtime-analysis literature (which proves the opposite for pheromone
samplers on needle-like landscapes), and the repository's own logs contain 14 satisfiable
instances (n = 1000, 4-SAT, MiniSat SAT in 0.02 s) on which it fails. -/

end UmacoSat


/-! ## 8. System C: the pheromone that integrates fitness (research/system_c/SystemC_3SAT_attack.md)

Discrete cores of Theorems 1, 4, 7 and Lemmas 6, 9 of that log. -/

namespace UmacoSat

/-- Theorem 1 (Lyapunov summand): `(a − b)(log a − log b) ≥ 0` for `a, b > 0`. Each term of
`d/dt log F̃(p)` along the replicator flow has this form times a nonnegative factor. -/
theorem mul_log_sub_nonneg (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
    0 ≤ (a - b) * (Real.log a - Real.log b) := by
  rcases le_total a b with h | h
  · have h1 : a - b ≤ 0 := by linarith
    have h2 : Real.log a - Real.log b ≤ 0 := by
      have := Real.log_le_log ha h; linarith
    nlinarith
  · have h1 : 0 ≤ a - b := by linarith
    have h2 : 0 ≤ Real.log a - Real.log b := by
      have := Real.log_le_log hb h; linarith
    exact mul_nonneg h1 h2

/-- Theorem 4 (escape time): a persistently violated clause has weight at least `1 + 5λt` after
`t` steps; once that exceeds the total critical weight `L` of any variable in it, the vertex is no
longer a local maximum. The threshold is `t > (L − 1)/(5λ)`. -/
theorem escape_time (lam L t : ℝ) (hlam : 0 < lam) (ht : (L - 1) / (5 * lam) < t) :
    L < 1 + 5 * lam * t := by
  have h5 : 0 < 5 * lam := by positivity
  have := (div_lt_iff₀ h5).mp ht
  linarith

/-- A literal's value under `x` and under `x*` differ iff the variable's values differ. -/
theorem lit_eval_ne_iff (x xs : Assignment) (l : Lit) :
    Lit.eval x l ≠ Lit.eval xs l ↔ x l.var ≠ xs l.var := by
  unfold Lit.eval
  cases l.pos <;> cases x l.var <;> cases xs l.var <;> simp

/-- Lemma 6 / Lemma 9 core (k arbitrary): a clause unsatisfied by `x` and satisfied by `x*`
contains a literal whose variable is *wrong* (`x v ≠ x* v`). -/
theorem unsat_sat_has_wrong (x xs : Assignment) (c : Clause)
    (hx : Clause.sat x c = false) (hxs : Clause.sat xs c = true) :
    ∃ l ∈ c, x l.var ≠ xs l.var := by
  unfold Clause.sat at hx hxs
  obtain ⟨l, hl, hls⟩ := List.any_eq_true.mp hxs
  refine ⟨l, hl, ?_⟩
  have hall := List.any_eq_false.mp hx
  have hxl : Lit.eval x l = false := by
    have := hall l hl; simpa using this
  rw [← lit_eval_ne_iff]
  rw [hxl, hls]; decide

/-- Toward-probability of a uniform choice in a `k`-literal clause is at least `1/k`
(Lemma 6 for `k = 2` gives `≥ 1/2`; for `k = 3` only `≥ 1/3`). Stated as a counting bound:
the number of wrong literals is at least one. -/
theorem toward_prob_ge_inv_k (c : Clause) (x xs : Assignment)
    (hx : Clause.sat x c = false) (hxs : Clause.sat xs c = true) :
    (1 : ℚ) / c.length ≤
      ((c.filter (fun l => decide (x l.var ≠ xs l.var))).length : ℚ) / c.length := by
  obtain ⟨l, hl, hne⟩ := unsat_sat_has_wrong x xs c hx hxs
  have hpos : 0 < c.length := List.length_pos_of_mem hl
  have h1 : 1 ≤ (c.filter (fun l => decide (x l.var ≠ xs l.var))).length := by
    have : l ∈ c.filter (fun l => decide (x l.var ≠ xs l.var)) := by
      rw [List.mem_filter]; exact ⟨hl, by simpa using hne⟩
    exact List.length_pos_of_mem this
  have hpos' : (0 : ℚ) < c.length := by exact_mod_cast hpos
  apply div_le_div_of_nonneg_right _ hpos'.le
  exact_mod_cast h1

/-- Theorem 7 (weights on the fixed clause cannot help): with acceptance probabilities
`A ≤ 1` on wrong variables and `A = 1` on right variables, the accepted-flip toward-probability
`Σ_wrong A / Σ_all A` is at most `|wrong| / (|wrong| + |right|)`, i.e. exactly WalkSAT's `W/k`
when all are accepted. -/
theorem toward_prob_le_of_accept {ι : Type*} [DecidableEq ι] (Wr Rt : Finset ι)
    (hdisj : Disjoint Wr Rt)
    (A : ι → ℝ) (hA0 : ∀ i ∈ Wr, 0 ≤ A i) (hA1 : ∀ i ∈ Wr, A i ≤ 1) (hR : ∀ i ∈ Rt, A i = 1)
    (hRpos : 0 < Rt.card) :
    (∑ i ∈ Wr, A i) / (∑ i ∈ Wr ∪ Rt, A i) ≤ (Wr.card : ℝ) / (Wr.card + Rt.card) := by
  have hsumR : ∑ i ∈ Rt, A i = Rt.card := by
    rw [Finset.sum_congr rfl hR]; simp
  have hsplit : ∑ i ∈ Wr ∪ Rt, A i = (∑ i ∈ Wr, A i) + Rt.card := by
    rw [Finset.sum_union hdisj, hsumR]
  set t := ∑ i ∈ Wr, A i with ht
  have ht0 : 0 ≤ t := Finset.sum_nonneg hA0
  have htW : t ≤ Wr.card := by
    have := Finset.sum_le_card_nsmul Wr A 1 hA1
    simpa [nsmul_eq_mul] using this
  have hR' : (0 : ℝ) < Rt.card := by exact_mod_cast hRpos
  rw [hsplit]
  -- t/(t+R) ≤ W/(W+R)  ⇔  t(W+R) ≤ W(t+R)  ⇔  tR ≤ WR
  rw [div_le_div_iff₀ (by linarith) (by linarith)]
  nlinarith

/-- Lemma 9 (break asymmetry): if a clause satisfied by `x*` is *critical* on a literal `l` under
`x` (that literal is its unique true literal) and `l`'s variable is wrong, then the clause contains
another literal whose variable is wrong. So flipping a wrong variable can only break clauses that
already contain a second wrong variable. -/
theorem critical_wrong_has_wrong (x xs : Assignment) (c : Clause) (l : Lit)
    (hcrit_true : Lit.eval x l = true)
    (hcrit_uniq : ∀ l' ∈ c, l' ≠ l → Lit.eval x l' = false)
    (hwrong : x l.var ≠ xs l.var)
    (hxs : Clause.sat xs c = true) :
    ∃ l' ∈ c, l' ≠ l ∧ x l'.var ≠ xs l'.var := by
  have hls : Lit.eval xs l = false := by
    have hne := (lit_eval_ne_iff x xs l).mpr hwrong
    rw [hcrit_true] at hne
    cases h : Lit.eval xs l
    · rfl
    · exact absurd h.symm hne
  unfold Clause.sat at hxs
  obtain ⟨l', hl', hl's⟩ := List.any_eq_true.mp hxs
  have hne : l' ≠ l := by
    intro h; rw [h] at hl's; rw [hls] at hl's; exact Bool.false_ne_true hl's
  refine ⟨l', hl', hne, ?_⟩
  have hxl' := hcrit_uniq l' hl' hne
  rw [← lit_eval_ne_iff, hxl', hl's]; decide

end UmacoSat


/-! ## 9. Planted-model closed forms (research/system_c/SystemC_3SAT_attack.md §8) -/

namespace UmacoSat

/-- Theorem 11: for uniform in-clause selection on the planted model, the toward-probability at
wrongness `δ` is `δ / (1 − (1−δ)³)`, and it is at least `1/2` iff `δ² − 3δ + 1 ≤ 0`, i.e. iff
`δ ≥ (3 − √5)/2 ≈ 0.382`. Below that distance the walk has negative drift: the tunnel. -/
theorem uniform_toward_crossover (δ : ℝ) (h0 : 0 < δ) (h1 : δ < 1) :
    (1 : ℝ) / 2 ≤ δ / (1 - (1 - δ) ^ 3) ↔ δ ^ 2 - 3 * δ + 1 ≤ 0 := by
  have hden : 0 < 1 - (1 - δ) ^ 3 := by
    have : (1 - δ) ^ 3 < 1 := by
      have h2 : 0 < 1 - δ := by linarith
      have h3 : 1 - δ < 1 := by linarith
      calc (1 - δ) ^ 3 < 1 ^ 3 := by gcongr
        _ = 1 := by norm_num
    linarith
  rw [div_le_div_iff₀ (by norm_num) hden]
  constructor
  · intro h; nlinarith
  · intro h; nlinarith

/-- The crossover point itself: `δ_c = (3 − √5)/2` is the root in `(0,1)` of `δ² − 3δ + 1`. -/
theorem crossover_root : ((3 - Real.sqrt 5) / 2) ^ 2 - 3 * ((3 - Real.sqrt 5) / 2) + 1 = 0 := by
  have h5 : Real.sqrt 5 ^ 2 = 5 := Real.sq_sqrt (by norm_num)
  nlinarith [h5]

/-- Theorem 12 (annealed pheromone drift, closed form). Enumerating the seven sign patterns of a
planted clause with the variable in slot 0, where a literal with planted truth value `t` is false
under the current assignment with probability `(1−t)(1−δ) + tδ`, the signed sum of
"other two literals false" probabilities is exactly `(1 − δ)²`. Multiplying by `3r/7` clauses per
pattern-slot gives `E[∂_v Q̃ toward x*_v] = (3r/7)(1−δ)² > 0`. -/
theorem annealed_drift_identity (δ : ℝ) :
    -- t₀ = 1 patterns: (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    ((1 - δ) * (1 - δ) + (1 - δ) * δ + δ * (1 - δ) + δ * δ)
    -- t₀ = 0 patterns: (0,0,1), (0,1,0), (0,1,1)
    - ((1 - δ) * δ + δ * (1 - δ) + δ * δ)
    = (1 - δ) ^ 2 := by ring

/-- Hence the annealed drift is strictly positive for every `δ < 1` and every density `r > 0`. -/
theorem annealed_drift_pos (r δ : ℝ) (hr : 0 < r) (hδ : δ < 1) :
    0 < (3 * r / 7) * (1 - δ) ^ 2 := by
  have : 0 < 1 - δ := by linarith
  positivity

/-- Theorem 14 core: at `δ = 0` the per-variable drift is exactly the criticality count
`crit*(v) ≥ 0`; the negative terms all carry a factor `δ`. Stated as the pattern polynomial. -/
theorem quenched_drift_at_zero (c n101 n110 n111 n010 n001 n011 : ℝ) :
    c * (1 - 0) ^ 2 + (n101 + n110) * 0 * (1 - 0) + n111 * 0 ^ 2
      - (n010 + n001) * 0 * (1 - 0) - n011 * 0 ^ 2 = c := by ring

end UmacoSat


/-! ## 10. XOR obstruction and repair cascade (research/system_c/SystemC_3SAT_attack.md §9–10) -/

namespace UmacoSat

/-- Theorem 15: the four clauses encoding `x₁ ⊕ x₂ ⊕ x₃ = 1` have total unsatisfaction probability
`P[parity = 0] = ½(1 − m₁m₂m₃)` under a product distribution, `m_i = 2p_i − 1`. The left side sums
`P[x = a]` over the four even-parity `a`. -/
theorem xor_encoding_is_three_spin (p1 p2 p3 : ℝ) :
    (1 - p1) * (1 - p2) * (1 - p3) + p1 * p2 * (1 - p3) + p1 * (1 - p2) * p3
      + (1 - p1) * p2 * p3
      = (1 / 2) * (1 - (2 * p1 - 1) * (2 * p2 - 1) * (2 * p3 - 1)) := by ring

/-- The annealed XOR drift `(1 − 2δ)²` vanishes exactly at `δ = ½`: the symmetric point is a fixed
point of the mean-field flow on every XOR instance. -/
theorem xor_annealed_drift_zero_at_half : (1 - 2 * (1 / 2 : ℝ)) ^ 2 = 0 := by norm_num

/-- Theorem 17: `u·e^{−u} ≤ 1/e` for all real `u`, hence the cascade branching factor
`b(r) ≤ (3r/7)e^{−3r/7} < 1` at every density. -/
theorem branching_factor_le_inv_e (u : ℝ) : u * Real.exp (-u) ≤ Real.exp (-1) := by
  have h := Real.add_one_le_exp (u - 1)
  have hpos : 0 < Real.exp (-u) := Real.exp_pos _
  calc u * Real.exp (-u) ≤ Real.exp (u - 1) * Real.exp (-u) := by
        apply mul_le_mul_of_nonneg_right _ hpos.le; linarith
    _ = Real.exp (-1) := by rw [← Real.exp_add]; ring_nf

/-- Strict subcriticality: `1/e < 1`. -/
theorem inv_e_lt_one : Real.exp (-1 : ℝ) < 1 := by
  have := Real.exp_lt_exp.mpr (show (-1 : ℝ) < 0 by norm_num)
  simpa using this

end UmacoSat


/-! ## 11. The multilinear extension as an object (research/system_c/SystemC_3SAT_attack.md §1–2)

`prodProb p x` is the product-measure weight of assignment `x` under marginals `p`;
`mlExt p F` is `E_p[F]`, the multilinear extension. -/

namespace UmacoSat

variable {n : ℕ}

/-- Product-measure weight of `x` under marginals `p`. -/
def prodProb (p : Fin n → ℝ) (x : Fin n → Bool) : ℝ :=
  ∏ v, if x v then p v else 1 - p v

/-- Multilinear extension `E_p[F]`. -/
def mlExt (p : Fin n → ℝ) (F : (Fin n → Bool) → ℝ) : ℝ :=
  ∑ x, prodProb p x * F x

/-- The weights are nonnegative on the cube. -/
theorem prodProb_nonneg (p : Fin n → ℝ) (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1) (x : Fin n → Bool) :
    0 ≤ prodProb p x := by
  unfold prodProb
  apply Finset.prod_nonneg
  intro v _
  split_ifs
  · exact (hp v).1
  · linarith [(hp v).2]

/-- The weights sum to one: `∑_x ∏_v g_v(x_v) = ∏_v (g_v 0 + g_v 1)` and each factor is `1`. -/
theorem prodProb_sum_one (p : Fin n → ℝ) : ∑ x, prodProb p x = 1 := by
  unfold prodProb
  have h := Finset.prod_univ_sum (t := fun _ : Fin n => (Finset.univ : Finset Bool))
    (f := fun v b => if b then p v else 1 - p v)
  rw [Fintype.piFinset_univ] at h
  rw [← h]
  simp

/-- Off `v`, `p` and `Function.update p v b` agree, so the product over `{v}ᶜ` is unchanged. -/
theorem prod_compl_update (p : Fin n → ℝ) (v : Fin n) (b : ℝ) (x : Fin n → Bool) :
    ∏ u ∈ {v}ᶜ, (if x u then Function.update p v b u else 1 - Function.update p v b u)
      = ∏ u ∈ {v}ᶜ, (if x u then p u else 1 - p u) := by
  apply Finset.prod_congr rfl
  intro u hu
  have : u ≠ v := by simpa using hu
  simp [Function.update_of_ne this]

/-- **Conditioning identity** (multilinearity in coordinate `v`):
`E_p[F] = (1 − p_v) · E_{p|v←0}[F] + p_v · E_{p|v←1}[F]`. -/
theorem mlExt_split (p : Fin n → ℝ) (F : (Fin n → Bool) → ℝ) (v : Fin n) :
    mlExt p F = (1 - p v) * mlExt (Function.update p v 0) F + p v * mlExt (Function.update p v 1) F := by
  unfold mlExt prodProb
  rw [Finset.mul_sum, Finset.mul_sum, ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro x _
  rw [Fintype.prod_eq_mul_prod_compl v, Fintype.prod_eq_mul_prod_compl v,
      Fintype.prod_eq_mul_prod_compl v, prod_compl_update, prod_compl_update]
  cases hx : x v <;> simp [hx] <;> ring

/-- **Partial derivative identity**: the extension is affine in `p_v` with slope
`E_{p|v←1}[F] − E_{p|v←0}[F]`. Stated as the finite-difference form used in Theorems 1–2. -/
theorem mlExt_slope (p : Fin n → ℝ) (F : (Fin n → Bool) → ℝ) (v : Fin n) (a : ℝ) :
    mlExt (Function.update p v a) F
      = mlExt (Function.update p v 0) F
        + a * (mlExt (Function.update p v 1) F - mlExt (Function.update p v 0) F) := by
  have h := mlExt_split (Function.update p v a) F v
  simp only [Function.update_idem, Function.update_self] at h
  rw [h]; ring

/-- At a vertex `x₀` (marginals in `{0,1}`), the extension equals the function value. -/
theorem prodProb_vertex (x₀ x : Fin n → Bool) :
    prodProb (fun v => if x₀ v then 1 else 0) x = if x = x₀ then 1 else 0 := by
  unfold prodProb
  by_cases h : x = x₀
  · subst h; rw [if_pos rfl]
    apply Finset.prod_eq_one; intro v _; cases hx : x v <;> simp [hx]
  · rw [if_neg h]
    obtain ⟨v, hv⟩ : ∃ v, x v ≠ x₀ v := by
      by_contra hcon; exact h (funext fun v => by by_contra hv; exact hcon ⟨v, hv⟩)
    apply Finset.prod_eq_zero (Finset.mem_univ v)
    cases hx : x v <;> cases hx0 : x₀ v <;> simp_all

theorem mlExt_vertex (x₀ : Fin n → Bool) (F : (Fin n → Bool) → ℝ) :
    mlExt (fun v => if x₀ v then 1 else 0) F = F x₀ := by
  unfold mlExt
  simp [prodProb_vertex]

/-- **Theorem 3 core**: if `F ≤ 1` on the cube and `F x* = 1`, then `E_p[F] ≤ 1 = F x*` for every
`p ∈ [0,1]^n`: a solution is a global maximum of the multilinear extension for every weighting. -/
theorem mlExt_le_one_of_le_one (p : Fin n → ℝ) (hp : ∀ v, 0 ≤ p v ∧ p v ≤ 1)
    (F : (Fin n → Bool) → ℝ) (hF : ∀ x, F x ≤ 1) :
    mlExt p F ≤ 1 := by
  unfold mlExt
  calc ∑ x, prodProb p x * F x ≤ ∑ x, prodProb p x * 1 := by
        apply Finset.sum_le_sum; intro x _
        exact mul_le_mul_of_nonneg_left (hF x) (prodProb_nonneg p hp x)
    _ = 1 := by simp [prodProb_sum_one]

end UmacoSat


/-! ## 12. Theorems 1 and 2 on the real object -/

namespace UmacoSat

variable {n : ℕ}

/-- The slope of the extension in coordinate `v`. -/
def slope (p : Fin n → ℝ) (F : (Fin n → Bool) → ℝ) (v : Fin n) : ℝ :=
  mlExt (Function.update p v 1) F - mlExt (Function.update p v 0) F

/-- `mlExt` is affine in `p_v`: `E_{p|v←a}[F] = E_{p|v←0}[F] + a · slope`. Restated. -/
theorem mlExt_update_eq (p : Fin n → ℝ) (F : (Fin n → Bool) → ℝ) (v : Fin n) (a : ℝ) :
    mlExt (Function.update p v a) F = mlExt (Function.update p v 0) F + a * slope p F v := by
  unfold slope; exact mlExt_slope p F v a

/-- **Theorem 1, discrete coordinate form.** One replicator step in coordinate `v`,
`p_v ← p_v + η p_v (1 − p_v) · slope`, changes the extension by exactly
`η p_v (1 − p_v) · slope² ≥ 0`. So the pheromone ascent is monotone, with equality iff
`p_v ∈ {0,1}` or the slope vanishes: the fixed-point characterization of Theorem 2. -/
theorem replicator_step_gain (p : Fin n → ℝ) (F : (Fin n → Bool) → ℝ) (v : Fin n) (η : ℝ) :
    mlExt (Function.update p v (p v + η * p v * (1 - p v) * slope p F v)) F - mlExt p F
      = η * p v * (1 - p v) * (slope p F v) ^ 2 := by
  have h1 := mlExt_update_eq p F v (p v + η * p v * (1 - p v) * slope p F v)
  have h2 := mlExt_update_eq p F v (p v)
  rw [Function.update_eq_self] at h2
  rw [h1, h2]; ring

theorem replicator_step_gain_nonneg (p : Fin n → ℝ) (F : (Fin n → Bool) → ℝ) (v : Fin n)
    (η : ℝ) (hη : 0 ≤ η) (hp : 0 ≤ p v ∧ p v ≤ 1) :
    0 ≤ mlExt (Function.update p v (p v + η * p v * (1 - p v) * slope p F v)) F - mlExt p F := by
  rw [replicator_step_gain]
  have : 0 ≤ 1 - p v := by linarith [hp.2]
  have := hp.1
  positivity

/-- Vertex marginals of an assignment. -/
def vertex (x : Fin n → Bool) : Fin n → ℝ := fun v => if x v then 1 else 0

theorem vertex_update_true (x : Fin n → Bool) (v : Fin n) :
    Function.update (vertex x) v 1 = vertex (Function.update x v true) := by
  funext u; unfold vertex
  by_cases h : u = v
  · subst h; simp
  · simp [Function.update_of_ne h]

theorem vertex_update_false (x : Fin n → Bool) (v : Fin n) :
    Function.update (vertex x) v 0 = vertex (Function.update x v false) := by
  funext u; unfold vertex
  by_cases h : u = v
  · subst h; simp
  · simp [Function.update_of_ne h]

/-- **Theorem 2, vertex form.** At a vertex `x` the slope in coordinate `v` is the finite
difference `F(x^{v←1}) − F(x^{v←0})`. Hence `x` is asymptotically stable for the replicator
flow iff `F(x) > F(x ⊕ e_v)` for every `v`, i.e. iff `x` is a strict 1-flip local maximum. -/
theorem slope_at_vertex (x : Fin n → Bool) (F : (Fin n → Bool) → ℝ) (v : Fin n) :
    slope (vertex x) F v = F (Function.update x v true) - F (Function.update x v false) := by
  unfold slope
  rw [vertex_update_true, vertex_update_false]
  unfold vertex
  rw [mlExt_vertex, mlExt_vertex]

/-- The extension at a vertex is the function value (restated with `vertex`). -/
theorem mlExt_at_vertex (x : Fin n → Bool) (F : (Fin n → Bool) → ℝ) :
    mlExt (vertex x) F = F x := mlExt_vertex x F

end UmacoSat


/-! ## 13. Bounded weights under decay (research/system_c/SystemC_3SAT_attack.md §11, lesson 2) -/

namespace UmacoSat

/-- **Theorem 19 (EMA weights are bounded regardless of history).** If the stubbornness obeys
`S(t+1) ≤ μ S(t) + λ` with `0 ≤ μ < 1`, then `S(t) ≤ λ/(1−μ) + S(0)` for all `t`. With the code's
EMA branch `S ← μS + (1−μ) f`, `λ = 1−μ`, so `S ≤ 1 + S(0)` and `w = 1 + 5S ≤ 6 + 5 S(0)`: the weight
ratio the escape mechanism can ever build in the EMA regime is at most ≈ 6, and a local optimum whose
critical weight exceeds that is never escaped. Only the (inert) additive branch is unbounded. -/
theorem ema_weight_bound (lam mu : ℝ) (hmu0 : 0 ≤ mu) (hmu1 : mu < 1) (hlam : 0 ≤ lam)
    (S : ℕ → ℝ) (hS0 : 0 ≤ S 0) (hstep : ∀ t, S (t + 1) ≤ mu * S t + lam) :
    ∀ t, S t ≤ lam / (1 - mu) + S 0 := by
  have h1 : 0 < 1 - mu := by linarith
  intro t
  induction t with
  | zero =>
    have : 0 ≤ lam / (1 - mu) := div_nonneg hlam h1.le
    linarith
  | succ t ih =>
    calc S (t + 1) ≤ mu * S t + lam := hstep t
      _ ≤ mu * (lam / (1 - mu) + S 0) + lam := by
          have := mul_le_mul_of_nonneg_left ih hmu0; linarith
      _ = lam / (1 - mu) + mu * S 0 := by field_simp; ring
      _ ≤ lam / (1 - mu) + S 0 := by
          have : mu * S 0 ≤ S 0 := by nlinarith
          linarith

/-- Instance with the code's constants: `μ = 0.87959`, EMA increment `λ = 1 − μ`, `S(0) = 0`
gives `S ≤ 1`, hence `w ≤ 6`. -/
theorem ema_weight_bound_code (S : ℕ → ℝ) (hS0 : S 0 = 0)
    (hstep : ∀ t, S (t + 1) ≤ 0.87959 * S t + (1 - 0.87959)) :
    ∀ t, 1 + 5 * S t ≤ 6 := by
  intro t
  have h := ema_weight_bound (1 - 0.87959) 0.87959 (by norm_num) (by norm_num) (by norm_num)
    S (by rw [hS0]) hstep t
  rw [hS0] at h
  have : (1 - 0.87959 : ℝ) / (1 - 0.87959) = 1 := by norm_num
  rw [this] at h
  linarith

end UmacoSat


/-! ## 14. Planted signal vs random symmetry (research/system_c/SystemC_3SAT_attack.md §13) -/

namespace UmacoSat

/-- Negating every literal of a clause. -/
def Clause.negate (c : Clause) : Clause := c.map (fun l => ⟨l.var, !l.pos⟩)

/-- Negating the assignment. -/
def Assignment.negate (x : Assignment) : Assignment := fun v => !(x v)

theorem Lit.eval_negate (x : Assignment) (l : Lit) :
    Lit.eval (Assignment.negate x) ⟨l.var, !l.pos⟩ = Lit.eval x l := by
  unfold Lit.eval Assignment.negate
  cases l.pos <;> cases x l.var <;> simp

/-- **Symmetry of the random ensemble.** A clause is satisfied by `x` iff its negation is satisfied
by the negation of `x`. So the map (formula, assignment) ↦ (negated formula, negated assignment) is
a bijection preserving satisfaction; it maps the unconditioned random ensemble to itself and sends the
drift toward any fixed `x*` to the drift toward its complement. Hence the expected drift toward any
fixed assignment at `p = ½` is zero on the random model, while on the planted model it is `3r/28`
(Theorem 12 at `δ = ½`). -/
theorem Clause.sat_negate (x : Assignment) (c : Clause) :
    Clause.sat (Assignment.negate x) (Clause.negate c) = Clause.sat x c := by
  unfold Clause.sat Clause.negate
  rw [List.any_map]
  congr 1
  funext l
  exact Lit.eval_negate x l

/-- At `p = ½`, the per-variable drift toward `x*` from one clause is `s · 2^{−(k−1)}` with `s = ±1`
the sign of `v`'s literal relative to `x*`; negating the clause flips `s`. The two contributions
cancel in expectation over an ensemble closed under negation. -/
theorem drift_cancels (a : ℝ) : a + (-a) = 0 := by ring

end UmacoSat


/-! ## 15. Critical variables converge (research/system_c/SystemC_3SAT_attack.md §13, Theorem 21) -/

namespace UmacoSat

/-- Theorem 21: with `c = crit*(v) ≥ 1` and the pattern counts `A, B, n011, n111 ≥ 0`, the quenched
drift `D_v(δ) = c(1−δ)² + B δ(1−δ) + n111 δ² − A δ(1−δ) − n011 δ²` is positive whenever
`c(1−δ) > (A−B)δ + n011 δ²/(1−δ)`, for `0 ≤ δ < 1`. -/
theorem critical_drift_pos (c A B n011 n111 δ : ℝ) (hc : 1 ≤ c) (hB : 0 ≤ B) (hn111 : 0 ≤ n111)
    (hδ0 : 0 ≤ δ) (hδ1 : δ < 1)
    (h : c * (1 - δ) > (A - B) * δ + n011 * δ ^ 2 / (1 - δ)) :
    0 < c * (1 - δ) ^ 2 + B * δ * (1 - δ) + n111 * δ ^ 2 - A * δ * (1 - δ) - n011 * δ ^ 2 := by
  have h1 : 0 < 1 - δ := by linarith
  have h2 : n011 * δ ^ 2 = (n011 * δ ^ 2 / (1 - δ)) * (1 - δ) := by field_simp
  have h3 : 0 ≤ n111 * δ ^ 2 := by positivity
  have h4 : 0 < (c * (1 - δ) - (A - B) * δ - n011 * δ ^ 2 / (1 - δ)) * (1 - δ) := by
    apply mul_pos _ h1; linarith
  nlinarith [h2, h3, h4]

end UmacoSat


/-! ## 16. Theorem 22: the fixed-weight ascent converges to the stationary set -/

namespace UmacoSat

variable {n : ℕ}

/-- The cube `[0,1]^n`. -/
def InCube (p : Fin n → ℝ) : Prop := ∀ v, 0 ≤ p v ∧ p v ≤ 1

theorem mlExt_nonneg (p : Fin n → ℝ) (hp : InCube p) (F : (Fin n → Bool) → ℝ)
    (hF : ∀ x, 0 ≤ F x) : 0 ≤ mlExt p F := by
  unfold mlExt
  apply Finset.sum_nonneg
  intro x _
  exact mul_nonneg (prodProb_nonneg p hp x) (hF x)

theorem inCube_update01 (p : Fin n → ℝ) (hp : InCube p) (v : Fin n) (b : ℝ) (hb : 0 ≤ b ∧ b ≤ 1) :
    InCube (Function.update p v b) := by
  intro u
  by_cases h : u = v
  · subst h; simp [hb]
  · simp [Function.update_of_ne h, hp u]

/-- The slope lies in `[−1, 1]` when `F` takes values in `[0,1]`. -/
theorem slope_bounds (p : Fin n → ℝ) (hp : InCube p) (F : (Fin n → Bool) → ℝ)
    (hF : ∀ x, 0 ≤ F x ∧ F x ≤ 1) (v : Fin n) :
    -1 ≤ slope p F v ∧ slope p F v ≤ 1 := by
  unfold slope
  have h0 := inCube_update01 p hp v 0 (by norm_num)
  have h1 := inCube_update01 p hp v 1 (by norm_num)
  have a0 := mlExt_nonneg _ h0 F (fun x => (hF x).1)
  have a1 := mlExt_nonneg _ h1 F (fun x => (hF x).1)
  have b0 := mlExt_le_one_of_le_one _ h0 F (fun x => (hF x).2)
  have b1 := mlExt_le_one_of_le_one _ h1 F (fun x => (hF x).2)
  constructor <;> linarith

/-- One coordinate replicator step. -/
def repStep (η : ℝ) (F : (Fin n → Bool) → ℝ) (p : Fin n → ℝ) (v : Fin n) : Fin n → ℝ :=
  Function.update p v (p v + η * p v * (1 - p v) * slope p F v)

/-- The step keeps the cube when `0 ≤ η ≤ 1` and `F ∈ [0,1]`. -/
theorem repStep_inCube (η : ℝ) (hη : 0 ≤ η ∧ η ≤ 1) (F : (Fin n → Bool) → ℝ)
    (hF : ∀ x, 0 ≤ F x ∧ F x ≤ 1) (p : Fin n → ℝ) (hp : InCube p) (v : Fin n) :
    InCube (repStep η F p v) := by
  unfold repStep
  apply inCube_update01 p hp v
  obtain ⟨hs0, hs1⟩ := slope_bounds p hp F hF v
  obtain ⟨hp0, hp1⟩ := hp v
  have hq : 0 ≤ p v * (1 - p v) := mul_nonneg hp0 (by linarith)
  constructor <;> nlinarith [mul_nonneg hη.1 hq, mul_le_mul_of_nonneg_left hs1 (mul_nonneg hη.1 hq),
    mul_le_mul_of_nonneg_left hs0 (mul_nonneg hη.1 hq), mul_nonneg hp0 hp0,
    mul_nonneg (sub_nonneg.mpr hp1) (sub_nonneg.mpr hp1)]

/-- A trajectory of coordinate steps along any coordinate schedule `vs`. -/
def traj (η : ℝ) (F : (Fin n → Bool) → ℝ) (p0 : Fin n → ℝ) (vs : ℕ → Fin n) : ℕ → (Fin n → ℝ)
  | 0 => p0
  | t + 1 => repStep η F (traj η F p0 vs t) (vs t)

theorem traj_inCube (η : ℝ) (hη : 0 ≤ η ∧ η ≤ 1) (F : (Fin n → Bool) → ℝ)
    (hF : ∀ x, 0 ≤ F x ∧ F x ≤ 1) (p0 : Fin n → ℝ) (hp0 : InCube p0) (vs : ℕ → Fin n) :
    ∀ t, InCube (traj η F p0 vs t) := by
  intro t
  induction t with
  | zero => exact hp0
  | succ t ih => exact repStep_inCube η hη F hF _ ih (vs t)

/-- The per-step gain of the extension. -/
def gain (η : ℝ) (F : (Fin n → Bool) → ℝ) (p0 : Fin n → ℝ) (vs : ℕ → Fin n) (t : ℕ) : ℝ :=
  mlExt (traj η F p0 vs (t + 1)) F - mlExt (traj η F p0 vs t) F

theorem gain_eq (η : ℝ) (F : (Fin n → Bool) → ℝ) (p0 : Fin n → ℝ) (vs : ℕ → Fin n) (t : ℕ) :
    gain η F p0 vs t
      = η * traj η F p0 vs t (vs t) * (1 - traj η F p0 vs t (vs t))
          * (slope (traj η F p0 vs t) F (vs t)) ^ 2 := by
  unfold gain
  show mlExt (repStep η F (traj η F p0 vs t) (vs t)) F - mlExt (traj η F p0 vs t) F = _
  unfold repStep
  exact replicator_step_gain _ F _ η

theorem gain_nonneg (η : ℝ) (hη : 0 ≤ η ∧ η ≤ 1) (F : (Fin n → Bool) → ℝ)
    (hF : ∀ x, 0 ≤ F x ∧ F x ≤ 1) (p0 : Fin n → ℝ) (hp0 : InCube p0) (vs : ℕ → Fin n) (t : ℕ) :
    0 ≤ gain η F p0 vs t := by
  rw [gain_eq]
  obtain ⟨h0, h1⟩ := traj_inCube η hη F hF p0 hp0 vs t (vs t)
  have : 0 ≤ 1 - traj η F p0 vs t (vs t) := by linarith
  have := hη.1
  positivity

/-- Telescoping: the partial sums of the gains are bounded by `1`. -/
theorem sum_gain_le_one (η : ℝ) (hη : 0 ≤ η ∧ η ≤ 1) (F : (Fin n → Bool) → ℝ)
    (hF : ∀ x, 0 ≤ F x ∧ F x ≤ 1) (p0 : Fin n → ℝ) (hp0 : InCube p0) (vs : ℕ → Fin n) (T : ℕ) :
    ∑ t ∈ Finset.range T, gain η F p0 vs t ≤ 1 := by
  have htel : ∑ t ∈ Finset.range T, gain η F p0 vs t
      = mlExt (traj η F p0 vs T) F - mlExt (traj η F p0 vs 0) F := by
    unfold gain
    exact Finset.sum_range_sub (fun t => mlExt (traj η F p0 vs t) F) T
  rw [htel]
  have hup := mlExt_le_one_of_le_one _ (traj_inCube η hη F hF p0 hp0 vs T) F (fun x => (hF x).2)
  have hlo := mlExt_nonneg _ (traj_inCube η hη F hF p0 hp0 vs 0) F (fun x => (hF x).1)
  linarith

/-- **Theorem 22.** Along any coordinate schedule, the gains are summable and tend to zero:
`η p_v(1−p_v) slope² → 0`. Every limit point of the trajectory is therefore stationary in the sense
of Theorem 2 (`p_v ∈ {0,1}` or vanishing slope along the visited coordinates). -/
theorem gain_tendsto_zero (η : ℝ) (hη : 0 ≤ η ∧ η ≤ 1) (F : (Fin n → Bool) → ℝ)
    (hF : ∀ x, 0 ≤ F x ∧ F x ≤ 1) (p0 : Fin n → ℝ) (hp0 : InCube p0) (vs : ℕ → Fin n) :
    Filter.Tendsto (gain η F p0 vs) Filter.atTop (nhds 0) := by
  have hs : Summable (gain η F p0 vs) :=
    summable_of_sum_range_le (gain_nonneg η hη F hF p0 hp0 vs) (sum_gain_le_one η hη F hF p0 hp0 vs)
  exact hs.tendsto_atTop_zero

end UmacoSat


/-! ## 17. Theorem 23 (counting half): one variable moves the weighted count by at most the
weight of the clauses containing it -/

namespace UmacoSat

/-- A clause with no literal on `v` is unaffected by changing `x v`. -/
theorem Clause.sat_update_of_not_mem (x : Assignment) (c : Clause) (v : ℕ) (b : Bool)
    (h : ∀ l ∈ c, l.var ≠ v) :
    Clause.sat (Function.update x v b) c = Clause.sat x c := by
  unfold Clause.sat
  apply Bool.eq_iff_iff.mpr
  simp only [List.any_eq_true]
  constructor
  · rintro ⟨l, hl, he⟩
    refine ⟨l, hl, ?_⟩
    unfold Lit.eval at he ⊢
    rwa [Function.update_of_ne (h l hl)] at he
  · rintro ⟨l, hl, he⟩
    refine ⟨l, hl, ?_⟩
    unfold Lit.eval at he ⊢
    rwa [Function.update_of_ne (h l hl)]

/-- Weighted-count difference bound: if two `{0,1}`-valued families `a, b` agree outside `T ⊆ s`
and the weights are nonnegative, then `|Σ_s w (a − b)| ≤ Σ_T w`. -/
theorem weighted_diff_le {ι : Type*} [DecidableEq ι] (s T : Finset ι) (hT : T ⊆ s)
    (w : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i)
    (a b : ι → ℝ) (ha : ∀ i ∈ s, 0 ≤ a i ∧ a i ≤ 1) (hb : ∀ i ∈ s, 0 ≤ b i ∧ b i ≤ 1)
    (hagree : ∀ i ∈ s, i ∉ T → a i = b i) :
    |∑ i ∈ s, w i * (a i - b i)| ≤ ∑ i ∈ T, w i := by
  have hsplit : ∑ i ∈ s, w i * (a i - b i) = ∑ i ∈ T, w i * (a i - b i) := by
    rw [← Finset.sum_sdiff hT]
    have : ∑ i ∈ s \ T, w i * (a i - b i) = 0 := by
      apply Finset.sum_eq_zero
      intro i hi
      rw [Finset.mem_sdiff] at hi
      rw [hagree i hi.1 hi.2]; ring
    rw [this, zero_add]
  rw [hsplit]
  calc |∑ i ∈ T, w i * (a i - b i)| ≤ ∑ i ∈ T, |w i * (a i - b i)| := Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ i ∈ T, w i := by
        apply Finset.sum_le_sum
        intro i hi
        have hi' := hT hi
        rw [abs_mul, abs_of_nonneg (hw i hi')]
        apply mul_le_of_le_one_right (hw i hi')
        rw [abs_le]
        constructor <;> linarith [(ha i hi').1, (ha i hi').2, (hb i hi').1, (hb i hi').2]

end UmacoSat


/-! ## 18. Pairwise pheromone on XOR: balance under a value-flipping involution
(research/system_c/SystemC_3SAT_attack.md §16, Theorem 25) -/

namespace UmacoSat

/-- If `σ` is an involution of `s` that flips the predicate `p`, then `p` is balanced on `s`.
Applied to the solution set of an XOR system (an affine subspace over GF(2)) with `σ = (· + d)`
for a direction `d` on which a pair functional `x_u ⊕ x_v` is `1`: the functional is balanced, so
the pairwise correlation of the uniform solution measure is `0`; otherwise it is constant, `±1`. -/
theorem card_filter_eq_of_involution {α : Type*} [DecidableEq α] (s : Finset α) (σ : α → α)
    (hσs : ∀ a ∈ s, σ a ∈ s) (hσσ : ∀ a ∈ s, σ (σ a) = a)
    (p : α → Prop) [DecidablePred p] (hflip : ∀ a ∈ s, (p (σ a) ↔ ¬ p a)) :
    (s.filter p).card = (s.filter (fun a => ¬ p a)).card := by
  apply Finset.card_bij (fun a _ => σ a)
  · intro a ha
    rw [Finset.mem_filter] at ha ⊢
    refine ⟨hσs a ha.1, ?_⟩
    intro hpσ
    exact (hflip a ha.1).mp hpσ ha.2
  · intro a ha b hb h
    have ha' := (Finset.mem_filter.mp ha).1
    have hb' := (Finset.mem_filter.mp hb).1
    calc a = σ (σ a) := (hσσ a ha').symm
      _ = σ (σ b) := by rw [h]
      _ = b := hσσ b hb'
  · intro b hb
    rw [Finset.mem_filter] at hb
    refine ⟨σ b, ?_, hσσ b hb.1⟩
    rw [Finset.mem_filter]
    exact ⟨hσs b hb.1, (hflip b hb.1).mpr hb.2⟩

/-- Consequently the signed count `#{p} − #{¬p}` vanishes: the uniform measure on `s` has zero
correlation with `p`. -/
theorem signed_count_zero_of_involution {α : Type*} [DecidableEq α] (s : Finset α) (σ : α → α)
    (hσs : ∀ a ∈ s, σ a ∈ s) (hσσ : ∀ a ∈ s, σ (σ a) = a)
    (p : α → Prop) [DecidablePred p] (hflip : ∀ a ∈ s, (p (σ a) ↔ ¬ p a)) :
    ((s.filter p).card : ℤ) - (s.filter (fun a => ¬ p a)).card = 0 := by
  rw [card_filter_eq_of_involution s σ hσs hσσ p hflip]; ring

end UmacoSat


/-! ## 19. Equivariant marginal dynamics are pinned at ½ on self-complementary formulas -/

namespace UmacoSat

/-- An abstract one-step marginal update `U : formula → marginals → marginals` is
**negation-equivariant** if negating the formula and reflecting the marginals commutes with it:
`U (neg φ) (1 − p) = 1 − U φ p`. Every rule in this document has this property (the product
sampler, the fitness-weighted deposit, the coverage weights, the min-break local search and the
breakout weights all treat a literal and its negation symmetrically). -/
def Equivariant {Φ : Type*} {n : ℕ} (neg : Φ → Φ) (U : Φ → (Fin n → ℝ) → (Fin n → ℝ)) : Prop :=
  ∀ φ p, U (neg φ) (fun v => 1 - p v) = fun v => 1 - U φ p v

/-- **Theorem 27.** If `U` is equivariant and `φ` is self-complementary (`neg φ = φ`), then the
uniform point `p ≡ ½` is a fixed point of `U φ`. So no equivariant marginal dynamics started at ½
ever moves on a self-complementary formula; only exogenous noise can. -/
theorem equivariant_fixed_half {Φ : Type*} {n : ℕ} (neg : Φ → Φ)
    (U : Φ → (Fin n → ℝ) → (Fin n → ℝ)) (hU : Equivariant neg U) (φ : Φ) (hφ : neg φ = φ) :
    U φ (fun _ => (1 : ℝ) / 2) = fun _ => (1 : ℝ) / 2 := by
  have h := hU φ (fun _ => (1 : ℝ) / 2)
  rw [hφ] at h
  have h' : (fun v : Fin n => 1 - (1 : ℝ) / 2) = (fun _ : Fin n => (1 : ℝ) / 2) := by
    funext v; norm_num
  rw [h'] at h
  funext v
  have hv := congrFun h v
  simp only at hv
  linarith

/-- Iterates stay at ½ as well. -/
theorem equivariant_iterate_half {Φ : Type*} {n : ℕ} (neg : Φ → Φ)
    (U : Φ → (Fin n → ℝ) → (Fin n → ℝ)) (hU : Equivariant neg U) (φ : Φ) (hφ : neg φ = φ) :
    ∀ t, (U φ)^[t] (fun _ => (1 : ℝ) / 2) = fun _ => (1 : ℝ) / 2 := by
  intro t
  induction t with
  | zero => rfl
  | succ t ih =>
    rw [Function.iterate_succ_apply', ih]
    exact equivariant_fixed_half neg U hU φ hφ

/-- Self-complementary satisfiable formulas exist: a clause and its negation are jointly satisfiable
whenever the clause has at least two literals on distinct variables, e.g. `(a ∨ b ∨ ¬c)` and
`(¬a ∨ ¬b ∨ c)` are both satisfied by `a = true, b = false`. -/
theorem self_complementary_example :
    let a : Lit := ⟨0, true⟩
    let b : Lit := ⟨1, true⟩
    let c : Lit := ⟨2, false⟩
    let x : Assignment := fun v => v = 0
    Clause.sat x [a, b, c] = true ∧ Clause.sat x (Clause.negate [a, b, c]) = true := by
  decide

end UmacoSat


/-! ## 20. Theorem 28: linearization at ½, and the XOR degeneracy -/

namespace UmacoSat

/-- A clause's contribution to `D_v` is affine in `p_u` with slope `−w s_v s_u q_w`:
`w · s_v · (½ − s_u (p − ½)) · q_w = w s_v q_w / 2 − (w s_v s_u q_w) (p − ½)`. -/
theorem clause_drift_affine (w sv su qw p : ℝ) :
    w * sv * (1 / 2 - su * (p - 1 / 2)) * qw
      = w * sv * qw / 2 - (w * sv * su * qw) * (p - 1 / 2) := by ring

/-- For one XOR, summing `(−1)^{a_u + a_v}` over the four forbidden assignments `a` (fixed parity)
gives `0`: the pairwise sign agreement of XOR-encoded clauses vanishes, so `M ≡ 0`. -/
theorem xor_pair_sign_sum_zero :
    ((List.filter (fun a : Bool × Bool × Bool => (a.1 != a.2.1) != a.2.2)
        [(false,false,false),(false,false,true),(false,true,false),(false,true,true),
         (true,false,false),(true,false,true),(true,true,false),(true,true,true)]).map
      (fun a => if a.1 == a.2.1 then (1 : ℤ) else -1)).sum = 0 := by
  decide

end UmacoSat


/-! ## 21. Locality: T iterates of a neighbourhood-local update depend only on the radius-T ball
(the hypothesis of Theorem 26 and of the overlap-gap obstruction) -/

namespace UmacoSat

variable {V : Type*} [DecidableEq V]

/-- `N v` is the (closed) neighbourhood of `v`. -/
def Local (N : V → Finset V) (F : (V → ℝ) → (V → ℝ)) : Prop :=
  ∀ p q v, (∀ u ∈ N v, p u = q u) → F p v = F q v

/-- Radius-`T` ball: `ball 0 v = {v}`, `ball (T+1) v = ⋃_{u ∈ N v} ball T u`. -/
def ball (N : V → Finset V) : ℕ → V → Finset V
  | 0, v => {v}
  | T + 1, v => (N v).biUnion (ball N T)

theorem mem_ball_self (N : V → Finset V) (hN : ∀ v, v ∈ N v) : ∀ T v, v ∈ ball N T v := by
  intro T
  induction T with
  | zero => intro v; simp [ball]
  | succ T ih =>
    intro v
    simp only [ball, Finset.mem_biUnion]
    exact ⟨v, hN v, ih v⟩

/-- **Locality lemma.** If `p` and `q` agree on `ball N T v`, then `F^[T] p v = F^[T] q v`. -/
theorem iterate_local (N : V → Finset V) (F : (V → ℝ) → (V → ℝ)) (hF : Local N F) :
    ∀ (T : ℕ) (p q : V → ℝ) (v : V), (∀ u ∈ ball N T v, p u = q u) → F^[T] p v = F^[T] q v := by
  intro T
  induction T with
  | zero =>
    intro p q v h
    simp only [Function.iterate_zero, id_eq]
    exact h v (by simp [ball])
  | succ T ih =>
    intro p q v h
    rw [Function.iterate_succ_apply', Function.iterate_succ_apply']
    apply hF
    intro u hu
    apply ih
    intro x hx
    apply h
    simp only [ball, Finset.mem_biUnion]
    exact ⟨u, hu, hx⟩

end UmacoSat


/-! ## 22. Theorem 30: annealed planted drift for general k -/

namespace UmacoSat

/-- Sum over all planted truth patterns of the other `k` literals of the probability that they are
all false under the current assignment: `∑_t ∏_i [(1−t_i)(1−δ) + t_i δ] = ((1−δ) + δ)^k = 1`. -/
theorem pattern_sum_one (δ : ℝ) (k : ℕ) :
    ∑ t : Fin k → Bool, ∏ i, (if t i then δ else 1 - δ) = 1 := by
  have h := Finset.prod_univ_sum (t := fun _ : Fin k => (Finset.univ : Finset Bool))
    (f := fun _ b => if b then δ else 1 - δ)
  rw [Fintype.piFinset_univ] at h
  rw [← h]
  simp

/-- Annealed drift for planted k-SAT: patterns with `v`'s literal true contribute the full sum `1`;
patterns with it false contribute everything except the all-false pattern, `1 − (1−δ)^k`; the
difference is `(1−δ)^k`, with `k` the number of other literals (`k = 2` gives Theorem 12). -/
theorem annealed_drift_general (δ : ℝ) (k : ℕ) :
    (∑ t : Fin k → Bool, ∏ i, (if t i then δ else 1 - δ))
      - ((∑ t : Fin k → Bool, ∏ i, (if t i then δ else 1 - δ)) - (1 - δ) ^ k)
      = (1 - δ) ^ k := by ring

/-- For `k = 2` the uniform-selection toward-probability `δ/(1−(1−δ)²) = 1/(2−δ) ≥ ½` for all
`δ ∈ (0,1]`: no crossover, Lemma 6. -/
theorem uniform_toward_k2 (δ : ℝ) (h0 : 0 < δ) (h1 : δ ≤ 1) :
    (1 : ℝ) / 2 ≤ δ / (1 - (1 - δ) ^ 2) := by
  have hden : 1 - (1 - δ) ^ 2 = δ * (2 - δ) := by ring
  have hpos : 0 < δ * (2 - δ) := mul_pos h0 (by linarith)
  rw [hden, div_le_div_iff₀ (by norm_num) hpos]
  nlinarith

end UmacoSat


/-! ## 23. System U (docs/system-u/SystemU_theorems.md): the literal walk -/

namespace UmacoSat

/-- U1b: on the flat field with `η ≡ 1`, step `t+1` of the walk chooses uniformly among the
`2(n−t)` literals of unassigned variables, so a full path has probability `1/(2^n · n!)`:
`∏_{t<n} 2(n−t) = 2^n · n!`. Since each assignment has `n!` orderings, `P(x) = 2^{−n}`. -/
theorem flat_walk_path_count (n : ℕ) :
    ∏ t ∈ Finset.range n, (2 * (n - t)) = 2 ^ n * n.factorial := by
  have h := Finset.prod_range_reflect (fun t => 2 * (t + 1)) n
  -- `prod_range_reflect : ∏ t in range n, f (n - 1 - t) = ∏ t in range n, f t`
  have h2 : ∀ t ∈ Finset.range n, 2 * (n - t) = 2 * (n - 1 - t + 1) := by
    intro t ht
    have := Finset.mem_range.mp ht
    omega
  rw [Finset.prod_congr rfl h2]
  rw [show (∏ t ∈ Finset.range n, 2 * (n - 1 - t + 1))
      = ∏ t ∈ Finset.range n, (fun t => 2 * (t + 1)) (n - 1 - t) from rfl, h]
  rw [Finset.prod_mul_distrib, Finset.prod_const, Finset.card_range,
      Finset.prod_range_add_one_eq_factorial]

theorem flat_walk_uniform (n : ℕ) :
    (n.factorial : ℚ) * (1 / ((2 : ℚ) ^ n * n.factorial)) = (1 / 2) ^ n := by
  have hf : (n.factorial : ℚ) ≠ 0 := by exact_mod_cast n.factorial_ne_zero
  have h2 : ((1 : ℚ) / 2) ^ n * 2 ^ n = 1 := by rw [← mul_pow]; norm_num
  field_simp
  linarith [h2]

/-- U2a, one step. For a softmax choice with logit `s` on the chosen option `b`, value `fb`, and
the other options contributing `C = Σ_{c≠b} e^{θ_c} f_c` and `D = Σ_{c≠b} e^{θ_c} > 0`, the expected
value `E = (e^s fb + C)/(e^s + D)` has derivative in `s` equal to `p_b · (fb − E)` where
`p_b = e^s/(e^s + D)`: the score-function identity `∂E/∂θ_b = E[f · (1[X=b] − p_b)]`. -/
theorem softmax_score_identity (fb C D s : ℝ) (hD : 0 < D) :
    HasDerivAt (fun s => (Real.exp s * fb + C) / (Real.exp s + D))
      ((Real.exp s / (Real.exp s + D)) * (fb - (Real.exp s * fb + C) / (Real.exp s + D))) s := by
  have hZ : Real.exp s + D ≠ 0 := by positivity
  have hnum : HasDerivAt (fun s => Real.exp s * fb + C) (Real.exp s * fb) s := by
    have := (Real.hasDerivAt_exp s).mul_const fb
    simpa using this.add_const C
  have hden : HasDerivAt (fun s => Real.exp s + D) (Real.exp s) s := by
    simpa using (Real.hasDerivAt_exp s).add_const D
  have h := hnum.div hden hZ
  have key : (Real.exp s / (Real.exp s + D)) * (fb - (Real.exp s * fb + C) / (Real.exp s + D))
      = (Real.exp s * fb * (Real.exp s + D) - (Real.exp s * fb + C) * Real.exp s)
          / (Real.exp s + D) ^ 2 := by
    field_simp
  rw [key]
  exact h

end UmacoSat

/-! ## 7. Axiom audit -/
#print axioms UmacoSat.soundness
#print axioms UmacoSat.cost_linear
#print axioms UmacoSat.decision_rule_correct_iff
#print axioms UmacoSat.far_vars_bound
#print axioms UmacoSat.deposit_ratio_bound
#print axioms UmacoSat.hamming_after_flips
#print axioms UmacoSat.mul_log_sub_nonneg
#print axioms UmacoSat.escape_time
#print axioms UmacoSat.toward_prob_ge_inv_k
#print axioms UmacoSat.toward_prob_le_of_accept
#print axioms UmacoSat.critical_wrong_has_wrong
#print axioms UmacoSat.uniform_toward_crossover
#print axioms UmacoSat.crossover_root
#print axioms UmacoSat.annealed_drift_identity
#print axioms UmacoSat.xor_encoding_is_three_spin
#print axioms UmacoSat.branching_factor_le_inv_e
#print axioms UmacoSat.mlExt_split
#print axioms UmacoSat.mlExt_le_one_of_le_one
#print axioms UmacoSat.replicator_step_gain
#print axioms UmacoSat.slope_at_vertex
#print axioms UmacoSat.ema_weight_bound
#print axioms UmacoSat.Clause.sat_negate
#print axioms UmacoSat.critical_drift_pos
#print axioms UmacoSat.gain_tendsto_zero
#print axioms UmacoSat.weighted_diff_le
#print axioms UmacoSat.card_filter_eq_of_involution
#print axioms UmacoSat.equivariant_iterate_half
#print axioms UmacoSat.xor_pair_sign_sum_zero

/-! ## 24. System U3: the burst as an operator (docs/system-u/SystemU_theorems.md §U3)

The burst adds `g · e^{iφ} · B` to `Φ = R + iM`, with `B = U_k Σ_k V_kᵀ` the rank-`k` part of `R`,
`g ≥ 0` the panic scale and `φ = angle(Ψ)`. The walk reads `w = ε + max(0, R² − M²)` entrywise.
Everything after the SVD is entrywise, so the regime table is an identity in `(g, cos φ, sin φ)`. -/

namespace UmacoSat

/-- The interference weight of an edge with attraction `r` and repulsion `m`. -/
noncomputable def iw (ε r m : ℝ) : ℝ := ε + max 0 (r ^ 2 - m ^ 2)

/-- U3a. On a captured edge (`B_ab = R_ab = r`, `M_ab = 0`) a burst of strength `g` at phase `φ`
(`c = cos φ`, `s = sin φ`) multiplies `r²` by `F(g,φ) = 1 + 2g cos φ + g² cos 2φ`. -/
theorem burst_factor (g c s : ℝ) (h : c ^ 2 + s ^ 2 = 1) :
    (1 + g * c) ^ 2 - (g * s) ^ 2 = 1 + 2 * g * c + g ^ 2 * (2 * c ^ 2 - 1) := by
  have hs : s ^ 2 = 1 - c ^ 2 := by linarith
  rw [show (g * s) ^ 2 = g ^ 2 * s ^ 2 by ring, hs]; ring

theorem burst_on_captured (ε r g c s : ℝ) (h : c ^ 2 + s ^ 2 = 1) :
    iw ε (r + g * c * r) (g * s * r)
      = ε + max 0 (r ^ 2 * (1 + 2 * g * c + g ^ 2 * (2 * c ^ 2 - 1))) := by
  unfold iw
  rw [show (r + g * c * r) ^ 2 - (g * s * r) ^ 2 = r ^ 2 * ((1 + g * c) ^ 2 - (g * s) ^ 2) by ring,
      burst_factor g c s h]

/-- At full strength `g = 1` the factor is `2 cos φ (1 + cos φ) = 4 cos φ cos²(φ/2)`. -/
theorem burst_full_factor (c s : ℝ) (h : c ^ 2 + s ^ 2 = 1) :
    (1 + c) ^ 2 - s ^ 2 = 2 * c * (1 + c) := by
  have hs : s ^ 2 = 1 - c ^ 2 := by linarith
  rw [hs]; ring

/-- U3b. At full strength a captured edge is erased to the floor iff `cos φ ≤ 0`, i.e. iff
`φ ≥ π/2`: the regimes π/2, (π/2, π) and π all erase; 0 and (0, π/2) do not. -/
theorem burst_full_erase_iff (c : ℝ) (hc : -1 ≤ c) :
    2 * c * (1 + c) ≤ 0 ↔ c ≤ 0 := by
  constructor
  · intro h
    by_contra hpos
    push Not at hpos
    nlinarith
  · intro h
    nlinarith

/-- The floor is reached exactly when repulsion matches attraction. Read forwards: erasure.
Read backwards (U3c): an edge carrying a persistent repulsion `m` stays at the floor until its
rebuilt attraction exceeds `|m|`. -/
theorem iw_eq_floor_iff (ε r m : ℝ) : iw ε r m = ε ↔ r ^ 2 ≤ m ^ 2 := by
  unfold iw
  rw [add_eq_left, max_eq_left_iff]
  constructor <;> intro h <;> linarith

/-- U3b (weak bursts). If `g² < 1/2` no phase erases a captured edge: the factor is positive. -/
theorem burst_no_erase_of_small (g c s : ℝ) (h : c ^ 2 + s ^ 2 = 1) (hg : g ^ 2 < 1 / 2) :
    0 < (1 + g * c) ^ 2 - (g * s) ^ 2 := by
  have hs : s ^ 2 = 1 - c ^ 2 := by linarith
  have key : (1 + g * c) ^ 2 - (g * s) ^ 2 = 2 * (g * c + 1 / 2) ^ 2 + (1 / 2 - g ^ 2) := by
    rw [show (g * s) ^ 2 = g ^ 2 * s ^ 2 by ring, hs]; ring
  rw [key]; nlinarith [sq_nonneg (g * c + 1 / 2)]

/-- At `φ = 3π/4` (`c = −s`, `c² = 1/2`) the factor is exactly `1 − √2·g`, so `g ≥ 1/√2` is
the sharp threshold for any erasure. -/
theorem burst_factor_three_quarter (g c s : ℝ) (h : c ^ 2 + s ^ 2 = 1) (hcs : s = -c) :
    (1 + g * c) ^ 2 - (g * s) ^ 2 = 1 + 2 * g * c := by
  have hc : c ^ 2 = 1 / 2 := by rw [hcs] at h; nlinarith
  rw [burst_factor g c s h, hc]; ring

/-- U3b (mixed regime). At full strength the sector `(0, π/2)` is *net reinforcing* on the
captured edges until `cos φ = (√3 − 1)/2`, i.e. `φ ≈ 68.5°`; only beyond that does the added
repulsion outweigh the added attraction. -/
theorem burst_full_net_reinforce_iff (c : ℝ) (hc : -1 ≤ c) :
    1 ≤ 2 * c * (1 + c) ↔ (Real.sqrt 3 - 1) / 2 ≤ c := by
  have h3 : Real.sqrt 3 ^ 2 = 3 := Real.sq_sqrt (by norm_num)
  have h3' : 0 ≤ Real.sqrt 3 := Real.sqrt_nonneg 3
  have hfac : 2 * c * (1 + c) - 1
      = 2 * (c - (Real.sqrt 3 - 1) / 2) * (c - (-Real.sqrt 3 - 1) / 2) := by
    ring_nf; rw [h3]; ring
  have hneg : 0 < c - (-Real.sqrt 3 - 1) / 2 := by nlinarith
  constructor
  · intro h
    have : 0 ≤ 2 * (c - (Real.sqrt 3 - 1) / 2) * (c - (-Real.sqrt 3 - 1) / 2) := by linarith
    nlinarith
  · intro h
    have : 0 ≤ 2 * (c - (Real.sqrt 3 - 1) / 2) * (c - (-Real.sqrt 3 - 1) / 2) := by
      apply mul_nonneg (mul_nonneg (by norm_num) (by linarith)) hneg.le
    linarith

/-- U3e (flat return). When every edge weight is the same constant `ε^α`, the walk's law is the
`η`-only law: `ε` and `α` cancel. So a full erasure of a rank-`≤k` field returns the walk to the
heuristic-guided flat walk of U1b in one step. -/
theorem flat_law_indep {ι : Type*} (S : Finset ι) (η : ι → ℝ) (b : ι) (c : ℝ) (hc : c ≠ 0) :
    c * η b / (∑ b' ∈ S, c * η b') = η b / ∑ b' ∈ S, η b' := by
  rw [← Finset.mul_sum, mul_div_mul_left _ _ hc]

/-- Any single entry is bounded by the Frobenius energy: after a full π burst, `R' = R − B_k`
has energy `Σ_{i>k} σ_i²`, so every edge weight is at most `ε + Σ_{i>k} σ_i²`. -/
theorem entry_sq_le_frob {m n : Type*} [Fintype m] [Fintype n] (E : Matrix m n ℝ) (i : m) (j : n) :
    E i j ^ 2 ≤ ∑ i', ∑ j', E i' j' ^ 2 := by
  calc E i j ^ 2 ≤ ∑ j', E i j' ^ 2 :=
        Finset.single_le_sum (fun _ _ => sq_nonneg _) (Finset.mem_univ j)
    _ ≤ ∑ i', ∑ j', E i' j' ^ 2 :=
        Finset.single_le_sum (fun _ _ => Finset.sum_nonneg (fun _ _ => sq_nonneg _))
          (Finset.mem_univ i)

open InnerProductSpace in
/-- U3e (directedness). If `B` is the orthogonal projection of `R` onto a subspace
(`⟪R,B⟫ = ‖B‖²`, true of every SVD truncation), then subtracting `B` removes exactly `‖B‖²`
of energy. -/
theorem burst_energy_drop {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] (R B : E)
    (h : ⟪R, B⟫_ℝ = ‖B‖ ^ 2) : ‖R - B‖ ^ 2 = ‖R‖ ^ 2 - ‖B‖ ^ 2 := by
  rw [norm_sub_sq_real, h]; ring

open InnerProductSpace in
/-- A perturbation orthogonal to the field (the mean behaviour of isotropic noise) *raises* the
energy: `‖R − Δ‖² = ‖R‖² + ‖Δ‖²`. The random part of the burst is therefore not a descent. -/
theorem orthogonal_noise_energy_rise {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (R Δ : E) (h : ⟪R, Δ⟫_ℝ = 0) : ‖R - Δ‖ ^ 2 = ‖R‖ ^ 2 + ‖Δ‖ ^ 2 := by
  rw [norm_sub_sq_real, h]; ring

open InnerProductSpace in
/-- Among perturbations `Δ` living in the captured subspace (`⟪R,Δ⟫ = ⟪B,Δ⟫`) of size at most
`‖B‖`, the projection `B` is the one most aligned with `R`: `−B` is the steepest descent
direction of the energy within that subspace. -/
theorem projected_best_direction {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (R B Δ : E) (hB : ⟪R, B⟫_ℝ = ‖B‖ ^ 2) (hΔ : ⟪R, Δ⟫_ℝ = ⟪B, Δ⟫_ℝ) (hn : ‖Δ‖ ≤ ‖B‖) :
    ⟪R, Δ⟫_ℝ ≤ ⟪R, B⟫_ℝ := by
  rw [hΔ, hB]
  calc ⟪B, Δ⟫_ℝ ≤ ‖B‖ * ‖Δ‖ := real_inner_le_norm B Δ
    _ ≤ ‖B‖ * ‖B‖ := by gcongr
    _ = ‖B‖ ^ 2 := by ring

/-- U3d. The field of a population that agrees on an assignment `x` but varies the order is,
in the limit of all orders, the all-ones block `J` on the `n` literals of `x` with the diagonal
removed: `J − 1`. It decomposes as `(n−1)·P − (1 − P)` with `P = J/n` an orthogonal projection,
so its singular values are `n − 1` (once) and `1` (`n − 1` times) and its rank-1 part is
`(n−1)·P`. -/
theorem ones_mul_ones (n : ℕ) :
    (Matrix.of fun _ _ : Fin n => (1 : ℝ)) * (Matrix.of fun _ _ : Fin n => (1 : ℝ))
      = (n : ℝ) • Matrix.of fun _ _ : Fin n => (1 : ℝ) := by
  ext i j; simp [Matrix.mul_apply]

theorem mode_block_decomp (n : ℕ) (hn : (n : ℝ) ≠ 0) :
    (Matrix.of fun _ _ : Fin n => (1 : ℝ)) - 1
      = ((n : ℝ) - 1) • ((1 / (n : ℝ)) • Matrix.of fun _ _ : Fin n => (1 : ℝ))
        - (1 - (1 / (n : ℝ)) • Matrix.of fun _ _ : Fin n => (1 : ℝ)) := by
  ext i j
  simp only [Matrix.sub_apply, Matrix.smul_apply, Matrix.of_apply, Matrix.one_apply, smul_eq_mul]
  split_ifs <;> field_simp <;> ring

theorem mode_projection_idem (n : ℕ) (hn : (n : ℝ) ≠ 0) :
    ((1 / (n : ℝ)) • Matrix.of fun _ _ : Fin n => (1 : ℝ))
      * ((1 / (n : ℝ)) • Matrix.of fun _ _ : Fin n => (1 : ℝ))
      = (1 / (n : ℝ)) • Matrix.of fun _ _ : Fin n => (1 : ℝ) := by
  rw [Matrix.smul_mul, Matrix.mul_smul, ones_mul_ones, smul_smul, smul_smul]
  congr 1
  field_simp

/-- U3d (the escape, exactly). A full π burst on the mode block `J − 1` (rank-1 part removed)
leaves every off-diagonal entry at `1/n`: the mode's edges drop from weight `1` to `1/n`,
i.e. from `w = ε + 1` to `w = ε + 1/n²`. -/
theorem mode_block_after_burst (n : ℕ) (hn : (n : ℝ) ≠ 0) (i j : Fin n) (hij : i ≠ j) :
    ((Matrix.of fun _ _ : Fin n => (1 : ℝ)) - 1
      - ((n : ℝ) - 1) • ((1 / (n : ℝ)) • Matrix.of fun _ _ : Fin n => (1 : ℝ))) i j
      = 1 / n := by
  simp only [Matrix.sub_apply, Matrix.smul_apply, Matrix.of_apply, Matrix.one_apply, smul_eq_mul,
    if_neg hij]
  field_simp
  ring

end UmacoSat


/-! ## 25. System U4/U5: symmetry, and what the literal field receives on XOR
(docs/system-u/SystemU_theorems.md §U4, §U5) -/

namespace UmacoSat

/-- U4a (equivariance of one walk step). Relabelling the literals by a bijection `σ`, and
transporting the field row and the heuristic along with the set of candidates, gives the same
transition probability. Hence the whole path law is equivariant. -/
theorem walk_step_equivariant {ι : Type*} (σ : ι ≃ ι) (w : ι → ι → ℝ) (η : ι → ℝ)
    (S : Finset ι) (a b : ι) (α β : ℝ) :
    (w (σ.symm (σ a)) (σ.symm (σ b))) ^ α * (η (σ.symm (σ b))) ^ β
        / ∑ b' ∈ S.map σ.toEmbedding, (w (σ.symm (σ a)) (σ.symm b')) ^ α * (η (σ.symm b')) ^ β
      = w a b ^ α * η b ^ β / ∑ b' ∈ S, w a b' ^ α * η b' ^ β := by
  simp [Finset.sum_map]

/-- Literals of two variables: `0 = x, 1 = ¬x, 2 = y, 3 = ¬y`; the pair flip `x ↦ ¬x, y ↦ ¬y`. -/
def pairFlip : Fin 4 → Fin 4 := ![1, 0, 3, 2]

/-- An "agreement" field: weight 2 between literals of different variables and equal polarity,
weight 1 otherwise. -/
def agreeField : Fin 4 → Fin 4 → ℕ :=
  fun i j => if i.val / 2 ≠ j.val / 2 ∧ i.val % 2 = j.val % 2 then 2 else 1

/-- U4b (invariance does not force flatness). The agreement field is invariant under the pair
flip and is not flat. For the product samplers, invariance under a variable flip forced the
marginal to `1/2` (the equivariance pin); on the literal graph an invariant field can carry
correlations, so no such pin exists for System U. -/
theorem invariant_field_not_flat :
    (∀ i j, agreeField (pairFlip i) (pairFlip j) = agreeField i j) ∧
    agreeField 0 2 ≠ agreeField 0 3 := by
  decide

/-- The four-clause CNF encoding of `x₀ ⊕ x₁ ⊕ x₂ = 1`. -/
def xorCNF : CNF :=
  [[⟨0, true⟩, ⟨1, true⟩, ⟨2, true⟩], [⟨0, true⟩, ⟨1, false⟩, ⟨2, false⟩],
   [⟨0, false⟩, ⟨1, true⟩, ⟨2, false⟩], [⟨0, false⟩, ⟨1, false⟩, ⟨2, true⟩]]

def asg3 (a b c : Bool) : Assignment := fun v => if v = 0 then a else if v = 1 then b else c

/-- U5a (pairwise balance of XOR). Fix any two of the three variables at any values; summing the
satisfied-clause count over the third gives `7` in every case. So `E[f | two literals]` is the same
on every literal-to-literal edge, and the flat field is an exact fixed point of the expected
deposit on an XOR constraint. -/
theorem xor_pair_balance :
    (∀ a b : Bool, xorCNF.satCount (asg3 a b false) + xorCNF.satCount (asg3 a b true) = 7) ∧
    (∀ a c : Bool, xorCNF.satCount (asg3 a false c) + xorCNF.satCount (asg3 a true c) = 7) ∧
    (∀ b c : Bool, xorCNF.satCount (asg3 false b c) + xorCNF.satCount (asg3 true b c) = 7) := by
  decide

/-- The same for a general parity target: given any two bits, the third makes the parity right
exactly once. This is the fact behind U5a for a whole XOR system (each constraint has a free
variable, so each is violated with probability exactly `1/2` given any two literals). -/
theorem parity_given_two_uniform (a b t : Bool) :
    ((if (a ^^ b) ^^ false = t then 1 else 0) + (if (a ^^ b) ^^ true = t then 1 else 0) : ℕ) = 1 := by
  cases a <;> cases b <;> cases t <;> decide

/-- Contrast: a single OR clause `x₀ ∨ x₁ ∨ x₂` has pairwise signal: fixing `x₀ = x₁ = true` the
clause is always satisfied, fixing `x₀ = x₁ = false` only half the time. -/
def orCNF : CNF := [[⟨0, true⟩, ⟨1, true⟩, ⟨2, true⟩]]

theorem or_clause_pair_signal :
    orCNF.satCount (asg3 true true false) + orCNF.satCount (asg3 true true true) = 2 ∧
    orCNF.satCount (asg3 false false false) + orCNF.satCount (asg3 false false true) = 1 := by
  decide

/-- U5c (η is a soft force). A literal that is the last free literal of one unsatisfied clause
has `η = 2` against `η = 1` for its negation, so it is chosen with odds `2^β : 1` when its
variable is assigned; `L` such forced steps succeed together with probability at most
`(2^β/(1+2^β))^L < 1`. Unit propagation is recovered only as `β → ∞`. -/
theorem soft_force_lt_one (β : ℝ) : (2 : ℝ) ^ β / (1 + (2 : ℝ) ^ β) < 1 := by
  have h : 0 < (2 : ℝ) ^ β := Real.rpow_pos_of_pos (by norm_num) β
  rw [div_lt_one (by linarith)]; linarith

theorem soft_force_pow_le (β : ℝ) (L : ℕ) (p : ℝ) (hp : p ≤ (2 : ℝ) ^ β / (1 + (2 : ℝ) ^ β))
    (hp0 : 0 ≤ p) : p ^ L ≤ ((2 : ℝ) ^ β / (1 + (2 : ℝ) ^ β)) ^ L :=
  pow_le_pow_left₀ hp0 hp L

end UmacoSat


/-! ## 26. The paper's argument as a theorem: O1 ∧ O2 ∧ O3 ⟹ SAT ∈ P ⟹ NP ⊆ P
(`benchmarks/benchmark_analysis/UMACO_Polynomial_SAT_Scaling_Paper.md` §5.2–5.3; docs/p-vs-np/P_eq_NP_lean.md)

An abstract model of computation, enough to state and prove the argument's spine:
problems with a size function; "polynomial" means `≤ c·(size+1)^k`; polynomial-time many-one
reductions; Cook–Levin taken as a hypothesis on a class `NP`. The solver side uses the `Solver N`
of §4 (soundness built in, completeness `Complete S δ` as in O3). -/

namespace UmacoSat

/-- `cost` is polynomially bounded in `size`. -/
def PolyBoundedBy {X : Type*} (size cost : X → ℕ) : Prop :=
  ∃ c k : ℕ, ∀ x, cost x ≤ c * (size x + 1) ^ k

theorem polyBounded_add {X : Type*} {size f g : X → ℕ}
    (hf : PolyBoundedBy size f) (hg : PolyBoundedBy size g) :
    PolyBoundedBy size (fun x => f x + g x) := by
  obtain ⟨c, k, hc⟩ := hf; obtain ⟨d, l, hd⟩ := hg
  refine ⟨c + d, max k l, fun x => ?_⟩
  have h1 : (size x + 1) ^ k ≤ (size x + 1) ^ max k l :=
    Nat.pow_le_pow_right (Nat.succ_pos _) (le_max_left _ _)
  have h2 : (size x + 1) ^ l ≤ (size x + 1) ^ max k l :=
    Nat.pow_le_pow_right (Nat.succ_pos _) (le_max_right _ _)
  calc f x + g x ≤ c * (size x + 1) ^ k + d * (size x + 1) ^ l := Nat.add_le_add (hc x) (hd x)
    _ ≤ c * (size x + 1) ^ max k l + d * (size x + 1) ^ max k l :=
        Nat.add_le_add (Nat.mul_le_mul_left _ h1) (Nat.mul_le_mul_left _ h2)
    _ = (c + d) * (size x + 1) ^ max k l := by ring

theorem polyBounded_const_mul {X : Type*} {size f : X → ℕ} (N : ℕ) (hf : PolyBoundedBy size f) :
    PolyBoundedBy size (fun x => N * f x) := by
  obtain ⟨c, k, hc⟩ := hf
  exact ⟨N * c, k, fun x => by rw [mul_assoc]; exact Nat.mul_le_mul_left _ (hc x)⟩

/-- Composition through a size-bounded map: if `cost` is polynomial in `size'` and `size' ∘ f` is
polynomial in `size`, then `cost ∘ f` is polynomial in `size`. -/
theorem polyBounded_comp {X Y : Type*} {size : X → ℕ} {size' cost : Y → ℕ} (f : X → Y)
    (hs : PolyBoundedBy size (fun x => size' (f x))) (hc : PolyBoundedBy size' cost) :
    PolyBoundedBy size (fun x => cost (f x)) := by
  obtain ⟨c, k, hck⟩ := hs; obtain ⟨d, l, hdl⟩ := hc
  refine ⟨d * (c + 1) ^ l, k * l, fun x => ?_⟩
  have hpos : 1 ≤ (size x + 1) ^ k := Nat.one_le_pow _ _ (Nat.succ_pos _)
  have h1 : size' (f x) + 1 ≤ (c + 1) * (size x + 1) ^ k := by
    calc size' (f x) + 1 ≤ c * (size x + 1) ^ k + 1 := Nat.add_le_add_right (hck x) 1
      _ ≤ c * (size x + 1) ^ k + (size x + 1) ^ k := Nat.add_le_add_left hpos _
      _ = (c + 1) * (size x + 1) ^ k := by ring
  calc cost (f x) ≤ d * (size' (f x) + 1) ^ l := hdl (f x)
    _ ≤ d * ((c + 1) * (size x + 1) ^ k) ^ l :=
        Nat.mul_le_mul_left _ (Nat.pow_le_pow_left h1 l)
    _ = d * (c + 1) ^ l * (size x + 1) ^ (k * l) := by rw [mul_pow, ← pow_mul]; ring

/-- A decision problem: inputs with a size, and the yes-instances. -/
structure Problem where
  Input : Type
  size : Input → ℕ
  yes : Input → Prop

/-- A polynomial-time decider for `P`. -/
structure PolyDecider (P : Problem) where
  run : P.Input → Bool
  cost : P.Input → ℕ
  poly : PolyBoundedBy P.size cost
  correct : ∀ x, run x = true ↔ P.yes x

def InP (P : Problem) : Prop := Nonempty (PolyDecider P)

/-- A polynomial-time many-one reduction `A ≤_p B`. -/
structure PolyReduction (A B : Problem) where
  f : A.Input → B.Input
  cost : A.Input → ℕ
  poly : PolyBoundedBy A.size cost
  sizePoly : PolyBoundedBy A.size (fun x => B.size (f x))
  correct : ∀ x, A.yes x ↔ B.yes (f x)

theorem inP_of_reduction {A B : Problem} (r : PolyReduction A B) (hB : InP B) : InP A := by
  obtain ⟨d⟩ := hB
  refine ⟨⟨fun x => d.run (r.f x), fun x => r.cost x + d.cost (r.f x), ?_, fun x => ?_⟩⟩
  · exact polyBounded_add r.poly (polyBounded_comp r.f r.sizePoly d.poly)
  · rw [d.correct, r.correct]

/-- SAT as a problem: input a CNF, size = clauses plus total literals, yes = satisfiable. -/
def CNF.size (φ : CNF) : ℕ := φ.length + (φ.map List.length).sum

def SATProblem : Problem := ⟨CNF, CNF.size, CNF.Satisfiable⟩

/-- **Cook–Levin, as a hypothesis on a class `NP`**: every problem in it reduces to SAT in
polynomial time. Standard; not proved here. -/
def CookLevin (NP : Set Problem) : Prop := ∀ A ∈ NP, Nonempty (PolyReduction A SATProblem)

/-- SAT ∈ P ⟹ NP ⊆ P (with P ⊆ NP by definition, this is P = NP). -/
theorem np_subset_p_of_sat_in_p (NP : Set Problem) (cook : CookLevin NP) (hSAT : InP SATProblem) :
    ∀ A ∈ NP, InP A := fun A hA => by
  obtain ⟨r⟩ := cook A hA
  exact inP_of_reduction r hSAT

/-- **O1 ∧ O2 ∧ O3 ⟹ SAT ∈ P.** From a sound solver (`Solver N`, O2 built in) with a finite seed
set, a per-run cost polynomial in the input (O1), and completeness with some `δ > 0` on that seed
set (O3): the deterministic decider "run every seed, answer SAT iff some seed reached 100 %" is
correct and polynomial. Soundness gives the ⇐ direction of correctness, completeness the ⇒. -/
theorem sat_in_P_of_solver (S : Solver N) (hN : 0 < N) (bound : CNF → ℕ)
    (hpoly : PolyBoundedBy CNF.size bound) (δ : ℝ) (hδ : 0 < δ) (hcomplete : Complete S δ) :
    InP SATProblem := by
  refine ⟨⟨fun φ => decide (0 < successes S φ), fun φ => N * bound φ,
    polyBounded_const_mul N hpoly, fun φ => ?_⟩⟩
  change decide (0 < successes S φ) = true ↔ φ.Satisfiable
  simp only [decide_eq_true_eq]
  constructor
  · intro h
    have hne : (Finset.univ.filter (fun s : Fin N => (S.run φ s).isSome)).Nonempty :=
      Finset.card_pos.mp h
    obtain ⟨s, hs⟩ := hne
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hs
    obtain ⟨a, ha⟩ := Option.isSome_iff_exists.mp hs
    exact ⟨a, S.sound φ s a ha⟩
  · intro hφ
    have h := hcomplete φ hφ
    have hNr : (0 : ℝ) < N := by exact_mod_cast hN
    have : (0 : ℝ) < successes S φ := lt_of_lt_of_le (mul_pos hδ hNr) h
    exact_mod_cast this

/-- **The paper's argument, as one theorem.** Hypotheses: Cook–Levin; a sound solver with a finite
seed set; a polynomial per-run cost bound (O1); completeness with some `δ > 0` on that seed set
(O3). Conclusion: every problem in `NP` is in `P`. Everything except O3 is either proved above
or standard; O3 is the claim the benchmark corpus is evidence for. -/
theorem paper_argument (NP : Set Problem) (cook : CookLevin NP) (S : Solver N) (hN : 0 < N)
    (bound : CNF → ℕ) (hpoly : PolyBoundedBy CNF.size bound)
    (δ : ℝ) (hδ : 0 < δ) (hcomplete : Complete S δ) :
    ∀ A ∈ NP, InP A :=
  np_subset_p_of_sat_in_p NP cook (sat_in_P_of_solver S hN bound hpoly δ hδ hcomplete)

/-- O1 for the fixed-budget solver: the operation count of §3 is polynomial (linear) in `n + m`. -/
theorem fixed_budget_poly (b : Budget) :
    PolyBoundedBy (fun p : ℕ × ℕ => p.1 + p.2) (fun p => cost b p.1 p.2) :=
  ⟨b.T * b.K * (1 + b.k * (1 + 2 * b.F)), 1, fun p => by
    rw [pow_one]
    exact le_trans (cost_linear b p.1 p.2) (Nat.mul_le_mul_left _ (Nat.le_succ _))⟩

end UmacoSat

#print axioms UmacoSat.iterate_local
#print axioms UmacoSat.pattern_sum_one
#print axioms UmacoSat.uniform_toward_k2
#print axioms UmacoSat.flat_walk_path_count
#print axioms UmacoSat.softmax_score_identity
#print axioms UmacoSat.burst_full_net_reinforce_iff
#print axioms UmacoSat.mode_block_after_burst
#print axioms UmacoSat.projected_best_direction
#print axioms UmacoSat.walk_step_equivariant
#print axioms UmacoSat.invariant_field_not_flat
#print axioms UmacoSat.xor_pair_balance
#print axioms UmacoSat.paper_argument
#print axioms UmacoSat.sat_in_P_of_solver
#print axioms UmacoSat.np_subset_p_of_sat_in_p
#print axioms UmacoSat.fixed_budget_poly
