No argument:
- intro, intros (arg only for naming)
- apply_instance
- assumption
- contradiction
- trivial
- exfalso
- reflexivity/refl
- symmetry
- transitivity (adds new variable)
- by_contradiction (arg not necessary)
- by_contra (same as above)
- constructor
- econstructor
- left 
- right 
- split
- ac_refl/ ac_reflexivity
- cc
- dedup
- apply_opt_param
- apply_auto_param

Expr arg:
- apply, fapply, eapply, apply_with_expr, apply_instance
- refine
- change
- exact, exacts
- generalise 
- specialise
- by_cases
- structured tacs (omit from system for now)
  - assume
  - have 
  - let (arg can be omitted)
  - suffices (arg can be omitted)
  - show
  - from
- existsi ?

id arg:
- introv * 
- rename *
- revert * (takes ids from current assumptions)
- clear *
- induction (variable in local context)
- cases (variable in local context)
- destruct (variable in local context)
- injection (can use exprs, but substiture for ids for now)
- rewrite/rw (use constant or id, but can be arbitrary expr)
- rwa
- erewrite/erw
- subst (takes hypothesis)
- simp * (many variants, taking local variables and hypotheses), dsimp, simp *, simp at , simp with, simp intros
- unfold * and variants, delta, unfold_projs


Not needed
- admit (closes with sorry)
- case (can use cases instead)
- injections (can be replaced by injection )
- tactic combinators (ignore for now)


## Approach
- Need to define premises, hypotheses and local variables as ids, and define which tactics can take what as arguments
  - tactic state contains comma separated hypotheses (with type) before |- and goal after (parse and process these as separate objects)
  - after |- can also have variables..

- Working with sexpressions
  - Can process entire goal state, with hyps and goals, as with HOList sexpressions
  - Possibly match variable names to types defined in HYPS?
  - Need to figure out how to find variables in sexpression (e.g. argument after :)
  - Map from lean-gym to sexpression

- Figure out how deal with braces in tactics?

- Could take sexpression format from lean_proof_recording and integrate with lean-step ?
- Or, hook lean-gym to output tactic states as s-expressions
