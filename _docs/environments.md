---
permalink: /docs/environments/
title: "Environments"
---

```terminal
├── environments
    ├── HOL4
    ├── HOList
    ├── INT
    ├── LeanDojo
```

# HOList

The HOList environment from ~paper. Provides an interactive interface to HOL-Light.
`environments.HOList.holist_env` provides a wrapper which enables compatibility with the abstract ~End-to-End
experiment.

## Additions/Modifications

- Modified to include pretty printed expressions in proof interaction, which
  are much more concise and closer to the expressions used in e.g. LeanDojo
- Modified proof logging to include pretty printed expressions.
    - Can create original HOList dataset over core and complex libraries, now with Pretty printing.
        - Done with `environments.HOList.gen_holist_data.sh`

## Future additions

- Update backend C++/O'Caml environment in `environments/HOList/hol-light` to accept
  more general tactics (currently quite restrictive, can't use general expressions, introduce assumptions, terms are
  only
  in s-expression format)
- Extend to arbitrary GitHub repos like LeanDojo
    - Test on flyspeck to reconstruct full HOList benchmark

# LeanDojo

Wrapper over the standard ~LeadDojo environment, implemented in `environments.LeanDojo.leandojo_env`.

LeanDojo is the most complete environment feature wise:

- It allows for proofs over arbitrary new GitHub repos
- No restrictions on the tactic input, allowing general expressions to be used
- Automatically manages Docker configurations internally

## Additions/Modifications

- Unlike the environment wrapper originally used in ReProver,
  this wrapper separate subgoals within a proof state.

# HOL4

The HOL4 environment used in ~TacticZero.

## Future additions

- Integrate with End-to-End environment, where all models/approaches and features can be used 
