---
layout: single
permalink: /
title: "BAIT: Benchmarking Architectures for Interactive Theorem-Proving"
order: 1
author_profile: false
classes: wide
---

BAIT is a platform for accelerating research in the application of AI to Interactive Theorem-Proving (AI-ITP).

$x + y$

To summarise, BAIT

- Integrates a (growing) collection of current AI-ITP approaches and benchmarks into a unified platform
- Streamlines developing and experimenting with the different components of AI-ITP architecture
- Facilitates reproducible and transparent results

BAIT was introduced as a AAAI24 Main Track paper, where it was used to study Embedding Architectures in AI-ITP.

# AI-ITP 
Interactive Theorem Proving (ITP) is an important paradigm of formal verification,
with broad applications ranging from pure mathematics to critical software. 

ITP requires a human expert for proof guidance, which limits its scalability and widespread adoption.
AI-ITP research aims to address this through methods which automate and assist ITP proofs.

The ITP task also serves as a challenge and benchmark for the broader AI field.
Requiring advanced reasoning capabilities, models successful in AI-ITP are likely to be transferable to other domains.


Current AI-ITP research is burdened by the fragmentation of results across ITP systems.
Benchmarks and environments exist for \system{HOL Light} \cite{bansal_holist_2019, kaliszyk_holstep_2017}, \system{HOL4} \cite{wu_tacticzero_2021}, \system{Lean} \cite{polu_formal_2022, han_proof_2021, yang_leandojo_2023}, Isabelle \cite{li_isarstep_2020} and Metamath \cite{kaliszyk_mizar_2015}.
These provide a broad set of tasks for benchmarking.
However being isolated to a single system complicates comparisons between them.
This is magnified by the variety and complexity of the learning algorithms, which vary over several axes.
For example, \alg{TacticZero} \cite{wu_tacticzero_2021} uses a seq2seq autoencoder for expressions,
and learns through online Reinforcement Learning (RL) with a custom goal selection algorithm.
\cite{bansal_holist_2019} instead use Breadth First Search (BFS) for goal selection,
with offline learning over labelled proof logs.


Research in the area is fragmented, with a diverse set of approaches being spread across several ITP systems.
This presents a significant challenge to the comparison of methods, which are often complex and difficult to replicate.

This is a growing area of interest, 



is designed to be a general framework for AI-ITP, with Data, Model and Environment modules implementing the setup in Figure~\ref{fig:ai-itp}. These are managed with an additional Experiment module,


There are four main modules: 
- Data
- Environments
- Experiments
- Models

