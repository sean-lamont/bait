---
layout: splash
permalink: /
title: ""
order: 1
masthead: false
include_footer: false
---
<p></p>

# BAIT: Benchmarking Architectures for Interactive Theorem-Proving
{: .text-center}
{: .text-center}

<a href="https://github.com/sean-lamont/bait" class="btn btn--primary">Code</a> <a href="/bait/assets/images/paper.pdf" class="btn btn--primary">Paper</a>
<a href="/bait/assets/images/poster.pdf" class="btn btn--primary">Poster</a> <a href="/bait/docs/" class="btn btn--primary">Docs</a>
{: .text-center}

<p></p>

## Overview
BAIT is a platform for accelerating research in the application of AI to Interactive Theorem-Proving (AI-ITP). 

BAIT was accepted as a AAAI24 Main Track paper, where it was used to study Embedding Architectures in AI-ITP.

To summarise, BAIT
- Integrates a (growing) collection of current AI-ITP approaches and benchmarks into a unified platform
- Streamlines developing and experimenting with the different components of AI-ITP architecture
- Facilitates reproducible and transparent results


[//]: # todo add a video

## Motivation
Interactive Theorem Proving (ITP) is an important paradigm of formal verification,
with broad applications ranging from pure mathematics to critical software.

ITP requires a human expert for proof guidance, which limits scalability and widespread adoption.
AI-ITP research aims to address this through methods which automate and assist ITP proofs.

The ITP task also serves as a challenging benchmark for the broader AI field.
Requiring advanced reasoning capabilities, successful AI-ITP models are likely to be transferable to other domains.

Current AI-ITP results are spread across several ITP systems and benchmarks, which complicates comparisons between them.
This is magnified by the variety and complexity of the approaches, which can vary over several axes.
This includes the search strategy, learning approach (Reinforcement Learning vs Supervised) and the model architecture used.

![alt]({{ site.url }}{{ site.baseurl }}/assets/images/approaches.png){: .align-center} 

## System overview
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/aitp.png){: .align-center style="width:70%"} 

Despite the large variety in approaches, the AI-ITP setup used in most approaches can be decomposed into several key modules. 

This motivates the design of BAIT, which aims to decouple the Data, Environment and Model.
These are combined into experiments, which represent tasks in ITP.


## Results
Please see the [paper]() for our results using BAIT to experiment with Embedding 
Architectures in AI-ITP.