---
layout: post
title:  "The Variational Fair Autoencoder"
date:   2019-05-02 10:59:24 +0200
tags: [representation learning, iclr, 2016]
categories:  [Representation Learning]
author: Louizos et al, ICLR 2016, <a href='https://arxiv.org/pdf/1511.00830.pdf' target='_blank'>[link]</a>
---



<div class="summary">
The goal of this work is to propose a variational autoencoder based model that learns latent representations which are independent from some sensitive knowledge  present in the data, while retaining enough information to solve the task at hand, e.g. classification. This independence constraint is incorporated via  loss term based on Maximum Mean Discrepancy (MMD).
<ul>
<li><span class="procons">Pros (+):</span> pros.</li>
<li><span class="procons">Cons (-):</span> cons.</li>
</ul>
</div>


<h3 class="section sota"> State-of-the-art </h3>
Stae-of-the-art

---

<h3 class="section proposed"> Proposed </h3>
Given input data $$x$$, the goal is to learn a representation of $$x$$, that factorizes out *nuisance* or *sensitive* variables, $$s$$, while retaining task-relevant content $$z$$. Working in the `VAE` (Variational Autoencoder) framework, this is modeled as a generative process:

$$
\begin{align}
z &\sim p(z)\\
x &\sim p_{\theta}(x | z, s)
\end{align}
$$

where the prior on latent $$z$$ is explicitly made invariant to the variables to filter out, $$s$$. Introducing decoder $$q_{\phi}: x, s \mapsto z$$, this model can be trained using the standard variational lower bound objective ($$\mathcal{L}_{\text{ELBO}}$$) <span class="citations">[1]</span>.


#### Semi-supervised model
In order to make the learned representations relevant to a specific task, the authors propose to incorporate *label knowledge* during the feature learning stage. This is particularly useful if the task target label $$y$$ is correlated with the sensitive information $$s$$, otherwise unsupervised learning could yield random representations in order to get rid of $$s$$ only. This is done by considering two source of information: 


$$
\begin{align}
z &\sim p(z)\\
x &\sim p_{\theta}(x | z, s)
\end{align}
$$

---

<h3 class="section dataset"> Datasets </h3>

Datasets


---

<h3 class="section experiments"> Experiments </h3>

Experiments

---

<h3 class="section followup">Closely related (follow-up work)</h3>

Follow-up work

---

<h3 class="section references"> References </h3>
* <span class="citations">[1]</span>Autoencoding Variational Bayes, <i>Kingma and Welling, ICLR 2014</i>
* <span class="citations">[1]</span> title, <i>info</i>
