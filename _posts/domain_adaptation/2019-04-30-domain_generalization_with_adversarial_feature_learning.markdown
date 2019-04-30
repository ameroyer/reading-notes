---
layout: post
title:  "Domain Generalization with Adversarial Feature Learning"
date:   2019-04-30 8:59:24 +0200
tags: [domain adaptation, domain generalization, cvpr, 2018]
categories:  [Domain Adaptation]
author: Li et al, CVPR 2018, <a href='http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Domain_Generalization_With_CVPR_2018_paper.pdf' target='_blank'>[link]</a>
---



<div class="summary">
In this paper, the authors tackle the problem of <b>Domain Generalization</b>: Given multiple source domains, the goal is to learn a joint aligned feature representation, hoping it would generalize to a new <b>unseen target</b> domain. This is closely related to the <b>Domain Adaptation</b> task, with the difference that no target data (even unlabeled) is available at training time. Most approaches rely on the idea of aligning the source domains distributions in a shared space. In this work, the authors propose to additionally match the source distributions to a known <b>prior distribution</b>.
<ul>
<li><span class="procons">Pros (+):</span> pros.</li>
<li><span class="procons">Cons (-):</span> cons.</li>
</ul>
</div>



<h3 class="section proposed"> Proposed model: MMD-AAE</h3>
The goal of domain generalization is to find a common *domain-invariant feature space* underlying the source and (unseen) target spaces, under the assumption that such a space exists.
To learn such space, the authors propose a variant of <span class="citations">[1]</span>, whose goal is to minimize the variance between the different source domains distributions using *Maximum Mean Discrepancy*. Additionally, the source distributions are aligned with a fixed *prior distribution*, with the hope that this reduces the risk of overfitting to the seen domains.

#### Adversarial Auto-encoder
The proposed model, `MMD-AAE` (Maximum Mean Discrepency Adversarial Auto-encoder) consists in an *encoder* $$Q: x \mapsto h$$, that maps inputs to latent codes, and a decoder $$P: h \mapsto x$$. These are equipped with a standard auto-encoding loss to make the model learn meaningful embeddings

$$
\begin{align}
\mathcal{L}_{\text{AE}}(x) = \| P(Q(x)) - x \|^2
\end{align}
$$

Based on the `AAE` framework <span class="citations">[1]</span>, we also want the learned latent codes to match a certain *prior* distribution, $$p(h)$$ (In practice, a Laplace distribution). This is done by introducing a `GAN` (Generative Adversarial Networks) loss terms on the generated embeddings, with the prior as the true, target, distribution. Introducing $$D$$, a discriminator with binary outputs, we have:

$$
\begin{align}
\mathcal{L}_{\text{GAN}}(x) = \mathbb{E}_{h \sim p(h)}(\log D(h)) + \mathbb{E}_{x \sim p(x)}(\log(1 - D(Q(x))))
\end{align}
$$

#### MMD Regularization
On top of the `AAE` objective, the authors propose to regularize the feature space using MMD, extended to the multi-domain setting. They key idea of the maximum mean discrepancy (`MMD`) is to compare two distributions using their mean statistics rather than density estimators. 
http://alex.smola.org/teaching/iconip2006/iconip_3.pdf


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
* <span class="citations">[1]</span> Adversarial Autoencoders, <i>Makhzani et al, ICLR Workshop, 2016</i>
