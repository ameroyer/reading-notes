---
layout: post
title:  "Domain Adversarial Training of Neural Networks"
date:   2019-05-23 10:59:24 +0200
tags: [domain adaptation, representation learning, adversarial, icml, jmlr, 2016]
categories:  [Domain Adaptation]
author: Ganin et al, JMLR 2016, <a href='https://arxiv.org/pdf/1505.07818.pdf' target='_blank'>[link]</a>
thumb: /images/thumbs/dann.png
year: 2016
---



<div class="summary">
In this article, the authors tackle the problem of <b>unsupervised domain adaptation</b>: Given labeled samples from a source distribution `\mathcal D_S` and unlabeled samples from target distribution `\mathcal D_T`, the goal is to learn a function that solves the task for both the source and target domain. In particular, the proposed model is trained on <b>both</b> source and target data jointly, and aims to directly learn an <b>aligned representation</b> of the domains, while retaining meaningful information with respect to the source labels.
<ul>
<li><span class="procons">Pros (+):</span> Theoretical justification, Simple model easy to implement.</li>
<li><span class="procons">Cons (-):</span> Some training unstability in practice.</li>
</ul>
</div>


<h3 class="section theory"> Generalized Bound on the Expected Risk </h3>
Several theoretical studies of the domain adaptation problem have proposed upper bounds of the *risk on the target domain*, involving the risk on the source domain and a noion of *distance* between the source and target distribution, $$\mathcal D_S$$ and $$\mathcal D_T$$. Here, the authors specifically consider the work of <span class="citations">[1]</span>. First, define the $$\mathcal H$$-divergence, a distance between distributions, as:

$$
\begin{align}
d_{\mathcal H}(\mathcal D_S, \mathcal D_T) = 2 \sup_{h \in \mathcal H} \left| \mathbb{E}_{x\sim\mathcal{D}_s} (h(x) = 1) -  \mathbb{E}_{x\sim\mathcal{D}_T} (h(x) = 1) \right| \tag{1}
\end{align}
$$

where $$\mathcal H$$ is a space of (here, binary) hypothesis functions. In the case where $$\mathcal H$$ is a *symmetric hypothesis class* (i.e., $$h \in \mathcal H \implies -h \in \mathcal H$$), one can reduce **(1)** to the empirical form:

$$
\begin{align}
d_{\mathcal H}(\mathcal D_S, \mathcal D_T) &\simeq 2 \sup_{h \in \mathcal H} \left|\frac{1}{|D_S|} \sum_{x \in D_S} [\!|h(x) = 1 |\!] - \frac{1}{|D_T|} \sum_{x \in D_T} [\!|h(x) = 1 |\!]  \right|\\
&= 2 \sup_{h \in \mathcal H} \left|\frac{1}{|D_S|} \sum_{x \in D_S} 1 -  [\!|h(x) = 0 |\!] - \frac{1}{|D_T|} \sum_{x \in D_T} [\!|h(x) = 1 |\!]  \right|\\
&=  2 - 2 \min_{h \in \mathcal H} \left|\frac{1}{|D_S|} \sum_{x \in D_S} [\!|h(x) = 0 |\!] + \frac{1}{|D_T|} \sum_{x \in D_T} [\!|h(x) = 1 |\!]  \right| \tag{2}
\end{align}
$$

It is difficult to estimate the minimum over the hypothesis class $$\mathcal H$$ even when it is "simple" (e.g., linear classifiers). Instead, <span class="citations">[1]</span> propose to *approximate* **(2)** by training a classifier $$\hat{h}$$ on samples $$\mathbf{x_S} \in \mathcal{D}_S$$ with label 0 and $$\mathbf{x_T} \in \mathcal D_T$$ wth label 1, and replacing the `minimum` term by the empirical risk of $$\hat h$$.
Given this definition of the $$\mathcal H$$-divergence, <span class="citations">[1]</span> further derives an *upper bound* on the empirical risk on the target domain, which in particular involves a trade-off between the empirical risk on the source domain, $$\mathcal{R}_{D_S}(h)$$, and the divergence between the source and target distributions, $$d_{\mathcal H}(D_S, D_T)$$. 

$$
\begin{align}
\mathcal{R}_{D_T}(h) \leq \mathcal{R}_{D_S}(h) + d_{\mathcal H}(D_S, D_T) + F\left(\mbox{VC}(\mathcal H), \frac{1}{n}\right) \tag{upper-bound}
\end{align}
$$

where $$\mbox{VC}$$ designates the Vapnik–Chervonenkis dimensions and $$n$$ the number of samples.
The rest of the paper directly stems from this intuition: in order to minimize the *target risk* the proposed *Domain Adversarial Neural Network* (`DANN`) aims to build an "<i>internal representation that contains no discriminative information about the origin of the input (source or target), while preserving a low risk on the source (labeled) examples</i>".


--- 

<h3 class="section proposed"> Proposed </h3>
The goal of the model is to learn a classifier $$\phi$$, which can be decomposed as $$\phi = G_y \circ G_f$$, where $$G_f$$ is a feature extractor and $$G_y$$ a small classifier on top that outputs the target label.  This architecture is trained with a standard classification objective to *minimize*:

$$
\begin{align}
\mathcal{L}_y(\theta_f, \theta_y) = \frac{1}{N_s} \sum_{(x, y) \in D_s} \ell(G_y(G_f(x)), y)
\end{align}
$$

Additionally `DANN` introduces a *domain prediction branch*, which is another classifier $$G_d$$ on top of the feature representation $$G_f$$ and whose goal is to approximate the domain discrepency as in **(2)**, which leads to the following training objective to *maximize*:

$$
\begin{align}
\mathcal{L}_d(\theta_f, \theta_d)  = \frac{1}{N_s} \sum_{x \in D_s} \ell(G_d(G_f(x)), s) + \frac{1}{N_t} \sum_{x \in D_t} \ell(G_d(G_f(x)), t)
\end{align}
$$

The ***final objective*** can thus be written as:

$$
\begin{align}
E(\theta_f, \theta_y, \theta_d) &= \mathcal{L}_y(\theta_f, \theta_y)  - \lambda \mathcal{L}_d(\theta_f, \theta_d) \tag{1}\\
\theta_f^\ast, \theta_y^\ast &= \arg\min E(\theta_f, \theta_y, \theta_d) \tag{2}\\
\theta_d^\ast &= \arg\max E(\theta_f, \theta_y, \theta_d) \tag{3} 
\end{align}
$$

#### Gradient Reversal Layer

Applying standard gradient descent, this problem leads to the following gradient update rules:

$$
\begin{align}
\theta_f &= \theta_f - \alpha \left( \frac{\partial \mathcal{L}_y}{\partial \theta_f} - \lambda \frac{\partial \mathcal{L}_d}{\partial \theta_f}  \right)\\
\theta_y &= \theta_y - \alpha \frac{\partial \mathcal{L}_y}{\partial \theta_y} \\
\theta_d &= \theta_d + \alpha \frac{- \lambda \partial \mathcal{L}_d}{\partial \theta_d} \\
\end{align}
$$

In the case of neural networks, the gradients of the loss with respect to parameters are obtained with the *backpropagation algorithm*. The current system equations are very similar to the standard backpropagation scheme, except for the opposite sign in the derivative of $$\mathcal{L}_d$$ with respect to $$\theta_d$$ and $$\theta_f$$. The authors introduce a ***gradient reversal layer*** (`GRL`) trick to evaluate both gradients in one standard backward pass.

The idea is that the output of $$\theta_f$$ is normally propagated to $$\theta_d$$, however during backpropagation, its gradient is multiplied by a negative constant:

$$
\begin{align}
\frac{\partial \mathcal L_d}{\partial \theta_f} = \frac{\mathbf{\color{red}{-}}  \partial \mathcal L_d}{\partial G_f(x)} \frac{\partial G_f(x)}{\partial \theta_f} 
\end{align}
$$

In other words, for $$\theta_d$$ parameters, the gradients of $$\mathcal L_d$$ are computed normally (*minimization*), but are propagated with a minus sign in the feature extraction part of the network (maximization).
Augmented with the gradient reversal layer, the final model is standardly trained by minimizing the sum of losses $$\mathcal L_d + \mathcal L_y$$ , which corresponds to the optimization problem in **(1-3)**.


<div class="figure">
<img src="{{ site.baseurl }}/images/posts/dann.png">
<p><b>Figure:</b> The proposed architecture includes a <span style="color: green">deep feature extractor</span> and a <span style="color: blue">deep label predictor</span>. 
Unsupervised domain adaptation is achieved by adding a <span style="color: fuchsia">domain classifier</span> connected to the feature extractor via a gradient reversal layer that multiplies
the gradient by a certain negative constant during backpropagation. </p>
</div>

---


<h3 class="section experiments"> Experiments </h3>

#### Datasets
The paper presents extensive results on the following settings:
 * **Toy dataset**: A toy example based on the *two half-moons dataset*, where the source domains consists in the standard binary classification tasks with the two half-moons, and the target is the same, but with a 30 degrees rotation. They compare the `DANN` to a `NN` model which has the same architecture but without the `GRL`: in other words, the baseline directly minimizes both the task and domain classificatoin losses.
 * **Sentiment Analysis**: These experiments are performed on the *Amazon reviews dataset* which contains product reviews from four different domains (hence 12 different source to target scenarios) which have to be classified as either positive or negative reviews.
* **Image classification**: Here the model is evaluated on various image classification task including MNIST $$\rightarrow$$ SVHN, or different domain pairs from the OFFICE dataset <span class="citations">[2]</span> .
* **Person re-identification**: The task of person identification across various visual domains.


#### Validation


#### Conclusions
in general, the proposed method seems to perform very well for aligning the source and target domains in an *unsupervised domain adaptation* framework. Its main advantage is its *simplicity*, both in terms of motivation and implementation, as the Gradient Reversal Layer is easily implemented in standard Deep Learning frameworks and can be added to any architectures. The main shortcomings are that **(i)** all experiments deal with only two sources and extensions to multiple domains might require some tweaks (e.g., considering the sum of pairwise discrepancies as an upper-bound) and **(ii)** in practice, training can be come *unstable* due to the adverary training scheme; In particular, the experiments sections show that some stability tricks have to be used during training, such as using momentum or slowly increasing the contribution of the domain classificatoin branch.


<div class="figure">
<img src="{{ site.baseurl }}/images/posts/dann_mnist_embeddings.png">
<p><b>Figure:</b> <code>t-SNE</code> projections of the embeddings for the <span style="color: blue">source</span> (MNIST) and <span style="color: red">target</span> (SVHN) datasets without (<b>left</b>) and with (<b>right</b>) <code>DANN</code> adaptation. </p>
</div>
---

<h3 class="section followup">Closely related (follow-up work)</h3>
<h4 style="margin-bottom: 0px"> Title.</h4>
<p style="text-align: right"><small>Authors, <a href="">[link]</a></small></p>
> Description

---

<h3 class="section references"> References </h3>
* <span class="citations">[1]</span> Analysis of representations for Domain Adaptation, <i>Ben-David et al, NIPS 2006</i>
* <span class="citations">[2]</span> Adapting visual category models to new domains, <i>Saenko et al, ECCV 2010</i>
