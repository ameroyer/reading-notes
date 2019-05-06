---
layout: post
title:  "Do Deep Generative Models Know what they don't Know ?"
date:   2019-05-06 10:59:24 +0200
tags: [adversarial examples, generative models, iclr, 2019]
categories:  [Adversarial Examples]
author: Nalisnick et al, ICLR 2019, <a href='https://arxiv.org/pdf/1810.09136.pdf' target='_blank'>[link]</a>
thumb: /images/thumbs/ddgmkwtdk.png
year: 2019
---



<div class="summary">
CNNs prediction landscapes are known to be very sensitive to adversarial examples, which are samples generated to be wrongly predicted with high confidence. On the other hand, <b>probabilistic generative models</b> such as `PixelCNN`s and `VAE`s learn a distribution over the input image domain hence could be used to <b>detect out-of-distribution</b> inputs, e.g., by estimating their likelihood under the data distribution. This paper provides interesting results showing that distributions learned by generative models are not robust enough yet to employ them in this way.
<ul>
<li><span class="procons">Pros (+):</span> Convincing experiments on multiple generative models, detailed analysis in the invertible flow case.</li>
<li><span class="procons">Cons (-):</span> It would be interesting to provide further results for different classes of domain shifts to observe if this is rather a property of the model or of the input data.</li>
</ul>
</div>



<h3 class="section proposed"> Methodology </h3>


Three classes of generative models are considered in this paper:
  * <b>Auto-regressive</b> models such as `PixelCNN` <span class="citations">[1]</span>
  * <b>Latent variable</b> models, such as `VAE` <span class="citations">[2]</span>
  * Generative models with `invertible flows` <span class="citations">[3]</span>, in particular <span class="citations">[4]</span>. 
  
The  first experiment the authors propose is to train a generative model $$G$$ on input data $$\mathcal X$$ and use it to evaluate the likelihood on both the training domain $$\mathcal X$$ and a different domain $$\tilde{\mathcal X}$$. Their first negative result is showing that ***a model trained on the CIFAR-10 dataset yields a higher likelihood when evaluated on the SVHN test dataset than on the CIFAR-10 test (or even train) split***. Interestingly, the  converse, when training on SVHN and evaluating on CIFAR, is not true. This result was consistantly observed for various architectures including <span class="citations">[1]</span>, <span class="citations">[2]</span> and <span class="citations">[4]</span>, although it is of lesser efect in the PixelCNN case.

Intuitively, this could come from the fact that both of these datasets contain natural images and that CIFAR-10 is strictly more diverse than SVHN in terms of semantic content. Nonetheless, these datasets vastly differ in appearance, and this result is *counter-intuitive* as it goes against the idea that generative models can reliably be use to detect out-of-distribution samples. Furthermore, this observation also confirms the general idea that higher likelihoods does not necessarily coincide with better generated samples <span class="citations">[5]</span>.


---

<h3 class="section theory"> Analysis in the Invertible Flow Models Case </h3>

The authors further study this phenomenon in the invertible flow models case as they provide a more rigorous analytical framework (e.g., exact likelihood estimation unlike VAEs which only provide a bound on the true likelihood). 

More specifically invertible flow models are characterized with a diffeomorphism,  $$f(x; \phi)$$, between input space $$\mathcal X$$ and latent space $$\mathcal Z$$, and choice of the latent distribution $$p(z; \psi)$$. The *change of variable formula* links the density of $$x$$ and $$z$$ as follows:

$$
\begin{align}
\int_x p_x(x)d_x = \int_x p_z(f(x)) \left| \frac{\partial f}{\partial x} \right| dx
\end{align}
$$

And the training objective under this transformation becomes

$$
\begin{align}
\arg\max_{\theta} \log p_x(\mathbf{x}; \theta) = \arg\max_{\phi, \psi} \sum_i \log p_z(f(x_i; \phi); \psi) + \log \left| \frac{\partial f_{\phi}}{\partial x_i} \right|
\end{align}
$$

Typically, $$p_z$$ is chosen to be Gaussian, and samples are build by inverting $$f$$, i.e.,$$\tilde(z) \sim p(\mathbf z),\ \tilde x = f^{-1}(\tilde z)$$. And $$f_{\phi}$$ is build such that computing the log determinant of the Jacobian in the previous equation is tractable.

First, they observe that contribution of the flow can be decomposed in a *density* element (left term) and a *volume* element (right term), resulting from the change of variables formula. Experiment results with `Glow` <span class="citations">[4]</span> show that the higher density  on SVHN mostly comes from the *volume element contribution*.

Interestingly this negative results seems quite robust: The authors also performed experiments with different types of flow formulation, e.g., *constant volume* flows (i.e., the term $$ \log \left\| \frac{\partial f_{\phi}}{\partial x_i} \right\| $$ is constant for all $$x$$) and with an ensemble of generative models, and still observe the same higher likelihood for SVHN. 
  
Secondly, they try to directly analyze the difference in likelihood between two domains $$\mathcal X$$ and $$\tilde{\mathcal X}$$; which can be done by a *second-order expansion* of the log-likelihood locally around the expectation of the distribution (assuming $$\mathbb{E} (\mathcal X) \sim \mathbb{E}(\tilde{\mathcal X})$$). For the constant volume Glow module, final analytical formula indeed confirms that the log-likelihood of SVHN should be higher than CIFAR's.
In some sense, SVHN is *included* in CIFAR (under the model distribution) and has lower variance, which explains the higher likelihood.


---

<h3 class="section references"> References </h3>
  * <span class="citations">[1]</span> Conditional Image Generation with PixelCNN Decoders, <i>van den Oord et al, NIPS 2016</i>
  * <span class="citations">[2]</span> Auto-Encoding Variational Bayes, <i>Kingma and Welling, ICLR 2013</i>
  * <span class="citations">[3]</span> Density estimation using Real NVP, <i>iDinh et al., ICLR 2015</i>
  * <span class="citations">[4]</span> Glow: Generative Flow with Invertible 1x1 Convolutions, <i>Kingma and Dhariwal, NIPS 2018</i>
  * <span class="citations">[5]</span> A Note on the Evaluation of Generative Models, <i>Theis et al., ICLR 2016</i>
