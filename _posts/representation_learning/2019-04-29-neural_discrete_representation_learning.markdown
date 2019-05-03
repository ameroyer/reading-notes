---
layout: post
title:  "Neural Discrete Representation Learning"
date:   2019-04-29 14:59:24 +0200
tags: [representation learning, neurips, 2017]
categories:  [Representation Learning]
author: Van den Oord et al, N(eur)IPS 2017, <a href='https://arxiv.org/abs/1711.00937' target='_blank'>[link]</a>
---



<div class="summary">
In this work, the authors propose <code>VQ-VAE</code>, a variant of the Variational Autoencoder (<code>VAE</code>) framework with a discrete latent space, using ideas from vector quantization. The two main motivations are <b>(i)</b> discrete variables are potentially better fit to capture the structure of data such as text and <b>(ii)</b> to prevent the posterior collapse in <code>VAE</code>s that leads to latent variables being ignored when the decoder is too powerful.
<ul>
<li><span class="procons">Pros (+):</span> pros.</li>
<li><span class="procons">Cons (-):</span> Paragraph about the learned prior is not very clear, and this module does not have any ablation experiments.</li>
</ul>
</div>


<h3 class="section proposed"> Proposed </h3>



### Discrete latent space

The model is based on `VAE` <span class="citations">[1]</span>, where image $$x$$ is generated from random latent variable $$z$$ by a *decoder* $$p(x\ \vert\ z)$$. The posterior (*encoder*) captures the latent variable distribution $$q_{\phi}(z\ \vert\ x)$$ and is generally trained to match a certain distribution $$p(z)$$ from which $$z$$ is sampled from at inference time.
Contrary to the standard framework, in this work *the latent space is discrete*, i.e., $$z \in \mathbb{R}^{K \times D}$$ where $$K$$ is the number of codes in the latent space and $$D$$ their dimensionality. More precisely, the input image is first fed to $$z_e$$, that outputs a continuous vector, which is then mapped to one of the latent codes in the discrete space via *nearest-neighbor search*.

$$
\begin{align}
q(z = z_k\ |\ x) = [\!| k = \arg\min_j \| z_e(x) - z_j \|^2 |\!]
\end{align}
$$

Adapting the $$\mathcal{L}_{\text{ELBO}}$$ to this formalism, the KL divergence term greatly simplifies and we obtain:

$$
\begin{align}
\mathcal{L}_{\text{ELBO}}(x) &= \text{KL}(q(z | x) \| p(z)) - \mathbb{E}_{z \sim q(\cdot | x)}(\log p(x | z))\\
&= - \log(p(z_k)) -  \log p(x | z_k)\\
\mbox{where }& z_k = z_q(x) = \arg\min_z \| z_e(x) - z \|^2 \tag{1}
\end{align}
$$

In practice, the authors use a categorical *uniform prior* for the latent codes, meaning the KL divergence is constant and the objective reduces to the reconstruction loss.


<div class="figure">
<img src="{{ site.baseurl }}/images/posts/vqvae.png">
<p><b>Figure:</b>  A figure describing the <code>VQ-VAE</code> (<b>left</b>). Visualisation of the embedding space (<b>right</b>)). The output of the encoder z(x) is mapped to the nearest point. The gradient (in <span style="color:red">red</span>) will push the
encoder to change its output, which could alter the configuration, hence the code assignment, in the next forward pass.</p>
</div>

#### Training Objective
As we mentioned previously, the $$\mathcal{L}_{\text{ELBO}}$$ objective reduces to the *reconstruction loss* and is used to learn the encoder and decoder parameters.  However the mapping from $$z_e$$ to $$z_q$$ is not straight-forward differentiable (Equation **(1)**). 
To palliate this, the authors use a *straight-through estimator*, meaning the gradients from the decoder input $$z_q(x)$$ (quantized) are directly copied to the encoder output $$z_e(x)$$ (continuous).
However, this means that the latent codes that intervene in the mapping from $$z_e$$ to $$z_q$$ do not receive gradient updates that way. 

Hence in order to train the discrete embedding space, the authors propose to use *Vector Quantization* (`VQ`), a dictionary learning technique, which uses mean squared error to make the latent code closer to the continuous vector it was matched to:

$$
\begin{align}
\mathcal{L}_{\text{VQ-VAE}}(x) = -  \log p(x | z_q(x)) + \| \overline{z_e(x)} - e \|^2 + \beta  \| z_e(x) - \bar{e} \|^2 
\end{align}
$$

where $$x \mapsto  \overline{x}$$ denotes the `stop gradient` operator. The first term is the reconstruction loss stemming from the ELBO, the second term is the vector quantization contribution. Finally, the last  term is a *commitment loss* to control  the volume of the latent space by forcing the encoder to "commit" to the latent code it matched with, and not grow its output space unbounded. 

#### Learned Prior
A second contribution of this work consists in *learning the prior distribution*. As mentionned, during the training phase, the prior $$p(z)$$ is a uniform categorical distribution. After the training is done, we fit an *autoregressive distribution* over the space of latent codes. This is in particular enabled by the fact that the latent space is discrete.

---

<h3 class="section experiments"> Experiments </h3>

The proposed model is mostly  compared to the standard continuous `VAE` framework. It seems to achieve similar log-likelihood and sample quality, while taking advantage of the discrete latent space. In particular 
For ImageNet for instance, they consider $$K = 512$$ latent codes with dimensions $$32 \times 32 \times 1$$

---

<h3 class="section references"> References </h3>
* <span class="citations">[1]</span> Autoencoding Variational Bayes, <i>Kingma and Welling, ICLR 2014</i>
* <span class="citations">[2]</span> A Kernel Method for the Two-Sample-Problem, <i>Gretton et al, NIPS 2006</i>
