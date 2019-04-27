---
layout: post
title: "Measuring Abstract Reasoning in Neural Networks"
date: 2019-04-26 14:59:24 +0200
tags: [Visual Reasoning]
categories: [Visual Reasoning]
author: Barrett et al., ICML 2018, <a href='https://arxiv.org/pdf/1807.04225.pdf' target='_blank'>[link]</a>

---

The authors introduce a new visual analogy dataset with the aim to analyze the reasoning abilities of ConvNets on higher abstract reasoning tasks such as small IQ tests.
* **Pros (+):** Introduces a new dataset for abstract reasoning and different evaluation procedures, considers a large range of baselines.
* **Cons(+):** The Relation Network considers only pairwise interactions which is simple yet might be too specific to the problem at hand. Also hard to interpret and measure actual reasoning of the network.

---

<h3 class="section dataset"> Dataset </h3>

This paper introduces  the *Procedurally Generated Matrices* (PGM) dataset. It is based on *Raven’s Progressive Matrices (RPM)* introduced by psychologist John Raven in 1936. Given an incomplete ***3x3*** matrix (missing the bottom right panel), the goal is to complete the matrix with an image picked ***out of 8 candidates***. Typically, several candidates are plausible but the subject has to select the one with the strongest justification.

<center><img src='https://drive.google.com/uc?export=view&id=1Nd7QA5574NeXmSimxhD1b4THfbYnXy_E'></center>
<br>

#### Construction

A PGM is defined as a set of triples $(r, o, a)$, each encoding a particular relation. For instance (`progression`, `lines`, `number`) means that the PGM contains a progression relation on the number of lines. In practice, the PGM dataset only contains ***1 to 4*** relations per PGM. The construction primitives are as follows:

* **relation types** ($R$, with elements $r$): `progression`, `XOR`, `OR`, `AND`, `consistent union`. The only relation that might require beyong binary correspondencies is the consistent union.
* **object types** ($O$, with elements $o$): `shape` or `lines`.
* **attribute types** ($A$, with elements $a$): `size`, `type`, `colour`, `position`, `number`. Each attribute takes values in a discrete set (e.g. 10 levels of gray intensity for colour).

Note that some relations are hard to define (for instance progresson on shape position ?), and hence ignored. In total, ***29*** possible relations triples are considered.

The attributes which are not involved in any of the relations of the PGM are called the ***nuisance attributes***. They are chosen either as a fixed value for all images in the squence, or randomly assigned (***distracting setting***). 


#### Evaluation Setting
The authors consider 8 generalization settings to evaluate on:

* **Neutral setting.** Standard random train/test split, no constraint on the relations

* **Interpolation** and **Extrapolation.** The values of the `colour` and `size` attributes are restricted to half the possible values in the training set, and take values in the remaining half options in the test set. Note that in this setting, the test set is built such that every sequence contains one of these two attributes, i.e. generalization is required for every image. The different between inter- and extrapolation lies in the ***discretized space split***: For interpolation, the split is uniform across the support (even-indexed values vs. odd-indexed values). In extrapolation, the values are split between lower half of the space and upper half of the space.

* **Held-out setting.** As the name indicates, this evaluation setting consists in keeping certain relations out of the training set and considering them only at test time (each of the test question contains at least one of the kept-out relations).
* *shape-colour.* Keep out any relation with $o=$ `shape` and $a =$ `colour`
* *line-type.* Keep out any relation with $o=$ `line` and $a =$ `type`
* *triples*. Take out **seven** relation triples (chosen such that every attribute is represented exactly once (**?: but there's only five attributes**)).
* *pairs of triples*. Same as before but considering pairs of triples this time and only generating PGM with at least two relations: in that way, some relation interactions will have never been seen on training time.
* *pairs of attributes*. Same as before but at the attribute level

---

<h3 class="section sota"> Baselines </h3>
The main contributions of the paper are to introduce the PGM dataset and evaluate several standard deep architectures on it:

* **CNN-MLP:** A standard 4-layers CNN, followed by 2 fully connected layers. It takes as inputs the 8 context panels of the matrix and the 8 panel candidates concatenated on the channel axis: i.e., inputs to the model are 80x80x16 images. It outputs the labels of the correct panels (8-labels classification task).

* **ResNet.** Same as before but with a ResNet architecture.

* **Wild resNet.** This time, the candidate panels are fed separately (i.e. 8 different input, each as a 9 channel image) and a score is output for each one of them. The candidate with the higest score is chosen.

* **Context-blind ResNet.** Rather a "sanity check" than a baseline, train a ResNet that only takes the candidate panels as inputs, no context.

* **LSTM.** First, each of the 16 panels is fed independently through a 4-layers CNN and the output feature maps is tagged with an index (following the sequence order). This sequence is fed through a LSTM, whose final hidden state is passed through one linear layer for the final classification.

* **RN network.** The authors propose a Relation Network based on recent work  [1]. Each context panel and candidate is fed through a CNN resulting in embeddings $\{x_1 \dots x_8\}$ and  $\{c_1 \dots c_8\}$ respectively. Then for each candidate panel $k$, the Relation Network outputs a score $s_k$:

$$
\begin{align}
s_k = f_{\phi} \left( \sum_{x, y \in \{x_1 \dots x_8, c_k\}^2 } g_{\theta}(x, y) \right)
\end{align}
$$

Additionally, they consider a semi-supervised variant where the model tries to additionally predict the relations  underlying the PGM (encoded as a one-hot vector) as a ***meta-target***. The total loss is a weighted average between the candidate classification loss term and the meta-target regression loss term.

---

<h3 class="section experiments"> Experiments </h3>

#### Overall results

The CNN-based  models perform consistantly badly, while LSTM provides an improvement but a small one. The Wild ResNet provides futher improvement over ResNet, which shows that using a panel scoring structure is more beneficial than direct classification of the correct candidate. Finally ***WReN outperforms all other baselines***, which could be expected as it makes use of pairwise interactions across panels. The main benefit of the method is its simplicity (*Note:* it could be interesting to compare agains other sequential architecture on ResNet).

### Different evaluation procedure

While the WReN achives satisfying accuracy on the **neutral** and **interpolation** splits (~ 60%), as one would expect this does not hold for the more challenging settings, e.g. it significantly drops to 17% on the ***extrapolation*** setting.

More generally, it seems that the model ***has troubles generalizing*** when some attributes are never seen during the training, (e.g., extrapolation or attr.rels settings) which seems to indicate the model probably more easily picks visual properties rather than  high-level abstract reasoning ones.


#### Detailed results
The authors also report results broken down by number of relations per matrix, relation types and ttribute types(when only one relation). As one would expect, one-relation are the easiest to solve, but, interestingly, it is slightly easier to solve three-relations matrices than four-relations one, which might be because it determines a more precise answer.

As for relations,`XOR` and `progression` are the hardest to solve although the model still performs decently well on those (50%).


---

<h3 class="section followup">Closely related (follow-up work)</h3>

#### 1. Improving Generalization for Abstract Reasoning Tasks Using Disentangled Feature Representations.
**Steenbrugge et al., [[link]](https://arxiv.org/abs/1811.04784)**

> The main observation is that the previously proposed model seems to ***disregard high-level abstract relations*** (e.g. considering the poor accuracy on the extrapolation set). this paper proposes to improve the encoding step by embedding the panel in a **"disentangled"** space using a $\beta$-VAE. (Note the apostrophes around *disentangled*, as, as far as I know, there is no guarantee of this property for $\beta$-VAE;  it has only been studied experimentally on specific datasets e.g. CelebA).

> There are also a few weird details in the experimental section. For instance, they claim the RN embedding has dimension 512, while it has dimension 256 (it only becomes 512 whe concatenating in $g_{\theta}$). Second, their own VAE embedding has latent dimension 64 and it's not clear why they wouldn't use more dimensions fo a fairer comparison (my guess is it might lose the "disentangled" property). In the end their encoder adds two additional fully-connected layers for no apparent good reasons.

> The model yields some improvement, especially on the more challenging settings (roughly 5% at best). They however omit results in the extrapolation regime, for unkown reasons.

---

<h3 class="section references"> References </h3>
* [1] A simple neural network module for relational reasoning, Santoro et al., NIPS 2017
