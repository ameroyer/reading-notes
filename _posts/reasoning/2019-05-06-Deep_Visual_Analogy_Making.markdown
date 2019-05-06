---
layout: post
title: "Deep Visual Analogy Making"
date: 2019-05-06 12:40:24 +0200
tags: [visual reasoning, 2015, nips, visalogy]
categories: [Visual Reasoning]
author: Reed et al., NIPS 2015, <a href='https://openreview.net/pdf?id=rJgMlhRctm' target='_blank'>[link]</a>
thumb: /images/thumbs/dvam.png
year: 2015
---


<div class="summary">
In this paper, the authors propose to learn <b>visual analogies</b> akin to the semantic and synctatic analogies emerging in the <code>Word2Vec</code> embedding <span class="citations">[1]</span>. In particular, they tackle the task of predicting the transformation of a source image under a certain analogy, inferred from another given (source, target) pair.
<ul>
<li><span class="procons">Pros (+):</span> Very intuitive, Introduces two datasets for the visual analogy task.</li>
<li><span class="procons">Cons (-):</span>  Only consider "local" scenarios, i.e. geometric transformations or single attributes, and very clean images (no background).</li>
</ul>
</div>


<h3 class="section proposed"> Proposed Model</h3>

***Definition.*** A visual analogy is denoted by "**a:b :: c:d**", meaning that the entity **a** is to **b** what the entity **c** is to **d**. In particular , this paper focuses on the problem of generating image **d** given the relation **a:b** and a source image **c**.

They propose to use an encoder-decoder based model for generation, and to learn to make analogy with *simple transformations of the latent space*, for instance addition or multiplication between vectors, as was the case in words embeddings such as `GloVe` <span class="citations">[2]</span> or `Word2Vec` <span class="citations">[1]</span>.


#### Learning to generate analogies via manipulation of the embedding space
**Additive objective.** Let $$f$$ be an encoder and $$g$$ the decoder. The first, most straightforward loss objective they propose is to learn analogy by addition in the latent space, in other words using the objective

$$
\begin{align}
\mathcal L_{\mbox{add}} = |d - g \left(f(b) - f(a) + f(c) \right)|
\end{align}
$$

One disadvantage of this *purely linear transformation* is that it cannot learn complex structures, for instance periodic transformations: For instance, if $$f(b) - f(a)$$ is a rotation, the decoded image (and embedding) should eventually comes back to $f(a)$ which is not possible when adding a non-zero vector. To capture more complex transformations, the authors introduce two variants of the previous objective.

**Multiplicative objective.**

$$
\begin{align}
\mathcal L_{\mbox{mult}} = |d - g \left( W \odot [f(b) - f(a)] \odot  f(c) \right)|
\end{align}
$$

where $$W \in \mathbb{R}^{K\times K\times K}$$ and the three-way multiplication is defined as $$(A \odot B \odot C)_k = \sum_{i, j} A_{ijk} B_i C_j$$

**Deep objective.**

$$
\begin{align}
\mathcal L_{\mbox{deep}} = |d - g \left( \mbox{MLP}([ f(b) - f(a),  f(c)]) \right)|
\end{align}
$$

where $$\mbox{MLP}$$ designates a Multi Layer Perceptron.

 
<div class="figure">
<img src="{{ site.baseurl }}/images/posts/deep_visual_analogy_1.png">
<p><b>Figure:</b>  Illustration of the network structure for analogy making. The top portion shows the encoder, transformation module, and decoder. The botton portion illustrates each of the transformation variants. We share weights with all three encoder networks shown on the top left</p>
</div>
 
#### Regularizer
 
 While the previous losses acted at the pixel-level between the decoded image and the target image **D**, the authors introduce an additional regularization loss that additionally matches the analogy between those two images *at the feature level* with the source analogy **a:b**:
 
 $$
 \begin{align}
 R = |(f(d) - f(c)) -  T(f(b) - f(a), f(c))|
 \end{align}
 $$
 
 Where $$T$$ is defined accordingly to match the chosen embedding, $$\mathcal L_{\mbox{add}}$$, $$\mathcal L_{\mbox{mult}}$$ or $$\mathcal L_{\mbox{deep}}$$. For intance, $$T: (x, y) \mapsto x$$ in the additive variant.
 
 
#### Disentangling the feature space
The authors consider another solution to the visual analogy problem which aims to learn a disentangled *feature space* that can be freely manipulated by selecting appropriate latent variables, rather than specific operation.

In that setting, the problem is slightly different, as we require additional supervision to control the different factors of variation. It can be denoted as **(a, b):s :: c**, which means: given two input images **a** and **b**, and a switch vector **s** controlling the latent space, retrieve image **c** which matches the features of **a** according to the pattern of **s**, and features of **b** on remaining latent variables. 

Let us denote by $$S$$ the number of possible axes of variations (e.g., change in illumination, elevation, rotation etc) then $$s \in \{0, 1\}^S$$, a one-hot block vector encoding the current transformation, called the *switch vector*. The disentangling objective is thus

$$
 \begin{align}
 \mathcal{L}_{\mbox{dis}} =  |c - g(f(a) \times + f(b) \times (1 - s))|
 \end{align}
 $$
 
 In other words the decoder tries to match **c**  by decoding separate information from **a** and **b**. Contrary to the previous analogy objectives, only three images are needed, but it also requires extra supervision in the form of a switch vector **s** which can be hard to obtain.
 
---

<h3 class="section experiments"> Experiments </h3>

 The authors consider three main experimental settings:
  * **Synthetic experiments on geometric shapes.** The dataset consists in 48 × 48 images scaled to [0, 1] with 4 shapes, 8 colors, 4rscales, 5 row and column positions, and 24 rotation angles. No disentangling training was performed in this setting.
  * **Sprites dataset.** The dataset consists of 60 × 60 color images of sprites scaled to [0, 1], with 7 attributes and 672 total unique characters. For each character, there are 5 animations each from 4 viewpoints: spellcast, thrust, walk, slash and shoot. Each animation has between 6 and 13 frames. We split the data by characters. In that case they test two disentanglement methods,:`dist`, where they only try to separate the pose from identity (collection of all attributes), and `dist+cls`, where they actually split the latent variables by attribute.
  
  * **3D Cars.** For each of the 199 car models, we generated 64 × 64 color renderings from 24 rotation angles each offset by 15 degrees.
  
The authors report results in terms of *pixel prediction error*. Out of the three manipulation method, $$\mathcal{L}_{\mbox deep}$$ usually performs best. However qualitative samples show that $$\mathcal L_{\mbox{add}}$$ and $$\mathcal L_{\mbox{mult}}$$ both also perform well, although they fail for the case of rotation in the first set of experiments, which justifies the use of more complex training objectives.

Disentanglement methods usually outperforms the other baselines, especially in *few-shots experiments*. In particular the `dist+cls` method usually wins by a large margin, which shows that the additional supervision really helps in learning a structured representation. However such supervisory signal sounds hard to obtain in practice in more generic scenarios.
  
  
<div class="figure">
<img src="{{ site.baseurl }}/images/posts/deep_visual_analogy_2.png">
<p><b>Figure 2:</b> Examples of samples from the three visual analogy datasets considered in experiments</p>
</div>


---


<h3 class="section followup"> Closely Related</h3>

<h4 style="margin-bottom: 0px"> Visalogy: Answering Visual Analogy Questions</h4>
<p style="text-align: right"><small>Sadeghi et al., <a href="https://arxiv.org/pdf/1510.08973.pdf">[link]</a></small></p>

> In this paper, the authors tackle the visual analogy problem in natural images by learning a joint embedding on relation and visual appearances using a *Siamese architecture*. The main idea is to learn an embedding space where the analogy transformation can be done by *simple latent vector transformations*. The model consists in a Siamese quadruple architecture, where the four heads corresponds to the three context and the candidate images of the visual analogy task. They do consider a *restrained set of analogies*, in particular those based on attributes or actions of animals or geometric view point changes. Given analogy problem $$I_1 : I_2 :: I_3 : I_4$$ with label $$y$$ (1 if $$I_4$$ fits the analogy, 0 otherwise), the model is trained with the following objective

$$
\begin{align}
\mathcal{L}(x_{1, 2}, x_{3, 4}) = y  (\| x_{1, 2} - x_{3, 4} \| -m+P) + (1 - y) \max (m_N - \| x_{1, 2} - x_{3, 4} \|, 0)
\end{align}
$$

> where $$x_{i, j}$$ refers to an embedding for pair $$i, j$$. Intuitively, the model pushes embeddings with a similar analogy close, and others apart (up to a certain margin $$m_N$$). The $$m_P$$ margin is a *heuristic to avoid overfittting*: Embeddings are only made closer if their distance is above the margin threshold $$m_P$$.The pairwise embeddings are obtained by subtracting the individual images embeddings. This implies the underlying assumption that $$x_2 = x_1 + r$$, where $$r$$ is the transformation from image $$I_1$$to image $$I_2$$.


> The authors additionally create a *visual analogy dataset*. Generating the dataset is rather intuitive as long as we have an attribute-style representation of the domain. Typically, the analogies considered act over *properties* (object, action, pose) of different *categories* (dog, cat, chair etc). As negative  data points, they consider **(i)** fully random quadruples, or (ii) valid quadruples where one of $$I_3$$ or  $$I_4$$ is swapped with a random image.

> The evaluation is done with *image retrieval metrics*, i.e. whether the correct image is retrieved. They consider generalization scenario: For instance remove analogy $$white \rightarrow black$$ during training, but the model will have seen e.g. $$white \rightarrow red$$ and $$green \rightarrow black$$.  There is a lack of details about the missing pairs to really get a full idea of the generalization ability of the model (i.e. if an analogy is missing from the training set does that mean its reverse also is ? or does "analogy" refers to the high-level relation or is it instantiated relatively to the category too ?)

---

  
<h3 class="section references"> References</h3>
   * <span class="citations">[1]</span> Distributed representations of words and phrases and their compositionality, <i>Mikolov et al., NIPS 2013</i>
   * <span class="citations">[2]</span> GloVe: Global Vectors for Word Representation, <i>Pennington et al., EMNLP 2014</i>
