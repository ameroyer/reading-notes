---
layout: post
title:  "A simple Neural Network Module for Relational Reasoning"
date:   2019-05-14 08:59:24 +0200
tags: [architectures, nips, 2017]
categories:  [Architectures]
author: Santoro et al, NIPS 2017, <a href='https://arxiv.org/abs/1706.01427' target='_blank'>[link]</a>
thumb: /images/thumbs/asnnmfrr.png
year: 2017
---



<div class="summary">

The authors propose a <b>relation module</b> to equip <code>CNN</code> architectures with notion of relational reasoning, particularly useful for tasks such as visual question answering, dynamics understanding etc.
<ul>
<li><span class="procons">Pros (+):</span> Simple architecture.</li>
<li><span class="procons">Cons (-):</span>  Still a black-box module, hard to quantify how much "reasoning" happens.</li>
</ul>
</div>


<h3 class="section proposed"> Proposed Model</h3>

The main idea of *Relation Networks* (`RN`) is to constrain the functional form of convolutional neural networks as to explicitly learn relations between entities, rather than hoping for this property to emerge in the representation during training.

$$
\begin{align}
\mbox{Let } O \mbox{ be a set of objects }& O = \{o_1 \dots o_n\} \mbox{ and define}\\
\mbox{RN}(O) =  f_{\phi}& \left(\sum_{i, j} g_{\theta}(o_i, o_j) \right)
\end{align}
$$

$$f_{\phi}$$ and $$g_{\theta}$$ are defined as *Multi Layer Perceptrons*. The key ideas in this Relation Network modules are that **(i)** it acts on all pairs of objects which implies exhaustivity, ***(ii)*** They operate directly on the set of objects, i.e., are not constrained to a specific organization of the data and ***(iii)*** they are data-efficient in the sense that only one function, $$g_{\theta}$$ is learned to capture all the possible relations.

The *object* building blocks are defined with regard to the task at hand, for instance: 
  * **Attending relations between objects in an image**: The image is first processed through a full convolution network ConvNet. Each of the resulting cell yields a feature of dimensions $$k$$ as an object, which is  additionally tagged with its position in the feature map.
  
  * **Sequence of images.** In that case, each image is simply fed through a feature extractor and the resulting embedding is used as an object. 
  
  
<div class="figure">
<img src="{{ site.baseurl }}/images/posts/relation_network.png">
<p><b>Figure:</b> Example of applying the Relation Network for <b>Visual Question Answeting</b>. Questions are processed with an <code>LSTM</code> to produce a question embedding, and images are processed with a <code>CNN</code> to produce a set of objects for the <code>RN</code>.</p>
</div>


---

<h3 class="section experiments"> Experiments </h3>
The evaluation is done on the CLEVR dataset.  The main message seems to be that their model is very simple and yet often benefits the model accuracy when added to various architectures (`CNN`, `CNN` with `LSTM` etc.), comparing to methods exhibited in <span class="citations">[1]</span>. It is mainly compared to Spatial Attention (`SA`) which is another simple method to integrate some form of relational reasoning in a neural architecture.
  
  ---
 
 <h3 class="section followup">Closely related (follow-up work)</h3>
  
  
<h4 style="margin-bottom: 0px"> Recurrent Relational Neural Networks</h4>
<p style="text-align: right"><small>Palm et al, <a href="https://arxiv.org/pdf/1711.08028.pdf">[link]</a></small></p>

> This paper builds on the Relation Network architecture and propose to explore  *more complex relational structures*,  defined in a graph, using a *message passing* approach. We're given a graph with vertices $$\mathcal V = \{v_i\}$$ and edges $$\mathcal E = \{e_{i, j}\}$$. By abuse of notations, $$v_i$$ also denotes the embedding for vertex $$i$$ (e.g. obtained via a CNN) and $$e_{i, j}$$  is 1 where  $$i$$ and $$j$$ are linked, 0 otherwise. To each node we associate a *hidden state* $$h_i^t$$ at iteration $$t$$, which will be updated via message passing. After a few iterations, the resulting state is passed through a `MLP`  $$r$$ to output the result (either for each node or for the whole graph):
  
  $$
  \begin{align}
  h_i^0 &= v_i\\
  h_i^{t + 1} &= f_{\phi} \left( h_i^t, v_i, \sum_{j} e_{i, j} g_{\theta}(h^t_i, h^t_j) \right)\\
  o_i &= r(h_i^T) \mbox{ or } o = r(\sum_i h_i^T)
  \end{align}
  $$
  
>  Comparing to the original Relation Network:
  * Each update rule is a Relation Network that only looks at *pairwise relations between linked vertices*. The message passing scheme additionally introduces the notion of recurrence, and the dependency on the previous hidden state.
  * The dependence on $$h_i^t$$ could *in theory* be avoided by adding self-edges from $$v_i$$ to $$v_i$$, to make it closer to the Relation Network formulation.
  * Adding $$v_i$$ as input of $$f_\phi$$ looks like a simple trick  to avoid long-term memory problems.
  
  
> The *experiments* essentially compares the proposed `RRNN` model to the Relational Network and various more classical sequential architectures such as `LSTM`. They consider three datasets: 
   * **Babi.** NLP question answering task with some reasoning involved. Solves 19.7 (out of 20) tasks on average, while simple RN solved around 18 of them reliably.
   * **Pretty CLEVR.** A CLEVR like dataset (only with simple 2D shapes) with questions involving various steps of reasoning, e.g. "which is the shape $$n$$ steps of the red circle ?"
   * **Sudoku.** the graph contains 81 nodes (one for each cell in the sudoku), with edges between cells belonging to the same row, column or block.
    
    

<h4 style="margin-bottom: 0px"> Multi-Layer Relation Neural Networks</h4>
<p style="text-align: right"><small>Jahrens and Martinetz, <a href="https://arxiv.org/pdf/1811.01838.pdf">[link]</a></small></p>

> This paper presents a very simple trick to make Relation Network consider higher order relations than pairwise, while retaining some efficiency. Essentially the model can be written as follow:

$$
\begin{align}
h_{i, j}^0 &= g^0_{\theta}(x_i, x_j) \\
h_{i, j}^t &= g^{t + 1}_{\theta}\left(\sum_k h_{i, k}^{t - 1}, \sum_k h_{j, k}^{t - 1}\right) \\
MLRN(O) &= f_{\phi}(\sum_{i, j} h^T_{i, j})
\end{align}
$$

> Though it is not clear while this model would be equivalent to explicitly considering higher-level relations (as it is rather *combining pairwise terms for a finite number of steps*). According to the experiments it seems that indeed this architecture could be better fitted for the studied tasks (e.g. over  the Relation Network or Recurrent Relation Network) but it also renders the model even more obscure.

---

<h3 class="section references">References</h3>
* <span class="citations">[1]</span> Inferring and executing programs for visual reasoning, <i>Johnson et al, ICCV 2017</i>
