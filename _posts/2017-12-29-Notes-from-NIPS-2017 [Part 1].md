---
title: Notes from NIPS 2017 (Part 1)
updated: 2015-12-01 15:56
---

> This is a summary of the tutorial, "Deep Learning Practice and Trends" by Oriol Vinyals and Scott Reeds at NIPS 2017. It's a rough draft, I will flesh out the post in the coming weeks.


I attended the 2017 NIPS Conference held at Long Beach as a Microsoft employee (thanks Microsoft India for sending me!). And it was great attending the full conference with tutorials & workshops (workshops were the best!). I found the first tutorial, "Deep learning Practice and Trends" very well prepared. As the title says, the talk was divided into two parts: Practice, which gave a nice high-level view of the Deep Learning toolbox and Trends, which explained 5 trends in deep learning research. 

<iframe width="560" height="400" src="https://www.youtube.com/embed/YJnddoa8sHk" frameborder="0" allowfullscreen></iframe> <style> .responsive-wrap iframe{ max-height: 100%;} </style>

## The Deep Learning Toolbox

Deep learning excels at structured data, e.g. images. This part covered the standard architectures/tools that form the building blocks of a deep learning model. The key message was that different neural net architectures need to have the right inductive biases, depending on the input & the task they are operating upon. For Images & Image classification, the key inductive biases are 1. Locality Invariance 2. Translation invariance which are realised in a Convolutional layer, which is a locally connected layer with weight sharing. 

Another key tool is depth, but training is a deep net is not easy: depth can't be parallelized (unlike convolutions, hence slow training) and there are too many parameters to optimize. So how to build very deep CNNs? The tricks are:
- SGD + Momentum (typical for CNNs)
- BatchNorm after each weight layer (It led to higher accuracy plus faster training for Inception v2) 
- Weight Initialisation (Glorot works well for VGG)
- Model Building: Use stacks of 3x3 Conv layers & ResNET: Add residual connections that skip 2-3 weight layers. The effect  is that since the gradient skips a few weight layers and does not vanish, a better gradient flow is maintained. Another related architecture is highway network and DenseNet (skip connections between everything and everything, resulting in a dense block which can be treated as just another building block) and UNET architecture. 

Further, moving to text, it is worth recalling that the 2 key ingredients that have brought DL into NLP are 1. Neural embeddings (key insight: vectorizing contexts) and RNN language models, which outperformed other approaches to language modelings, since RNNs have persistent memory i.e. a state variable for arbitrarily large contexts. 

Seq2Seq models (easy to implement in Tensorflow) are an extension to RNNs, however they have the limitation that increasing sentence length causes a bottleneck; which leads to a sharp drop in BLEU scores in the NMT task. Attention is a mechanism to alleviate this information bottleneck. The idea in attention (Bahdanau paper, 2015) is that the decoder can query the encoder at every time step. It is really simple to implement and is a way of differentiable content based memory addressing. There are cool extensions to this model as well (R/W Memory as in Neural Turing Machines, Pointer, Key-Value Memory etc.). Finally, some tricks to train sequence models:

1. Long Sequences? Use attention, bigger state and reverse inputs (e.g. in translation)
2. Can't overfit? Bigger hidden state, deep LSTM + Skip Connections
3. Overfit? Dropout + Ensembles
4. Tuning: decrease the learning rate, Initialize parameters (-Uniform(0.05, 0.05)), clip the gradients.

Now, the cool part of the talk was the Trends part, which discussed a number of models that are really taking deep learning to new levels.

## Trend 1: Autoregressive models

The idea is quite simple: The joint distribution to be learnt can be written as $$ P(x;theta) = ? $$, and each factor can be parameterized by theta, which can be shared, as long as the ordering and grouping is consistent (i.e. it doesn't violate causality). Each factor can be parameterized by a DNN (just like in DQNs, which learned to play Atari games by parametering the Q function with a neural net). So, the key questions are 1. how do you order and group the variables and 2. how do you parameterize them? 
#### Part 1: Modeling Raw Waveforms using Causal Convolutions
Key thing is that each output depends only on the input from prior time steps. And if you want to get more context for every prediction, you can use dilation. So, you have causal dilated convolutions. And of course, you can stack them. But more tricks are needed: If you have many many possible values the cross entropy loss has a large memory consumption. So they modeled the loss using a discretized mixture of logistic losses (a mixture of sigmoids).
Another question is: how do you speed up the sampling? To sample from these models you have to go from left to right which is an O(N) operation. But you can first distill a student net from a teacher net i.e. pretrain a Wavenet teacher in the usual way and then train the student net (kin of like GANs).
####  Part 2: Modeling Text
Self attention: the weights are adaptive and it has a more flexible architecture than convolutions (which have the same kernel and same weights).  
#### Part 3: Modeling Images
You can do it pixel by pixel or group by group (with conditional independence b/w groups, you lose expressiveness but then you can do parallel sampling in O(logN)). 

So, to summarize Autoregressive models, we had 2 kinds of models:
   1. Fully Sequential Models: You can factorize per sample or per pixel, this gives you fast scoring but O(N) sampling. You can make conditional independence assumptions, which enable faster sampling.
   2. Distilled Models: Examples are Wavenet, Parallel NMT. They have O(N) scoring and O(1) sampling.
      
## Trend 2: Domain Alignment

Scott Reed said this was the most promising thing in unsupervised learning. And it absolutely is. And it is the method behind two of my favorite papers in NLP this year: Language Style Transfer and Unsupervised Word Translation. The idea behind this trend is also very simple and the problem is immensely practical.

So let's see the problem/model description: 

-  Input: Input is a set of images with some shared structure but there exists no pairing between the images in one domain and the other (i.e. there is no direct alignment). For text, you could have a text corpora in different languages but you don't have the matching sentences.
- Architectures: They aren't fancy but are cleverly hooked up. 
- Losses: are wired with a loss so that an alignment emerges between the two domains. So, the losses could be in the latent space (e.g. you want the latent representations to be indistinguishable) or in the raw observation space (e.g. pixel space). Adversarial objectives and max likelihood both work well.

#### Approach 1: Weakly Supervised (e.g. Visual Domain Alignment b/w SVHN and MNIST)
So you could do the alignment in a weakly supervised way: share layers with different modalities, train them for some downstream task and then neurons with activations over the same semantic units occur! Then you can do cross-domain retrieval i.e. you can query the model and retrieve the images in other domains.
  
#### Approach 2: Adversarial Learning
Here, you are aligning the domains by construction. You can share the encoder over several domains, so that the features learnt are invariant to a domain. The paper Domain Adversarial Training of NNs explains it very well.
  
Examples:
  1. Unsupervised cross domain Image generation
  2. Cycle Consistent Loss (Unpaired Image to Image Translation)
  3. Unsupervised Image to Image translation
  4. DiscoGAN
  5. GraspGAN
  6. Unsupervised NMT
  7. Unsupervised NMT using Monolingual Corpora only
  
## Trend 3: Meta Learning
It is about a loss that models another loss. The idea is to go beyond a train and test paradigm. So, meta-learning adds a twist to the standard paradigm. The task between the training set and the test set changes, so the model has to learn to learn.
- There are three approaches to meta-learning:
  1. Model Based: Model is conditioned on a support set.
  2. Metric Based: Support set is kept in memory (differentiable nearest neighbour).
  3. Optimization Based
     
## Trend 4: Deep Learning Over Graphs
Graphs are hard to represent as tensors. There are two main architectures: message passing and (CNNs, RNNs, attention). But again the question to ask is: What is the right inductive bias for  graphs. We want our model to be order invariant. So, X = permutation(X). Further, with graphs batching will be quite tricky.
  
## Trend 5: Program Induction with NNs
So you have 3 ways:
  1. You can think of NN as a program: embed a program as weights of a NN.
  2. NN that generates source code. (Learning to execute paper by Zaremba et al. Follows under the paradigm of "Apply Seq2Seq to everthing", Deep Coder, RobustFill: shows that Neural program induction works).
  3. Probabilistic Programming with NNs.

To conclude the trends part: 
- Deep AR models and CNNs are in production in consumer apps.
- Inductive biases are very useful: CNNs (spatial), RNNs (time) and Graphs (Perm. Invariance).
- Simple tricks like Resnets/skip-connections are very helpful too.
- Adversarial Nets + Unsupervised domain adaptations work well.
- Meta-learning: more and more lifecycle of a model will be the part of the end to end training.
- Program synthesis and deep geometric learning will become very important. 
     
