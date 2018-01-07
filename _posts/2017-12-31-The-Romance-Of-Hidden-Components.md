---
title: The Romance of Hidden Components
updated: 2015-12-01 15:56
---

> It is amazing that deep learning works. Billions of parameters simultaneously converging to a good solution, not getting trapped in local minima. In convex optimization, the theory is very clear regarding the convergence of gradient descent (i.e. how many iterations?). But the regime in which deep learning operates is very different, besides just non-convexity. We do not have good intuitions about very high dimensional spaces. Another reason we should understand high-dimensional spaces is that, in the data that we observe, is high dimensionality just nominal? The manifold hypothesis certainly says so. And if the high dimensions are just nominal, how can we find the real manifold where the data resides. This post covers some properties of high-dimensional spaces, how can we extract the real data manifold from a high-dimensional description and some connections to deep neural nets.


## Properties of High Dimensional Spaces

- Why study high-dimensional spaces?

  Well, they are spaces where real world objects reside like images. A 5 MP image (in the pixel space) resides in a 5x10^6 dimensional space. Similarly, tuning neural nets is an optimization problem in a parameter space of billions of parameters.
- How are they different from low-dimensional spaces?

  There are some fundamental differences. There are curses as well as blessings.
- It seems impossible to optimise a function with parameters in this space?

  Um.. SGD works well. These spaces are dominated by saddle points. And gradient descent with some noise can escape saddle points. It is an active research area. 
- And what is the true space where the data resides?

  Yes, that's the crux. If we know where data resides, we can study it, manipulate it. The subject is called manifold learning. PCA is based on certain linear assumptions (data resides in a linear subspace). And then there are non-linear algorithms. And there are neural net approaches too (autoencoders).


... To be continued..
