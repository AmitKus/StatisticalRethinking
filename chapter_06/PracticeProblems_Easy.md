## Problem: 6E1

Define three motivating criteria that define information entropy?

**Information:** The reduction in uncertainty derived from learning an outcome. 
To use this definition we need to define uncertainty in a principled way.

Some desired properties that a measure of uncertainty should possess.

1. The measure of uncertainty should be continuous.
2. The measure of uncertainty should increase as the number of possible events increases.
3. The measure of uncertainty should be additive.
(On page 177-178. )

There is only one function that satisfies these desiderata. The functions is usually known as **INFORMATION ENTROPY**. If there are $n$ different possible events and each event $i$ has probability $p_i$, and we call the list of probabilities $p$, then the unique measure of uncertainty we seek is:
$ H(p) = -\bold{E} log(p_i) = -\sum_i^n p_i log(p_i)$



