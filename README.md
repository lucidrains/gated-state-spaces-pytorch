## Gated State Spaces - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2206.13947">Gated State Spaces</a>, from the paper "Long Range Language Modeling via Gated State Spaces", in Pytorch

It will also contain a few more settings to compare state spaces to a sequence-wise GLU depthwise conv, and even simpler, a parameterized exponential moving average along the sequence dimension. So we get to the bottom of whether state spaces are worth it. Results will be shared in the readme.

## Citations

```bibtex
@inproceedings{Mehta2022LongRL,
    title   = {Long Range Language Modeling via Gated State Spaces},
    author  = {Harsh Mehta and Ankit Gupta and Ashok Cutkosky and Behnam Neyshabur},
    year    = {2022}
}
```
