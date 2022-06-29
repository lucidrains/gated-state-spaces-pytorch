<img src="./gss.png" width="400png"></img>

## Gated State Spaces - Pytorch (wip)

Implementation of Gated State Spaces, from the paper <a href="https://arxiv.org/abs/2206.13947">Long Range Language Modeling via Gated State Spaces</a>, in Pytorch. In particular, it will contain the hybrid version containing local self attention with the long-range GSS.

It will also contain a few more settings to compare state spaces to a sequence-wise GLU depthwise conv, and even simpler, a parameterized exponential moving average along the sequence dimension. So we get to the bottom of whether state spaces are worth it, or whether it is really all about the `O(L log(L))` FFT convolution trick. Results will be shared in the readme.

## Citations

```bibtex
@inproceedings{Mehta2022LongRL,
    title   = {Long Range Language Modeling via Gated State Spaces},
    author  = {Harsh Mehta and Ankit Gupta and Ashok Cutkosky and Behnam Neyshabur},
    year    = {2022}
}
```
