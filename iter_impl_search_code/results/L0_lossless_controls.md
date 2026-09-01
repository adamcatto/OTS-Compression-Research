# L0 — lossless controls

Experiment `experiment_001` is a 1,000-step WikiText-2 causal Transformer
trace (19,491,431,115 bytes). All ten codec/mode results round-tripped exactly.

Zstandard level 3 was the strongest exact control: 1.07520x online and
1.07492x offline compression (about 7.0% space saved), with 66.98 s and 62.70
s encode time respectively. DEFLATE had comparable size but required roughly
11 minutes; LZ4 was effectively neutral and Snappy expanded the trace slightly.

This establishes that generic byte codecs find little redundancy in the raw
FP32 stream. D1 therefore targets controlled numeric approximation and measures
its downstream effect instead of pursuing another lossless byte codec.
