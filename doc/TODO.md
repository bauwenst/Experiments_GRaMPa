# TODO
- Some findings about tuning hyperparameters:
  - You can't tune temperature with Rényi efficiency, because higher temperatures bias the segmentations out of uniformity
    and has the same effect on token usage uniformity (which is what RE measures) hence dropping RE for any temperature
    that isn't 1.0. Also, on top of this, the RE barely changes when temperature changes, so even if the effect was positive,
    you could barely notice.
  - You can't tune multiplexing-p with Rényi efficiency, because the multiplexed tokenisers likely have different ranges of RE and then you just get p=0 or p=1 depending on which range is above the other range.
  - You can't tune Kudo alpha with Rényi efficiency, because the alphas with better downstream performance have lower Rényi efficiency,
    so the typical correlation between RE and downstream no longer holds and so anything can hold.
- I wonder how the following metrics vary based on BPE-dropout rate and Kudo alpha:
  - Fraction of paths observed. This is also macro/micro-averageable across words.
    - I'm looking for a reason that BPE-dropout's best-performing parameter has only a regularisation rate of 0.1.
      You *could* hypothesise this is because seeing many different segmentations is bad. However, there is no guarantee
      that more dropout leads to a more *uniform or diverse* sample from the segmentations. That's the whole point of the Cognetta paper.
      So instead of choosing the ULM alpha such that it matches BPE-dropout's regularisation rate (and also choosing the
      multiplexing p to be the regularisation rate, i.e. 0.1, which means 90% of words do not see GRaMPa...), instead try to
      match some other metric that matches BPE-dropout's p=0.1 setting. (It is true that the multiplexing p cannot be chosen
      easily then, but that 90% is insane.)
  - Entropy of path distribution (can observe all paths but still be very skewed), which is not the same as RE which measures the entropy of the token distribution. This is not averageable across words, although possible if you normalise by the maximal entropy, you can.
  - Average token amount (which should perhaps be normalised by word length to get segmentality).
    - We have a plot for GRaMPa temperature about this.
    - Higher Kudo alphas cause more skew towards the argmax path, but is that necessarily the shortest path? You would think in a sense that shorter paths are advantaged given that they have fewer tokens, hence a product across fewer probabilities. This is the same argument as to why BoMMa needed its multiplicatively balanced transform and why uniform arc sampling in the DAG doesn't produce globally uniform segmentations.
- Re-tune parameters as follows:
  - [x] BPE-dropout: we already have p=0.1. This is also roughly the p with highest RE.
  - [ ] ULM-sampling: ~~RE(alpha) for alphas above 1.0, since RE kept rising. => It did keep rising... which means you can't trust RE for finding alpha since larger alphas just cause the ULM argmax to be selected, rather than sampling effectively.~~ 
    Kudo suggests using alpha = 0.1 (almost fully uniform) for those k where every segmentation is meaningful, so not k=infty, and
    otherwise making alpha higher to make the distribution skew towards the argmax. In fact, he says that for low-resource settings,
    you actually want MORE argmax, not less.
    The problem with alpha be-> 0 is that it's I opt for keeping the alpha = 0.3.
  - Deterministic BPE+GRaMPa: 
    - [ ] **Choose temperatures** based on histograms and/or examples: (-x, 1, +y).
    - [x] ~~RE(p), which is independent of temperature (or, we know it drops for all temps that aren't 1, so just use 1 always), says p=0.55.~~ You can't tune multiplexing p with RE because the scale for RE is different across tokenisers, so the tuning will point to either one tokeniser or the other tokeniser unless they coincidentally (and meaninglessly) have the same scale of RE.
    - [x] It's not obvious that you want to multiplex with dropout's p=0.1 regularisation rate (namely 0.1), because Cognetta already shows that BPE+uniform performs better with higher regularisation rate, proving the above theory that **it's not high quantity of regularisation but low quality of regularisation** that hurts you.
  - Deterministic ULM+GRaMPa: 
    - [x] Same temperatures as above.
    - [ ] ~~RE(p), again with t=1. We fucked this tuning up last time by choosing alpha=0.5, k=64. Should've been alpha=1.0, k=1.~~ As stated above, this is pointless.
    - [x] Multiplex with alpha=0.3 regularisation rate, which is 0.25.
- Print DeBERTa model to make sure it looks like what we want it to look like (e.g. no absolute positionals)
- Throw one onto A100 debug to find a working batch size.
- Throw all 8 onto H100.
  - Implement GLUE/SuperGLUE
  - Process paper comments
- Finetune
