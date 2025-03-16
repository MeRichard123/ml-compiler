**Up-Sweep Phase:**
- The `torch.cumsum(log_coeffs, dim=1)` operation corresponds to the up-sweep phase of the Blelloch scan, where partial sums are computed.

**Down-Sweep Phase:**
- The `torch.logcumsumexp(log_values - a_star, dim=1)` operation corresponds to the down-sweep phase of the Blelloch scan, where partial sums are propagated back down the tree.

**Exclusive Scan:**
- The padding (F.pad) and slicing ([:, 1:]) ensure that the result is an exclusive scan, where the first element is 0.

**Numerical Stability:**
- The use of logcumsumexp ensures that the computation is numerically stable, which is particularly important when working with probabilities or log-domain values.
