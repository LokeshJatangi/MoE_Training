# Narrow-expert continuation results

Measured run. Matched tokens/data order and reset Adam; FFN matrix MACs exclude attention, activation, routing selection and dispatch overhead.

| Seed | Arm | Total / active parameters | Conversion CE | Final validation CE | Loss drops / recovers dense |
|---|---|---|---|---|---|
| 17 | dense_control | 2289024 / 2289024 | 4.278877 | 4.051691 | True / True |
| 17 | full_copy | 4946304 / 2292096 | 4.278877 | 4.081297 | True / True |
| 17 | granular | 4955520 / 2301312 | 4.350333 | 4.065797 | True / True |

See summary.json for conversion drift, load fractions, gradient checks, checkpoint recovery, timings and parameter changes. No statistical quality or GPU efficiency claim follows from a one-seed CPU pilot.
