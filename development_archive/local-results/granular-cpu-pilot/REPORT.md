# Narrow-expert continuation results

Measured run. Matched tokens/data order and reset Adam; FFN matrix MACs exclude attention, activation, routing selection and dispatch overhead.

| Seed | Arm | Total / active parameters | Conversion CE | Final validation CE | Loss drops / recovers dense |
|---|---|---|---|---|---|
| 17 | dense_control | 90816 / 90816 | 2.624921 | 2.505817 | True / True |
| 17 | full_copy | 238784 / 91328 | 2.624921 | 2.511586 | True / True |
| 17 | granular | 240320 / 92864 | 2.655241 | 2.498899 | True / True |

See summary.json for conversion drift, load fractions, gradient checks, checkpoint recovery, timings and parameter changes. No statistical quality or GPU efficiency claim follows from a one-seed CPU pilot.
