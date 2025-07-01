# hoomd_chimes
ChIMES interatomic potential (IAP) is a general purpose machine learned potential, diverging from the conventional potential used in hoomd to simulate colloids, such as LJ potential. Historically, it's targeting the use cases of generating an IAP by learning the density-functional theory simulation data, e.g. energy, force and virial. As a results, many components of ChIMES IAP are different from the other potential in hoomd-rs. I picture it would be constructed in a different manner so a hoomd-chimes crate might be a good idea to store all the related implementaion.

See [ChIMES 2.0](https://www.nature.com/articles/s41524-024-01497-y) paper and the paper about [CG model to simulate colloids using ChIMES](https://chemrxiv.org/engage/chemrxiv/article-details/6838ce27c1cb1ecda034de24).

## Key components

### Chebyshev polynomials
ChIMES uses Chebyshev polynomials to fit to the energy, force and virial data. The `hoomd_chimes::cheby` is implemented for calculating the polynomials and it's derivatives.

In principal, the `hoomd_chimes::cheby` struct can also be used in other places requires a one-dimensional complete basis set.

### Transformation style
Since Chebyshev polynomials' domain is conventioanlly defined within `[-1, 1]`. The pairwise distance `r` used to calculate potential must be transformed into a new coordinate `s` fall in `[-1, 1]`. This requirement gives rise to the ChIMES transformation style, which is a function acting on `r` and the resulting `s` will be used to calculate Chebyshev polynomials.

A trait `hoomd_chimes::transformtation::Transformation` is placed to ensure the consistent implementation and to provide a way for user to custom their transformation style.

So far, the most popular Morse style transformation is implemented as `hoomd_chimes::transformation::MorseTransformation`. It is also served as an example of how to do it in hoomd-rs.

`TODO`:

1. Direct transformation.
2. Inverse transformation.

### Parameter text file parser
`TODO`:
The ChIMES IAP models (hyperparameters) are stored as a text file. A parser would enable a more convenient use of ChIMES model.

### Wrapper
`TODO`:
Due to a higher complexity of ChIMES IAP, many struct and fucntions are constructed and a wrapper to coordiante the their use would enable a more convenient use.

## Related components in hoomd-interaction

### Combine chebyshev polynomials and fitting coefficients
A struct `hoomd_interaction::pairwise::Chimes2b` is implemented for such purpose.

### Smoothing fucntion
ChIMES IAP use a smoothing fucntion multiply on the potential to ensure the potential smoothly decrease to 0 at sufficient long distance to make sure the stability of simulation. Similar to the `hoomd_interaction::pairwise::Shifted` and `hoomd_interaction::pairwise::Xplor` of hoomd-rs. The tersoff style smoothing `hoomd_interaction::pairwise::TersoffSmooth` is implemented.

`TODO`:

1. Cubic smooth

### Penalty fucntion
ChIMES IAP is essentially a bounded potential defined only in a interval between inner and outter distance cutoffs. When particle fall within inner cutoff, a penalty function is used to add energy penalty to the interaction and push particles away from each other and to prevent erroneous results. The `hoomd_interaction::pairwise::ChimesPenalty` is implemented for such purpose.

### Three- and Four body compoenents
`TODO`:

ChIMES IAP is essentially a many-body potential. Three- and Four-body parts should be implemented once the infrastructure of hoomd-rs is ready.


## Implemntation validation
`TODO`:

[ChIMES-Calculator](https://github.com/LindseyLab-umich/chimes_calculator-LLfork/tree/main/serial_interface/tests/force_fields) repo provides tons of force field models for benchmarking. In future, we should use those models to run MD/MC simulations in hoomd-rs to validate the implementation.
