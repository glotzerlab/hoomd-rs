# Release notes

## Next release

*Added:*

* `[hoomd-mc]`: Add `TuneOptions` type that describes move size tuning options.
* `[hoomd-mc]`: Added `Tune::tune_with_options` associated method that tunes trial move
  sizes with options passed via `TuneOptions`.
* `[hoomd-mc]`: Added `Sweep::apply_with_filter` associated method that applies trial
  moves only to bodies that match a filter. For example, use `apply_with_filter`
  to keep a crystal seed fixed during the simulation.
* `[hoomd-mc]`: Added `Sweep::tune_with_options_and_filter` associated method that
  tunes trial move sizes while only applying trial moves to bodies that match
  a filter. Use `tune_with_options_and_filter` with the same filter given to
  `apply_with_filter` to accurately tune move sizes.
* `[tutorial]`: Added *"Seeded Self-Assembly"* tutorial.

*Changed:*

*Deprecated:*

* `[hoomd-mc]`: Deprecated `Tune::tune`. Use `tune_with_options`.
* `[hoomd-mc]`: Deprecated `Tune::tune_default`. Use `tune_with_options(..., &TuneOptions::default())`.

*Removed:*

*Fixed:*

## 1.0.2 (2026-03-20)

*Fixed:*

* Trusted publishing workflow.

## 1.0.1 (2026-03-20)

*Fixed:*

* `hoomd-microstate` documentation builds on docs.rs (#253).
* Unit tests pass on the Windows native platform (#253).

## 1.0.0 (2026-03-17)

*Initial release*.
