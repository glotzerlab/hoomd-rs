# Release notes

## Next release

*Added:*

* Implement `IntersectsAt` for `ConvexPolygon` intersection tests.
  The separating planes method is faster for small *n* than the Xenocollide
  algorithm implemented for `Convex<ConvexPolygon>` (#260).

*Changed:*

* Remove unnecessary trait bounds on `IntersectsAt` implementation (#260).
* Adjust benchmark parameters to make accurate comparisons with HOOMD-blue (#260).

*Deprecated:*

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
