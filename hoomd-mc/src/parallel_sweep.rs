

// ParallelSweep
// 1. Make checkerboard (or update a cached one for efficiency)
// 2. Place body indices in checkerboard spaces.
// 3. Loop until a sufficient number of trial moves is performed.
// 4. Loop over all space indices by color.
// 5. Prepare trial bodies vec (same len the current space indices)
// 6. for each space index (of the current color) in parallel:
//    * Choose a body randomly.
//    * Propose a trial move.
//    * Reject if the body center leaves the current space.
//    * Accept or reject the move as in Sweep.
//    * Store the result in the trial bodies
// 7. Process the trial bodies output and apply the accepted moves.
