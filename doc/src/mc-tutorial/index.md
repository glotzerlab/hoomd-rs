# Monte Carlo Tutorial

The [`hoomd-mc`] crate implements Monte Carlo simulations.

In a Monte Carlo (MC) simulation, you chose the Hamiltonian (`$H$`) and what types
of trial moves to propose. *hoomd-rs* will:
1. Propose random trial moves that change the microstate: `$S -> S^\prime$`.
2. Compute the change in energy made by each trial move: `$\Delta E = H(S^\prime) - H(S)$`.
3. Then *accept* or *reject* the trial move based on criteria specific to that
   trial move.

For each of these steps, you can use built-in code provided by [`hoomd-mc`] or
use your own custom code.Read the following tutorials to learn how to run and
customize MC simulations using *hoomd-rs*.

[`hoomd-mc`]: api/hoomd_mc/index.html
