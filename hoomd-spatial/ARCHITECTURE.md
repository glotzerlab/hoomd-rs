# Spatial Data Structures Subcrate – Architecture Overview

This subcrate provides specialized spatial data structures for particle systems. It is designed to be independent of system infrastructure while supporting both real and ghost particles.

## System Components

- **Data Arrays:**
  - **Sites:** Array of particle positions.
  - **Ghosts:** Array for ghost particles.
- **Particle Types:**
  - Enumeration distinguishing between ghost and real particles.

## Cell List Data Structure

The cell list is the core data structure for neighbor search operations. It maps space
into a Cartesian grid of cube cells and leverages hash maps for storage and lookup. The
struct is independent of spatial dimension and works only for N-dimensional Cartesian space.

### Members

- **Contains:**
  - **Cell Width**
    Uses cube boxes to partition space. The cell size is user-defined and exposed as a parameter.
  - **Primary Hash Map:**
    Maps cell indices (keys, represented as an array of cell indices) to vectors of particle indices (values).
    *Note: This structure may be extended to use tuples to distinguish between ghost and real particles.*
  - **Secondary Hash Map:**
    Provides a mapping from individual particle indices to their corresponding cell index.

### Methods and Operations

- **Creation:**
   - Requires only points and cell width. PBCs are implemented via ghosts - if ghosts
     are provided we might have to store a tuple instead of just index (index, type enum).
   - Mapping of coordinates to cells is done via:
    `cell = floor(x / width)` using array from fn
   - This information gets encoded in two Hash Maps.

- **Locate Cell:**
   - Identify the cell that contains the particle of interest based on its position or
     tag.

- **Neighbor Search:**
   - Given a particle position and a ball radius cutoff, determine all neighboring cells to consider.
   - Retrieve and filter candidate particles from these cells for potential interactions
     based on cutoff.

- **Adding a Particle:**
  - Insert the particle into the appropriate cell.
  - Update both hash maps accordingly.

- **Removing a Particle:**
  - Delete the particle entry from the relevant cell.
  - Update both hash maps accordingly.

- **Moving (Translating) a Particle:**
  - **If the particle crosses cell boundaries:**
    - Remove it from the old cell and add it to the new one.
  - **If the particle remains in the same cell:**
    - No action is required.

- **Rebuilding the Cell List:**
  - Provide a convenience method to rebuild both hash maps.

## Future Enhancements

- **Parallelism Support:**
  - Introduce an origin offset for translating cube coordinates, which will aid in parallel computation.
  - Develop an efficient method for iterating over all particle indices within a given cell.
