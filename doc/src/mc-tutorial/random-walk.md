# Random Walk

<script type="module">
import init from './random-walk.js'

init().catch((error) => {
  if (!error.message.startsWith("Using exceptions for control flow, don't mind me. This isn't actually an error!")) {
    throw error;
  }
});
</script>

<canvas id="hoomd-example" width="750" height="421"></canvas>

```rust,ignore
{{#include ../../../examples/mc-tutorial/random-walk.rs:simulation_struct}}
```
