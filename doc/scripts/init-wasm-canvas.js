let canvas = document.getElementById('hoomd-example')
canvas.addEventListener("keydown", function(e) {
  e.stopPropagation();
});

await init().catch((error) => {
  if (!error.message.startsWith("Using exceptions for control flow, don't mind me. This isn't actually an error!")) {
    throw error;
  }
});
