let canvas = document.getElementById('hoomd-example')
canvas.addEventListener("keydown", function(e) {
  e.stopPropagation();
});

await init().catch((error) => {
  if (!error.message.startsWith("Using exceptions for control flow, don't mind me. This isn't actually an error!")) {
    throw error;
  }
});
// for unknown reasons, the web builds focus themselves on start.
// The `await` above waits for `init` to complete, then `blur`
// removes that focus to avoid disrupting the reader.
canvas.blur();
