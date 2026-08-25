# Human judgment with an LLM copilot

The experiment used an LLM to propose an initial hyperparameter search space. The useful suggestions were optimizer, hidden-layer width, batch size, learning rate, and dropout.

The final decision was narrower than the suggestion:

- Accepted: Adam/RMSprop, 64/128/256 neurons, and batch sizes 32/64.
- Rejected for this run: learning rate and dropout.
- Reason: adding two learning rates would double the search from 36 to 72 fits, while dropout would introduce another architectural axis. The assignment called for a compact but meaningful search on a time-limited T4 session.

This is not presented as autonomous model design. The LLM supplied candidates; the developer set the computational budget, chose the final grid, kept the official test set untouched, and interpreted class-level errors. The weakest class was `Shirt` (F1 0.67), which is visually close to T-shirt/top, pullover, and coat.

