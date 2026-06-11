"""Model-evaluation pipeline (my part of the shared study).

models/ -> registry: load + preprocess + metadata, one entry per candidate model.
eval/   -> input loader for the pre-corrupted sets, metrics, the pure eval atom,
           and the fan-out driver that writes the tidy results table.
configs/-> which models x which corruption sets.

Expensive work (GPU eval) lives upstream of the results table; downstream
analysis/plots only ever read it. A model is never recomputed to redraw a figure.
"""
