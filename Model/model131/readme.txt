Model131: LightOnOCR instruct-base variant - PREPARED, NOT RUN
==============================================================
Idea: train the established recipe from lightonai/LightOnOCR-2-1B (the
instruction-tuned release) instead of the -base checkpoint every other LightOn
member starts from. A different starting point should decorrelate its errors
from the existing LightOn family while keeping their strength.

config_instruct.json is Model129's recipe with only the base path changed;
queue.txt runs all five folds with test predictions.

Deliberately deprioritized rather than forgotten. Model132 showed that a second
member of an architecture already in the vote can make it worse (a second Kraken
cost -0.0006 on fold 0), and the LightOn family already contributes three
members. The GPU budget went instead to Model136's Qwen2.5-VL-7B - a genuinely
different family - which gained +0.0037 on fold 0. If budget remains after
Model136's five folds, this is the next cheapest thing to try, and it must be
gated on fold 0 first exactly as Model132 and Model136 were.

Status: configuration ready; nothing trained, no score claimed.
