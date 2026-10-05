Model102: Model101 versus training-dictionary candidate
=======================================================

Model102 applies the unchanged leakage-safe selector to Model101 and Model89.
Model89's dictionary was learned fold-safely from supplied training text only.
Every evaluated row is excluded from the selector and language-model fit. No
GPU, external data, or leaderboard feedback is used.

OOF result
----------

  Model101: 0.8998601038879951
  Model102: 0.9000848759914764
  delta:   +0.0002247721034813

Folds 0, 1, 2, and 4 improve; fold 3 regresses slightly. Model89's dictionary
configuration was selected using aggregate OOF evidence, so this result has
more model-selection risk than Model101 and is treated as experimental.

A 5,000-sample paired row bootstrap versus Model101 gives a 95% interval of
[+0.0000141059, +0.0004433157], positive in 98.28% of resamples. The candidate
is therefore retained, but Model101 remains the cleaner validation result.

Candidate
---------

  submissions/model102_dictionary_selector.csv
  SHA-256 560861f7fe2e2d8907a1a90ead87e5e9645d17177fdca8983df4abb1b7db3f9b

It differs from Model101 on 38/1,374 test rows and is schema-validated against
the official sample submission. Nothing is submitted automatically.
