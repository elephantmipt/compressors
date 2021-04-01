Callbacks
=========

.. toctree::


Logits difference callbacks
---------------------------
Callbacks which use difference between probabilities distribution over last layers.
Useful for classification task.

.. automodule:: compressors.distillation.callbacks.logits_diff
    :members:
    :exclude-members: on_batch_end


Wrappers
--------
Wrappers is useful when your callback inputs something different then hidden states or logits,
but you don't want to modify batch in your runner.

.. automodule:: compressors.distillation.callbacks.wrappers
    :members:
    :exclude-members: on_batch_end

Preprocessors
-------------

Inplace analogs of wrappers. Preprocess your ``runner.batch`` before applying callbacks.

.. autoclass:: compressors.distillation.callbacks.LambdaPreprocessCallback
    :members:
    :exclude-members: on_batch_end

.. autoclass:: compressors.distillation.callbacks.HiddenStatesSelectCallback
    :members:
    :exclude-members: on_batch_end


Hiddens states
-------------------------

.. automodule:: compressors.distillation.callbacks.hidden_states
    :members:
    :exclude-members: on_batch_end