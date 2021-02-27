Callbacks
=========

.. toctree::


Logits difference callbacks
---------------------------
Callbacks which use difference between probabilities distribution over last layers. 
Useful for classification task.

.. automodule:: src.distillation.callbacks.logits_diff
    :members:
    :exclude-members: on_batch_end

Hiddens mapping callbacks
-------------------------

.. automodule:: src.distillation.callbacks.hiddens_mapping
    :members:
    :exclude-members: on_batch_end

Hiddens difference callbacks
----------------------------

.. automodule:: src.distillation.callbacks.hiddens_diff
    :members:
    :exclude-members: on_batch_end

