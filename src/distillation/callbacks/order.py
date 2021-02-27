class CallbackOrder:
    """Callback usage order during training.
    Catalyst executes Callbacks with low `CallbackOrder`
    **before** Callbacks with high `CallbackOrder`.
    Predefined orders:
    - **Internal** (0) - some Catalyst Extras,
      like PhaseCallbacks (used in GANs).
    - **Metric** (20) - Callbacks with metrics and losses computation.
    - **MetricAggregation** (40) - metrics aggregation callbacks,
      like sum different losses into one.
    - **Optimizer** (60) - optimizer step,
      requires computed metrics for optimization.
    - **Scheduler** (80) - scheduler step,
      in `ReduceLROnPlateau` case
      requires computed validation metrics for optimizer schedule.
    - **External** (100) - additional callbacks with custom logic,
      like InferenceCallbacks
    Nevertheless, you always can create CustomCallback with any order,
    for example::
        >>> class MyCustomCallback(Callback):
        >>>     def __init__(self):
        >>>         super().__init__(order=42)
        >>>     ...
        # MyCustomCallback will be executed after all `Metric`-Callbacks
        # but before all `MetricAggregation`-Callbacks.
    .. _Alchemy: https://alchemy.host
    """

    Internal = internal = 0  # noqa: WPS115
    HiddensSlct = hiddens_slct = 5  # noqa: WPS115
    HiddensMapping = hiddens_mapping = 10  # noqa: WPS115
    Metric = metric = 20  # noqa: WPS115
    MetricAggregation = metric_aggregation = 40  # noqa: WPS115
    Optimizer = optimizer = 60  # noqa: WPS115
    Scheduler = scheduler = 80  # noqa: WPS115
    External = external = 100  # noqa: WPS115


__all__ = ["CallbackOrder"]
