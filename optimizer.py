from chainer.optimizer import WeightDecay


def apply_weightdecay_only_w(model, rate):
    """
    use after setup optimizer
    """

    for p in model.params():
        if p.name == 'W':
            if hasattr(p.hyperparam, 'weight_decay_rate'):
                p.hyperparam.weight_decay_rate = rate
            else:
                p.update_rule.add_hook(WeightDecay(rate))