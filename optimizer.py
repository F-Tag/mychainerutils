from chainer.optimizer import WeightDecay


def apply_weightdecay_only_w(model, rate):
    """
    use after setup optimizer
    """

    for p in model.params():
        if p.name == 'W':
            p.update_rule.add_hook(WeightDecay(rate))