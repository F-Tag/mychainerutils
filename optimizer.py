from chainer.optimizer import WeightDecay


def apply_weightdecay_only_w(model, rate):
    """
    use after setup optimizer
    """

    if hasattr(model, 'params') :
        for p in model.params():
            if p.name == 'W':
                if hasattr(p.update_rule.hyperparam, 'weight_decay_rate'):
                    p.update_rule.hyperparam.weight_decay_rate = rate
                else:
                    p.update_rule.add_hook(WeightDecay(rate))