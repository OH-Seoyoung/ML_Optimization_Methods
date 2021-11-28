from keras.optimizers import Optimizer
from keras.legacy import interfaces
from keras import backend as K

class custom_Adam(Optimizer):

    def __init__(self, lr = 0.001, beta1 = 0.9, beta2 = 0.999,
                 epsilon = 1e-10, decay = 0, **kwargs):
        
        # store hyperparameters
        super(Adam, self).__init__(**kwargs)
        
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype = 'int64', name = 'iterations')
            self.lr = K.variable(lr, name = 'lr')
            self.beta1 = K.variable(beta1, name = 'beta1')
            self.beta2 = K.variable(beta2, name = 'beta2')
            self.decay = K.variable(decay, name = 'decay')
            self.epsilon = epsilon
            self.initial_decay = decay

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1 / (1 + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta2, t)) /
                     (1 - K.pow(self.beta1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros((1,1)) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta1 * m) + (1. - self.beta1) * g
            v_t = (self.beta2 * v) + (1. - self.beta2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta1': float(K.get_value(self.beta1)),
                  'beta2': float(K.get_value(self.beta2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(Adam, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))