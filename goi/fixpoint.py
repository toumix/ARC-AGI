"""Family 3: one neighbourhood cell iterated over the grid, to a fixpoint.

The wiring is the grid: one cell per pixel, reading its Moore
neighbourhood, the same cell at every pixel and every round -- a
cellular automaton whose rule is learned. A round is one pass of the
cell over the grid; `rounds = 1` is the one-pass neighbourhood family of
discopy#703, and recolour is its `rounds = 1` at radius zero. What the
survey showed a one-pass table cannot do -- let a colour travel -- the
rounds do.

The cell keeps the quotiented interface of the CLRS study (clrs#1's
`key1 > key2`): beside the colours it reads, for every neighbour, whether
it equals the centre, is background or lies past the border, and it
*writes* relatively -- keep the centre, take a neighbour's colour, or an
absolute colour -- so a rule such as "take the colour of the neighbour
that is not background" is one row whatever the colours are. The readout
is the colour distribution itself, no learned decoder, so a table read
back off the cell cannot absorb a permutation of colours (lesson 5).

Training is end to end on the demonstrations alone, the only rung ARC
has: the state is a distribution over colours per pixel, every round
mixes by the cell's own probabilities, and the loss is the cross-entropy
of the demo's output under the final distribution. At test the routing
is hard, the state one-hot after every round, the trained cell dropped
into the exact automaton unchanged.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

COLOURS = 10
SIZE = 30
OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
           if (dy, dx) != (0, 0)]
FEATURES = 2 * COLOURS + len(OFFSETS) * (COLOURS + 3)
CODES = 1 + len(OFFSETS) + COLOURS


def pad(pairs):
    """Demos as arrays on one canvas: input one-hots, targets, masks."""
    x0 = np.zeros((len(pairs), SIZE, SIZE, COLOURS), np.float32)
    target = np.zeros((len(pairs), SIZE, SIZE), np.int32)
    valid = np.zeros((len(pairs), SIZE, SIZE), np.float32)
    weight = np.zeros(len(pairs), np.float32)
    for k, pair in enumerate(pairs):
        grid = np.asarray(pair['input'])
        height, width = grid.shape
        x0[k, :height, :width] = np.eye(COLOURS)[grid]
        valid[k, :height, :width] = 1
        weight[k] = 1
        if 'output' in pair:
            target[k, :height, :width] = np.asarray(pair['output'])
    return jnp.array(x0), jnp.array(target), jnp.array(valid), jnp.array(weight)


def shift(array, dy, dx):
    """The array seen from the neighbour at offset (dy, dx), zero past it."""
    mask = edge(array.shape[1], dy)[:, None] * edge(array.shape[2], dx)[None, :]
    return jnp.roll(array, (-dy, -dx), axis=(1, 2)) \
        * mask.reshape((1, *mask.shape, *(1,) * (array.ndim - 3)))


def edge(size, delta):
    """Which positions have a neighbour at `delta` inside the canvas."""
    index = jnp.arange(size) + delta
    return ((index >= 0) & (index < size)).astype(jnp.float32)


def features(state, x0, valid):
    """What the cell reads at every pixel: the quotiented interface."""
    parts, neighbours = [state, x0], []
    for dy, dx in OFFSETS:
        colours = shift(state, dy, dx)
        inside = shift(valid, dy, dx)
        equal = jnp.sum(colours * state, axis=-1, keepdims=True)
        parts += [colours, 1 - inside[..., None], equal,
                  colours[..., :1]]
        neighbours.append((colours, inside))
    return jnp.concatenate(parts, axis=-1), neighbours


def init(key, hidden):
    key1, key2 = jax.random.split(key)
    return {'w1': jax.random.normal(key1, (FEATURES, hidden)) * 0.1,
            'b1': jnp.zeros(hidden),
            'w2': jax.random.normal(key2, (hidden, CODES)) * 0.1,
            'b2': jnp.zeros(CODES)}


def cell(params, inputs):
    hidden = jax.nn.relu(inputs @ params['w1'] + params['b1'])
    return hidden @ params['w2'] + params['b2']


def step(params, state, x0, valid, hard):
    """One round: every pixel rewritten by the cell's choice."""
    inputs, neighbours = features(state, x0, valid)
    choice = jax.nn.softmax(cell(params, inputs))
    new = choice[..., :1] * state
    for k, (colours, inside) in enumerate(neighbours):
        taken = inside[..., None] * colours + (1 - inside[..., None]) * state
        new = new + choice[..., 1 + k:2 + k] * taken
    new = new + choice[..., 1 + len(OFFSETS):]
    new = new * valid[..., None]
    if hard:
        new = jax.nn.one_hot(jnp.argmax(new, -1), COLOURS) * valid[..., None]
    return new


@functools.partial(jax.jit, static_argnames=('rounds', 'hard'))
def run(params, x0, valid, rounds, hard):
    """The automaton from the input, `rounds` rounds, soft or hard."""
    def body(state, _):
        return step(params, state, x0, valid, hard), None
    state, _ = jax.lax.scan(body, x0, None, length=rounds)
    return state


def loss(params, x0, target, valid, weight, rounds):
    state = run(params, x0, valid, rounds, False)
    logp = jnp.log(jnp.take_along_axis(state, target[..., None], -1)[..., 0]
                   + 1e-6)
    mask = valid * weight[:, None, None]
    return -jnp.sum(logp * mask) / jnp.sum(mask)


@functools.partial(jax.jit, static_argnames=('rounds',))
def update(params, opt, data, count, rounds):
    """One Adam step on the demos, the learning rate decaying by count."""
    grads = jax.grad(loss)(params, *data, rounds)
    rate = 3e-3 * jnp.minimum(1., 10. / (count + 1)) ** 0.5
    new_params, new_opt = {}, {}
    for key in params:
        first = 0.9 * opt[key][0] + 0.1 * grads[key]
        second = 0.999 * opt[key][1] + 0.001 * grads[key] ** 2
        new_opt[key] = (first, second)
        new_params[key] = params[key] \
            - rate * first / (jnp.sqrt(second) + 1e-8)
    return new_params, new_opt


def fit(pairs, rounds, seed, steps=300, hidden=64, tail=50):
    """Train the cell on the pairs, Polyak-averaging the last iterates."""
    data = pad(pairs)
    params = init(jax.random.PRNGKey(seed), hidden)
    opt = {key: (jnp.zeros_like(value), jnp.zeros_like(value))
           for key, value in params.items()}
    mean = None
    for count in range(steps):
        params, opt = update(params, opt, data, count, rounds)
        if count >= steps - tail:
            mean = params if mean is None \
                else {key: mean[key] + params[key] for key in params}
    return {key: value / tail for key, value in mean.items()}


def predict(params, pairs, rounds):
    """The hard automaton's output grids, cropped to each input's shape."""
    x0, _, valid, _ = pad(pairs)
    state = np.asarray(run(params, x0, valid, rounds, True))
    grids = []
    for k, pair in enumerate(pairs):
        height, width = np.asarray(pair['input']).shape
        grids.append(np.argmax(state[k, :height, :width], -1).tolist())
    return grids


def exact(params, pairs, rounds):
    """How many of the pairs the hard automaton reproduces exactly."""
    return sum(grid == pair['output']
               for grid, pair in zip(predict(params, pairs, rounds), pairs))
