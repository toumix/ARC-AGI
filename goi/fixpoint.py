"""Family 3: one neighbourhood cell iterated over the grid, to a fixpoint.

The wiring is the grid: one cell per pixel, reading its Moore
neighbourhood, the same cell at every pixel and every round -- a
cellular automaton whose rule is learned. A round is one pass of the
cell over the grid; `rounds = 1` is the one-pass neighbourhood family of
discopy#703, and recolour is its `rounds = 1` at radius zero. What the
survey showed a one-pass table cannot do -- let a colour travel -- the
rounds do.

The cell keeps the quotiented interface of the CLRS study (clrs#1's
`key1 > key2`): beside the colours it reads, for every port, whether
it equals the centre, is background or lies past the border, and it
*writes* relatively -- keep the centre, take a port's colour, or an
absolute colour -- so a rule such as "take the colour of the neighbour
that is not background" is one row whatever the colours are. The readout
is the colour distribution itself, no learned decoder, so a table read
back off the cell cannot absorb a permutation of colours (lesson 5).

`ports = 'moore'` is the eight neighbours. `ports = 'all'` adds what the
verifiers said the neighbourhood lacks (`families.py`): four *mirror*
ports, the pixel across the grid's horizontal, vertical, central and
diagonal axis, and four *rays*, the first non-background colour seen
along each row and column direction, or none. A mirror is one wire per
pixel; a ray is a scan along the row, so a line reaches its end in one
round instead of thirty.

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
MIRRORS = ('hmirror', 'vmirror', 'rot180', 'transpose')
RAYS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PORTS = {'moore': len(OFFSETS),
         'all': len(OFFSETS) + len(MIRRORS) + len(RAYS)}


def features_count(ports):
    return 2 * COLOURS + PORTS[ports] * (COLOURS + 3)


def codes_count(ports):
    return 1 + PORTS[ports] + COLOURS


def mirrors(height, width):
    """For every pixel, the pixel each mirror port reads, and whether it
    exists: the transpose of a rectangle is not the same grid."""
    i, j = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing='ij')
    inside = (i < height) & (j < width)
    flips = [(height - 1 - i, j), (i, width - 1 - j),
             (height - 1 - i, width - 1 - j), (j, i)]
    index = np.zeros((len(MIRRORS), SIZE, SIZE, 2), np.int32)
    ok = np.zeros((len(MIRRORS), SIZE, SIZE), np.float32)
    for m, (a, b) in enumerate(flips):
        ok[m] = inside & (m < 3 or height == width)
        index[m, ..., 0] = np.where(ok[m], a, i)
        index[m, ..., 1] = np.where(ok[m], b, j)
    return index, ok


def arrays(inputs, targets, shapes, weight):
    """The canvas: grids `[P, SIZE, SIZE]` with their shapes and weights
    as the arrays a fit reads, one-hot inputs, masks and mirror indices."""
    inputs, targets = np.asarray(inputs), np.asarray(targets)
    slots = len(inputs)
    i, j = np.meshgrid(np.arange(SIZE), np.arange(SIZE), indexing='ij')
    valid = ((i[None] < shapes[:, :1, None]) & (j[None] < shapes[:, 1:, None]))
    index = np.zeros((slots, len(MIRRORS), SIZE, SIZE, 2), np.int32)
    ok = np.zeros((slots, len(MIRRORS), SIZE, SIZE), np.float32)
    for k, (height, width) in enumerate(shapes):
        index[k], ok[k] = mirrors(height, width)
    return {'x0': jnp.array(np.eye(COLOURS, dtype=np.float32)[inputs]
                            * valid[..., None]),
            'target': jnp.array(targets.astype(np.int32)),
            'valid': jnp.array(valid.astype(np.float32)),
            'weight': jnp.array(np.asarray(weight, np.float32)),
            'index': jnp.array(index), 'inside': jnp.array(ok)}


def pad(pairs, weights=None, slots=None):
    """Pairs as arrays on one canvas, `slots` of them, padded by empty ones."""
    slots = slots or len(pairs)
    inputs = np.zeros((slots, SIZE, SIZE), np.int8)
    targets = np.zeros((slots, SIZE, SIZE), np.int8)
    shapes = np.zeros((slots, 2), np.int32)
    weight = np.zeros(slots, np.float32)
    for k, pair in enumerate(pairs):
        grid = np.asarray(pair['input'])
        height, width = shapes[k] = grid.shape
        inputs[k, :height, :width] = grid
        weight[k] = 1 if weights is None else weights[k]
        if 'output' in pair:
            targets[k, :height, :width] = np.asarray(pair['output'])
    return arrays(inputs, targets, shapes, weight)


def shift(array, dy, dx):
    """The array seen from the neighbour at offset (dy, dx), zero past it."""
    mask = edge(array.shape[1], dy)[:, None] * edge(array.shape[2], dx)[None, :]
    return jnp.roll(array, (-dy, -dx), axis=(1, 2)) \
        * mask.reshape((1, *mask.shape, *(1,) * (array.ndim - 3)))


def edge(size, delta):
    """Which positions have a neighbour at `delta` inside the canvas."""
    index = jnp.arange(size) + delta
    return ((index >= 0) & (index < size)).astype(jnp.float32)


def gather(state, index):
    """The state at another pixel of the same grid, per pixel."""
    pairs = jnp.arange(state.shape[0])[:, None, None]
    return state[pairs, index[..., 0], index[..., 1]]


def ray(state, valid, dy, dx):
    """What a pixel sees looking along (dy, dx): the first non-background
    colour, or none -- a scan along the axis, the wall past the grid
    stopping it. Soft when the state is: each cell lets the ray through
    with its background mass and stops it with the rest."""
    axis = 1 if dy else 2
    sequence = jnp.moveaxis(state * valid[..., None], axis, 0)

    def body(carry, cell):
        hit, free = carry
        return (cell[..., 1:] + cell[..., :1] * hit, cell[..., 0] * free), carry

    zero = (jnp.zeros(sequence.shape[1:-1] + (COLOURS - 1,)),
            jnp.ones(sequence.shape[1:-1]))
    _, (hit, _) = jax.lax.scan(body, zero, sequence, reverse=(dy + dx) > 0)
    hit = jnp.moveaxis(hit, 0, axis)
    none = 1 - jnp.sum(hit, -1, keepdims=True)
    return jnp.concatenate([none, hit], -1), 1 - none[..., 0]


def reads(state, valid, data, ports):
    """Every port's colour distribution and whether it reads anything."""
    for dy, dx in OFFSETS:
        yield shift(state, dy, dx), shift(valid, dy, dx)
    if ports == 'all':
        for m in range(len(MIRRORS)):
            yield gather(state, data['index'][:, m]), data['inside'][:, m]
        for dy, dx in RAYS:
            yield ray(state, valid, dy, dx)


def features(state, data, ports):
    """What the cell reads at every pixel, and what each port offers."""
    parts, taken = [state, data['x0']], []
    for colours, inside in reads(state, data['valid'], data, ports):
        equal = jnp.sum(colours * state, axis=-1, keepdims=True)
        parts += [colours, 1 - inside[..., None], equal, colours[..., :1]]
        taken.append(inside[..., None] * colours
                     + (1 - inside[..., None]) * state)
    return jnp.concatenate(parts, axis=-1), taken


def init(key, hidden, ports='moore'):
    key1, key2 = jax.random.split(key)
    return {'w1': jax.random.normal(key1, (features_count(ports), hidden)) * 0.1,
            'b1': jnp.zeros(hidden),
            'w2': jax.random.normal(key2, (hidden, codes_count(ports))) * 0.1,
            'b2': jnp.zeros(codes_count(ports))}


def cell(params, inputs):
    hidden = jax.nn.relu(inputs @ params['w1'] + params['b1'])
    return hidden @ params['w2'] + params['b2']


def step(params, state, data, hard, ports):
    """One round: every pixel rewritten by the cell's choice."""
    inputs, taken = features(state, data, ports)
    choice = jax.nn.softmax(cell(params, inputs))
    new = choice[..., :1] * state
    for k, colours in enumerate(taken):
        new = new + choice[..., 1 + k:2 + k] * colours
    new = new + choice[..., 1 + len(taken):]
    new = new * data['valid'][..., None]
    if hard:
        new = jax.nn.one_hot(jnp.argmax(new, -1), COLOURS) \
            * data['valid'][..., None]
    return new


@functools.partial(jax.jit, static_argnames=('rounds', 'hard', 'ports'))
def run(params, data, rounds, hard, ports='moore'):
    """The automaton from the input: the state after every round."""
    @jax.checkpoint
    def body(state, _):
        state = step(params, state, data, hard, ports)
        return state, state
    _, states = jax.lax.scan(body, data['x0'], None, length=rounds)
    return states


def loss(params, data, rounds, ports='moore'):
    """Deep supervision: the target is a fixpoint, so every round of the
    second half is asked to show it, not the last one alone."""
    states = run(params, data, rounds, False, ports)[rounds // 2:]
    logp = jnp.log(jnp.take_along_axis(
        states, data['target'][None, ..., None], -1)[..., 0] + 1e-6)
    mask = (data['valid'] * data['weight'][:, None, None])[None]
    return -jnp.sum(logp * mask) / jnp.maximum(jnp.sum(mask), 1) / len(states)


def adam(params, opt, grads, count):
    """One Adam step, the learning rate decaying by count, elementwise so
    that it applies to one cell or to a batch of them alike."""
    rate = 3e-3 * jnp.minimum(1., 10. / (count + 1)) ** 0.5
    new_params, new_opt = {}, {}
    for key in params:
        first = 0.9 * opt[key][0] + 0.1 * grads[key]
        second = 0.999 * opt[key][1] + 0.001 * grads[key] ** 2
        new_opt[key] = (first, second)
        new_params[key] = params[key] \
            - rate * first / (jnp.sqrt(second) + 1e-8)
    return new_params, new_opt


@functools.partial(jax.jit, static_argnames=('rounds', 'ports'))
def update(params, opt, data, count, rounds, ports='moore'):
    """One Adam step on the demos."""
    return adam(params, opt, jax.grad(loss)(params, data, rounds, ports), count)


def fit(pairs, rounds, seed, steps=600, hidden=64, tail=100, start=None,
        ports='moore'):
    """Train the cell on the pairs, Polyak-averaging the last iterates,
    from `start` when given -- the cell fitted at fewer rounds, so that
    the rounds are a curriculum rather than a restart."""
    data = pad(pairs)
    params = start or init(jax.random.PRNGKey(seed), hidden, ports)
    opt = {key: (jnp.zeros_like(value), jnp.zeros_like(value))
           for key, value in params.items()}
    mean = None
    for count in range(steps):
        params, opt = update(params, opt, data, count, rounds, ports)
        if count >= steps - tail:
            mean = params if mean is None \
                else {key: mean[key] + params[key] for key in params}
    return {key: value / tail for key, value in mean.items()}


def predict(params, pairs, rounds, ports='moore'):
    """The hard automaton's output grids, cropped to each input's shape."""
    data = pad(pairs)
    state = np.asarray(run(params, data, rounds, True, ports)[-1])
    grids = []
    for k, pair in enumerate(pairs):
        height, width = np.asarray(pair['input']).shape
        grids.append(np.argmax(state[k, :height, :width], -1).tolist())
    return grids


def exact(params, pairs, rounds, ports='moore'):
    """How many of the pairs the hard automaton reproduces exactly."""
    return sum(grid == pair['output']
               for grid, pair in zip(predict(params, pairs, rounds, ports), pairs))
