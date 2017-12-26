import tensorflow as tf


GRIDS = {25: [5, 5], 50: [5, 10], 100: [10, 10]}


def img_grid(imgs, rows, cols):
    """Pack image-like tensors into a grid for nicer TensorBoard visualization.
    
    Parameters:
        imgs: List of tensors batch x height x width x channels.
                E.g. if you have 32 filter maps batch x height x width x 32
                you should pack those into a list of 32 elements 
                batch x height x width and re-add a "fake" channel axis of size
                1 at the end.
                For 32 filters of shape 32 x height x width x n_channels, again
                pack them into a list but add a fake batch axis in front.
        rows: Duh.
        cols: DUh.

    Returns:
         4d tensor you can put into tf.summary.image (assuming the number of
         channels is okay).
    """
    if len(imgs) != rows * cols:
        raise ValueError("Grid doesn't match the number of images!")

    # make white border things
    max_val = tf.reduce_max([tf.reduce_max(img) for img in imgs])
    col_border = tf.fill([tf.shape(imgs[0])[0], tf.shape(imgs[0])[1], 1,
                          int(imgs[0].shape[3])], max_val)

    # first create the rows
    def make_row(ind):
        base = imgs[ind:(ind + cols)]
        _borders = [col_border] * len(base)
        _interleaved = [elem for pair in zip(base, _borders) for
                        elem in pair][:-1]  # remove last border
        return _interleaved

    grid_rows = [tf.concat(make_row(ind), axis=2) for
                 ind in range(0, len(imgs), cols)]

    # then stack them
    row_border = tf.fill([tf.shape(imgs[0])[0], 1, tf.shape(grid_rows[0])[2],
                          int(imgs[0].shape[3])], max_val)
    borders = [row_border] * len(grid_rows)
    interleaved = [elem for pair in zip(grid_rows, borders) for
                   elem in pair][:-1]
    grid = tf.concat(interleaved, axis=1)
    return grid
