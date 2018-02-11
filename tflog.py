import tensorflow as tf
from io import StringIO
from scipy.misc import imsave
import numpy as np


def copy_documentation(function_to_use):
    def decorator(f):
        f.__doc__ = function_to_use.__doc__
        return f

    return decorator


def log_scalar(writer, tag, value, step):
    """Log a scalar variable.
    Parameter
    ----------
    tag : basestring
        Name of the scalar
    value
    step : int
        training iteration
    """
    summary = tf.Summary(value=[tf.Summary.Value(
        tag=tag,
        simple_value=value
    )])
    writer.add_summary(summary, step)


def log_images(writer, tag, images, step):
    """Logs a list of images."""

    im_summaries = []
    for nr, img in enumerate(images):
        # Write the image to a string
        s = StringIO()
        imsave(s, img, format='png')

        # Create an Image object
        img_sum = tf.Summary.Image(
            encoded_image_string=s.getvalue(),
            height=img.shape[0],
            width=img.shape[1]
        )
        # Create a Summary value
        im_summaries.append(
            tf.Summary.Value(tag='%s/%d' % (tag, nr), image=img_sum)
        )

    # Create and write Summary
    summary = tf.Summary(value=im_summaries)
    writer.add_summary(summary, step)


def log_histogram(writer, tag, values, step, bins=1000, min=None, max=None, density=False):
    """Logs the histogram of a list/vector of values."""
    values = np.array(values)
    # Fill fields of histogram proto
    hist = tf.HistogramProto()
    hist.min = float(np.min(values)) if min is None else float(min)
    hist.max = float(np.max(values)) if min is None else float(max)

    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))

    # Create histogram using numpy
    counts, bin_edges = np.histogram(values, bins=bins, range=(hist.min, hist.max), density=density)

    # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
    # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
    # Thus, we drop the start of the first bin
    bin_edges = bin_edges[1:]

    # Add bin edges and counts
    for edge in bin_edges:
        hist.bucket_limit.append(edge)
    for c in counts:
        hist.bucket.append(c)

    # Create and write Summary
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
    writer.add_summary(summary, step)
    writer.flush()


class TFEventsLogger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir=None, writer=None):
        if writer is None:
            assert log_dir is not None, 'Must provide a logdir or a summary writer'
            self.writer = tf.summary.FileWriter(log_dir)
        else:
            self.writer = writer

    @copy_documentation(log_scalar)
    def log_scalar(self, tag, value, step):
        __doc__ = log_scalar.__doc__  # NOQA
        log_scalar(self.writer, tag, value, step)

    @copy_documentation(log_images)
    def log_images(self, tag, images, step):
        __doc__ = log_images.__doc__  # NOQA
        log_images(self.writer, tag, images, step)

    @copy_documentation(log_histogram)
    def log_histogram(self, tag, values, step, bins=1000, min=None, max=None, density=False):
        __doc__ = log_histogram.__doc__  # NOQA
        log_histogram(self.writer, tag, values, step, bins, min, max, density)
