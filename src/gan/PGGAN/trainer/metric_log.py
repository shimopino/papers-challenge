# https://github.com/kwotsin/mimicry/blob/22497cb3738214b212cd2c2a7b0867e7836b1f82/torch_mimicry/training/metric_log.py


class MetricLog:
    """
    A dictionary-like object for storing logs, and include an extra dict to map the metrics
    to its group name, if any, and the corresponding precision to print out
    """

    def __init__(self, **kwargs):
        self.metrics_dict = {}

    def add_metric(self, name, value, group=None, precision=4):
        """
        Logs metric to internal dict, but with an additional option
        of grouping certain metrics together.

        Arguments:
            name {str} -- name of metric
            value {Tensor/Float} -- value of metric

        Keyword Arguments:
            group {str} -- group to classify different metrics together (default: {None})
            precision {int} -- floating point precision to represent the value (default: {4})
        """
        try:
            value = value.items()
        except AttributeError:
            value = value

        self.metrics_dict[name] = dict(value=value, group=group, precision=precision)

    def __getitem__(self, key):
        return round(
            self.metrics_dict[key]["value"], self.metrics_dict[key]["precision"]
        )

    def get_group_name(self, name):
        """
        Obtains the group name of a particular metric. For example, errD and errG
        which represents the discriminator/generator losses could fall under a
        group name called "loss".

        Arguments:
            name (str): The name of the metric to retrieve group name.

        Returns:
            str: A string representing the group name of the metric.
        """
        return self.metrics_dict[name]["group"]

    def keys(self):
        """
        Dict like functionality for retrieving keys.
        """
        return self.metrics_dict.keys()

    def items(self):
        """
        Dict like functionality for retrieving items.
        """
        return self.metrics_dict.items()
