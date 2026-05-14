"""
Implements the abstract class for graph components for simulators.

Copyright © 2022 QuantrolOx Ltd
"""

import os
from multiprocessing import Pool
from typing import Dict, List

import numpy as np

from simulator.computational_graph.graph_nodes.node_configs import BaseNodeConfig


# pylint: disable-next=too-many-instance-attributes
class AbstractGraphNode:
    """
    Abstract class for components of the computational graph.
    """

    def __init__(self, node_config: BaseNodeConfig) -> None:
        """Instantiate the node with config parameters

        Args:
            node_config (BaseNodeConfig): configuration parameters of the node.
        """
        self.name = node_config.name
        self.input_data_names = node_config.connections.input_data_names
        self.output_data_names = node_config.connections.output_data_names
        self.input_data_dimensions = node_config.input_dimensions
        self.output_data_dimensions = node_config.output_dimensions

        self.supports_batching = node_config.supports_batching
        self.num_cpu = node_config.num_cpu
        self.config = node_config

        assert (
            len(self.output_data_names) > 0
        ), f"No output data nodes provided for node {self.name}"

        all_nodes = self.input_data_names + self.output_data_names

        if len(all_nodes) > len(set(all_nodes)):
            raise ValueError(f"Not all node names are unique in node: {self.name}")

    def __call__(self, **kwargs) -> List:
        if self.supports_batching or not self._is_batch(**kwargs):
            return self.compute(*[kwargs[name] for name in self.input_data_names])

        return self._handle_batch(**kwargs)

    def _is_batch(self, **kwargs) -> bool:
        """
        Identifies whether the provided data is batched.
        """
        assert isinstance(
            self.input_data_dimensions, List
        ), "If node doesn't support batching, dims must be provided."

        # get the difference in dimensions between provided arrays and expectations
        excess_dims = [
            len(kwargs[n].shape) - d
            for n, d in zip(self.input_data_names, self.input_data_dimensions)
            if isinstance(kwargs[n], np.ndarray) and not isinstance(kwargs[n][0], List)
        ]

        # Either the excess dims are all 0, or all 1. If 0, the data isn't batched.
        if not (len(set(excess_dims)) == 1 and excess_dims[0] in [0, 1]):
            raise ValueError("Badly batched data. Check dimensions.")

        # If there are excess dimensions, the data is batched.
        return excess_dims[0] > 0

    def _handle_batch(self, **kwargs) -> List:
        """
        Handles a batch of inputs.

        Args:
            **kwargs: dict of arrays to be processed. Each array has shape [batch_size, data_shape]

        Returns:
            List of processed arrays, each with shape [batch_size, data_shape]
        """
        # Check batch size consistency
        batch_sizes = [x.shape[0] for x in kwargs.values()]
        if len(set(batch_sizes)) > 1:
            raise ValueError("Batch sizes inconsistent.")
        batch_size = batch_sizes[0]

        # Generate batches from the input data
        batch_list = [{k: kwargs[k][z] for k in self.input_data_names} for z in range(batch_size)]

        result_lists = self._compile_results(batch_list)

        # Stack the results
        num_returns = len(result_lists[0])

        try:
            return [np.stack([sublist[i] for sublist in result_lists]) for i in range(num_returns)]

        except ValueError as exc:
            if "all input arrays must have the same shape" not in exc.args:
                raise exc
            return [
                np.array([sublist[i] for sublist in result_lists], dtype=object)
                for i in range(num_returns)
            ]

    def _compile_results(self, batch_list: List[Dict]) -> List[List]:
        """Distributes executions of the compute function and compiles the results

        Args:
            batch_list (List[Dict]): List of dictionaries to be fed to the compute function

        Returns:
            List[List]: compiled list of result lists
        """

        # If not using multiple cpus, compute sequentially
        if self.num_cpu == 1:
            return [self.compute(*[b[n] for n in self.input_data_names]) for b in batch_list]

        num_cpus_to_use = min(len(batch_list), self.num_cpu)
        process_args = [(self.input_data_names, b) for b in batch_list]
        with Pool(processes=num_cpus_to_use) as pool:
            results = pool.starmap(self._process_single, process_args)

            return results

    def compute(self) -> List:
        raise NotImplementedError("Nodes must implement compute function.")

    def _process_single(self, input_keys: List[str], batch: Dict) -> List:
        """Wrapper for parallelised execution of the compute function

        Args:
            input_keys (_type_): _description_
            batch (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Prepare arguments for my_function
        args = [batch[n] for n in input_keys]
        # Call the instance method my_function
        return self.compute(*args)

    def set_num_cpu(self, num_cpu: int) -> None:
        """sets the number of cpus to use for evaluation

        Args:
            num_cpu (int): number of cpus to use
        """
        cpu_count = os.cpu_count()

        if num_cpu < 0:
            raise ValueError("Must have positive num_cpu.")

        if isinstance(cpu_count, int) and num_cpu > cpu_count:
            raise ValueError(f"Num_cpu must be less than cpu_count: {cpu_count}")

        self.num_cpu = num_cpu
