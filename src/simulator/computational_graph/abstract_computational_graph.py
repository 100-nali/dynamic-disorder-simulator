"""
Base class for computational graphs for simulators.

Copyright © 2022 QuantrolOx Ltd
"""

import os
from copy import deepcopy
from functools import partial
from graphlib import CycleError
from typing import Dict, List, Set, Union

import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore
import numpy as np

from simulator.computational_graph import graph_nodes
from simulator.computational_graph.graph_config import GraphConfig
from simulator.computational_graph.graph_nodes.abstract_graph_node import AbstractGraphNode
from simulator.computational_graph.graph_nodes.node_configs import NodeConfigsGroup
from simulator.computational_graph.graph_nodes.nodes.initial_electrostatic_potential import (
    InitialElectroStaticPotential,
)


# pylint: disable=too-many-locals
class ComputationalGraph:
    """
    Class for computational graphs for simulators
    Specific computational workflows will inherit from this class
    """

    def __init__(self, graph_config: GraphConfig) -> None:
        """
        Iniitialise graph with components.

        Args:
            components: list of AbstractGraphNode objects comprising the computational graph
        """

        self.name = graph_config.name
        self.components = [self._instantiate_node_from_config(c) for c in graph_config.node_configs]
        self.data_dependencies: Dict = {}
        self.device_config = graph_config.device_config

        self._process_components()
        self._construct_data_dependencies()

        self.num_gates: int = 0
        for c in self.components:
            if isinstance(c, InitialElectroStaticPotential):
                self.num_gates = len(c.gate_split)
                break

    def _instantiate_node_from_config(self, node_config: NodeConfigsGroup) -> AbstractGraphNode:
        """
        Instantiate graph with components determined by node_configs listed in graph_config.

        Args:
            node_config: Node-specific configuration file, subclass of BaseNodeConfig

        Returns the instantiated graph node.
        """

        try:
            node_class = getattr(graph_nodes, node_config.node_class)

        except Exception as e:
            raise KeyError(
                f"Unable to find Node of type: {node_config.node_class}. Make sure that it is "
                "available in file computational_graph/graph_nodes/__init__.py"
            ) from e
        return node_class(node_config)

    def _process_components(self) -> None:
        """
        Adds each component to the graph.

        For each component, the output data node is added as an attribute to the instance.

        The name of the attribute is "_" + component.output_data_node, and the method used
        to compute it is the component call. The arguments of the call are either provided
        by the user, or computed by an upstream component.
        """
        for component in self.components:
            for idx, suboutput in enumerate(component.output_data_names):
                setattr(
                    self,
                    "_" + suboutput,
                    partial(self._value_getter, component=component, idx=idx),
                )
                self.data_dependencies[suboutput] = component.input_data_names.copy()

    def _construct_data_dependencies(self) -> None:
        """
        Populates the self.data_dependencies dict.
        """
        for component in self.components:
            input_data = component.input_data_names
            output_data = component.output_data_names

            for data_node, dependecies in self.data_dependencies.items():
                if data_node in input_data:
                    for suboutput in output_data:
                        self.data_dependencies[suboutput].extend(dependecies)

                        if suboutput in self.data_dependencies[suboutput]:
                            raise CycleError("Graph is cyclic")

    def _value_getter(
        self,
        set_value_dict: Dict[str, np.ndarray],
        component: AbstractGraphNode,
        idx: int = 0,
    ) -> np.ndarray:
        """
        Returns the value returned by a component.
        If inputs to the component are supplied by the set_value_dict, it uses
        those as input to the component, otherwise it will recursively compute them.

        Args:
            set_value_dict: Dictionary supplying possible inputs to the component
            component: a component of the graph
            idx: index of the value required within the list of outputs

        Returns:
            the output data node of the component
        """
        input_values = {}

        target_variable = component.output_data_names[idx]

        # Variable has already been computed
        if target_variable in set_value_dict.keys():
            return set_value_dict[target_variable]

        for k in component.input_data_names:
            if k in set_value_dict.keys():
                input_values[k] = set_value_dict[k]
            else:
                try:
                    input_values[k] = getattr(self, "_" + k)(set_value_dict)
                except AttributeError as e:
                    raise KeyError(f"Missing value for {k} in run call.") from e

        value_list = component(**input_values)

        for suboutput, value in zip(component.output_data_names, value_list):
            set_value_dict[suboutput] = value

        return value_list[idx]

    @property
    def _component_inputs(self) -> set[str]:
        input_names: Set = set()
        for component in self.components:
            input_names.update(component.input_data_names)
        return input_names

    @property
    def _component_outputs(self) -> set[str]:
        output_names: Set = set()
        for component in self.components:
            output_names.update(component.output_data_names)
        return output_names

    @property
    def graph_inputs(self) -> List[str]:
        """
        Returns a list of variable names that cannot be computed by upstream components.
        """
        return [x for x in self._component_inputs if x not in self._component_outputs]

    @property
    def graph_outputs(self) -> List[str]:
        """
        Returns a list of variable names that are not used by downstream components.
        """
        return [x for x in self._component_outputs if x not in self._component_inputs]

    @property
    def all_data_names(self) -> List[str]:
        """
        Returns a list of all named data variables in the graph.
        """
        return self.graph_inputs + list(self._component_outputs)

    def run(
        self, target: Union[str, List[str]], set_value_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Computes the target variable(s) using the set_value_dict items as dependency values.

        Args:
            string or list of strings of data names to compute
            set_value_dict: dictionary of any dependencies, typically gate voltages

        Returns:
            dict with k: v = data name: value for k in targets (list) or k = target (str)
        """

        # Make a copy of the dictionary to avoid modifying the dict outside of this scope.
        value_dict_copy = deepcopy(set_value_dict)

        if isinstance(target, str):
            target = [target]

        output_dict = {}
        for t in target:
            if t in value_dict_copy.keys():
                output_dict[t] = value_dict_copy[t]
            else:
                output_dict[t] = getattr(self, "_" + t)(value_dict_copy)

        return output_dict

    def run_component_outputs(self, set_value_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Computes all component outputs using values in set_value_dict as dependencies.

        Args:
            set_value_dict: the inputs to the graph execution

        Returns:
            a dict of k, v = output_data_name, value for all component outputs

        """
        return self.run(target=list(self._component_outputs), set_value_dict=set_value_dict)

    def run_graph_outputs(
        self,
        set_value_dict: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        Computes all graph output nodes using values in set_value_dict as dependencies.
        Graph outputs are all data values which are not used by downstream components.

        Args:
            set_value_dict: the inputs to the graph execution (typically gate_voltages)

        Returns:
            a dict of k, v = output_data_name, value for all graph outputs
        """
        return self.run(target=list(self.graph_outputs), set_value_dict=set_value_dict)

    def run_all_variables(self, set_value_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Computes and returns all data values in the graph.

        Args:
            set_value_dict: the inputs to the graph execution. Must include all graph inputs.

        Returns:
            a dict of k, v = output_data_name, value for all graph inputs and outputs.
        """
        return self.run(self.all_data_names, set_value_dict)

    def set_num_cpu(self, num_cpu: int) -> None:
        """sets the number of cpus used by each node to num_cpu

        Args:
            num_cpu (int): number of cpus to use
        """
        for c in self.components:
            c.set_num_cpu(num_cpu=num_cpu)

    def set_max_cpu(self, buffer: int = 0, raise_error=True) -> None:
        """Sets the number of cpus to the maximum value.

        Args:
            buffer (int): If non-zero, reduce the number of cpus by this value.
        """
        cpu_count = os.cpu_count()
        if isinstance(cpu_count, int):
            if buffer < 0 or (cpu_count - buffer) < 0:
                raise ValueError(f"Buffer value {buffer} should be between 0 and {cpu_count}")
            self.set_num_cpu(cpu_count - buffer)
        elif raise_error:
            raise ValueError("Could not evaluate os.cpu_count(), try setting cpu_count explicitly")
        else:
            print("Could not establish number of cpus, num_cpu unchanged.")

    def save_graph(self, fname: str = "graph.png") -> None:
        """
        Plots a representation of the graph.
        TODO: make this look cleaner.
        """
        graph_plot = nx.DiGraph()

        # Add component nodes
        for component in self.components:
            graph_plot.add_node(component.name)

        # Connect components if data names match
        for component in self.components:
            for input_vector_name in component.input_data_names:
                unplotted = True
                for other_component in self.components:
                    if (
                        other_component != component
                        and input_vector_name in other_component.output_data_names
                    ):
                        graph_plot.add_edge(
                            other_component.name, component.name, label=input_vector_name
                        )
                        unplotted = False
                if unplotted:
                    new_name = f"input_{input_vector_name}"
                    graph_plot.add_node(new_name)
                    graph_plot.add_edge(new_name, component.name, label=input_vector_name)

            for output_vector_name in component.output_data_names:
                unplotted = True
                for other_component in self.components:
                    if (
                        other_component != component
                        and output_vector_name in other_component.input_data_names
                    ):
                        unplotted = False
                        break

                if unplotted:
                    new_name = f"output_{output_vector_name}"
                    graph_plot.add_node(new_name)
                    graph_plot.add_edge(component.name, new_name, label=output_vector_name)

        # Draw the graph
        pos = nx.planar_layout(graph_plot)
        nx.draw_networkx_nodes(graph_plot, pos, node_color="lightblue", node_size=100)
        nx.draw_networkx_edges(graph_plot, pos, edge_color="gray")
        nx.draw_networkx_labels(graph_plot, pos, font_size=5)
        nx.draw_networkx_edge_labels(
            graph_plot, pos, edge_labels=nx.get_edge_attributes(graph_plot, "label"), font_size=5
        )
        # plt.show()
        plt.savefig(fname)
