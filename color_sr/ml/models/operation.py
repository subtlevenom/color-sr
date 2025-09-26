from abc import ABC
from typing import Any
from concurrent.futures import Future, ThreadPoolExecutor
from queue import Queue
import graphtik as gk
from gazenet.io.logger import logger
from gazenet.models.pipeline.node import Node
from gazenet.models.pipeline.graph import Graph
from gazenet.models.pipeline.calculator import Calculator
from gazenet.models.pipeline.packet import Packet

log = logger(__name__)


def gk_operation(node: Node) -> gk.base.Operation:
    """Creates a graphkit node operation.

    Args:
        node (Node): the node description.

    Returns:
        gk.Operation: resulting graphkit operation.
    """
    return gk.operation(node.calculator(),
                        name=node.name,
                        needs=node.inputs,
                        provides=node.outputs)


def gk_compose(graph: Graph) -> gk.pipeline.Pipeline:
    """Makes a graphkit composition (computation graph)
    
    Internally we use the graphkit framework as an engine for
    the graph calculations. It allows to split the algorithm
    into the sequence of the computation units (calculators)
    and express it as a computation graph.

    Here we convert a graph description into the graphkit 
    composition operation (computation graph).
    We add a special attribute __calculators__ storing
    the list of graph calculators.

    Args:
        name (str): the computation graph name.
        nodes (list[Node]): node descriptions.

    Returns:
        gk.Operation: graphkit composition opertation.
    """
    if graph is None:
        return None

    operations = [gk_operation(node) for node in graph.nodes]
    return gk.compose(graph.name, *operations, outputs=graph.outputs)

def gk_compile(operation: gk.pipeline.Pipeline):
    """Compiles the graphtik pipeline and makes an execution plan

    The execturion plan contains the subgraph DAG after the
    unsatisfied operation pruning, i.e. data eviction. 
    For the execution plan details see
    https://graphtik.readthedocs.io/en/latest/pipelines.html

    Args:
        operation (gk.pipeline.Pipeline): the gk pipeline

    Returns:
        ExecutionPlan: the resulting execution plan
    """
    if operation is None:
        return None
    return operation.compile()

FAULT = Packet(_state=-1)
DONE = Packet(_state=1)


class Operation(ABC):
    """Graphkit composition operation wrapper.
    
    The graphkit composition operation (computation graph)
    acts like a function, accepting input arguments and 
    providing a dictionary of outputs (one entry for each node).
    Here we need to orgnize a continues communication with the sources
    and sinks in the pipeline.

    The operation takes inputs from the input queue and outputs into 
    the output queue. On a call the operation runs the computation graph
    in the infinite loop in a separate thread, waits for the inputs, 
    passes the input values to the computation graph and outputs the 
    calculation result into the output queue.
    The infinite loop breakes when the operation gets a None input
    value. As a final step the operations send None to the output
    to notify the underlying operations in the pipeline.
    The child classes: source, sink, worker, repeater provides their own
    specific and slightly differs in behaviour from above.

    The abstract class.
    """

    _input: Queue = None
    _output: Queue = None
    _operation: gk.base.Operation = None
    _executor: ThreadPoolExecutor

    def __init__(self,
                 graph: Graph,
                 input: Queue = None,
                 output: Queue = None) -> None:
        self._operation = gk_compose(graph)
        self._plan = gk_compile(self._operation) 
        self._input = input
        self._output = output
        self._executor = ThreadPoolExecutor(1)

    # queue methods

    def _get(self) -> Packet:
        return self._input.get()

    def _put(self, packet: Packet) -> None:
        self._output.put(packet)

    def _fault(self) -> None:
        self._put(FAULT)

    def _done(self) -> None:
        self._put(DONE)

    # calculator methods

    def _open(self) -> None:
        self._input.queue.clear()
        self._output.queue.clear()
        for op in self._plan.ops:
            op.fn.open()

    def _close(self) -> None:
        for op in self._plan.ops:
            op.fn.close()

    def _compute(self, packet: Packet) -> dict:
        return self._operation.compute(packet)

    # processing

    def _process(self) -> None:
        raise NotImplementedError()

    def __call__(self) -> Future:
        self._open()
        task = self._executor.submit(self._process)
        task.add_done_callback(lambda _: self._close())
        return task
