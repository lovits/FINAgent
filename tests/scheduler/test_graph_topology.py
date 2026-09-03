from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup
from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.policy import CallableSchedulerPolicy


class _DummyLLM:
    def with_structured_output(self, *args, **kwargs):
        return self

    def bind_tools(self, *args, **kwargs):
        return self


def _setup() -> GraphSetup:
    tool_nodes = {
        key: (lambda state: {})
        for key in ("market", "social", "news", "fundamentals")
    }
    return GraphSetup(_DummyLLM(), _DummyLLM(), tool_nodes, ConditionalLogic())


def _edge_pairs(workflow) -> set[tuple[str, str]]:
    graph = workflow.compile().get_graph()
    return {(edge.source, edge.target) for edge in graph.edges}


def test_static_graph_keeps_original_entry_and_has_no_scheduler() -> None:
    workflow = _setup().setup_graph(("market", "news"))
    graph = workflow.compile().get_graph()
    edges = {(edge.source, edge.target) for edge in graph.edges}

    assert "Scheduler" not in graph.nodes
    assert ("__start__", "Market Analyst") in edges
    assert ("Portfolio Manager", "__end__") in edges


def test_dynamic_graph_routes_experts_back_to_scheduler() -> None:
    policy = CallableSchedulerPolicy(lambda _: SchedulerAction.MARKET)
    workflow = _setup().setup_scheduler_graph(("market", "news"), policy)
    graph = workflow.compile().get_graph()
    edges = _edge_pairs(workflow)

    assert ("__start__", "Scheduler") in edges
    assert ("Msg Clear Market", "Scheduler") in edges
    assert ("Msg Clear News", "Scheduler") in edges
    assert ("Bull Researcher", "Scheduler") in edges
    assert ("Portfolio Manager", "Scheduler") in edges
    assert ("Scheduler", "__end__") in edges
    assert "Sentiment Analyst" not in graph.nodes


def test_dynamic_analyst_tool_loop_stays_inside_expert() -> None:
    policy = CallableSchedulerPolicy(lambda _: SchedulerAction.MARKET)
    workflow = _setup().setup_scheduler_graph(("market",), policy)
    edges = _edge_pairs(workflow)

    assert ("Market Analyst", "tools_market") in edges
    assert ("tools_market", "Market Analyst") in edges
    assert ("tools_market", "Scheduler") not in edges
