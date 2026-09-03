from tradingagents.graph import setup as graph_setup_module
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.policy import CallableSchedulerPolicy


def _node_factory(*args, **kwargs):
    return lambda state: {}


def _graph_setup(monkeypatch):
    factories = (
        "create_aggressive_debator",
        "create_bear_researcher",
        "create_bull_researcher",
        "create_conservative_debator",
        "create_fundamentals_analyst",
        "create_market_analyst",
        "create_msg_delete",
        "create_neutral_debator",
        "create_news_analyst",
        "create_portfolio_manager",
        "create_research_manager",
        "create_sentiment_analyst",
        "create_trader",
    )
    for name in factories:
        monkeypatch.setattr(graph_setup_module, name, _node_factory)
    tools = {key: _node_factory() for key in ("market", "social", "news", "fundamentals")}
    return graph_setup_module.GraphSetup(None, None, tools, ConditionalLogic())


def _edges(workflow):
    return {
        (edge.source, edge.target, edge.conditional)
        for edge in workflow.compile().get_graph().edges
    }


def test_static_mode_is_the_original_graph(monkeypatch):
    setup = _graph_setup(monkeypatch)
    selected = ("market", "social", "news", "fundamentals")

    via_dispatch = _edges(setup.setup_graph(selected, scheduler_mode="static"))
    direct = _edges(setup.setup_static_graph(selected))

    assert via_dispatch == direct
    assert ("__start__", "Market Analyst", False) in direct
    assert ("Msg Clear Market", "Sentiment Analyst", False) in direct
    assert ("Msg Clear Sentiment", "News Analyst", False) in direct
    assert ("Msg Clear News", "Fundamentals Analyst", False) in direct
    assert ("Msg Clear Fundamentals", "Bull Researcher", False) in direct
    assert ("Research Manager", "Trader", False) in direct
    assert ("Trader", "Aggressive Analyst", False) in direct
    assert ("Portfolio Manager", "__end__", False) in direct
    assert not any(source == "Scheduler" or target == "Scheduler" for source, target, _ in direct)


def test_learned_mode_routes_experts_back_to_scheduler(monkeypatch):
    setup = _graph_setup(monkeypatch)
    policy = CallableSchedulerPolicy(lambda context: context.valid_actions[0])
    edges = _edges(
        setup.setup_graph(
            ("market", "news"),
            scheduler_mode="learned",
            scheduler_policy=policy,
        )
    )

    assert ("__start__", "Scheduler", False) in edges
    assert ("Scheduler", "Market Analyst", True) in edges
    assert ("Scheduler", "News Analyst", True) in edges
    assert ("Scheduler", "Sentiment Analyst", True) not in edges
    assert ("Market Analyst", "tools_market", True) in edges
    assert ("tools_market", "Market Analyst", False) in edges
    assert ("Msg Clear Market", "Scheduler", False) in edges
    assert ("Bull Researcher", "Scheduler", False) in edges
    assert ("Portfolio Manager", "Scheduler", False) in edges
    assert ("Scheduler", "__end__", True) in edges


def test_learned_mode_requires_policy(monkeypatch):
    setup = _graph_setup(monkeypatch)

    try:
        setup.setup_graph(("market",), scheduler_mode="learned")
    except ValueError as exc:
        assert "requires scheduler_policy" in str(exc)
    else:
        raise AssertionError("learned mode accepted a missing policy")


def test_unknown_mode_is_rejected(monkeypatch):
    setup = _graph_setup(monkeypatch)

    try:
        setup.setup_graph(("market",), scheduler_mode="other")
    except ValueError as exc:
        assert "unknown scheduler_mode" in str(exc)
    else:
        raise AssertionError("unknown scheduler mode was accepted")


def test_action_enum_still_has_exactly_one_terminal_action():
    assert [action for action in SchedulerAction if action is SchedulerAction.STOP] == [
        SchedulerAction.STOP
    ]
