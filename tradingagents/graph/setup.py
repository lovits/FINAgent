# TradingAgents/graph/setup.py

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_market_analyst,
    create_msg_delete,
    create_neutral_debator,
    create_news_analyst,
    create_portfolio_manager,
    create_research_manager,
    create_sentiment_analyst,
    create_trader,
)
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.agent_registry import registry_for_analysts
from tradingagents.scheduler.policy import SchedulerPolicy
from tradingagents.scheduler.scheduler_node import (
    SCHEDULER_NODE_NAME,
    SchedulerNode,
    route_scheduler_action,
)

from .analyst_execution import build_analyst_execution_plan
from .conditional_logic import ConditionalLogic

# Every target a shared conditional router can return. Each edge driven by the
# router maps all of them, so a fall-through return (e.g. under prompt/i18n/
# refactor drift in the speaker labels) can never hit a missing path_map entry
# and crash LangGraph mid-run (#1088).
DEBATE_PATH_MAP = {
    "Bull Researcher": "Bull Researcher",
    "Bear Researcher": "Bear Researcher",
    "Research Manager": "Research Manager",
}
RISK_ANALYSIS_PATH_MAP = {
    "Aggressive Analyst": "Aggressive Analyst",
    "Conservative Analyst": "Conservative Analyst",
    "Neutral Analyst": "Neutral Analyst",
    "Portfolio Manager": "Portfolio Manager",
}


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic

    def setup_graph(
        self,
        selected_analysts=("market", "social", "news", "fundamentals"),
        *,
        scheduler_mode: str = "static",
        scheduler_policy: SchedulerPolicy | None = None,
        scheduler_max_steps: int = 16,
        scheduler_on_decision: Callable | None = None,
    ):
        """Build either the original static graph or the learned scheduler graph."""

        if scheduler_mode == "static":
            return self.setup_static_graph(selected_analysts)
        if scheduler_mode == "learned":
            if scheduler_policy is None:
                raise ValueError("learned scheduler mode requires scheduler_policy")
            return self.setup_learned_graph(
                selected_analysts,
                scheduler_policy,
                scheduler_max_steps=scheduler_max_steps,
                scheduler_on_decision=scheduler_on_decision,
            )
        raise ValueError(f"unknown scheduler_mode: {scheduler_mode!r}")

    def setup_static_graph(
        self, selected_analysts=("market", "social", "news", "fundamentals")
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        plan = build_analyst_execution_plan(selected_analysts)

        analyst_factories = {
            "market": lambda: create_market_analyst(self.quick_thinking_llm),
            "social": lambda: create_sentiment_analyst(self.quick_thinking_llm),
            "news": lambda: create_news_analyst(self.quick_thinking_llm),
            "fundamentals": lambda: create_fundamentals_analyst(self.quick_thinking_llm),
        }

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm)
        research_manager_node = create_research_manager(self.deep_thinking_llm)
        trader_node = create_trader(self.quick_thinking_llm)

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm)

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for spec in plan.specs:
            workflow.add_node(spec.agent_node, analyst_factories[spec.key]())
            workflow.add_node(spec.clear_node, create_msg_delete())
            workflow.add_node(spec.tool_node, self.tool_nodes[spec.key])

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # Define edges
        # Start with the first analyst
        workflow.add_edge(START, plan.specs[0].agent_node)

        # Connect analysts in sequence
        for i, spec in enumerate(plan.specs):
            current_analyst = spec.agent_node
            current_tools = spec.tool_node
            current_clear = spec.clear_node

            # Add conditional edges for current analyst
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{spec.key}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst or to Bull Researcher if this is the last analyst
            if i < len(plan.specs) - 1:
                workflow.add_edge(current_clear, plan.specs[i + 1].agent_node)
            else:
                workflow.add_edge(current_clear, "Bull Researcher")

        # Both research-debate edges share the complete DEBATE_PATH_MAP (#1088).
        for debate_node in ("Bull Researcher", "Bear Researcher"):
            workflow.add_conditional_edges(
                debate_node,
                self.conditional_logic.should_continue_debate,
                DEBATE_PATH_MAP,
            )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        # All three risk edges share the complete RISK_ANALYSIS_PATH_MAP (#1088).
        for risk_node in ("Aggressive Analyst", "Conservative Analyst", "Neutral Analyst"):
            workflow.add_conditional_edges(
                risk_node,
                self.conditional_logic.should_continue_risk_analysis,
                RISK_ANALYSIS_PATH_MAP,
            )

        workflow.add_edge("Portfolio Manager", END)

        return workflow

    def setup_learned_graph(
        self,
        selected_analysts,
        scheduler_policy: SchedulerPolicy,
        *,
        scheduler_max_steps: int = 16,
        scheduler_on_decision: Callable | None = None,
    ):
        """Build a scheduler-centered graph while preserving expert internals."""

        plan = build_analyst_execution_plan(selected_analysts)
        analyst_factories = {
            "market": lambda: create_market_analyst(self.quick_thinking_llm),
            "social": lambda: create_sentiment_analyst(self.quick_thinking_llm),
            "news": lambda: create_news_analyst(self.quick_thinking_llm),
            "fundamentals": lambda: create_fundamentals_analyst(self.quick_thinking_llm),
        }
        workflow = StateGraph(AgentState)
        scheduler = SchedulerNode(
            scheduler_policy,
            selected_analysts,
            max_steps=scheduler_max_steps,
            max_debate_rounds=self.conditional_logic.max_debate_rounds,
            max_risk_rounds=self.conditional_logic.max_risk_discuss_rounds,
            on_decision=scheduler_on_decision,
        )
        workflow.add_node(SCHEDULER_NODE_NAME, scheduler)

        for spec in plan.specs:
            workflow.add_node(spec.agent_node, analyst_factories[spec.key]())
            workflow.add_node(spec.clear_node, create_msg_delete())
            workflow.add_node(spec.tool_node, self.tool_nodes[spec.key])
            workflow.add_conditional_edges(
                spec.agent_node,
                getattr(self.conditional_logic, f"should_continue_{spec.key}"),
                [spec.tool_node, spec.clear_node],
            )
            workflow.add_edge(spec.tool_node, spec.agent_node)
            workflow.add_edge(spec.clear_node, SCHEDULER_NODE_NAME)

        expert_nodes = {
            "Bull Researcher": create_bull_researcher(self.quick_thinking_llm),
            "Bear Researcher": create_bear_researcher(self.quick_thinking_llm),
            "Research Manager": create_research_manager(self.deep_thinking_llm),
            "Trader": create_trader(self.quick_thinking_llm),
            "Aggressive Analyst": create_aggressive_debator(self.quick_thinking_llm),
            "Conservative Analyst": create_conservative_debator(self.quick_thinking_llm),
            "Neutral Analyst": create_neutral_debator(self.quick_thinking_llm),
            "Portfolio Manager": create_portfolio_manager(self.deep_thinking_llm),
        }
        for node_name, node in expert_nodes.items():
            workflow.add_node(node_name, node)
            workflow.add_edge(node_name, SCHEDULER_NODE_NAME)

        route_map = {
            action.value: spec.node_name
            for action, spec in registry_for_analysts(tuple(selected_analysts)).items()
        }
        route_map[SchedulerAction.STOP.value] = END
        workflow.add_edge(START, SCHEDULER_NODE_NAME)
        workflow.add_conditional_edges(
            SCHEDULER_NODE_NAME,
            route_scheduler_action,
            route_map,
        )
        return workflow
