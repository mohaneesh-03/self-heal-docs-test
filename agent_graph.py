"""
agent_graph.py
--------------
LangGraph Actor-Critic workflow for self-healing documentation.

Two-node graph:
  Writer  → drafts/rewrites the README based on the PR diff
  Critic  → evaluates accuracy; loops back to Writer or exits

Uses ChatGroq for fast inference and Pydantic for structured critic output.
"""

from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


# ── State Schema ───────────────────────────────────────────────────────

class GraphState(TypedDict):
    """Shared state flowing through the Actor-Critic graph."""
    pr_diff: str
    current_readme: str
    draft_readme: str
    critic_feedback: str
    iteration_count: int
    is_accurate: bool
    pr_summary: str


# ── Pydantic model for structured Critic output ───────────────────────

class CriticEvaluation(BaseModel):
    """Structured evaluation from the Critic node."""
    is_accurate: bool = Field(
        description="True if the draft README accurately reflects the PR diff "
                    "without hallucinating features or missing key changes."
    )
    feedback: str = Field(
        description="Specific, actionable feedback for the Writer. "
                    "If is_accurate is True, this should be 'LGTM'."
    )


# ── LLM Setup ─────────────────────────────────────────────────────────

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,
)

critic_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.0,
)


# ── Prompts ────────────────────────────────────────────────────────────

WRITER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior technical writer. Your job is to update a project's README.md 
to accurately reflect code changes from a merged Pull Request.

RULES:
- Only add, modify, or remove documentation that is directly supported by the PR diff.
- Do NOT hallucinate features, endpoints, or behaviors not shown in the diff.
- Preserve the existing structure and tone of the README as much as possible.
- If the README is empty, create a clean, professional README from scratch based on the diff.
- Use proper Markdown formatting (headings, code blocks, lists).
- Be concise but thorough."""),

    ("human", """## Current README
{current_readme}

## PR Diff (code changes)
```
{pr_diff}
```

## Critic Feedback (from previous iteration)
{critic_feedback}

Please write the updated README.md content. Output ONLY the raw Markdown — no wrapping, 
no explanations, no code fences around the entire output."""),
])

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a strict documentation reviewer. Your job is to verify that a draft 
README accurately reflects the code changes in a PR diff.

You must check for:
1. **Hallucinations**: Does the README mention features, APIs, or behaviors NOT present in the diff?
2. **Omissions**: Does the README miss important changes shown in the diff?
3. **Accuracy**: Are code examples, function signatures, and descriptions correct per the diff?
4. **Quality**: Is the Markdown well-structured and professional?

Be strict. If there are ANY inaccuracies, mark is_accurate as false and provide specific feedback."""),

    ("human", """## PR Diff (source of truth)
```
{pr_diff}
```

## Draft README to evaluate
{draft_readme}

Evaluate this draft strictly against the PR diff."""),
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You write concise PR descriptions. Output 2-3 sentences max."),
    ("human", """Summarize the documentation changes made to README.md based on this diff:
```
{pr_diff}
```
This summary will be used as the body of a Pull Request."""),
])


# ── Node Functions ─────────────────────────────────────────────────────

def writer_node(state: GraphState) -> dict:
    """Writer (Actor): Drafts or revises the README based on diff + feedback."""
    print(f"\n[Writer] Iteration {state['iteration_count'] + 1} — "
          f"generating draft...")

    feedback = state.get("critic_feedback", "None — this is the first draft.")

    chain = WRITER_PROMPT | llm
    response = chain.invoke({
        "current_readme": state["current_readme"] or "(No existing README)",
        "pr_diff": state["pr_diff"],
        "critic_feedback": feedback,
    })

    draft = response.content.strip()
    print(f"[Writer] Draft generated ({len(draft)} chars)")

    return {"draft_readme": draft}


def critic_node(state: GraphState) -> dict:
    """Critic (Evaluator): Checks the draft README against the PR diff."""
    print(f"[Critic] Evaluating draft (iteration {state['iteration_count'] + 1})...")

    chain = CRITIC_PROMPT | critic_llm.with_structured_output(CriticEvaluation)
    evaluation: CriticEvaluation = chain.invoke({
        "pr_diff": state["pr_diff"],
        "draft_readme": state["draft_readme"],
    })

    print(f"[Critic] Accurate: {evaluation.is_accurate}")
    print(f"[Critic] Feedback: {evaluation.feedback[:200]}")

    return {
        "critic_feedback": evaluation.feedback,
        "iteration_count": state["iteration_count"] + 1,
        "is_accurate": evaluation.is_accurate,
    }


def summary_node(state: GraphState) -> dict:
    """Generate a short PR summary once the README is finalized."""
    print("[Summary] Generating PR description...")

    chain = SUMMARY_PROMPT | llm
    response = chain.invoke({"pr_diff": state["pr_diff"]})

    return {"pr_summary": response.content.strip()}


# ── Routing Logic ──────────────────────────────────────────────────────

def should_continue(state: GraphState) -> str:
    """Route after the Critic: loop back to Writer or finalize."""
    is_accurate = state.get("is_accurate", False)
    iteration = state["iteration_count"]

    if is_accurate:
        print(f"[Router] APPROVED - Critic approved at iteration {iteration}. Finalizing.")
        return "accepted"
    elif iteration >= 3:
        print(f"[Router] MAX ITERS - Max iterations ({iteration}) reached. "
              "Using best draft.")
        return "accepted"
    else:
        print(f"[Router] REVISE - Critic rejected. Sending back to Writer "
              f"(iteration {iteration}/3).")
        return "revise"


# ── Graph Construction ─────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Construct and compile the Actor-Critic LangGraph workflow."""
    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("writer", writer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("summary", summary_node)

    # Entry: start with the Writer
    graph.add_edge(START, "writer")

    # Writer always goes to Critic
    graph.add_edge("writer", "critic")

    # Critic conditionally routes
    graph.add_conditional_edges(
        "critic",
        should_continue,
        {
            "revise": "writer",     # loop back
            "accepted": "summary",  # finalize
        },
    )

    # Summary goes to END
    graph.add_edge("summary", END)

    return graph.compile()


# ── Public API ─────────────────────────────────────────────────────────

def run_workflow(pr_diff: str, current_readme: str) -> dict:
    """
    Run the Actor-Critic workflow end-to-end.

    Args:
        pr_diff:        The unified diff from the merged PR.
        current_readme: The current README.md content (may be empty).

    Returns:
        dict with keys: 'draft_readme', 'pr_summary', 'iteration_count'
    """
    app = build_graph()

    initial_state: GraphState = {
        "pr_diff": pr_diff,
        "current_readme": current_readme,
        "draft_readme": "",
        "critic_feedback": "",
        "iteration_count": 0,
        "pr_summary": "",
    }

    print("\n" + "=" * 60)
    print("  Self-Healing Docs - Actor-Critic Workflow")
    print("=" * 60)

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 60)
    print(f"  COMPLETE - Workflow finished after {final_state['iteration_count']} iteration(s)")
    print("=" * 60 + "\n")

    return {
        "draft_readme": final_state["draft_readme"],
        "pr_summary": final_state["pr_summary"],
        "iteration_count": final_state["iteration_count"],
    }
