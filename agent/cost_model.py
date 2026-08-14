"""Cost model for catalyst discovery campaign actions.

Defines cost units for each action type. All agents must use these constants
to ensure budget tracking is consistent across the loop.
"""

from __future__ import annotations

from typing import Literal


# ---------------------------------------------------------------------------
# Cost units (absolute, not ratios)
# ---------------------------------------------------------------------------

# Budget caps
INITIAL_BUDGET = 100.0
MAX_ACTIONS_PER_CAMPAIGN = 50

# Action costs
KG_LOOKUP_COST: float = 1.0          # Retriever: KG traversal
SURROGATE_COST: float = 5.0          # Predictor: MACE/CGCNN inference
EXPERIMENT_COST: float = 10.0        # Escalation: DFT/UMA verification

# Fixed costs (always deducted)
CAMPAIGN_OVERHEAD_COST: float = 0.5  # Journal logging, MLflow write


# ---------------------------------------------------------------------------
# Cost categories (for aggregation/logging)
# ---------------------------------------------------------------------------

class ActionCategory(str, Literal):
    """Action type for cost categorization."""
    KG_LOOKUP = "kg_lookup"
    SURROGATE_QUERY = "surrogate_query"
    EXPERIMENT_ESCALATION = "experiment_escalation"


# ---------------------------------------------------------------------------
# Cost tracker (for campaign analysis)
# ---------------------------------------------------------------------------

class BudgetTracker:
    """Track spending across a campaign.

    Attributes:
        remaining_budget: Starting budget minus overhead
        costs_by_category: Dict of ActionCategory -> total cost
        actions_by_category: Dict of ActionCategory -> count
    """

    def __init__(self, initial_budget: float = INITIAL_BUDGET):
        self.initial_budget = initial_budget
        self.remaining_budget = initial_budget - CAMPAIGN_OVERHEAD_COST
        self.costs_by_category: dict[str, float] = {
            ActionCategory.KG_LOOKUP: 0.0,
            ActionCategory.SURROGATE_QUERY: 0.0,
            ActionCategory.EXPERIMENT_ESCALATION: 0.0,
        }
        self.actions_by_category: dict[str, int] = {
            ActionCategory.KG_LOOKUP: 0,
            ActionCategory.SURROGATE_QUERY: 0,
            ActionCategory.EXPERIMENT_ESCALATION: 0,
        }

    def deduct(self, category: ActionCategory | str, cost: float) -> bool:
        """Deduct cost from budget. Returns False if budget exhausted.

        Args:
            category: Action category to charge
            cost: Cost amount

        Returns:
            True if deduction successful, False if budget exhausted
        """
        if self.remaining_budget < cost:
            return False

        self.remaining_budget -= cost
        self.costs_by_category[category] += cost
        self.actions_by_category[category] += 1
        return True

    def can_afford(self, category: ActionCategory | str, cost: float) -> bool:
        """Check if budget can afford this action.

        Args:
            category: Action category
            cost: Required cost

        Returns:
            True if budget >= cost
        """
        return self.remaining_budget >= cost

    def get_efficiency(self) -> dict[str, float]:
        """Calculate cost-per-action efficiency metrics.

        Returns:
            Dict of category -> cost_per_action (float or inf if no actions)
        """
        result = {}
        for cat in self.actions_by_category:
            count = self.actions_by_category[cat]
            total_cost = self.costs_by_category[cat]
            result[cat] = (
                total_cost / count if count > 0 else float("inf")
            )
        return result

    def to_dict(self) -> dict[str, Any]:
        """Export tracker state for journal logging.

        Returns:
            Dict with budget summary, costs, and efficiency metrics
        """
        return {
            "initial_budget": self.initial_budget,
            "remaining_budget": round(self.remaining_budget, 2),
            "total_spent": round(
                self.initial_budget - self.remaining_budget, 2
            ),
            "costs_by_category": {k: round(v, 2) for k, v in self.costs_by_category.items()},
            "actions_by_category": dict(self.actions_by_category),
            "efficiency": {k: round(v, 4) if v != float("inf") else None for k, v in self.get_efficiency().items()},
        }


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def create_tracker(initial_budget: float = INITIAL_BUDGET) -> BudgetTracker:
    """Create a new budget tracker.

    Args:
        initial_budget: Starting budget amount

    Returns:
        Configured BudgetTracker instance
    """
    return BudgetTracker(initial_budget=initial_budget)


__all__ = [
    # Constants
    "INITIAL_BUDGET",
    "MAX_ACTIONS_PER_CAMPAIGN",
    "KG_LOOKUP_COST",
    "SURROGATE_COST",
    "EXPERIMENT_COST",
    "CAMPAIGN_OVERHEAD_COST",
    # Enum
    "ActionCategory",
    # Classes
    "BudgetTracker",
    # Factory
    "create_tracker",
]
