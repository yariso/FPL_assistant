"""
Transfer Value Index for Horizon-Aware Transfer Decisions.

Elite FPL managers evaluate transfers over 3-6 week horizons, not just
immediate gains. This module calculates the true value of a transfer
considering:
- Immediate xP gain
- Future fixture runs
- Price change risk
- Transfer banking value (flexibility)
- Squad constraints (3-per-team, valid formations, bench quality)
"""

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from ..data.models import Fixture, Player, Position, Team

if TYPE_CHECKING:
    from ..data.models import Squad

logger = logging.getLogger(__name__)


# =============================================================================
# Squad Constraint Validation
# =============================================================================


@dataclass
class ConstraintViolation:
    """A single constraint violation."""

    constraint_name: str
    severity: str  # "blocking" (can't do transfer) or "warning" (suboptimal)
    message: str
    affected_players: list[int] = field(default_factory=list)


@dataclass
class ConstraintCheckResult:
    """Result of squad constraint validation."""

    is_valid: bool  # True if no blocking violations
    violations: list[ConstraintViolation] = field(default_factory=list)
    warnings: list[ConstraintViolation] = field(default_factory=list)

    @property
    def blocking_violations(self) -> list[ConstraintViolation]:
        """Get only blocking violations."""
        return [v for v in self.violations if v.severity == "blocking"]

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def summary(self) -> str:
        """Human-readable summary."""
        if self.is_valid and not self.has_warnings:
            return "Transfer passes all constraints"

        parts = []
        if not self.is_valid:
            parts.append(f"BLOCKED: {len(self.blocking_violations)} violation(s)")
            for v in self.blocking_violations:
                parts.append(f"  - {v.message}")
        if self.has_warnings:
            parts.append(f"WARNINGS: {len(self.warnings)}")
            for w in self.warnings:
                parts.append(f"  - {w.message}")
        return "\n".join(parts)


class SquadConstraintChecker:
    """
    Validates transfers against FPL squad constraints.

    Constraints checked:
    1. 3-per-team limit
    2. Valid formation (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD for starting XI)
    3. Bench quality (especially around BB weeks)
    4. Future flexibility (doesn't block premium targets)
    """

    # Formation constraints
    MIN_GK = 1
    MAX_GK = 1
    MIN_DEF = 3
    MAX_DEF = 5
    MIN_MID = 2
    MAX_MID = 5
    MIN_FWD = 1
    MAX_FWD = 3

    # Must have exactly 2 GKs, 5 DEFs, 5 MIDs, 3 FWDs in full squad
    SQUAD_POSITION_COUNTS = {
        Position.GK: 2,
        Position.DEF: 5,
        Position.MID: 5,
        Position.FWD: 3,
    }

    def __init__(
        self,
        players: dict[int, Player],
        upcoming_bb_gw: int | None = None,
        premium_targets: list[int] | None = None,
    ):
        """
        Initialize constraint checker.

        Args:
            players: Dict of player_id -> Player
            upcoming_bb_gw: Gameweek of planned Bench Boost (if any)
            premium_targets: Player IDs you might want to target in future
        """
        self.players = players
        self.upcoming_bb_gw = upcoming_bb_gw
        self.premium_targets = set(premium_targets or [])

    def check_transfer(
        self,
        current_squad_ids: set[int],
        player_out_id: int,
        player_in_id: int,
        available_budget: float | None = None,
        selling_price: float | None = None,
    ) -> ConstraintCheckResult:
        """
        Check if a transfer violates any constraints.

        Args:
            current_squad_ids: Current squad player IDs
            player_out_id: Player to sell
            player_in_id: Player to buy
            available_budget: Bank balance (if None, budget check skipped)
            selling_price: Selling price of player_out (if None, uses current price)

        Returns:
            ConstraintCheckResult with all violations/warnings
        """
        violations = []
        warnings = []

        player_out = self.players.get(player_out_id)
        player_in = self.players.get(player_in_id)

        if not player_out or not player_in:
            violations.append(ConstraintViolation(
                constraint_name="player_exists",
                severity="blocking",
                message="Player not found in player database",
            ))
            return ConstraintCheckResult(is_valid=False, violations=violations)

        # CRITICAL: Budget validation
        if available_budget is not None:
            sell_price = selling_price if selling_price is not None else player_out.price
            transfer_cost = player_in.price - sell_price
            if transfer_cost > available_budget:
                violations.append(ConstraintViolation(
                    constraint_name="budget",
                    severity="blocking",
                    message=f"Cannot afford: need £{transfer_cost:.1f}m but only £{available_budget:.1f}m available",
                    affected_players=[player_in_id],
                ))

        # Build new squad
        new_squad_ids = (current_squad_ids - {player_out_id}) | {player_in_id}

        # 1. Check position match (must replace same position)
        if player_out.position != player_in.position:
            violations.append(ConstraintViolation(
                constraint_name="position_match",
                severity="blocking",
                message=f"Cannot replace {player_out.position.name} with {player_in.position.name}",
                affected_players=[player_out_id, player_in_id],
            ))

        # 2. Check 3-per-team limit
        team_counts = self._count_players_by_team(new_squad_ids)
        for team_id, count in team_counts.items():
            if count > 3:
                team_players = [pid for pid in new_squad_ids
                               if self.players.get(pid) and self.players[pid].team_id == team_id]
                violations.append(ConstraintViolation(
                    constraint_name="team_limit",
                    severity="blocking",
                    message=f"Would have {count} players from same team (max 3)",
                    affected_players=team_players,
                ))

        # 3. Check player already in squad
        if player_in_id in current_squad_ids:
            violations.append(ConstraintViolation(
                constraint_name="player_unique",
                severity="blocking",
                message=f"{player_in.web_name} is already in your squad",
                affected_players=[player_in_id],
            ))

        # 4. Check squad composition (15 players with right position counts)
        position_counts = self._count_players_by_position(new_squad_ids)
        for position, required in self.SQUAD_POSITION_COUNTS.items():
            actual = position_counts.get(position, 0)
            if actual != required:
                violations.append(ConstraintViolation(
                    constraint_name="squad_composition",
                    severity="blocking",
                    message=f"Squad must have {required} {position.name}s, would have {actual}",
                ))

        # === WARNINGS (suboptimal but allowed) ===

        # 5. Bench Boost week warning
        if self.upcoming_bb_gw:
            # Check if incoming player might not play (BB needs all 15 to play)
            if player_in.chance_of_playing is not None and player_in.chance_of_playing < 75:
                warnings.append(ConstraintViolation(
                    constraint_name="bb_bench_quality",
                    severity="warning",
                    message=f"BB planned GW{self.upcoming_bb_gw}: {player_in.web_name} has {player_in.chance_of_playing}% play chance",
                    affected_players=[player_in_id],
                ))

        # 6. Future flexibility warning (blocks premium target)
        if self.premium_targets:
            player_in_team = player_in.team_id
            blocked_targets = []
            for target_id in self.premium_targets:
                target = self.players.get(target_id)
                if target and target.team_id == player_in_team:
                    # Check if adding player_in would make 3 from this team
                    current_from_team = sum(
                        1 for pid in new_squad_ids
                        if self.players.get(pid) and self.players[pid].team_id == player_in_team
                    )
                    if current_from_team >= 3:
                        blocked_targets.append(target_id)

            if blocked_targets:
                target_names = [self.players[tid].web_name for tid in blocked_targets if tid in self.players]
                warnings.append(ConstraintViolation(
                    constraint_name="future_flexibility",
                    severity="warning",
                    message=f"Would block future move to: {', '.join(target_names)} (3 from same team)",
                    affected_players=blocked_targets,
                ))

        # 7. Selling nailed player warning
        if player_out.minutes > 0:
            games_estimate = max(1, player_out.minutes / 90)
            avg_mins = player_out.minutes / games_estimate
            if avg_mins >= 85 and player_out.status == "a":
                warnings.append(ConstraintViolation(
                    constraint_name="selling_nailed",
                    severity="warning",
                    message=f"Selling nailed starter {player_out.web_name} (avg {avg_mins:.0f} mins/game)",
                    affected_players=[player_out_id],
                ))

        is_valid = len([v for v in violations if v.severity == "blocking"]) == 0
        return ConstraintCheckResult(
            is_valid=is_valid,
            violations=violations,
            warnings=warnings,
        )

    def _count_players_by_team(self, squad_ids: set[int]) -> dict[int, int]:
        """Count players per team."""
        counts: dict[int, int] = {}
        for pid in squad_ids:
            player = self.players.get(pid)
            if player:
                counts[player.team_id] = counts.get(player.team_id, 0) + 1
        return counts

    def _count_players_by_position(self, squad_ids: set[int]) -> dict[Position, int]:
        """Count players per position."""
        counts: dict[Position, int] = {}
        for pid in squad_ids:
            player = self.players.get(pid)
            if player:
                counts[player.position] = counts.get(player.position, 0) + 1
        return counts

    def check_valid_starting_xi(
        self,
        starting_xi_ids: list[int],
    ) -> ConstraintCheckResult:
        """
        Check if a starting XI has valid formation.

        Args:
            starting_xi_ids: 11 player IDs for starting XI

        Returns:
            ConstraintCheckResult
        """
        violations = []

        if len(starting_xi_ids) != 11:
            violations.append(ConstraintViolation(
                constraint_name="starting_xi_count",
                severity="blocking",
                message=f"Starting XI must have 11 players, has {len(starting_xi_ids)}",
            ))
            return ConstraintCheckResult(is_valid=False, violations=violations)

        position_counts = self._count_players_by_position(set(starting_xi_ids))

        gk_count = position_counts.get(Position.GK, 0)
        def_count = position_counts.get(Position.DEF, 0)
        mid_count = position_counts.get(Position.MID, 0)
        fwd_count = position_counts.get(Position.FWD, 0)

        # Must have exactly 1 GK
        if gk_count != 1:
            violations.append(ConstraintViolation(
                constraint_name="gk_count",
                severity="blocking",
                message=f"Must have exactly 1 GK in starting XI, has {gk_count}",
            ))

        # Must have 3-5 DEF
        if def_count < self.MIN_DEF or def_count > self.MAX_DEF:
            violations.append(ConstraintViolation(
                constraint_name="def_count",
                severity="blocking",
                message=f"Must have {self.MIN_DEF}-{self.MAX_DEF} DEF, has {def_count}",
            ))

        # Must have 2-5 MID
        if mid_count < self.MIN_MID or mid_count > self.MAX_MID:
            violations.append(ConstraintViolation(
                constraint_name="mid_count",
                severity="blocking",
                message=f"Must have {self.MIN_MID}-{self.MAX_MID} MID, has {mid_count}",
            ))

        # Must have 1-3 FWD
        if fwd_count < self.MIN_FWD or fwd_count > self.MAX_FWD:
            violations.append(ConstraintViolation(
                constraint_name="fwd_count",
                severity="blocking",
                message=f"Must have {self.MIN_FWD}-{self.MAX_FWD} FWD, has {fwd_count}",
            ))

        is_valid = len(violations) == 0
        return ConstraintCheckResult(is_valid=is_valid, violations=violations)


class TransferRecommendation(StrEnum):
    """Transfer recommendation classification."""

    DO_NOW = "do_now"         # Transfer is valuable, make it now
    WAIT = "wait"             # Valuable but no urgency
    SKIP = "skip"             # Not worth it
    URGENT = "urgent"         # Do immediately (price change imminent)


@dataclass
class TransferValueIndex:
    """
    Comprehensive transfer value assessment.

    Instead of just "player A is better than B this week",
    this evaluates the full transfer value over a horizon.
    """

    player_out: Player
    player_in: Player

    # Component values
    immediate_xp_gain: float    # This gameweek
    horizon_xp_gain: float      # Over 5 gameweeks total
    price_change_value: float   # Expected price impact (+ = player rising)
    flexibility_cost: float     # Value of saving the FT

    # Calculated scores
    net_tvi: float              # Total Transfer Value Index
    confidence: float           # How confident in this assessment

    # Recommendation
    recommendation: TransferRecommendation
    reasoning: str

    # Constraint validation (NEW)
    constraint_result: ConstraintCheckResult | None = None

    @property
    def should_do(self) -> bool:
        """Whether to make this transfer."""
        # Block if constraints violated
        if self.constraint_result and not self.constraint_result.is_valid:
            return False
        return self.recommendation in [
            TransferRecommendation.DO_NOW,
            TransferRecommendation.URGENT,
        ]

    @property
    def is_urgent(self) -> bool:
        """Whether transfer needs to be made immediately."""
        return self.recommendation == TransferRecommendation.URGENT

    @property
    def is_blocked(self) -> bool:
        """Whether transfer is blocked by constraints."""
        return self.constraint_result is not None and not self.constraint_result.is_valid

    @property
    def has_warnings(self) -> bool:
        """Whether transfer has constraint warnings."""
        return self.constraint_result is not None and self.constraint_result.has_warnings


# Value of banking a free transfer (having 2 FTs is valuable)
FT_BANKING_VALUE = 2.0  # Estimated xP value of having 2 FTs


class TransferValueCalculator:
    """
    Calculates Transfer Value Index for potential moves.

    Key insight: A transfer should be evaluated over the horizon
    it will impact, not just the immediate week.
    """

    def __init__(
        self,
        players: list[Player],
        teams: list[Team],
        fixtures: list[Fixture],
        projections: dict[int, dict[int, float]],  # player_id -> gw -> xP
        price_predictions: dict[int, float] | None = None,  # player_id -> predicted change
        current_squad_ids: set[int] | None = None,  # For constraint validation
        upcoming_bb_gw: int | None = None,  # Planned Bench Boost gameweek
        premium_targets: list[int] | None = None,  # Future transfer targets
        available_budget: float | None = None,  # Bank balance for budget validation
        selling_prices: dict[int, float] | None = None,  # player_id -> selling price
    ):
        """
        Initialize calculator.

        Args:
            players: All players
            teams: All teams
            fixtures: All fixtures
            projections: Projections per player per gameweek
            price_predictions: Optional price change predictions
            current_squad_ids: Current squad player IDs (for constraint checking)
            upcoming_bb_gw: Gameweek of planned Bench Boost (optional)
            premium_targets: Player IDs you might want in future (optional)
            available_budget: Bank balance for budget validation (optional)
            selling_prices: Dict of player_id -> selling price (optional)
        """
        self.players = {p.id: p for p in players}
        self.teams = {t.id: t for t in teams}
        self.fixtures = fixtures
        self.projections = projections
        self.price_predictions = price_predictions or {}
        self.current_squad_ids = current_squad_ids or set()
        self.available_budget = available_budget
        self.selling_prices = selling_prices or {}

        # Group fixtures by gameweek for quick lookup
        self._fixtures_by_gw: dict[int, list[Fixture]] = {}
        for f in fixtures:
            if f.gameweek not in self._fixtures_by_gw:
                self._fixtures_by_gw[f.gameweek] = []
            self._fixtures_by_gw[f.gameweek].append(f)

        # Initialize constraint checker
        self._constraint_checker = SquadConstraintChecker(
            players=self.players,
            upcoming_bb_gw=upcoming_bb_gw,
            premium_targets=premium_targets,
        )

    def calculate_tvi(
        self,
        player_out: Player,
        player_in: Player,
        current_gw: int,
        horizon: int = 5,
        free_transfers: int = 1,
        squad_ids: set[int] | None = None,
    ) -> TransferValueIndex:
        """
        Calculate Transfer Value Index for a potential move.

        Args:
            player_out: Player to sell
            player_in: Player to buy
            current_gw: Current gameweek
            horizon: Gameweeks to evaluate (default 5)
            free_transfers: Current FTs available
            squad_ids: Current squad (uses self.current_squad_ids if not provided)

        Returns:
            TransferValueIndex with full assessment
        """
        # Use provided squad or default
        current_squad = squad_ids or self.current_squad_ids

        # 0. Validate constraints FIRST (including budget)
        constraint_result = None
        if current_squad:
            # Get selling price for the outgoing player
            selling_price = self.selling_prices.get(player_out.id, player_out.price)

            constraint_result = self._constraint_checker.check_transfer(
                current_squad_ids=current_squad,
                player_out_id=player_out.id,
                player_in_id=player_in.id,
                available_budget=self.available_budget,
                selling_price=selling_price,
            )

            # Add hit warning if no free transfers
            if free_transfers <= 0 and constraint_result:
                constraint_result.warnings.append(ConstraintViolation(
                    constraint_name="hit_required",
                    severity="warning",
                    message="This transfer requires a -4 point hit",
                    affected_players=[],
                ))

        # 1. Immediate xP gain (this week only)
        out_gw_xp = self._get_projection(player_out.id, current_gw)
        in_gw_xp = self._get_projection(player_in.id, current_gw)
        immediate_gain = in_gw_xp - out_gw_xp

        # 2. Horizon xP gain (over all weeks)
        horizon_out_xp = sum(
            self._get_projection(player_out.id, gw)
            for gw in range(current_gw, current_gw + horizon)
        )
        horizon_in_xp = sum(
            self._get_projection(player_in.id, gw)
            for gw in range(current_gw, current_gw + horizon)
        )
        horizon_gain = horizon_in_xp - horizon_out_xp

        # 3. Price change value
        price_value = self._calculate_price_value(player_out, player_in)

        # 4. Flexibility cost (value of saving the FT)
        flexibility_cost = self._calculate_flexibility_cost(free_transfers, current_gw)

        # 5. Hit cost if no free transfers available
        hit_cost = 4.0 if free_transfers <= 0 else 0.0

        # 6. Calculate net TVI (factoring in hit cost!)
        net_tvi = (
            horizon_gain
            + price_value
            - flexibility_cost
            - hit_cost
        )

        # 7. Determine recommendation
        recommendation, reasoning = self._get_recommendation(
            immediate_gain,
            horizon_gain,
            price_value,
            net_tvi,
            player_in,
            player_out,
        )

        # Add hit info to reasoning if applicable
        if hit_cost > 0:
            reasoning += f" (includes -4 hit)"

        # Override recommendation if constraints blocked
        if constraint_result and not constraint_result.is_valid:
            recommendation = TransferRecommendation.SKIP
            reasoning = f"BLOCKED: {constraint_result.blocking_violations[0].message}"
        elif constraint_result and constraint_result.has_warnings:
            # Add warning to reasoning
            warning_msgs = [w.message for w in constraint_result.warnings]
            reasoning += f" | WARNING: {'; '.join(warning_msgs)}"

        # 8. Calculate confidence
        confidence = self._calculate_confidence(horizon, current_gw)

        return TransferValueIndex(
            player_out=player_out,
            player_in=player_in,
            immediate_xp_gain=round(immediate_gain, 2),
            horizon_xp_gain=round(horizon_gain, 2),
            price_change_value=round(price_value, 2),
            flexibility_cost=round(flexibility_cost, 2),
            net_tvi=round(net_tvi, 2),
            confidence=round(confidence, 2),
            recommendation=recommendation,
            reasoning=reasoning,
            constraint_result=constraint_result,
        )

    def _get_projection(self, player_id: int, gameweek: int) -> float:
        """Get projection for player in gameweek."""
        if player_id in self.projections:
            return self.projections[player_id].get(gameweek, 0)
        return 0.0

    def _calculate_price_value(
        self,
        player_out: Player,
        player_in: Player,
    ) -> float:
        """
        Calculate price change value component.

        Factors:
        - If player_in is rising, get them before rise
        - If player_out is falling, sell before fall
        - Price gain from selling high
        """
        value = 0.0

        # Incoming player price prediction
        in_change = self.price_predictions.get(player_in.id, 0)
        if in_change > 0:
            # Player rising - buying now saves money
            value += in_change * 0.5  # Partial credit (not guaranteed)

        # Outgoing player price prediction
        out_change = self.price_predictions.get(player_out.id, 0)
        if out_change < 0:
            # Player falling - selling now saves value
            value += abs(out_change) * 0.5

        return value

    def _calculate_flexibility_cost(
        self,
        free_transfers: int,
        current_gw: int,
    ) -> float:
        """
        Calculate the cost of using a transfer (lost flexibility).

        Having 2 FTs is valuable - it gives options for:
        - Double transfer plays
        - Emergency injury response
        - Better timing
        """
        if free_transfers >= 2:
            # Using one of 2 FTs - moderate flexibility cost
            return FT_BANKING_VALUE * 0.5
        else:
            # Only have 1 FT - higher flexibility cost
            return FT_BANKING_VALUE

    def _get_recommendation(
        self,
        immediate_gain: float,
        horizon_gain: float,
        price_value: float,
        net_tvi: float,
        player_in: Player,
        player_out: Player,
    ) -> tuple[TransferRecommendation, str]:
        """Determine recommendation and reasoning."""

        # Check for urgent price scenarios
        in_change = self.price_predictions.get(player_in.id, 0)
        out_change = self.price_predictions.get(player_out.id, 0)

        if in_change >= 0.1 or out_change <= -0.1:
            if net_tvi > 0:
                return (
                    TransferRecommendation.URGENT,
                    f"Price change imminent! {player_in.web_name} rising or {player_out.web_name} falling. Act now.",
                )

        # Normal recommendations based on TVI
        if net_tvi >= 5:
            return (
                TransferRecommendation.DO_NOW,
                f"Strong transfer: +{horizon_gain:.1f} xP over horizon. Clear improvement.",
            )
        elif net_tvi >= 2:
            return (
                TransferRecommendation.DO_NOW,
                f"Good transfer: +{horizon_gain:.1f} xP over horizon. Worth making.",
            )
        elif net_tvi >= 0:
            return (
                TransferRecommendation.WAIT,
                f"Marginal gain (+{net_tvi:.1f} TVI). Consider waiting for better options.",
            )
        else:
            return (
                TransferRecommendation.SKIP,
                f"Not recommended: {player_out.web_name} projects better over horizon.",
            )

    def _calculate_confidence(self, horizon: int, current_gw: int) -> float:
        """Calculate confidence in TVI calculation."""
        # Confidence decreases with longer horizons
        base_confidence = 0.8

        # Reduce for longer horizons
        horizon_penalty = (horizon - 1) * 0.05
        confidence = base_confidence - horizon_penalty

        # Reduce near end of season (fixtures more uncertain)
        if current_gw > 30:
            confidence -= 0.1

        return max(0.3, min(0.95, confidence))

    def get_best_transfers(
        self,
        current_squad_ids: set[int],
        budget: float,
        current_gw: int,
        position: Position | None = None,
        limit: int = 5,
        include_blocked: bool = False,
    ) -> list[TransferValueIndex]:
        """
        Find the best transfers for the current squad.

        Args:
            current_squad_ids: IDs of current squad players
            budget: Available budget (bank + player selling price)
            current_gw: Current gameweek
            position: Optional position filter
            limit: Max transfers to return
            include_blocked: Whether to include blocked transfers in results

        Returns:
            List of best TransferValueIndex options
        """
        results = []

        # Get squad players
        squad_players = [self.players[pid] for pid in current_squad_ids if pid in self.players]

        # For each squad player, find best replacement
        for out_player in squad_players:
            if position and out_player.position != position:
                continue

            # Find affordable replacements
            for in_player in self.players.values():
                if in_player.id in current_squad_ids:
                    continue
                if in_player.position != out_player.position:
                    continue
                if in_player.price > out_player.price + budget:
                    continue
                if in_player.status not in ["a", "d"]:  # Available or doubtful only
                    continue

                # Calculate TVI with constraint validation
                try:
                    tvi = self.calculate_tvi(
                        out_player,
                        in_player,
                        current_gw,
                        squad_ids=current_squad_ids,
                    )

                    # Filter blocked transfers unless explicitly included
                    if tvi.is_blocked and not include_blocked:
                        continue

                    if tvi.should_do or tvi.net_tvi > -2:  # Include marginal options
                        results.append(tvi)
                except Exception as e:
                    logger.debug(f"TVI calculation failed: {e}")

        # Sort by net TVI (valid transfers first, then by value)
        results.sort(key=lambda x: (not x.is_blocked, x.net_tvi), reverse=True)
        return results[:limit]


def calculate_transfer_value(
    player_out: Player,
    player_in: Player,
    all_players: list[Player],
    teams: list[Team],
    fixtures: list[Fixture],
    projections: dict[int, dict[int, float]],
    current_gw: int,
    horizon: int = 5,
) -> TransferValueIndex:
    """Convenience function for single transfer evaluation."""
    calculator = TransferValueCalculator(all_players, teams, fixtures, projections)
    return calculator.calculate_tvi(player_out, player_in, current_gw, horizon)
