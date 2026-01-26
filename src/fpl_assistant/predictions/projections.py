"""
Data-Driven Player Projections Engine.

Generates expected points using multiple factors:
- Recent form and points history
- Fixture difficulty rating (FDR)
- Home/Away performance splits
- ICT Index (Influence, Creativity, Threat)
- Minutes played consistency
- Team attack/defense strength
- Opponent strength analysis
- Blank/Double gameweek handling
- Position-specific scoring patterns
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..data.models import (
    Fixture,
    GameweekInfo,
    Player,
    PlayerProjection,
    PlayerStatus,
    Position,
    Team,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Position-Specific Scoring Weights
# =============================================================================

# Average points per appearance by position (historical FPL data)
POSITION_BASE_POINTS = {
    Position.GK: 3.8,
    Position.DEF: 3.9,
    Position.MID: 4.2,
    Position.FWD: 4.0,
}

# Clean sheet probability weight by position
CLEAN_SHEET_WEIGHTS = {
    Position.GK: 4.0,   # 4 pts for CS
    Position.DEF: 4.0,  # 4 pts for CS
    Position.MID: 1.0,  # 1 pt for CS
    Position.FWD: 0.0,  # No CS points
}

# Goal scoring multiplier by position (FPL points per goal)
GOAL_WEIGHTS = {
    Position.GK: 6.0,
    Position.DEF: 6.0,
    Position.MID: 5.0,
    Position.FWD: 4.0,
}

# Assist points (same for all)
ASSIST_POINTS = 3.0

# Bonus points average by form
BONUS_MULTIPLIER = 0.15  # ~15% of form translates to bonus

# =============================================================================
# xG-Based Projection Constants (THE KEY TO ACCURACY!)
# =============================================================================

# xG conversion rates - how much of xG actually converts to FPL points
# Based on historical analysis: xG is the BEST predictor of future goals
XG_GOAL_CONVERSION = {
    Position.GK: 0.5,   # GKs rarely score even when they have xG
    Position.DEF: 0.85,  # Defenders convert well on set pieces
    Position.MID: 0.95,  # Midfielders are most consistent
    Position.FWD: 0.90,  # Forwards have variance but high volume
}

# xA conversion rates
XA_ASSIST_CONVERSION = {
    Position.GK: 0.3,   # GK assists are rare
    Position.DEF: 0.75,  # Long balls sometimes assist
    Position.MID: 0.90,  # Creative midfielders assist consistently
    Position.FWD: 0.80,  # Target men get hockey assists
}

# =============================================================================
# POSITION-SPECIFIC XGI MULTIPLIERS - Key for proper valuation!
# =============================================================================
# Attackers should be valued MORE for their xGI than defenders
# This is because:
# 1. Goals/assists are the main source of attacking player points
# 2. Defenders get clean sheet points as a "floor" regardless of xGI
# 3. Captain picks should prioritize high-ceiling attackers

XGI_POSITION_MULTIPLIER = {
    Position.GK: 0.5,   # GKs rarely score - xGI almost irrelevant
    Position.DEF: 1.0,   # Defenders - xGI is secondary to clean sheets
    Position.MID: 2.5,   # Midfielders - xGI is PRIMARY value driver
    Position.FWD: 3.0,   # Forwards - xGI is EVERYTHING, no CS floor
}

# Bonus points correlation with xGI (expected goal involvement)
# High xGI players get more bonus points even when they don't score
XGI_BONUS_MULTIPLIER = 0.25

# Regression factor for players outperforming their xG
# If someone has scored 10 goals on 5 xG, they'll likely regress
REGRESSION_FACTOR = 0.3  # How much we adjust towards xG


# =============================================================================
# 2025/26 Defensive Contributions - NEW SCORING RULE
# =============================================================================
# DEF: 2 pts for 10+ defensive contributions (clearances + blocks + interceptions + tackles)
# MID/FWD: 2 pts for 12+ defensive contributions (adds ball recoveries)

DEFENSIVE_CONTRIBUTION_POINTS = 2  # Points for hitting threshold
DEFENSIVE_CONTRIBUTION_THRESHOLD = {
    Position.GK: 99,   # GKs don't get this (threshold unreachable)
    Position.DEF: 10,  # 10+ CBIT (clearances, blocks, interceptions, tackles)
    Position.MID: 12,  # 12+ CBIRT (adds ball recoveries)
    Position.FWD: 12,  # 12+ CBIRT
}

# Base probability of hitting defensive contribution threshold (per 90 mins)
# These are estimates based on typical PL defensive action rates
DEFENSIVE_CONTRIBUTION_BASE_PROBABILITY = {
    Position.GK: 0.0,    # GKs don't accumulate these
    Position.DEF: 0.35,  # ~35% of DEFs hit 10+ per game
    Position.MID: 0.10,  # ~10% of MIDs hit 12+ per game
    Position.FWD: 0.02,  # ~2% of FWDs hit 12+ per game
}

# =============================================================================
# Field-specific zero-is-suspicious policy
# =============================================================================
# Some fields legitimately can be 0 even with significant playing time
# (e.g., red cards, penalties missed). Others being 0 is suspicious.

# Configurable threshold: minutes played before zero becomes suspicious
# 450 mins = ~5 full matches. Lower early-season (e.g., 180-270) if needed.
SUSPICIOUS_ZERO_MINUTES = 450

ZERO_IS_SUSPICIOUS = {
    'bonus_per_90': True,       # Very rare for active player to have 0 bonus
    'yellow_card_prob': False,  # Clean players exist (but borderline)
    'red_card_prob': False,     # Most players never get red cards
    'saves_per_90': True,       # GKs with 450+ mins should have saves
    'goals_conceded_per_90': True,  # GK/DEF with 450+ mins should have data
}

# Sanity clamps for derived fields - prevents corrupted data from breaking optimizer
# Format: field_name -> (min_value, max_value)
DERIVED_FIELD_BOUNDS = {
    'bonus_per_90': (0.0, 2.0),         # Max ~2 bonus per 90 is elite
    'yellow_card_prob': (0.0, 0.3),     # 30% per match is extreme
    'red_card_prob': (0.0, 0.05),       # 5% per match is extreme
    'saves_per_90': (0.0, 7.0),         # 7 saves per 90 is very busy GK
    'goals_conceded_per_90': (0.0, 4.0),  # 4 goals per 90 is disaster
}


@dataclass
class DerivedPlayerStats:
    """
    Pre-computed derived statistics for a Player.

    Use this to avoid repeated fallback calculations during projection loops.
    Created by ProjectionEngine.normalise_player().
    """
    player_id: int
    equiv_90s: float
    bonus_per_90: float
    yellow_card_prob: float
    red_card_prob: float
    saves_per_90: float  # Only meaningful for GKs


@dataclass
class FixtureAnalysis:
    """Analysis of a fixture for projection purposes."""

    fixture: Fixture
    opponent_id: int
    is_home: bool
    difficulty: int  # 1-5 FDR
    opponent_attack_strength: float = 1.0
    opponent_defense_strength: float = 1.0
    clean_sheet_probability: float = 0.3
    scoring_probability: float = 0.5


@dataclass
class PlayerAnalysis:
    """Comprehensive analysis of a player."""

    player: Player
    minutes_probability: float = 1.0
    form_score: float = 0.0
    ict_score: float = 0.0
    consistency_score: float = 0.0  # How consistently they play 90 mins
    fixture_analyses: list[FixtureAnalysis] = field(default_factory=list)


@dataclass
class ProjectionComponents:
    """
    Breakdown of xP projection into components.

    All values are in points units for transparency.
    Sum of components = total xP
    """

    appearance: float = 0.0      # Base points from playing
    goals: float = 0.0           # Expected points from goals
    assists: float = 0.0         # Expected points from assists
    clean_sheet: float = 0.0     # Expected CS points
    defensive_contrib: float = 0.0  # 2025/26 DC points
    bonus: float = 0.0           # Expected bonus points
    saves: float = 0.0           # GK save points
    cards: float = 0.0           # Expected card deductions (negative)
    conceded: float = 0.0        # Goals conceded deductions (negative)

    @property
    def total(self) -> float:
        """Total expected points."""
        return (
            self.appearance + self.goals + self.assists +
            self.clean_sheet + self.defensive_contrib + self.bonus +
            self.saves + self.cards + self.conceded
        )

    def breakdown_str(self) -> str:
        """Human-readable breakdown."""
        parts = []
        if self.appearance > 0:
            parts.append(f"App: {self.appearance:.1f}")
        if self.goals > 0:
            parts.append(f"Goals: {self.goals:.1f}")
        if self.assists > 0:
            parts.append(f"Assists: {self.assists:.1f}")
        if self.clean_sheet > 0:
            parts.append(f"CS: {self.clean_sheet:.1f}")
        if self.defensive_contrib > 0:
            parts.append(f"DC: {self.defensive_contrib:.1f}")
        if self.bonus > 0:
            parts.append(f"Bonus: {self.bonus:.1f}")
        if self.saves > 0:
            parts.append(f"Saves: {self.saves:.1f}")
        if self.cards < 0:
            parts.append(f"Cards: {self.cards:.1f}")
        if self.conceded < 0:
            parts.append(f"Conceded: {self.conceded:.1f}")
        return " | ".join(parts) if parts else "No data"


class ProjectionEngine:
    """
    Data-driven projection engine for FPL players.

    Uses xG-based methodology for accurate predictions:
    1. xG (expected goals) - THE most predictive stat for attackers
    2. xA (expected assists) - Key for creative players
    3. xGI per 90 - Normalized scoring potential
    4. Fixture difficulty adjustment
    5. Home/Away performance
    6. Minutes probability
    7. Team strength factors
    8. Clean sheet probability (for defenders)
    9. Regression to mean for over/underperformers

    The xG approach is what FPL Review and other paid services use.
    We get this data FREE from the FPL API!
    """

    # Default weights for different factors (can be overridden by adaptive system)
    DEFAULT_FORM_WEIGHT = 0.15      # Reduced - form is lagging indicator
    DEFAULT_ICT_WEIGHT = 0.10       # Reduced - less predictive than xG
    DEFAULT_FDR_WEIGHT = 0.20       # Keep moderate
    DEFAULT_CONSISTENCY_WEIGHT = 0.10
    DEFAULT_TEAM_STRENGTH_WEIGHT = 0.10
    DEFAULT_XG_WEIGHT = 0.35        # NEW - highest weight for xG!

    # Instance weights (can be customized)
    FORM_WEIGHT = DEFAULT_FORM_WEIGHT
    ICT_WEIGHT = DEFAULT_ICT_WEIGHT
    FDR_WEIGHT = DEFAULT_FDR_WEIGHT
    CONSISTENCY_WEIGHT = DEFAULT_CONSISTENCY_WEIGHT
    TEAM_STRENGTH_WEIGHT = DEFAULT_TEAM_STRENGTH_WEIGHT
    XG_WEIGHT = DEFAULT_XG_WEIGHT

    # FDR adjustment multipliers (difficulty 1-5)
    FDR_MULTIPLIERS = {
        1: 1.25,  # Very easy fixture
        2: 1.10,  # Easy fixture
        3: 1.00,  # Medium fixture
        4: 0.85,  # Hard fixture
        5: 0.70,  # Very hard fixture
    }

    # Home advantage multiplier
    HOME_ADVANTAGE = 1.08

    def __init__(
        self,
        players: list[Player],
        teams: list[Team],
        fixtures: list[Fixture],
        gameweeks: list[GameweekInfo] | None = None,
        use_adaptive_weights: bool = True,
        custom_weights: dict[str, float] | None = None,
    ):
        """
        Initialize the projection engine.

        Args:
            players: List of all players
            teams: List of all teams
            fixtures: List of all fixtures
            gameweeks: List of gameweek info (for blank/double detection)
            use_adaptive_weights: Whether to load optimized weights from adaptive system
            custom_weights: Optional dict of custom weight overrides
        """
        self.players = {p.id: p for p in players}
        self.teams = {t.id: t for t in teams}
        self.fixtures = fixtures
        self.gameweeks = {gw.id: gw for gw in (gameweeks or [])}

        # Load weights - priority: custom > adaptive > defaults
        if custom_weights:
            self.FORM_WEIGHT = custom_weights.get("form_weight", self.DEFAULT_FORM_WEIGHT)
            self.ICT_WEIGHT = custom_weights.get("ict_weight", self.DEFAULT_ICT_WEIGHT)
            self.FDR_WEIGHT = custom_weights.get("fdr_weight", self.DEFAULT_FDR_WEIGHT)
            self.CONSISTENCY_WEIGHT = custom_weights.get("consistency_weight", self.DEFAULT_CONSISTENCY_WEIGHT)
            self.TEAM_STRENGTH_WEIGHT = custom_weights.get("team_strength_weight", self.DEFAULT_TEAM_STRENGTH_WEIGHT)
            self.XG_WEIGHT = custom_weights.get("xg_weight", self.DEFAULT_XG_WEIGHT)
        elif use_adaptive_weights:
            try:
                from .adaptive import get_optimized_weights
                weights = get_optimized_weights()
                self.FORM_WEIGHT = weights.form_weight
                self.ICT_WEIGHT = weights.ict_weight
                self.FDR_WEIGHT = weights.fdr_weight
                self.CONSISTENCY_WEIGHT = weights.consistency_weight
                self.TEAM_STRENGTH_WEIGHT = weights.team_strength_weight
                # xG weight not in adaptive yet, use default
                self.XG_WEIGHT = getattr(weights, 'xg_weight', self.DEFAULT_XG_WEIGHT)
                logger.debug(f"Using adaptive weights: form={self.FORM_WEIGHT:.2f}, xg={self.XG_WEIGHT:.2f}")
            except Exception as e:
                logger.debug(f"Could not load adaptive weights, using defaults: {e}")

        # Pre-compute team strength metrics
        self._team_attack_strength: dict[int, float] = {}
        self._team_defense_strength: dict[int, float] = {}
        self._compute_team_strengths()

        # Initialize minutes predictor for sophisticated minutes estimation
        try:
            from .minutes import MinutesPredictor
            self._minutes_predictor = MinutesPredictor(players)
            logger.debug("Minutes predictor initialized for enhanced minutes estimation")
        except Exception as e:
            logger.debug(f"Minutes predictor not available: {e}")
            self._minutes_predictor = None

    def _compute_team_strengths(self) -> None:
        """Compute normalized attack/defense strength for each team."""
        for team_id, team in self.teams.items():
            # Normalize strength values (FPL uses ~1000-1400 range)
            avg_attack = (team.strength_attack_home + team.strength_attack_away) / 2
            avg_defense = (team.strength_defence_home + team.strength_defence_away) / 2

            # Normalize to 0.5-1.5 range (1.0 = average)
            self._team_attack_strength[team_id] = avg_attack / 1200 if avg_attack > 0 else 1.0
            self._team_defense_strength[team_id] = avg_defense / 1200 if avg_defense > 0 else 1.0

    # =========================================================================
    # Fallback methods for computed_field properties (backward compatibility)
    # These replicate the logic from Player model for older Player objects
    # =========================================================================

    def _get_equiv_90s(self, player: Player) -> float:
        """
        Calculate equivalent 90-minute appearances from total minutes.

        This is distinct from literal matches played - a player with 450 mins
        has equiv_90s=5.0 regardless of whether that was 5 full games or 10 cameos.
        """
        if player.minutes == 0:
            return 0.0
        return player.minutes / 90.0

    def _is_valid_derived_field(
        self,
        value: float | None,
        player: Player,
        field_name: str = "",
    ) -> bool:
        """
        Check if a derived field value is valid (not missing/corrupted).

        Args:
            value: The field value to check
            player: The player instance
            field_name: Name of the field (for field-specific suspicious-zero policy)

        Handles cases where:
        - Value is None (attribute missing)
        - Value is 0 but player has minutes AND zero is suspicious for this field

        Uses ZERO_IS_SUSPICIOUS dict for field-specific policy:
        - bonus_per_90, saves_per_90: zero is suspicious (active players should have data)
        - red_card_prob, yellow_card_prob: zero is legitimate (clean players exist)
        """
        if value is None:
            return False

        # Check field-specific suspicious-zero policy
        # Use <= 0.0 for resilience against weird loader behavior (-0.0, tiny negatives)
        if value <= 0.0 and player.minutes >= SUSPICIOUS_ZERO_MINUTES:
            # Only flag as suspicious if this field is in the "zero is suspicious" list
            zero_suspicious = ZERO_IS_SUSPICIOUS.get(field_name, False)
            if zero_suspicious:
                return False  # Suspicious - likely missing data

        return True

    def _clamp_derived_field(self, value: float, field_name: str) -> float:
        """Clamp a derived field value to its sanity bounds."""
        bounds = DERIVED_FIELD_BOUNDS.get(field_name)
        if bounds:
            min_val, max_val = bounds
            return max(min_val, min(max_val, value))
        return value

    def _calculate_bonus_per_90_fallback(self, player: Player) -> float:
        """Fallback calculation for bonus_per_90 with regression-to-mean."""
        bonus = getattr(player, 'bonus', 0) or 0
        equiv_90s = self._get_equiv_90s(player)

        if equiv_90s < 1.0:
            return 0.3  # League average when insufficient data

        raw_rate = bonus / equiv_90s
        # Regression to mean: Bayesian-style blend with prior
        # rate = (observed + prior_rate * prior_weight) / (equiv_90s + prior_weight)
        league_avg = 0.3
        prior_weight = 10.0  # Equivalent to 10 games of prior belief
        return (bonus + league_avg * prior_weight) / (equiv_90s + prior_weight)

    def _calculate_yellow_card_prob_fallback(self, player: Player) -> float:
        """Fallback calculation for yellow_card_prob with regression-to-mean."""
        yellow_cards = getattr(player, 'yellow_cards', 0) or 0
        equiv_90s = self._get_equiv_90s(player)

        if equiv_90s < 1.0:
            return 0.08  # League average ~8% per game

        # Bayesian regression to mean
        league_avg = 0.08
        prior_weight = 8.0  # Cards are rarer, less prior weight needed
        return (yellow_cards + league_avg * prior_weight) / (equiv_90s + prior_weight)

    def _calculate_red_card_prob_fallback(self, player: Player) -> float:
        """Fallback calculation for red_card_prob with regression-to-mean."""
        red_cards = getattr(player, 'red_cards', 0) or 0
        equiv_90s = self._get_equiv_90s(player)

        if equiv_90s < 1.0:
            return 0.005  # League average ~0.5% per game

        # Bayesian regression to mean (reds are very rare, strong prior)
        league_avg = 0.005
        prior_weight = 15.0  # Strong prior for rare events
        return (red_cards + league_avg * prior_weight) / (equiv_90s + prior_weight)

    def _calculate_saves_per_90_fallback(self, player: Player) -> float:
        """Fallback calculation for saves_per_90 (GK only)."""
        saves = getattr(player, 'saves', 0) or 0
        equiv_90s = self._get_equiv_90s(player)

        if equiv_90s < 1.0 or player.minutes == 0:
            return 3.0  # League average for GKs

        # No regression needed for saves - it's a counting stat, not a probability
        return saves / equiv_90s

    def _estimate_xg_per_90(self, player: Player) -> float:
        """
        Estimate xG per 90 based on price and position when data is missing.

        Historical FPL data suggests xG per 90 scales roughly:
        - FWDs: £15m Haaland ~0.85, £10m ~0.45, £7m ~0.30, £5m ~0.15
        - MIDs: £14m Salah ~0.55, £10m ~0.35, £7m ~0.20, £5m ~0.10
        - DEFs: £7m premium ~0.08, £5m ~0.04, £4m ~0.02
        - GKs: negligible
        """
        price = player.price

        if player.position == Position.FWD:
            # FWD xG scales strongly with price
            # Linear: xG = 0.06 * price - 0.15, clamped
            xg = 0.06 * price - 0.15
            return max(0.10, min(1.0, xg))

        elif player.position == Position.MID:
            # MID xG also scales with price but slightly lower
            # Linear: xG = 0.045 * price - 0.10
            xg = 0.045 * price - 0.10
            return max(0.05, min(0.70, xg))

        elif player.position == Position.DEF:
            # DEF xG is low, slight price scaling
            # Linear: xG = 0.015 * price - 0.02
            xg = 0.015 * price - 0.02
            return max(0.02, min(0.15, xg))

        else:  # GK
            return 0.005  # GKs rarely score

    def _estimate_xa_per_90(self, player: Player) -> float:
        """
        Estimate xA per 90 based on price and position when data is missing.

        Historical FPL data suggests xA per 90:
        - FWDs: £15m Haaland ~0.25, £10m ~0.18, £7m ~0.12
        - MIDs: £14m Salah ~0.35, £10m ~0.25, £7m ~0.15
        - DEFs: £7m premium ~0.15, £5m ~0.08, £4m ~0.04
        - GKs: negligible
        """
        price = player.price

        if player.position == Position.FWD:
            # FWD xA - moderate scaling
            # Linear: xA = 0.02 * price + 0.02
            xa = 0.02 * price + 0.02
            return max(0.08, min(0.40, xa))

        elif player.position == Position.MID:
            # MID xA - highest potential
            # Linear: xA = 0.03 * price + 0.05
            xa = 0.03 * price + 0.05
            return max(0.10, min(0.50, xa))

        elif player.position == Position.DEF:
            # DEF xA - moderate for attacking FBs
            # Linear: xA = 0.025 * price - 0.02
            xa = 0.025 * price - 0.02
            return max(0.03, min(0.20, xa))

        else:  # GK
            return 0.01  # GKs rarely assist

    def normalise_player(self, player: Player) -> DerivedPlayerStats:
        """
        Pre-compute all derived fields for a Player, returning a DerivedPlayerStats object.

        Call this at load time to ensure all derived fields exist before projection.
        This avoids repeated fallback calculations during projection loops.

        Example:
            derived = engine.normalise_player(player)
            bonus_per_90 = derived.bonus_per_90
        """
        # Helper to get valid value or fallback, then clamp to sanity bounds
        def get_or_fallback(field_name: str, fallback_fn) -> float:
            value = getattr(player, field_name, None)
            if self._is_valid_derived_field(value, player, field_name):
                result = value
            else:
                result = fallback_fn(player)
            # Apply sanity clamps to prevent corrupted data from breaking optimizer
            return self._clamp_derived_field(result, field_name)

        return DerivedPlayerStats(
            player_id=player.id,
            equiv_90s=self._get_equiv_90s(player),
            bonus_per_90=get_or_fallback('bonus_per_90', self._calculate_bonus_per_90_fallback),
            yellow_card_prob=get_or_fallback('yellow_card_prob', self._calculate_yellow_card_prob_fallback),
            red_card_prob=get_or_fallback('red_card_prob', self._calculate_red_card_prob_fallback),
            saves_per_90=(
                get_or_fallback('saves_per_90', self._calculate_saves_per_90_fallback)
                if player.position == Position.GK else 0.0
            ),
        )

    def _get_fixtures_for_player(
        self,
        player: Player,
        start_gw: int,
        end_gw: int,
    ) -> list[FixtureAnalysis]:
        """Get fixture analyses for a player's team in a gameweek range."""
        analyses = []

        for fixture in self.fixtures:
            if fixture.gameweek < start_gw or fixture.gameweek > end_gw:
                continue

            is_home = fixture.home_team_id == player.team_id
            is_away = fixture.away_team_id == player.team_id

            if not is_home and not is_away:
                continue

            opponent_id = fixture.away_team_id if is_home else fixture.home_team_id
            difficulty = fixture.home_difficulty if is_away else fixture.away_difficulty
            opponent = self.teams.get(opponent_id)

            # Calculate opponent strength factors
            opp_attack = self._team_attack_strength.get(opponent_id, 1.0)
            opp_defense = self._team_defense_strength.get(opponent_id, 1.0)

            # Clean sheet probability based on opponent attack strength
            cs_prob = max(0.1, min(0.6, 0.35 / opp_attack))

            # Scoring probability based on opponent defense strength
            score_prob = max(0.2, min(0.8, 0.5 / opp_defense))

            analyses.append(FixtureAnalysis(
                fixture=fixture,
                opponent_id=opponent_id,
                is_home=is_home,
                difficulty=difficulty,
                opponent_attack_strength=opp_attack,
                opponent_defense_strength=opp_defense,
                clean_sheet_probability=cs_prob,
                scoring_probability=score_prob,
            ))

        return analyses

    def _get_minutes_prediction(self, player: Player):
        """
        Get full minutes prediction for a player.

        Returns MinutesPrediction with p_start, p_60_plus, e_minutes.
        Falls back to basic estimation if predictor unavailable.
        """
        from .minutes import MinutesPrediction, RotationRisk

        # Use enhanced minutes model if available
        if self._minutes_predictor is not None:
            try:
                return self._minutes_predictor.predict_minutes(player)
            except Exception as e:
                logger.debug(f"Minutes prediction failed for {player.web_name}: {e}")
                # Fall through to basic calculation

        # Basic fallback calculation
        status_probs = {
            PlayerStatus.AVAILABLE: 0.95,
            PlayerStatus.DOUBTFUL: 0.50,
            PlayerStatus.INJURED: 0.10,
            PlayerStatus.SUSPENDED: 0.00,
            PlayerStatus.UNAVAILABLE: 0.05,
            PlayerStatus.NOT_AVAILABLE: 0.00,
        }
        p_start = status_probs.get(player.status, 0.5)

        # Use chance_of_playing if available (0-100)
        if player.chance_of_playing is not None:
            p_start = min(p_start, player.chance_of_playing / 100)

        # Estimate p_60_plus from historical minutes
        if player.minutes > 0:
            games_estimate = player.minutes / 90
            if games_estimate > 0:
                avg_mins = player.minutes / games_estimate
                if avg_mins >= 85:
                    p_60_given_start = 0.95
                elif avg_mins >= 70:
                    p_60_given_start = 0.85
                elif avg_mins >= 55:
                    p_60_given_start = 0.70
                else:
                    p_60_given_start = 0.50
            else:
                p_60_given_start = 0.80
        else:
            p_60_given_start = 0.80  # Default for unknown

        p_60_plus = p_start * p_60_given_start
        e_minutes = p_start * 75 + (1 - p_start) * 0.3 * 15  # Simplified

        return MinutesPrediction(
            player_id=player.id,
            player_name=player.web_name,
            p_start=p_start,
            p_60_plus=p_60_plus,
            p_90_plus=p_60_plus * 0.7,
            e_minutes=e_minutes,
            rotation_risk=RotationRisk.UNKNOWN,
            risk_factors=[],
            confidence=0.5,
            data_points=0,
        )

    def _calculate_minutes_probability(self, player: Player) -> float:
        """
        Calculate expected minutes fraction based on sophisticated minutes model.

        Uses MinutesPredictor for P(start), E[minutes], and rotation risk analysis.
        Returns E[minutes]/90 to properly weight projections.
        """
        prediction = self._get_minutes_prediction(player)
        return prediction.e_minutes / 90.0

    def _calculate_form_score(self, player: Player) -> float:
        """
        Calculate form-based score contribution.

        Form in FPL is average points over last ~5 games.
        """
        if player.form <= 0:
            return 0.0

        # Normalize form (typically 0-10 range)
        normalized = player.form / 5.0  # 5.0 = average good form

        # Apply diminishing returns at high form
        return min(2.0, normalized)

    def _calculate_ict_score(self, player: Player) -> float:
        """
        Calculate ICT index contribution.

        ICT Index combines Influence, Creativity, and Threat metrics.
        """
        if player.ict_index <= 0:
            return 0.0

        # ICT typically ranges 0-500+ for top players
        # Normalize to 0-2 range
        normalized = player.ict_index / 200.0

        return min(2.0, normalized)

    def _calculate_consistency_score(self, player: Player) -> float:
        """Calculate how consistently the player plays full games."""
        if player.minutes == 0:
            return 0.5  # Unknown, assume average

        # Estimate games from total points and PPG
        if player.points_per_game > 0:
            games = player.total_points / player.points_per_game
        else:
            games = player.minutes / 90

        if games < 1:
            return 0.5

        avg_mins = player.minutes / games

        # Score based on average minutes
        if avg_mins >= 85:
            return 1.0  # Very consistent
        elif avg_mins >= 70:
            return 0.85
        elif avg_mins >= 55:
            return 0.70
        else:
            return 0.55  # Rotation risk

    def _calculate_xg_score(self, player: Player) -> tuple[float, float, float]:
        """
        Calculate xG-based expected scoring contribution.

        This is THE key method for accurate predictions. xG is the best
        predictor of future scoring - better than form, ICT, or actual goals.

        Returns:
            Tuple of (expected_goals_contribution, expected_assists_contribution, xg_bonus)
        """
        # Get xG/xA per 90 (normalized for playing time)
        # Handle players loaded before xG fields were added
        xg_per_90 = getattr(player, 'expected_goals_per_90', 0.0) or 0.0
        xa_per_90 = getattr(player, 'expected_assists_per_90', 0.0) or 0.0
        expected_goals = getattr(player, 'expected_goals', 0.0) or 0.0
        expected_assists = getattr(player, 'expected_assists', 0.0) or 0.0

        # If we don't have per-90 stats, calculate from totals
        if xg_per_90 == 0 and expected_goals > 0 and player.minutes > 0:
            xg_per_90 = (expected_goals / player.minutes) * 90
        if xa_per_90 == 0 and expected_assists > 0 and player.minutes > 0:
            xa_per_90 = (expected_assists / player.minutes) * 90

        # Apply position-specific conversion rates
        goal_conversion = XG_GOAL_CONVERSION.get(player.position, 0.9)
        assist_conversion = XA_ASSIST_CONVERSION.get(player.position, 0.85)

        # Calculate expected goals and assists for one game
        expected_goals = xg_per_90 * goal_conversion
        expected_assists = xa_per_90 * assist_conversion

        # Apply regression adjustment for over/underperformers
        # If player has scored way more than their xG, expect regression
        if expected_goals > 0 and player.goals_scored > 0:
            goals_over_xg = player.goals_scored - expected_goals
            regression = goals_over_xg * REGRESSION_FACTOR * -0.1  # Slight negative for overperformers
            expected_goals = max(0, expected_goals + regression)

        # Convert to FPL points
        goal_points = GOAL_WEIGHTS.get(player.position, 4.0)
        goals_contribution = expected_goals * goal_points
        assists_contribution = expected_assists * ASSIST_POINTS

        # xGI bonus - high xGI players get more bonus points
        xgi_per_90 = xg_per_90 + xa_per_90
        xg_bonus = xgi_per_90 * XGI_BONUS_MULTIPLIER

        return goals_contribution, assists_contribution, xg_bonus

    def project_single_player(
        self,
        player: Player,
        gameweek: int,
    ) -> float:
        """
        Project expected points for a single player in a gameweek.

        Args:
            player: Player to project
            gameweek: Gameweek number

        Returns:
            Expected points
        """
        # Get fixtures for this gameweek
        fixture_analyses = self._get_fixtures_for_player(player, gameweek, gameweek)

        if not fixture_analyses:
            # Blank gameweek for this team
            return 0.0

        total_xp = 0.0

        for fix_analysis in fixture_analyses:
            # Get full minutes prediction (includes p_60_plus)
            mins_pred = self._get_minutes_prediction(player)
            mins_prob = mins_pred.e_minutes / 90.0  # E[minutes]/90
            p_plays = mins_pred.p_start  # P(plays any minutes)
            p_60_plus = mins_pred.p_60_plus  # P(plays 60+) - ACTUAL from model!

            # Form contribution (reduced weight - lagging indicator)
            form_score = self._calculate_form_score(player)

            # Consistency contribution
            consistency = self._calculate_consistency_score(player)

            # xG-BASED SCORING - THE KEY TO ACCURACY!
            xg_goals, xg_assists, xg_bonus = self._calculate_xg_score(player)

            # Fixture difficulty adjustment
            fdr_mult = self.FDR_MULTIPLIERS.get(fix_analysis.difficulty, 1.0)

            # Home advantage
            home_mult = self.HOME_ADVANTAGE if fix_analysis.is_home else 1.0

            # Team strength factor
            team_attack = self._team_attack_strength.get(player.team_id, 1.0)
            team_defense = self._team_defense_strength.get(player.team_id, 1.0)

            # Position-specific clean sheet contribution
            if player.position in [Position.GK, Position.DEF]:
                cs_xp = CLEAN_SHEET_WEIGHTS[player.position] * fix_analysis.clean_sheet_probability * team_defense
            elif player.position == Position.MID:
                cs_xp = CLEAN_SHEET_WEIGHTS[player.position] * fix_analysis.clean_sheet_probability * team_defense
            else:  # FWD
                cs_xp = 0.0

            # =================================================================
            # COMPONENT-BASED xP CALCULATION (Corrected)
            # Each component is in FPL points units, directly additive
            # =================================================================

            # Form as a small rate adjuster (±10% on conversion)
            form_modifier = 1.0 + (form_score - 0.5) * 0.2
            form_modifier = max(0.9, min(1.1, form_modifier))

            # =================================================================
            # COMPONENT BREAKDOWN (all in FPL points units)
            # =================================================================

            # 1. APPEARANCE POINTS - Using actual FPL scoring rules
            # FPL gives: 0 pts if 0 mins, 1 pt if 1-59 mins, 2 pts if 60+ mins
            # p_plays = P(plays any minutes), p_60_plus = P(plays 60+ minutes)
            appearance_pts = 1 * p_plays + 1 * p_60_plus  # 1pt for playing + 1pt bonus for 60+

            # 2. GOAL POINTS - xg_goals is ALREADY in points (xG × GOAL_WEIGHTS)
            # Apply fixture/team adjustments, scaled by minutes probability
            goal_pts = xg_goals * fdr_mult * team_attack * form_modifier * mins_prob

            # 3. ASSIST POINTS - xg_assists is ALREADY in points (xA × ASSIST_POINTS)
            assist_pts = xg_assists * fdr_mult * team_attack * form_modifier * mins_prob

            # 4. CLEAN SHEET POINTS - requires playing 60+ minutes
            cs_pts = cs_xp * p_60_plus

            # 5. DEFENSIVE CONTRIBUTION POINTS (2025/26)
            # CONSERVATIVE: Set to 0 until we have historical DC data
            dc_pts = 0.0

            # 6. BONUS POINTS - Use regressed bonus_per_90 from Player model
            # The Player.bonus_per_90 property already includes regression-to-mean
            # Use getattr + validity check for backward compatibility
            bonus_per_90 = getattr(player, 'bonus_per_90', None)
            if not self._is_valid_derived_field(bonus_per_90, player, 'bonus_per_90'):
                # Fallback calculation if computed_field not available or invalid
                bonus_per_90 = self._calculate_bonus_per_90_fallback(player)
            bonus_per_90 = self._clamp_derived_field(bonus_per_90, 'bonus_per_90')
            bonus_pts = bonus_per_90 * p_60_plus * fdr_mult

            # 7. NEGATIVE EVENTS - Cards (per-match events, not per-minute!)
            # Yellow cards: -1 pt, Red cards: -3 pts
            # Card probabilities are per-match (conditional on playing), so scale by P(plays)
            # Use getattr + validity check for backward compatibility
            yellow_prob = getattr(player, 'yellow_card_prob', None)
            if not self._is_valid_derived_field(yellow_prob, player, 'yellow_card_prob'):
                yellow_prob = self._calculate_yellow_card_prob_fallback(player)
            yellow_prob = self._clamp_derived_field(yellow_prob, 'yellow_card_prob')
            red_prob = getattr(player, 'red_card_prob', None)
            if not self._is_valid_derived_field(red_prob, player, 'red_card_prob'):
                red_prob = self._calculate_red_card_prob_fallback(player)
            red_prob = self._clamp_derived_field(red_prob, 'red_card_prob')
            cards_pts = -(yellow_prob * 1 + red_prob * 3) * p_plays

            # Opponent attack strength for negative calculations
            opp_attack = fix_analysis.opponent_attack_strength

            # 8. GOALS CONCEDED (GK/DEF only) - -1 pt per 2 goals conceded
            if player.position in [Position.GK, Position.DEF]:
                # Expected goals conceded based on:
                # - League average (1.3 goals/game per team)
                # - Our team's defensive strength (higher = fewer conceded)
                # - Opponent's attacking strength (higher = more conceded)
                # team_defense is normalized around 1.0 (from strength/1200)
                avg_league_conceded = 1.3
                # Invert team_defense: higher defense = fewer goals conceded
                defence_factor = 1.0 / max(0.7, team_defense)  # Clamp to avoid division issues
                expected_conceded = avg_league_conceded * defence_factor * opp_attack
                # -1 pt per 2 goals conceded, only if playing 60+
                conceded_pts = -(expected_conceded / 2) * p_60_plus
            else:
                conceded_pts = 0.0

            # 9. SAVES (GK only) - 1 pt per 3 saves
            if player.position == Position.GK:
                # Expected saves based on opponent attack strength
                # More opponent attack = more saves expected
                # Use getattr + validity check for backward compatibility
                saves_per_90 = getattr(player, 'saves_per_90', None)
                if not self._is_valid_derived_field(saves_per_90, player, 'saves_per_90'):
                    saves_per_90 = self._calculate_saves_per_90_fallback(player)
                saves_per_90 = self._clamp_derived_field(saves_per_90, 'saves_per_90')
                expected_saves = saves_per_90 * opp_attack
                saves_pts = (expected_saves / 3) * mins_prob
            else:
                saves_pts = 0.0

            # TOTAL: Sum of all components (no double-counting!)
            weighted_xp = (
                appearance_pts
                + goal_pts
                + assist_pts
                + cs_pts
                + dc_pts
                + bonus_pts
                + cards_pts
                + conceded_pts
                + saves_pts
            )

            # Apply home modifier
            weighted_xp *= home_mult * consistency

            total_xp += weighted_xp

        return round(total_xp, 2)

    def project_all_players(
        self,
        start_gw: int,
        end_gw: int,
    ) -> list[PlayerProjection]:
        """
        Project expected points for all players across gameweeks.

        Args:
            start_gw: Starting gameweek
            end_gw: Ending gameweek

        Returns:
            List of PlayerProjection objects
        """
        projections = []

        for gw in range(start_gw, end_gw + 1):
            logger.info(f"Projecting gameweek {gw}...")

            for player_id, player in self.players.items():
                try:
                    xp = self.project_single_player(player, gw)

                    projections.append(PlayerProjection(
                        player_id=player_id,
                        gameweek=gw,
                        expected_points=xp,
                        minutes_probability=self._calculate_minutes_probability(player),
                        source="internal",
                        updated_at=datetime.now(),
                    ))
                except Exception as e:
                    logger.warning(f"Error projecting player {player_id} for GW{gw}: {e}")

        logger.info(f"Generated {len(projections)} projections")
        return projections

    def get_top_picks(
        self,
        gameweek: int,
        position: Position | None = None,
        limit: int = 10,
    ) -> list[tuple[Player, float]]:
        """
        Get top projected players for a gameweek.

        Args:
            gameweek: Gameweek number
            position: Filter by position (optional)
            limit: Number of players to return

        Returns:
            List of (Player, expected_points) tuples
        """
        picks = []

        for player in self.players.values():
            if position and player.position != position:
                continue

            if not player.is_available:
                continue

            xp = self.project_single_player(player, gameweek)
            picks.append((player, xp))

        # Sort by expected points
        picks.sort(key=lambda x: x[1], reverse=True)

        return picks[:limit]

    def get_captain_picks(
        self,
        gameweek: int,
        limit: int = 5,
        owned_player_ids: set[int] | None = None,
        min_minutes: int = 180,
    ) -> list[tuple[Player, float]]:
        """
        Get top captain picks for a gameweek.

        Uses xGI (Expected Goal Involvement) as the PRIMARY factor.
        xGI per 90 is the BEST predictor of who will score/assist.

        Args:
            gameweek: Gameweek number
            limit: Number of picks to return
            owned_player_ids: If provided, only consider these players (your squad)
            min_minutes: Minimum minutes played to be considered (avoids inflated per-90 stats)

        Returns:
            List of (Player, expected_points) tuples
        """
        picks = []

        for player in self.players.values():
            # Filter to owned players if specified
            if owned_player_ids is not None and player.id not in owned_player_ids:
                continue

            # Captain picks prioritize MID/FWD (but allow high-xG DEF like Trent)
            if player.position == Position.GK:
                continue

            if not player.is_available:
                continue

            # Filter out players with very few minutes (inflated per-90 stats)
            # Use 180 min (2 full games) as minimum to have meaningful data
            if player.minutes < min_minutes:
                continue

            # Get base projection
            xp = self.project_single_player(player, gameweek)

            # xGI-BASED CAPTAIN SCORING - THE KEY TO ACCURACY!
            # Handle players without xG data (loaded before update)
            xg_per_90 = getattr(player, 'expected_goals_per_90', 0.0) or 0.0
            xa_per_90 = getattr(player, 'expected_assists_per_90', 0.0) or 0.0
            xgi_per_90 = xg_per_90 + xa_per_90

            # Get fixture for this gameweek
            fixture_analyses = self._get_fixtures_for_player(player, gameweek, gameweek)

            # Captain bonus based on xGI and fixture
            if xgi_per_90 > 0:
                # High xGI = high ceiling player, great captain pick
                xgi_bonus = xgi_per_90 * 1.5  # 1.5x multiplier for captain value

                # Adjust for fixture difficulty
                if fixture_analyses:
                    avg_fdr = sum(f.difficulty for f in fixture_analyses) / len(fixture_analyses)
                    # Easy fixtures (1-2 FDR) boost captain value
                    if avg_fdr <= 2:
                        xgi_bonus *= 1.3
                    elif avg_fdr >= 4:
                        xgi_bonus *= 0.8

                    # Home games boost
                    if fixture_analyses[0].is_home:
                        xgi_bonus *= 1.1

                # Double gameweek = double captaincy value
                if len(fixture_analyses) >= 2:
                    xgi_bonus *= 1.8  # Massive boost for DGW

                captain_score = xp + xgi_bonus
            else:
                # No xG data - fall back to form/ICT
                ict_bonus = min(0.5, player.ict_index / 400)
                captain_score = xp * (1 + ict_bonus)

            # Defenders can only be captain picks if xGI is very high (like Trent)
            if player.position == Position.DEF and xgi_per_90 < 0.3:
                continue

            picks.append((player, captain_score))

        picks.sort(key=lambda x: x[1], reverse=True)

        return picks[:limit]

    def get_simulation_params(
        self,
        player: Player,
        gameweek: int,
    ):
        """
        Get simulation parameters for a player in a specific gameweek.

        This bridges the projection engine's calculations with the EventSimulator.

        Args:
            player: Player to get params for
            gameweek: Gameweek number

        Returns:
            SimulationParams for use with EventSimulator
        """
        from .simulation import SimulationParams

        # Get minutes prediction
        mins_pred = self._get_minutes_prediction(player)
        p_start = mins_pred.p_start
        p_60_plus = mins_pred.p_60_plus
        e_minutes = mins_pred.e_minutes
        p_plays = p_start + (1 - p_start) * 0.2  # Rough sub probability

        # Get fixture analysis
        fixture_analyses = self._get_fixtures_for_player(player, gameweek, gameweek)
        if not fixture_analyses:
            # No fixture - return minimal params
            return SimulationParams(
                player_id=player.id,
                position=player.position,
                p_start=0.0,
                p_sub=0.0,
                e_minutes_if_start=0,
                e_minutes_if_sub=0,
                lambda_goals=0.0,
                lambda_assists=0.0,
                p_clean_sheet=0.0,
                lambda_goals_conceded=0.0,
                lambda_saves=0.0,
                p_yellow=0.0,
                p_red=0.0,
                base_bonus=0.0,
            )

        fix_analysis = fixture_analyses[0]

        # Calculate fixture multiplier
        fdr_mult = self.FDR_MULTIPLIERS.get(fix_analysis.difficulty, 1.0)
        if fix_analysis.is_home:
            fdr_mult *= self.HOME_ADVANTAGE

        # Team strengths
        team_attack = self._team_attack_strength.get(player.team_id, 1.0)
        opp_attack = fix_analysis.opponent_attack_strength

        # Get derived fields with fallbacks
        xg_per_90 = getattr(player, 'xg_per_90', 0.0) or 0.0
        xa_per_90 = getattr(player, 'xa_per_90', 0.0) or 0.0

        # CRITICAL: If xG/xA data is missing, use price-based estimates
        # Otherwise simulation produces 0% P(haul) for all players
        if xg_per_90 <= 0.01:
            xg_per_90 = self._estimate_xg_per_90(player)
        if xa_per_90 <= 0.01:
            xa_per_90 = self._estimate_xa_per_90(player)

        yellow_prob = getattr(player, 'yellow_card_prob', None)
        if not self._is_valid_derived_field(yellow_prob, player, 'yellow_card_prob'):
            yellow_prob = self._calculate_yellow_card_prob_fallback(player)
        yellow_prob = self._clamp_derived_field(yellow_prob, 'yellow_card_prob')

        red_prob = getattr(player, 'red_card_prob', None)
        if not self._is_valid_derived_field(red_prob, player, 'red_card_prob'):
            red_prob = self._calculate_red_card_prob_fallback(player)
        red_prob = self._clamp_derived_field(red_prob, 'red_card_prob')

        saves_per_90 = getattr(player, 'saves_per_90', None)
        if not self._is_valid_derived_field(saves_per_90, player, 'saves_per_90'):
            saves_per_90 = self._calculate_saves_per_90_fallback(player)
        saves_per_90 = self._clamp_derived_field(saves_per_90, 'saves_per_90')

        bonus_per_90 = getattr(player, 'bonus_per_90', None)
        if not self._is_valid_derived_field(bonus_per_90, player, 'bonus_per_90'):
            bonus_per_90 = self._calculate_bonus_per_90_fallback(player)
        bonus_per_90 = self._clamp_derived_field(bonus_per_90, 'bonus_per_90')

        # Clean sheet probability
        team_defense = self._team_defense_strength.get(player.team_id, 1.0)
        p_cs = fix_analysis.clean_sheet_probability * team_defense

        # Goals conceded (for GK/DEF)
        avg_league_conceded = 1.3
        defence_factor = 1.0 / max(0.7, team_defense)
        lambda_goals_conceded = avg_league_conceded * defence_factor * opp_attack

        # Calculate minutes breakdown
        if p_start > 0.8:
            e_mins_if_start = min(90, e_minutes / p_start) if p_start > 0 else 85
            e_mins_if_sub = 20
            p_sub = 0.1
        elif p_start > 0.3:
            e_mins_if_start = 80
            e_mins_if_sub = 25
            p_sub = max(0.0, min(0.8, (p_60_plus - p_start * 0.9) / 0.5))
        else:
            e_mins_if_start = 75
            e_mins_if_sub = max(15, e_minutes / max(0.1, 1 - p_start))
            p_sub = max(0.1, p_60_plus - p_start)

        # Apply fixture multiplier to attacking rates
        fixture_mult = fdr_mult * team_attack
        lambda_goals = xg_per_90 * fixture_mult
        lambda_assists = xa_per_90 * fixture_mult

        # Adjust saves by opponent attack
        lambda_saves = saves_per_90 * opp_attack if player.position == Position.GK else 0.0

        return SimulationParams(
            player_id=player.id,
            position=player.position,
            p_start=p_start,
            p_sub=p_sub,
            e_minutes_if_start=e_mins_if_start,
            e_minutes_if_sub=e_mins_if_sub,
            lambda_goals=lambda_goals,
            lambda_assists=lambda_assists,
            p_clean_sheet=p_cs,
            lambda_goals_conceded=lambda_goals_conceded,
            lambda_saves=lambda_saves,
            p_yellow=yellow_prob * p_plays,  # Scale by P(plays)
            p_red=red_prob * p_plays,
            base_bonus=bonus_per_90,
        )

    def get_enhanced_projections(
        self,
        gameweek: int,
        player_ids: list[int] | None = None,
        n_sims: int = 5000,
    ) -> list[dict]:
        """
        Get projections with full simulation metrics for specified players.

        This is the RECOMMENDED way to get projections - includes:
        - Expected points (xP)
        - P(haul) - probability of 10+ points
        - P(blank) - probability of 2 or fewer points
        - Ceiling (90th percentile)
        - Floor (10th percentile)

        Args:
            gameweek: Gameweek number
            player_ids: List of player IDs (None = all players)
            n_sims: Number of simulations per player (default 5000 for speed)

        Returns:
            List of dicts with player data and simulation metrics
        """
        from .simulation import EventSimulator

        simulator = EventSimulator()
        results = []

        target_players = (
            [self.players[pid] for pid in player_ids if pid in self.players]
            if player_ids
            else list(self.players.values())
        )

        for player in target_players:
            if not player.is_available:
                continue

            try:
                # Get base projection
                xp = self.project_single_player(player, gameweek)

                # Get simulation params and run simulation
                params = self.get_simulation_params(player, gameweek)
                sim_result = simulator.simulate_player(params, n_sims=n_sims)

                results.append({
                    "player_id": player.id,
                    "player": player,
                    "xp": xp,
                    "sim_xp": sim_result.expected_points,
                    "std_dev": sim_result.std_dev,
                    "p_haul": sim_result.p_haul,
                    "p_returns": sim_result.p_returns,
                    "p_blank": sim_result.p_blank,
                    "floor": sim_result.median_points - sim_result.std_dev * 1.28,  # ~10th percentile
                    "ceiling": sim_result.percentile_90,
                    "ceiling_95": sim_result.percentile_95,
                    "ceiling_score": sim_result.ceiling_score,
                })

            except Exception as e:
                logger.warning(f"Error projecting player {player.id}: {e}")
                # Add basic projection without simulation
                try:
                    xp = self.project_single_player(player, gameweek)
                    results.append({
                        "player_id": player.id,
                        "player": player,
                        "xp": xp,
                        "sim_xp": xp,
                        "std_dev": 0.0,
                        "p_haul": 0.0,
                        "p_returns": 0.0,
                        "p_blank": 0.0,
                        "floor": xp * 0.5,
                        "ceiling": xp * 1.5,
                        "ceiling_95": xp * 2.0,
                        "ceiling_score": xp,
                    })
                except Exception:
                    pass

        return results

    def get_top_xgi_players(
        self,
        position: Position | None = None,
        limit: int = 10,
    ) -> list[tuple[Player, float]]:
        """
        Get players with highest xGI per 90 (expected goal involvement).

        This is the BEST metric for identifying scoring potential.
        These are your captain options and transfer targets.

        Args:
            position: Filter by position (optional)
            limit: Number of players to return

        Returns:
            List of (Player, xGI_per_90) tuples
        """
        picks = []

        for player in self.players.values():
            if position and player.position != position:
                continue

            if not player.is_available:
                continue

            if player.minutes < 180:  # Need at least 2 games of data
                continue

            # Handle players without xG data
            xg_per_90 = getattr(player, 'expected_goals_per_90', 0.0) or 0.0
            xa_per_90 = getattr(player, 'expected_assists_per_90', 0.0) or 0.0
            xgi_per_90 = xg_per_90 + xa_per_90

            picks.append((player, xgi_per_90))

        picks.sort(key=lambda x: x[1], reverse=True)

        return picks[:limit]

    def get_underperformers(
        self,
        limit: int = 10,
    ) -> list[tuple[Player, float, float]]:
        """
        Get players underperforming their xG (due for goals).

        These are BUY candidates - statistical regression suggests
        they will score more in the future.

        Returns:
            List of (Player, xG, actual_goals) tuples
        """
        candidates = []

        for player in self.players.values():
            if not player.is_available:
                continue

            if player.position == Position.GK:
                continue

            # Handle players without xG data
            player_xg = getattr(player, 'expected_goals', 0.0) or 0.0

            if player_xg < 1.5:  # Need meaningful sample
                continue

            # Underperformer = xG significantly higher than actual goals
            xg_diff = player_xg - player.goals_scored

            if xg_diff >= 0.5:  # At least 0.5 goals "owed"
                candidates.append((player, player_xg, player.goals_scored, xg_diff))

        # Sort by most underperforming
        candidates.sort(key=lambda x: x[3], reverse=True)

        return [(p, xg, g) for p, xg, g, _ in candidates[:limit]]

    def get_overperformers(
        self,
        limit: int = 10,
    ) -> list[tuple[Player, float, float]]:
        """
        Get players overperforming their xG (due for regression).

        These are SELL candidates - statistical regression suggests
        their scoring rate will drop.

        Returns:
            List of (Player, xG, actual_goals) tuples
        """
        candidates = []

        for player in self.players.values():
            if not player.is_available:
                continue

            if player.position == Position.GK:
                continue

            # Handle players without xG data
            player_xg = getattr(player, 'expected_goals', 0.0) or 0.0

            if player_xg < 1.0:  # Need meaningful sample
                continue

            # Overperformer = actual goals significantly higher than xG
            xg_diff = player.goals_scored - player_xg

            if xg_diff >= 1.0:  # At least 1 goal over xG
                candidates.append((player, player_xg, player.goals_scored, xg_diff))

        # Sort by most overperforming
        candidates.sort(key=lambda x: x[3], reverse=True)

        return [(p, xg, g) for p, xg, g, _ in candidates[:limit]]

    def get_differential_picks(
        self,
        gameweek: int,
        max_ownership: float = 10.0,
        limit: int = 10,
        min_minutes: int = 180,
    ) -> list[tuple[Player, float]]:
        """
        Get high-value differential picks (low ownership, high expected).

        Args:
            gameweek: Gameweek number
            max_ownership: Maximum ownership percentage
            limit: Number of picks to return
            min_minutes: Minimum minutes played to be considered

        Returns:
            List of (Player, expected_points) tuples
        """
        picks = []

        for player in self.players.values():
            if player.selected_by_percent > max_ownership:
                continue

            if not player.is_available:
                continue

            # Filter out players with very few minutes (need meaningful data)
            if player.minutes < min_minutes:
                continue

            xp = self.project_single_player(player, gameweek)

            # Value score = xp per million
            value = xp / max(player.price, 4.0)

            picks.append((player, xp, value))

        # Sort by value score
        picks.sort(key=lambda x: x[2], reverse=True)

        return [(p, xp) for p, xp, _ in picks[:limit]]

    def get_differential_captain_value(
        self,
        gameweek: int,
        db=None,
        league_position: str = "mid",
        risk_tolerance: str = "medium",
        owned_player_ids: set[int] | None = None,
    ):
        """
        Get captain recommendation with effective ownership (EO) analysis.

        This is the key method for differential captaincy decisions.
        Analyzes whether to go with the template captain or pick a differential.

        Args:
            gameweek: Gameweek number
            db: Database instance (for ownership tracking)
            league_position: "leading", "chasing", or "mid"
            risk_tolerance: "low", "medium", or "high"
            owned_player_ids: Set of player IDs you own (captain must be from your squad!)

        Returns:
            DifferentialRecommendation with full analysis
        """
        from ..analysis.differentials import (
            CaptainDifferentialAnalyzer,
            DifferentialRecommendation,
        )
        from ..data.ownership import OwnershipTracker

        # Get captain candidates (MID/FWD with high projections) - ONLY from owned players!
        captain_picks = self.get_captain_picks(gameweek, limit=10, owned_player_ids=owned_player_ids)

        if not captain_picks:
            raise ValueError("No captain candidates available")

        if db is None:
            # Return basic recommendation without EO analysis
            best_pick = captain_picks[0]
            from ..analysis.differentials import CaptainStrategy
            return DifferentialRecommendation(
                recommended_captain=best_pick[0],
                recommended_strategy=CaptainStrategy.TEMPLATE,
                template_captain=best_pick[0],
                template_captain_eo=best_pick[0].selected_by_percent,
                template_captain_xp=best_pick[1],
                differential_captain=captain_picks[1][0] if len(captain_picks) > 1 else None,
                differential_captain_eo=captain_picks[1][0].selected_by_percent if len(captain_picks) > 1 else 0,
                differential_captain_xp=captain_picks[1][1] if len(captain_picks) > 1 else 0,
                expected_rank_gain_template=0,
                expected_rank_gain_differential=0,
                reasoning=f"{best_pick[0].web_name} has highest projected points ({best_pick[1]:.1f} xP)",
                confidence=0.7,
            )

        # Use ownership tracker for EO analysis
        tracker = OwnershipTracker(db)
        analyzer = CaptainDifferentialAnalyzer(tracker)

        return analyzer.get_recommendation(
            candidates=captain_picks,
            gameweek=gameweek,
            league_position=league_position,
            risk_tolerance=risk_tolerance,
        )

    def get_captain_picks_with_eo(
        self,
        gameweek: int,
        db=None,
        limit: int = 5,
        owned_player_ids: set[int] | None = None,
    ) -> list[dict]:
        """
        Get captain picks with effective ownership data.

        Returns a list of captain options with EO analysis for each.

        Args:
            gameweek: Gameweek number
            db: Database instance (for ownership tracking)
            limit: Number of picks to return
            owned_player_ids: Set of player IDs you own (captain must be from your squad!)

        Returns:
            List of dicts with player, xP, captain_eo, regular_eo, differential_value
        """
        captain_picks = self.get_captain_picks(gameweek, limit=limit, owned_player_ids=owned_player_ids)

        if not captain_picks:
            return []

        results = []

        if db is not None:
            from ..data.ownership import OwnershipTracker
            tracker = OwnershipTracker(db)

            for player, xp in captain_picks:
                captain_eo = tracker.estimate_captain_eo(player, gameweek)

                results.append({
                    "player": player,
                    "expected_points": xp,
                    "captain_eo": captain_eo.overall_eo,
                    "regular_eo": captain_eo.regular_ownership,
                    "captain_delta": captain_eo.captain_delta,
                    "is_template": captain_eo.overall_eo >= 30.0,
                    "is_differential": captain_eo.overall_eo < 15.0,
                })
        else:
            # Without DB, use basic ownership data
            for player, xp in captain_picks:
                # Estimate captain EO as ~1.5x regular ownership for popular picks
                estimated_captain_eo = min(90.0, player.selected_by_percent * 1.5)

                results.append({
                    "player": player,
                    "expected_points": xp,
                    "captain_eo": estimated_captain_eo,
                    "regular_eo": player.selected_by_percent,
                    "captain_delta": estimated_captain_eo - player.selected_by_percent,
                    "is_template": estimated_captain_eo >= 30.0,
                    "is_differential": estimated_captain_eo < 15.0,
                })

        return results


# =============================================================================
# Convenience Functions
# =============================================================================

def generate_projections(
    players: list[Player],
    teams: list[Team],
    fixtures: list[Fixture],
    gameweeks: list[GameweekInfo],
    start_gw: int,
    horizon: int = 5,
) -> list[PlayerProjection]:
    """
    Generate projections for all players.

    Args:
        players: List of players
        teams: List of teams
        fixtures: List of fixtures
        gameweeks: List of gameweek info
        start_gw: Starting gameweek
        horizon: Number of weeks to project

    Returns:
        List of PlayerProjection objects
    """
    engine = ProjectionEngine(players, teams, fixtures, gameweeks)
    return engine.project_all_players(start_gw, start_gw + horizon - 1)


def get_best_xi(
    players: list[Player],
    teams: list[Team],
    fixtures: list[Fixture],
    gameweek: int,
) -> list[tuple[Player, float]]:
    """
    Get the best projected XI for a gameweek.

    Returns 11 players with valid formation (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD).
    """
    engine = ProjectionEngine(players, teams, fixtures)

    # Get top players by position
    gks = engine.get_top_picks(gameweek, Position.GK, 2)
    defs = engine.get_top_picks(gameweek, Position.DEF, 5)
    mids = engine.get_top_picks(gameweek, Position.MID, 5)
    fwds = engine.get_top_picks(gameweek, Position.FWD, 3)

    # Build XI with 1 GK, 4 DEF, 4 MID, 2 FWD (balanced formation)
    xi = []
    xi.extend(gks[:1])
    xi.extend(defs[:4])
    xi.extend(mids[:4])
    xi.extend(fwds[:2])

    # Sort by expected points
    xi.sort(key=lambda x: x[1], reverse=True)

    return xi
