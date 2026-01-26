"""
Set Piece Integration for FPL Projections.

Uses FPL's set piece notes to boost penalty and free kick takers.
Penalty takers have significantly higher xG, FK takers have higher xA.

Based on research:
- Penalty conversion rate: ~75-80%
- FK direct goals: ~5% conversion
- FK assists (corners, indirect): ~2-3% per attempt
"""

import logging
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class SetPieceRole(StrEnum):
    """Set piece responsibility types."""
    PENALTY = "penalty"
    FREE_KICK_DIRECT = "fk_direct"      # Direct FK shots
    FREE_KICK_INDIRECT = "fk_indirect"  # Crosses/corners
    CORNER = "corner"


@dataclass
class SetPieceBonus:
    """xP bonus for set piece responsibility."""
    player_id: int
    player_name: str
    team_id: int

    # Roles
    is_penalty_taker: bool
    is_fk_taker: bool
    is_corner_taker: bool

    # Calculated bonuses per game
    penalty_xg_boost: float     # Extra xG from penalties
    fk_xg_boost: float          # Extra xG from direct FKs
    fk_xa_boost: float          # Extra xA from indirect FKs/corners

    @property
    def total_xgi_boost(self) -> float:
        """Total xGI boost from set pieces."""
        return self.penalty_xg_boost + self.fk_xg_boost + self.fk_xa_boost

    @property
    def total_xp_boost(self) -> float:
        """
        Estimated FPL points boost from set pieces.

        Penalties: ~3 pts per game on average (0.5 pens/game * 0.75 conv * 4-6 pts)
        Direct FKs: ~0.3 pts (rare)
        Corners/Indirect: ~0.5 pts (consistent assists)
        """
        # Assume MID/FWD (5/4 pts per goal, 3 pts per assist)
        goal_pts = 4.5  # Average between MID (5) and FWD (4)
        assist_pts = 3

        return (
            self.penalty_xg_boost * goal_pts
            + self.fk_xg_boost * goal_pts
            + self.fk_xa_boost * assist_pts
        )


# Expected penalties per game per team (PL average ~0.3 per team per game)
PENALTIES_PER_TEAM_PER_GAME = 0.3
PENALTY_CONVERSION_RATE = 0.76

# Free kicks per game estimates
DIRECT_FK_SHOTS_PER_GAME = 0.5  # Per taker
DIRECT_FK_CONVERSION = 0.05

# Corner/indirect FK assists
CORNERS_PER_GAME = 5.0  # Per team
CORNER_ASSIST_RATE = 0.03


class SetPieceAnalyzer:
    """
    Analyzes set piece responsibilities and calculates xP boosts.

    Uses FPL's set piece notes to identify key players.
    """

    def __init__(self, set_piece_notes: list[dict] | None = None):
        """
        Initialize analyzer.

        Args:
            set_piece_notes: Raw set piece notes from FPL API
        """
        self._notes = set_piece_notes or []
        self._player_roles: dict[int, SetPieceBonus] = {}
        self._parse_notes()

    def _parse_notes(self) -> None:
        """Parse set piece notes into player roles."""
        if not self._notes:
            logger.debug("No set piece notes available")
            return

        for note in self._notes:
            team_id = note.get("team")
            info = note.get("info_message", "")
            external_link = note.get("external_link", "")

            # Extract player IDs from the note
            # Notes format varies, this is a best-effort parse
            # TODO: Improve parsing based on actual API response format

            # For now, we'll use the element IDs if available
            # The actual structure needs to be verified from live API

            logger.debug(f"Team {team_id}: {info}")

    def get_bonus_for_player(self, player_id: int) -> SetPieceBonus | None:
        """Get set piece bonus for a player."""
        return self._player_roles.get(player_id)

    def calculate_penalty_taker_bonus(
        self,
        player_id: int,
        player_name: str,
        team_id: int,
        is_primary: bool = True,
    ) -> SetPieceBonus:
        """
        Calculate bonus for penalty taker.

        Args:
            player_id: Player FPL ID
            player_name: Player web name
            team_id: Team ID
            is_primary: Whether they're the primary taker

        Returns:
            SetPieceBonus with calculated values
        """
        # Primary takers get full bonus, secondary get reduced
        multiplier = 1.0 if is_primary else 0.3

        penalty_xg = PENALTIES_PER_TEAM_PER_GAME * PENALTY_CONVERSION_RATE * multiplier

        return SetPieceBonus(
            player_id=player_id,
            player_name=player_name,
            team_id=team_id,
            is_penalty_taker=True,
            is_fk_taker=False,
            is_corner_taker=False,
            penalty_xg_boost=round(penalty_xg, 3),
            fk_xg_boost=0.0,
            fk_xa_boost=0.0,
        )

    def calculate_fk_taker_bonus(
        self,
        player_id: int,
        player_name: str,
        team_id: int,
        is_direct: bool = True,
        is_corner_taker: bool = False,
    ) -> SetPieceBonus:
        """
        Calculate bonus for free kick taker.

        Args:
            player_id: Player FPL ID
            player_name: Player web name
            team_id: Team ID
            is_direct: Whether they take direct FKs
            is_corner_taker: Whether they take corners

        Returns:
            SetPieceBonus with calculated values
        """
        fk_xg = DIRECT_FK_SHOTS_PER_GAME * DIRECT_FK_CONVERSION if is_direct else 0.0
        fk_xa = CORNERS_PER_GAME * CORNER_ASSIST_RATE if is_corner_taker else 0.0

        return SetPieceBonus(
            player_id=player_id,
            player_name=player_name,
            team_id=team_id,
            is_penalty_taker=False,
            is_fk_taker=is_direct,
            is_corner_taker=is_corner_taker,
            penalty_xg_boost=0.0,
            fk_xg_boost=round(fk_xg, 3),
            fk_xa_boost=round(fk_xa, 3),
        )


# Known penalty takers for 2025/26 (hardcoded fallback)
# Updated periodically based on actual takers
KNOWN_PENALTY_TAKERS_2025_26 = {
    # Format: player_web_name: priority (1 = primary, 2 = backup)
    # Arsenal
    "Saka": 1,
    # Aston Villa
    "Watkins": 1,
    # Bournemouth
    "Kluivert": 1,
    # Brentford
    "Mbeumo": 1,
    # Brighton
    "Joao Pedro": 1,
    # Chelsea
    "Palmer": 1,
    # Crystal Palace
    "Eze": 1,
    # Everton
    "Calvert-Lewin": 1,
    # Fulham
    "Andreas": 1,
    # Ipswich
    "Hutchinson": 1,
    # Leicester
    "Vardy": 1,
    # Liverpool
    "M.Salah": 1,
    # Man City
    "Haaland": 1,
    # Man United
    "B.Fernandes": 1,
    # Newcastle
    "Isak": 1,
    # Nottm Forest
    "Wood": 1,
    # Southampton
    "Aribo": 1,
    # Spurs
    "Son": 1,
    # West Ham
    "Bowen": 1,
    # Wolves
    "Neto": 1,
}


def get_penalty_taker_boost(player_web_name: str) -> float:
    """
    Get xP boost for known penalty taker.

    Args:
        player_web_name: Player's web name

    Returns:
        xP boost (typically 0.5-1.0 pts per game)
    """
    priority = KNOWN_PENALTY_TAKERS_2025_26.get(player_web_name)
    if priority is None:
        return 0.0

    # Primary takers: ~0.8 xP boost
    # Secondary takers: ~0.2 xP boost
    if priority == 1:
        return PENALTIES_PER_TEAM_PER_GAME * PENALTY_CONVERSION_RATE * 4.5  # ~1.0 xP
    else:
        return PENALTIES_PER_TEAM_PER_GAME * PENALTY_CONVERSION_RATE * 4.5 * 0.3  # ~0.3 xP


def enhance_projection_with_set_pieces(
    base_xp: float,
    player_web_name: str,
) -> tuple[float, str | None]:
    """
    Enhance a player's projection with set piece bonus.

    Args:
        base_xp: Base expected points
        player_web_name: Player's web name

    Returns:
        Tuple of (enhanced_xp, reason_if_boosted)
    """
    boost = get_penalty_taker_boost(player_web_name)

    if boost > 0:
        return base_xp + boost, "Penalty taker (+{:.1f} xP)".format(boost)

    return base_xp, None
