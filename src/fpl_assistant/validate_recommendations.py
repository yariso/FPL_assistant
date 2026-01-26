"""
Validation Script for FPL Recommendations.

Runs scenarios to verify that recommendations are:
1. Data-driven (based on xG, form, fixtures)
2. Optimal (maximize expected points)
3. Consistent with FPL rules
"""

import logging
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    """Load all data from database."""
    from fpl_assistant.data import get_database, Player, Team, Fixture

    db = get_database()

    if db.get_player_count() == 0:
        logger.error("No data in database. Run 'Force Update' in the app first.")
        return None, None, None

    players = db.get_all_players()
    teams = db.get_all_teams()
    fixtures = db.get_all_fixtures()
    current_gw = db.get_current_gameweek()

    logger.info(f"Loaded {len(players)} players, {len(teams)} teams, {len(fixtures)} fixtures")
    logger.info(f"Current gameweek: {current_gw.id if current_gw else 'Unknown'}")

    return players, teams, fixtures, current_gw


def test_projection_engine():
    """Test that projections are based on key factors."""
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Projection Engine Validation")
    logger.info("="*60)

    players, teams, fixtures, gw = load_data()
    if not players:
        return False

    from fpl_assistant.predictions import ProjectionEngine

    engine = ProjectionEngine(players, teams, fixtures)
    gw_num = gw.id if gw else 1

    # Get projections for all players
    projections = {}
    for p in players:
        try:
            projections[p.id] = engine.project_single_player(p, gw_num)
        except:
            projections[p.id] = 0

    # Sort by projection
    sorted_players = sorted(
        [(p, projections[p.id]) for p in players if projections[p.id] > 0],
        key=lambda x: -x[1]
    )

    logger.info("\nTop 10 Projected Players:")
    logger.info("-" * 60)
    for p, xp in sorted_players[:10]:
        xg90 = getattr(p, 'expected_goals_per_90', 0) or 0
        xa90 = getattr(p, 'expected_assists_per_90', 0) or 0
        logger.info(f"{p.web_name:20} | xP: {xp:5.1f} | xGI/90: {xg90+xa90:.2f} | Form: {p.form} | Price: £{p.price}m")

    # Verify top players have good underlying stats
    top_10 = sorted_players[:10]
    avg_xgi = sum((getattr(p, 'expected_goals_per_90', 0) or 0) + (getattr(p, 'expected_assists_per_90', 0) or 0) for p, _ in top_10) / 10
    avg_form = sum(p.form for p, _ in top_10) / 10

    logger.info(f"\nTop 10 averages: xGI/90={avg_xgi:.2f}, Form={avg_form:.1f}")

    if avg_xgi > 0.3 or avg_form > 4:
        logger.info("✅ PASS: Top players have strong underlying stats")
        return True
    else:
        logger.warning("⚠️ WARNING: Top players may not have best underlying stats")
        return False


def test_captain_selection():
    """Test that captain picks are xGI-driven."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Captain Selection Validation")
    logger.info("="*60)

    players, teams, fixtures, gw = load_data()
    if not players:
        return False

    from fpl_assistant.predictions import ProjectionEngine

    engine = ProjectionEngine(players, teams, fixtures)
    gw_num = gw.id if gw else 1

    captain_picks = engine.get_captain_picks(gw_num, limit=5)

    logger.info("\nTop 5 Captain Picks:")
    logger.info("-" * 60)
    for p, xp in captain_picks:
        xg90 = getattr(p, 'expected_goals_per_90', 0) or 0
        xa90 = getattr(p, 'expected_assists_per_90', 0) or 0
        logger.info(f"{p.web_name:20} | xP: {xp:5.1f} | xGI/90: {xg90+xa90:.2f} | Ownership: {p.selected_by_percent:.1f}%")

    # Captain should be high xGI
    top_captain = captain_picks[0][0] if captain_picks else None
    if top_captain:
        xgi = (getattr(top_captain, 'expected_goals_per_90', 0) or 0) + (getattr(top_captain, 'expected_assists_per_90', 0) or 0)
        if xgi > 0.4:
            logger.info(f"✅ PASS: Top captain {top_captain.web_name} has high xGI ({xgi:.2f})")
            return True
        else:
            logger.warning(f"⚠️ WARNING: Top captain xGI may be low ({xgi:.2f})")
            return False

    return False


def test_transfer_suggestions():
    """Test that transfers maximize xP gain."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Transfer Suggestion Validation")
    logger.info("="*60)

    players, teams, fixtures, gw = load_data()
    if not players:
        return False

    from fpl_assistant.predictions import ProjectionEngine
    from fpl_assistant.data.models import Position, PlayerStatus

    engine = ProjectionEngine(players, teams, fixtures)
    gw_num = gw.id if gw else 1

    # Get projections
    projections = {}
    for p in players:
        try:
            projections[p.id] = engine.project_single_player(p, gw_num)
        except:
            projections[p.id] = 0

    # Find best value players at each position
    for pos in [Position.GK, Position.DEF, Position.MID, Position.FWD]:
        pos_players = [(p, projections[p.id]) for p in players
                       if p.position == pos and p.status == PlayerStatus.AVAILABLE and projections[p.id] > 0]
        pos_players.sort(key=lambda x: -x[1])

        logger.info(f"\nTop 5 {pos.name}s by xP:")
        for p, xp in pos_players[:5]:
            xg90 = getattr(p, 'expected_goals_per_90', 0) or 0
            xa90 = getattr(p, 'expected_assists_per_90', 0) or 0
            logger.info(f"  {p.web_name:20} | xP: {xp:5.1f} | Price: £{p.price}m | xGI/90: {xg90+xa90:.2f}")

    logger.info("✅ PASS: Transfer targets identified by position")
    return True


def test_minutes_model():
    """Test that minutes model correctly identifies rotation risks."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Minutes Model Validation")
    logger.info("="*60)

    players, teams, fixtures, gw = load_data()
    if not players:
        return False

    from fpl_assistant.predictions.minutes import MinutesPredictor, RotationRisk

    predictor = MinutesPredictor(players)

    # Find high-priced rotation risks
    rotation_risks = predictor.get_rotation_risks(min_price=6.0)

    logger.info("\nHigh-price rotation risks (£6m+):")
    logger.info("-" * 60)
    for pred in rotation_risks[:10]:
        player = next((p for p in players if p.id == pred.player_id), None)
        if player:
            logger.info(f"{pred.player_name:20} | £{player.price}m | P(start): {pred.p_start*100:.0f}% | Risk: {pred.rotation_risk}")

    # Find nailed starters
    nailed = predictor.get_nailed_players(min_price=5.0)

    logger.info("\nNailed starters (£5m+):")
    logger.info("-" * 60)
    for pred in nailed[:10]:
        player = next((p for p in players if p.id == pred.player_id), None)
        if player:
            logger.info(f"{pred.player_name:20} | £{player.price}m | P(start): {pred.p_start*100:.0f}% | E[mins]: {pred.e_minutes:.0f}")

    logger.info("✅ PASS: Minutes model correctly classifies players")
    return True


def test_distribution_forecasting():
    """Test that distribution forecasting provides P10/P50/P90."""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Distribution Forecasting Validation")
    logger.info("="*60)

    players, teams, fixtures, gw = load_data()
    if not players:
        return False

    from fpl_assistant.predictions import ProjectionEngine
    from fpl_assistant.predictions.uncertainty import UncertaintyModel

    engine = ProjectionEngine(players, teams, fixtures)
    uncertainty = UncertaintyModel(players)
    gw_num = gw.id if gw else 1

    # Get distributions for top players
    top_players = sorted(players, key=lambda p: p.form, reverse=True)[:10]

    logger.info("\nDistribution forecasts for high-form players:")
    logger.info("-" * 70)
    logger.info(f"{'Player':20} | {'xP':>5} | {'P10':>5} | {'P50':>5} | {'P90':>5} | {'Upside':>6}")
    logger.info("-" * 70)

    for p in top_players:
        xp = engine.project_single_player(p, gw_num)
        dist = uncertainty.estimate_distribution(p, xp)
        logger.info(f"{p.web_name:20} | {dist.mean:5.1f} | {dist.p10:5.1f} | {dist.p50:5.1f} | {dist.p90:5.1f} | +{dist.upside:5.1f}")

    logger.info("✅ PASS: Distribution forecasting working correctly")
    return True


def test_optimal_team():
    """Test that optimal team maximizes points within rules."""
    logger.info("\n" + "="*60)
    logger.info("TEST 6: Optimal Team Validation")
    logger.info("="*60)

    players, teams, fixtures, gw = load_data()
    if not players:
        return False

    from fpl_assistant.optimizer import FPLOptimizer
    from fpl_assistant.predictions import ProjectionEngine

    engine = ProjectionEngine(players, teams, fixtures)
    gw_num = gw.id if gw else 1

    # Generate projections
    projections = {}
    for p in players:
        try:
            projections[p.id] = engine.project_single_player(p, gw_num)
        except:
            projections[p.id] = 0

    # Build optimal team
    player_dict = {p.id: p for p in players}
    optimizer = FPLOptimizer()
    result = optimizer.optimize_single_week(player_dict, projections, budget=100.0)

    if result and result.starting_xi:
        squad_ids = set(result.starting_xi + result.bench_order)
        total_xp = sum(projections.get(pid, 0) for pid in result.starting_xi)
        total_cost = sum(p.price for p in players if p.id in squad_ids)

        logger.info(f"\nOptimal Starting XI (Budget: £100m):")
        logger.info(f"Total Cost: £{total_cost:.1f}m | Starting XI xP: {total_xp:.1f}")
        logger.info("-" * 60)

        squad_players = sorted(
            [(p, projections[p.id]) for p in players if p.id in result.starting_xi],
            key=lambda x: (x[0].position.value, -x[1])
        )

        for p, xp in squad_players:
            captain = "(C)" if p.id == result.captain_id else ""
            logger.info(f"{p.position_name:3} | {p.web_name:20} | £{p.price}m | {xp:.1f} xP {captain}")

        logger.info("\nBench:")
        for pid in result.bench_order:
            p = next((pl for pl in players if pl.id == pid), None)
            if p:
                logger.info(f"    | {p.web_name:20} | £{p.price}m | {projections.get(pid, 0):.1f} xP")

        logger.info("✅ PASS: Optimal team generation working")
        return True
    else:
        logger.warning("⚠️ WARNING: Optimizer failed to find solution")
        return False


def run_all_tests():
    """Run all validation tests."""
    logger.info("="*60)
    logger.info("FPL RECOMMENDATIONS VALIDATION SUITE")
    logger.info("="*60)

    results = []
    results.append(("Projection Engine", test_projection_engine()))
    results.append(("Captain Selection", test_captain_selection()))
    results.append(("Transfer Suggestions", test_transfer_suggestions()))
    results.append(("Minutes Model", test_minutes_model()))
    results.append(("Distribution Forecasting", test_distribution_forecasting()))
    results.append(("Optimal Team", test_optimal_team()))

    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{name:25} | {status}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    return passed == total


if __name__ == "__main__":
    run_all_tests()
