"""
FPL Assistant Streamlit Web Application.

Main entry point for the web interface.
Run with: streamlit run src/fpl_assistant/app.py
"""

import sys
from pathlib import Path

# Add src to path for direct execution
_src_path = Path(__file__).parent.parent
if str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import streamlit as st

from fpl_assistant.data import get_database
from fpl_assistant.api import CachedFPLClient, SyncFPLClient, get_cache
from fpl_assistant.data import (
    process_bootstrap_static,
    process_fixtures,
    identify_blank_double_gameweeks,
    update_gameweeks_with_blank_double,
    PlayerStatus,
    Position,
)
from fpl_assistant.config import get_settings

# Page config
st.set_page_config(
    page_title="FPL Assistant",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path("data")


@st.cache_resource
def get_db():
    """Get cached database connection."""
    return get_database(get_data_dir() / "fpl.db")


def get_manager_id() -> int | None:
    """Get manager ID from session state, settings, or default.

    Priority:
    1. Session state (user entered in UI)
    2. Environment/settings
    3. Default value (1178030)
    """
    # Check session state first (user entered)
    if "manager_id" in st.session_state and st.session_state.manager_id:
        return st.session_state.manager_id

    # Fall back to settings/environment
    settings = get_settings()
    if settings.fpl.manager_id > 0:
        return settings.fpl.manager_id

    # Default value
    return 1178030


def load_data_from_api(force: bool = False):
    """Load data from FPL API."""
    cache = get_cache()
    raw_client = SyncFPLClient()
    client = CachedFPLClient(raw_client, cache)
    db = get_db()

    try:
        with st.spinner("Fetching player data..."):
            bootstrap = client.get_bootstrap_static(force_refresh=force)
            players, teams, gameweeks = process_bootstrap_static(bootstrap)

        with st.spinner("Fetching fixtures..."):
            fixtures_data = client.get_fixtures(force_refresh=force)
            fixtures = process_fixtures(fixtures_data)

        with st.spinner("Analyzing gameweeks..."):
            blank_double = identify_blank_double_gameweeks(fixtures)
            gameweeks = update_gameweeks_with_blank_double(gameweeks, blank_double)

        with st.spinner("Saving to database..."):
            db.upsert_players(players)
            db.upsert_teams(teams)
            db.upsert_fixtures(fixtures)
            for gw in gameweeks:
                db.upsert_gameweek(gw)

        # Record ownership snapshot for price prediction and EO analysis
        with st.spinner("Recording ownership & price data..."):
            try:
                from fpl_assistant.data.ownership import OwnershipTracker
                from fpl_assistant.predictions.prices import PricePredictor
                current_gw = next((gw for gw in gameweeks if gw.is_current), None)
                if current_gw:
                    tracker = OwnershipTracker(db)
                    tracker.record_ownership_snapshot(players, current_gw.id)
                    # Also record price snapshot for tracking
                    predictor = PricePredictor(db)
                    predictor.save_price_snapshot(players)
            except ImportError:
                pass  # Skip if imports fail (not critical)

        return True, f"Loaded {len(players)} players, {len(teams)} teams, {len(fixtures)} fixtures"

    except Exception as e:
        return False, str(e)
    finally:
        raw_client.close()


def main():
    """Main application."""
    # Sidebar
    with st.sidebar:
        st.title("⚽ FPL Assistant")
        st.markdown("---")

        # Navigation - WEEKLY ADVICE is the main page
        page = st.radio(
            "Navigation",
            ["📋 Weekly Advice", "🏆 Best Squad", "👤 My Team", "🔍 Scout Tips", "📆 Fixtures", "🎯 Rivals", "👥 Players", "📊 Backtest", "📈 Performance"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        # Manager ID input - allows users to enter their own ID
        st.subheader("Your Team")

        # Default to user's stored ID or the preset default
        default_manager_id = 1178030  # Default ID

        # Get current value from session state or settings or default
        current_manager_id = st.session_state.get("manager_id", default_manager_id)

        # Input field for manager ID
        new_manager_id = st.number_input(
            "FPL Manager ID",
            min_value=1,
            value=current_manager_id,
            help="Find your ID at fantasy.premierleague.com/entry/[YOUR_ID]/history",
            key="manager_id_input"
        )

        # Update session state if changed
        if new_manager_id != st.session_state.get("manager_id"):
            st.session_state.manager_id = new_manager_id
            # Clear cached team data when ID changes
            if "my_team" in st.session_state:
                del st.session_state.my_team
            if "fetch_debug" in st.session_state:
                del st.session_state.fetch_debug

        st.markdown("---")

        # Data refresh
        st.subheader("Data")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Update", use_container_width=True):
                success, msg = load_data_from_api(force=False)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
        with col2:
            if st.button("Force", use_container_width=True):
                success, msg = load_data_from_api(force=True)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

        # Refresh My Team button - to get latest picks from FPL
        if "my_team" in st.session_state and st.session_state.my_team:
            if st.button("🔄 Refresh My Team", use_container_width=True, help="Re-fetch your team picks from FPL"):
                mgr_id = get_manager_id()
                if mgr_id:
                    fetch_my_team(mgr_id)
                    st.rerun()

        # Database stats
        db = get_db()
        player_count = db.get_player_count()
        if player_count > 0:
            st.caption(f"📊 {player_count} players in database")
            current_gw = db.get_current_gameweek()
            if current_gw:
                st.caption(f"📅 {current_gw.name}")

        st.markdown("---")
        st.caption("Built with Streamlit & PuLP")

    # Main content based on navigation
    if page == "📋 Weekly Advice":
        show_weekly_advice()
    elif page == "🏆 Best Squad":
        show_best_squad()
    elif page == "👤 My Team":
        show_my_team()
    elif page == "🔍 Scout Tips":
        show_scout_tips()
    elif page == "📆 Fixtures":
        show_fixture_ticker()
    elif page == "🎯 Rivals":
        show_rival_analysis()
    elif page == "👥 Players":
        show_players()
    elif page == "📊 Backtest":
        show_backtest()
    elif page == "📈 Performance":
        show_performance_tracking()


def show_weekly_advice():
    """THE MAIN PAGE - tells you exactly what to do this week."""
    st.title("📋 What To Do This Week")

    db = get_db()
    if db.get_player_count() == 0:
        st.warning("No data loaded. Click 'Update' in the sidebar first.")
        return

    manager_id = get_manager_id()

    # Show any previous errors
    if "fetch_error" in st.session_state:
        st.error(f"Failed to fetch team: {st.session_state.fetch_error}")
        with st.expander("Error details"):
            st.code(st.session_state.get("fetch_error_detail", "No details"))

    # Show fetch debug info
    if "fetch_debug" in st.session_state:
        debug = st.session_state.fetch_debug
        source = debug.get('picks_source', 'unknown')

        # Show user-friendly note about API delay when using fallback
        if source == "fallback":
            st.info("⏳ **Note:** The FPL API may take some time to reflect your most recent transfers. If your squad looks outdated, try refreshing again in a few minutes.")

        with st.expander("🔧 Debug: Fetch Info", expanded=False):
            st.write(f"**Picks source:** {source} | Target GW: {debug.get('target_gw')}")
            if source == "fallback":
                st.write(f"Fallback from GW: {debug.get('base_gw')}")
                st.write(f"GWs with transfers: {debug.get('unique_events')}")
                st.write(f"Transfers applied: {debug.get('transfers_for_gw')}")
            st.write(f"Final squad IDs: {debug.get('pick_ids')}")

    # Check if team is loaded
    if "my_team" not in st.session_state or not st.session_state.my_team:
        st.info("👆 First, let's load your team to give personalized advice.")
        if st.button("🔄 Load My Team", type="primary"):
            fetch_my_team(manager_id)
            st.rerun()
        return

    # Get all the data we need
    players = db.get_all_players()
    teams = db.get_all_teams()
    fixtures = db.get_all_fixtures()
    teams_dict = {t.id: t for t in teams}
    player_dict = {p.id: p for p in players}

    from fpl_assistant.predictions import ProjectionEngine
    from datetime import datetime

    engine = ProjectionEngine(players, teams, fixtures)
    current_gw = db.get_current_gameweek()
    gw = current_gw.id if current_gw else 1

    # ==========================================================================
    # CHECK IF DEADLINE PASSED - Switch to NEXT gameweek if so
    # ==========================================================================
    deadline_dt = current_gw.deadline if current_gw else None
    deadline_passed = False

    if deadline_dt:
        now = datetime.now(deadline_dt.tzinfo) if deadline_dt.tzinfo else datetime.now()
        if (deadline_dt - now).total_seconds() <= 0:
            deadline_passed = True
            # Switch to next gameweek for all advice
            next_gw = db.get_gameweek(gw + 1)
            if next_gw:
                gw = next_gw.id
                current_gw = next_gw
                deadline_dt = next_gw.deadline

    # ==========================================================================
    # DEADLINE COUNTDOWN & QUICK SUMMARY (Top of page)
    # ==========================================================================

    # Get deadline info (GameweekInfo.deadline is already a datetime)
    if deadline_dt:
        try:
            now = datetime.now(deadline_dt.tzinfo) if deadline_dt.tzinfo else datetime.now()
            time_remaining = deadline_dt - now

            if time_remaining.total_seconds() > 0:
                days = time_remaining.days
                hours = time_remaining.seconds // 3600
                minutes = (time_remaining.seconds % 3600) // 60

                # Format countdown
                if days > 0:
                    countdown = f"{days}d {hours}h {minutes}m"
                    urgency = "info"
                elif hours > 6:
                    countdown = f"{hours}h {minutes}m"
                    urgency = "info"
                elif hours > 1:
                    countdown = f"{hours}h {minutes}m"
                    urgency = "warning"
                else:
                    countdown = f"{minutes} minutes!"
                    urgency = "error"

                # Display deadline box
                deadline_container = st.container()
                with deadline_container:
                    if urgency == "error":
                        st.error(f"⏰ **GW{gw} DEADLINE: {countdown}** - ACT NOW!")
                    elif urgency == "warning":
                        st.warning(f"⏰ **GW{gw} Deadline: {countdown}** - Make your moves soon!")
                    else:
                        st.info(f"⏰ **GW{gw} Deadline: {countdown}** ({deadline_dt.strftime('%a %d %b, %H:%M')})")
            else:
                st.success(f"✅ GW{gw} deadline passed. Good luck!")
        except Exception as e:
            pass  # Silently skip if deadline parsing fails

    # Generate all projections
    projections = {}
    for p in players:
        try:
            projections[p.id] = engine.project_single_player(p, gw)
        except:
            projections[p.id] = p.form * 2

    # Get user's team
    picks = st.session_state.my_team["picks"]
    entry_history = st.session_state.my_team.get("entry_history", {})
    manager = st.session_state.my_team.get("manager", {})
    my_player_ids = {p["element"] for p in picks}
    bank = entry_history.get("bank", 0) / 10  # Convert to millions

    # Get actual free transfers from API
    transfers_data = st.session_state.my_team.get("transfers", {})
    ft_limit = transfers_data.get("limit", 1) if transfers_data else 1
    ft_made = transfers_data.get("made", 0) if transfers_data else 0
    actual_free_transfers = max(0, ft_limit - ft_made)

    # Get chips status from API
    chips_data = st.session_state.my_team.get("chips", [])
    chips_used = {chip.get("name") for chip in chips_data if chip.get("event")}
    all_chips = {"wildcard", "freehit", "bboost", "3xc"}
    chips_remaining = all_chips - chips_used

    # ==========================================================================
    # CHIP STATUS BAR
    # ==========================================================================
    chip_icons = {
        "wildcard": ("WC", "🔄"),
        "freehit": ("FH", "🎯"),
        "bboost": ("BB", "📈"),
        "3xc": ("TC", "👑"),
    }

    st.markdown("---")
    st.markdown("**Chips Available:**")
    chip_cols = st.columns(4)
    for i, chip_name in enumerate(["wildcard", "freehit", "bboost", "3xc"]):
        with chip_cols[i]:
            short, icon = chip_icons.get(chip_name, (chip_name, ""))
            if chip_name in chips_remaining:
                st.success(f"{icon} {short}")
            else:
                st.caption(f"~~{short}~~ (used)")

    # ==========================================================================
    # DYNAMIC INFO SECTION - What you need to know this week
    # ==========================================================================
    with st.container():
        st.markdown("---")

        # Get key alerts and info
        alerts = []
        tips_info = []
        algorithm_info = []

        # Check for blank/double gameweeks
        gw_fixtures = [f for f in fixtures if f.gameweek == gw]
        teams_playing = set()
        for f in gw_fixtures:
            teams_playing.add(f.home_team_id)
            teams_playing.add(f.away_team_id)

        # Teams with blanks
        all_team_ids = {t.id for t in teams}
        blank_teams = all_team_ids - teams_playing
        if blank_teams:
            blank_names = [teams_dict[tid].short_name for tid in blank_teams if tid in teams_dict]
            if blank_names:
                alerts.append(f"⚠️ **BLANK GW**: {', '.join(blank_names)} don't play this week!")

        # Teams with doubles (more than 1 fixture)
        from collections import Counter
        team_fixture_count = Counter()
        for f in gw_fixtures:
            team_fixture_count[f.home_team_id] += 1
            team_fixture_count[f.away_team_id] += 1
        double_teams = [tid for tid, count in team_fixture_count.items() if count > 1]
        if double_teams:
            double_names = [teams_dict[tid].short_name for tid in double_teams if tid in teams_dict]
            if double_names:
                alerts.append(f"🔥 **DOUBLE GW**: {', '.join(double_names)} play twice!")

        # Check for injured/doubtful players in user's team
        injured_players = []
        doubtful_players = []
        for pick in picks:
            player = player_dict.get(pick["element"])
            if player:
                if player.status == PlayerStatus.INJURED:
                    injured_players.append(player.web_name)
                elif player.status == PlayerStatus.DOUBTFUL:
                    doubtful_players.append(f"{player.web_name} ({player.chance_of_playing}%)")
                elif player.status == PlayerStatus.SUSPENDED:
                    injured_players.append(f"{player.web_name} (suspended)")

        if injured_players:
            alerts.append(f"🚨 **INJURED/OUT**: {', '.join(injured_players)}")
        if doubtful_players:
            alerts.append(f"⚠️ **DOUBTFUL**: {', '.join(doubtful_players)}")

        # Check if any user players have a blank
        blank_players = []
        for pick in picks:
            player = player_dict.get(pick["element"])
            if player and player.team_id in blank_teams:
                blank_players.append(player.web_name)
        if blank_players:
            alerts.append(f"🔴 **YOUR PLAYERS WITH BLANK**: {', '.join(blank_players)}")

        # Rotation risk alerts
        try:
            from fpl_assistant.predictions.minutes import MinutesPredictor, RotationRisk

            minutes_predictor = MinutesPredictor(players)
            rotation_risks = []
            for pick in picks:
                player = player_dict.get(pick["element"])
                if player and player.status == PlayerStatus.AVAILABLE:
                    pred = minutes_predictor.predict_minutes(player)
                    if pred.rotation_risk == RotationRisk.HIGH:
                        rotation_risks.append(f"{player.web_name} (P(start)={pred.p_start*100:.0f}%)")
                    elif pred.rotation_risk == RotationRisk.MEDIUM and pred.p_start < 0.7:
                        rotation_risks.append(f"{player.web_name} (P(start)={pred.p_start*100:.0f}%)")
            if rotation_risks:
                alerts.append(f"🔄 **ROTATION RISK**: {', '.join(rotation_risks)} - may not start!")
        except Exception as e:
            pass  # Minutes prediction optional

        # Price change alerts
        try:
            from fpl_assistant.predictions.prices import PricePredictor

            price_predictor = PricePredictor(db)

            # Get predictions for owned players
            my_players_list = [player_dict.get(p["element"]) for p in picks if player_dict.get(p["element"])]
            price_alerts = price_predictor.get_price_alerts(my_players_list, players)

            # Urgent alerts
            if price_alerts["urgent_sell"]:
                urgent_names = [f"{p.player_name} (falls tonight!)" for p in price_alerts["urgent_sell"]]
                alerts.append(f"🚨 **PRICE DROP IMMINENT**: {', '.join(urgent_names)} - Transfer out NOW!")

            # Rising targets
            if price_alerts["rising_targets"]:
                rising_names = [f"{p.player_name}" for p in price_alerts["rising_targets"][:3]]
                tips_info.append(f"📈 **Rising Tonight**: {', '.join(rising_names)} - Buy before price increase!")

        except Exception as e:
            pass  # Price prediction optional

        # Algorithm info - what factors are being weighted (NOW WITH xG!)
        algorithm_info.append(f"📊 **xG: {engine.XG_WEIGHT*100:.0f}%** | Form: {engine.FORM_WEIGHT*100:.0f}% | FDR: {engine.FDR_WEIGHT*100:.0f}% | ICT: {engine.ICT_WEIGHT*100:.0f}%")
        algorithm_info.append("🎯 *xG (Expected Goals) is now the PRIMARY predictor - what FPL Review charges £5/mo for!*")

        # Check if weights have been optimized
        try:
            from fpl_assistant.predictions.adaptive import get_adaptive_manager
            adaptive_manager = get_adaptive_manager()
            summary = adaptive_manager.get_performance_summary()
            if summary["num_historical_results"] > 0:
                weights_info = summary["current_weights"]
                if weights_info.get("correlation"):
                    algorithm_info.append(f"🧠 Weights optimized from {summary['num_historical_results']} backtests (correlation: {weights_info['correlation']:.2f})")
        except:
            pass

        # Get community/tip site info (simulated for now - could be web scraped)
        # Top captaincy choices based on xGI (Expected Goal Involvement) - FROM YOUR SQUAD ONLY
        all_captain_picks = engine.get_captain_picks(gw, limit=20)  # Get more to filter
        # Filter to only players YOU own
        top_captains = [(p, xp) for p, xp in all_captain_picks if p.id in my_player_ids][:3]
        if top_captains:
            captain_names = [f"{p.web_name} ({xp:.1f}xP, xGI:{(getattr(p, 'expected_goals_per_90', 0) or 0) + (getattr(p, 'expected_assists_per_90', 0) or 0):.2f}/90)" for p, xp in top_captains]
            tips_info.append(f"👑 **Top Captain Picks (by xGI)**: {', '.join(captain_names)}")

        # Top differentials
        differentials = engine.get_differential_picks(gw, max_ownership=5.0, limit=3)
        if differentials:
            diff_names = [f"{p.web_name} ({p.selected_by_percent:.1f}%)" for p, xp in differentials]
            tips_info.append(f"💎 **Differentials (<5% owned)**: {', '.join(diff_names)}")

        # Display the info boxes
        if alerts:
            st.error("\n\n".join(alerts))

        if tips_info:
            st.info("\n\n".join(tips_info))

        if algorithm_info:
            with st.expander("🔧 Algorithm Settings", expanded=False):
                for info in algorithm_info:
                    st.caption(info)
                st.caption("*Weights can be tuned based on backtest results*")

        st.markdown("---")

    # ==========================================================================
    # QUICK ACTIONS SUMMARY BOX - Everything you need to do in one place
    # ==========================================================================
    st.markdown("### 📋 QUICK ACTIONS CHECKLIST")

    # Build quick actions based on gathered data (with safety checks)
    quick_actions = []

    # 1. Captain recommendation - use highest xP from YOUR squad
    try:
        # Find highest xP player from your squad (same source as main section)
        my_squad_with_xp = []
        for pick in picks:
            p = player_dict.get(pick["element"])
            if p:
                xp_val = projections.get(p.id, 0)
                my_squad_with_xp.append((p, xp_val))
        my_squad_with_xp.sort(key=lambda x: -x[1])

        if my_squad_with_xp:
            best_cap = my_squad_with_xp[0]
            quick_actions.append(f"👑 **Captain:** {best_cap[0].web_name} ({best_cap[1]:.1f} xP)")
    except Exception:
        pass

    # 2. Transfer if any alert for injured/suspended
    try:
        if injured_players:
            quick_actions.append(f"🚨 **Transfer OUT:** {injured_players[0]} (unavailable)")
        elif doubtful_players:
            quick_actions.append(f"⚠️ **Monitor:** {doubtful_players[0].split('(')[0].strip()} before deadline")
        else:
            quick_actions.append(f"🔄 **Transfers:** {actual_free_transfers} FT available (bank £{bank:.1f}m)")
    except NameError:
        quick_actions.append(f"🔄 **Transfers:** {actual_free_transfers} FT available (bank £{bank:.1f}m)")

    # 3. Blank/Double GW warning
    try:
        if blank_teams:
            blank_team_names = [teams_dict[tid].short_name for tid in blank_teams if tid in teams_dict][:2]
            if blank_team_names:
                quick_actions.append(f"⚠️ **Blank:** {', '.join(blank_team_names)} don't play!")
        elif double_teams:
            double_team_names = [teams_dict[tid].short_name for tid in double_teams if tid in teams_dict][:2]
            if double_team_names:
                quick_actions.append(f"🔥 **Double:** {', '.join(double_team_names)} play twice!")
    except NameError:
        pass

    # 4. Chip suggestion (simplified)
    try:
        upcoming_gws_check = [db.get_gameweek(gw + i) for i in range(6)]
        upcoming_gws_check = [g for g in upcoming_gws_check if g]
        double_gws_check = [g for g in upcoming_gws_check if g.is_double]

        if double_gws_check and "bboost" in chips_remaining:
            quick_actions.append(f"🎴 **Chip Tip:** BB for GW{double_gws_check[0].id}")
        elif double_gws_check and "3xc" in chips_remaining:
            quick_actions.append(f"🎴 **Chip Tip:** TC for GW{double_gws_check[0].id}")
    except Exception:
        pass

    # Display quick actions as a compact list
    if quick_actions:
        num_cols = min(len(quick_actions), 4)
        action_cols = st.columns(num_cols)
        for i, action in enumerate(quick_actions):
            with action_cols[i % num_cols]:
                st.markdown(action)
    else:
        st.info("All good! No urgent actions needed.")

    st.markdown("---")

    # Current team info
    st.markdown(f"### {manager.get('name', 'Your Team')} | GW{gw}")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Rank", f"{manager.get('summary_overall_rank', 0):,}")
    with col2:
        st.metric("Points", manager.get("summary_overall_points", 0))
    with col3:
        st.metric("Bank", f"£{bank:.1f}m")
    with col4:
        st.metric("Free Transfers", actual_free_transfers)

    st.markdown("---")

    # ===========================================
    # 1. CAPTAIN PICK - Most important decision
    # ===========================================
    st.header("👑 1. CAPTAIN PICK")

    my_players_with_xp = []
    for pick in picks:
        p = player_dict.get(pick["element"])
        if p:
            xp = projections.get(p.id, 0)
            my_players_with_xp.append((p, xp, pick))

    # Sort by xP
    my_players_with_xp.sort(key=lambda x: -x[1])

    if my_players_with_xp:
        # Get differential captain analysis - ONLY from players you own!
        try:
            captain_recommendation = engine.get_differential_captain_value(
                gameweek=gw,
                db=db,
                league_position="mid",  # TODO: Get from mini-league data
                risk_tolerance="medium",
                owned_player_ids=my_player_ids,  # Only consider your squad!
            )
            has_eo_analysis = True
        except Exception as e:
            captain_recommendation = None
            has_eo_analysis = False

        best_captain = my_players_with_xp[0]
        p, xp, pick = best_captain
        team = teams_dict.get(p.team_id)

        # Get fixture info for explanation
        gw_fixtures = [f for f in fixtures if f.gameweek == gw]
        player_fixture = next((f for f in gw_fixtures if f.home_team_id == p.team_id or f.away_team_id == p.team_id), None)

        # ONE CLEAR CAPTAIN RECOMMENDATION - Always highest xP
        st.success(f"**👑 CAPTAIN: {p.web_name}** ({team.short_name if team else '?'}) - Projected: **{xp:.1f}** points")

        # Explain WHY (always show) - NOW WITH xG!
        reasons = []
        # Handle players without xG data (need to Force Update to get it)
        xg_per_90 = getattr(p, 'expected_goals_per_90', 0.0) or 0.0
        xa_per_90 = getattr(p, 'expected_assists_per_90', 0.0) or 0.0
        xgi_val = xg_per_90 + xa_per_90
        reasons.append(f"xGI/90: {xgi_val:.2f}")  # Most important stat!
        reasons.append(f"Form: {p.form}")
        if player_fixture:
            is_home = player_fixture.home_team_id == p.team_id
            opp_id = player_fixture.away_team_id if is_home else player_fixture.home_team_id
            opp = teams_dict.get(opp_id)
            difficulty = player_fixture.away_difficulty if is_home else player_fixture.home_difficulty
            venue = "HOME" if is_home else "AWAY"
            diff_text = {1: "Very Easy", 2: "Easy", 3: "Medium", 4: "Hard", 5: "Very Hard"}.get(difficulty, "")
            reasons.append(f"Plays {opp.short_name if opp else '?'} ({venue}) - {diff_text} fixture (FDR {difficulty})")
        reasons.append(f"Ownership: {p.selected_by_percent:.1f}%")
        st.caption("**Why:** " + " | ".join(reasons))

        if len(my_players_with_xp) > 1:
            p2, xp2, _ = my_players_with_xp[1]
            team2 = teams_dict.get(p2.team_id)
            st.info(f"Vice Captain: {p2.web_name} ({team2.short_name if team2 else '?'}) - {xp2:.1f} xP")

        # ===========================================
        # EVENT-BASED SIMULATION - More accurate captain analysis
        # ===========================================
        with st.expander("🎰 Event Simulation (Poisson/Bernoulli) - Most Accurate", expanded=False):
            try:
                from fpl_assistant.predictions.simulation import EventSimulator

                # Get top 5 captain candidates by xP
                top_captain_ids = [p.id for p, xp, _ in my_players_with_xp[:5] if xp > 0]

                if top_captain_ids:
                    st.markdown("**Simulating goals, assists, CS, cards as individual events (10k runs)**")
                    st.caption("Uses Poisson distribution for goals/assists/saves, Bernoulli for CS/cards")

                    # Run event-based simulation
                    event_sim = EventSimulator()
                    sim_results = []

                    for pid in top_captain_ids:
                        player_obj = player_dict.get(pid)
                        if player_obj:
                            try:
                                # Get simulation params from engine
                                params = engine.get_simulation_params(player_obj, gw)
                                result = event_sim.simulate_player(params, n_sims=10000)
                                sim_results.append((player_obj, result))
                            except Exception as e:
                                logger.warning(f"Simulation failed for {player_obj.web_name}: {e}")

                    if sim_results:
                        # Sort by ceiling score
                        sim_results.sort(key=lambda x: x[1].ceiling_score, reverse=True)

                        sim_data = []
                        for i, (player_obj, result) in enumerate(sim_results[:5]):
                            medal = {0: "🥇", 1: "🥈", 2: "🥉"}.get(i, "")
                            team_obj = teams_dict.get(player_obj.team_id)

                            sim_data.append({
                                "Rank": f"{medal} {i+1}",
                                "Player": player_obj.web_name,
                                "Team": team_obj.short_name if team_obj else "?",
                                "xP": f"{result.expected_points:.1f}",
                                "P(Haul)": f"{result.p_haul*100:.0f}%",
                                "P(Blank)": f"{result.p_blank*100:.0f}%",
                                "Ceiling (P90)": f"{result.percentile_90:.1f}",
                                "Floor (P10)": f"{result.median_points - result.std_dev*1.28:.1f}",
                            })

                        st.dataframe(sim_data, use_container_width=True, hide_index=True)

                        # Recommendation
                        best_player, best_result = sim_results[0]
                        if best_result.p_haul >= 0.3:
                            st.success(f"**{best_player.web_name}** has {best_result.p_haul*100:.0f}% haul probability - excellent captain!")
                        elif best_result.p_haul >= 0.2:
                            st.info(f"**{best_player.web_name}** has decent {best_result.p_haul*100:.0f}% haul chance")
                        else:
                            st.warning(f"No clear haul candidate - {best_player.web_name} only {best_result.p_haul*100:.0f}%")

                        # Show P(haul) leader if different from xP leader
                        haul_leader = max(sim_results, key=lambda x: x[1].p_haul)
                        if haul_leader[0].id != best_player.id:
                            st.caption(f"💡 Highest haul probability: {haul_leader[0].web_name} ({haul_leader[1].p_haul*100:.0f}%)")
                    else:
                        st.caption("Could not simulate captain candidates")
                else:
                    st.caption("No valid captain candidates found")
            except Exception as e:
                st.caption(f"Event simulation unavailable: {e}")

        # ===========================================
        # MONTE CARLO CAPTAIN COMPARISON - LEGACY
        # ===========================================
        with st.expander("🎲 Monte Carlo Analysis (Normal distribution - legacy)", expanded=False):
            try:
                from fpl_assistant.predictions.uncertainty import MonteCarloSimulator

                # Get top 5 captain candidates by xP
                top_captain_ids = [p.id for p, xp, _ in my_players_with_xp[:5] if xp > 0]

                if top_captain_ids:
                    # Run Monte Carlo simulation
                    mc_simulator = MonteCarloSimulator(
                        players=players,
                        projections=projections,
                    )
                    mc_results = mc_simulator.compare_captains(top_captain_ids, n_simulations=10000)

                    if mc_results:
                        st.markdown("**Who wins most often across 10,000 simulated gameweeks?**")
                        st.caption("Win Rate = % of simulations where this player scored highest. Higher ceiling (P90) = better upside.")

                        # Display results
                        mc_data = []
                        for i, result in enumerate(mc_results[:5]):
                            # Medal for top 3
                            medal = {0: "🥇", 1: "🥈", 2: "🥉"}.get(i, "")
                            player_obj = player_dict.get(result.player_id)
                            team_obj = teams_dict.get(player_obj.team_id) if player_obj else None

                            mc_data.append({
                                "Rank": f"{medal} {i+1}",
                                "Player": result.player_name,
                                "Team": team_obj.short_name if team_obj else "?",
                                "Win Rate": f"{result.captain_win_rate:.1f}%",
                                "Top 3 Rate": f"{result.top_3_rate:.1f}%",
                                "xP": f"{result.mean_points:.1f}",
                                "Ceiling (P90)": f"{result.p90:.1f}",
                                "Floor (P10)": f"{result.p10:.1f}",
                            })

                        st.dataframe(mc_data, use_container_width=True, hide_index=True)

                        # Recommendation based on win rate
                        best = mc_results[0]
                        if best.captain_win_rate >= 40:
                            st.success(f"**Clear leader:** {best.player_name} wins {best.captain_win_rate:.0f}% of simulations - confident pick!")
                        elif best.captain_win_rate >= 30:
                            st.info(f"**Slight edge:** {best.player_name} wins {best.captain_win_rate:.0f}% - good pick, but close race")
                        else:
                            st.warning(f"**Tight race:** No clear winner (best is {best.captain_win_rate:.0f}%) - go with gut or highest ceiling")

                        # If #2 has higher ceiling, mention it
                        if len(mc_results) > 1 and mc_results[1].p90 > mc_results[0].p90:
                            alt = mc_results[1]
                            st.caption(f"💡 Note: {alt.player_name} has higher ceiling ({alt.p90:.1f} vs {best.p90:.1f}) - consider for differential punt")
                    else:
                        st.caption("No simulation results available")
                else:
                    st.caption("No valid captain candidates found")
            except Exception as e:
                st.caption(f"Monte Carlo analysis unavailable: {e}")

        # Captain Options Table - optional detail
        with st.expander("📊 All Captain Options (optional - click to expand)", expanded=False):
            try:
                captain_options = engine.get_captain_picks_with_eo(gw, db=db, limit=5, owned_player_ids=my_player_ids)
                if captain_options:
                    # Get uncertainty model for distribution forecasting
                    try:
                        from fpl_assistant.predictions.uncertainty import UncertaintyModel
                        uncertainty_model = UncertaintyModel(players)
                        has_uncertainty = True
                    except:
                        has_uncertainty = False

                    cap_data = []
                    for opt in captain_options:
                        player = opt["player"]
                        team = teams_dict.get(player.team_id)

                        # Calculate xGI with fallback for old data
                        p_xg = getattr(player, 'expected_goals_per_90', 0.0) or 0.0
                        p_xa = getattr(player, 'expected_assists_per_90', 0.0) or 0.0
                        p_xgi = p_xg + p_xa

                        # Get distribution for P10/P90
                        if has_uncertainty:
                            dist = uncertainty_model.estimate_distribution(player, opt['expected_points'])
                            floor = f"{dist.p10:.1f}"
                            ceiling = f"{dist.p90:.1f}"
                        else:
                            floor = "-"
                            ceiling = "-"

                        cap_data.append({
                            "Player": player.web_name,
                            "Team": team.short_name if team else "?",
                            "xGI/90": f"{p_xgi:.2f}",
                            "xP": f"{opt['expected_points']:.1f}",
                            "Floor": floor,
                            "Ceiling": ceiling,
                            "EO": f"{opt['captain_eo']:.0f}%",
                        })
                    st.dataframe(cap_data, use_container_width=True, hide_index=True)
            except Exception as e:
                st.caption("Detailed analysis unavailable")

        # xG Analysis Expander
        with st.expander("📊 xG Analysis - Buy/Sell Signals", expanded=False):
            st.markdown("*Based on Expected Goals (xG) - the same methodology FPL Review uses*")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🎯 BUY: Underperforming xG** (Due to score)")
                st.caption("High xG but low actual goals = unlucky, will regress upwards")
                try:
                    underperformers = engine.get_underperformers(limit=5)
                    if underperformers:
                        for player, xg, goals in underperformers:
                            team = teams_dict.get(player.team_id)
                            diff = xg - goals
                            in_team = "✓" if player.id in my_player_ids else ""
                            st.write(f"• {player.web_name} ({team.short_name if team else '?'}): {goals} goals on {xg:.1f} xG (+{diff:.1f} owed) {in_team}")
                    else:
                        st.caption("No significant underperformers found")
                except Exception as e:
                    st.caption(f"Error: {e}")

            with col2:
                st.markdown("**⚠️ SELL: Overperforming xG** (Due to regress)")
                st.caption("Low xG but high actual goals = lucky, will regress downwards")
                try:
                    overperformers = engine.get_overperformers(limit=5)
                    if overperformers:
                        for player, xg, goals in overperformers:
                            team = teams_dict.get(player.team_id)
                            diff = goals - xg
                            in_team = "⚠️ YOU OWN" if player.id in my_player_ids else ""
                            st.write(f"• {player.web_name} ({team.short_name if team else '?'}): {goals} goals on {xg:.1f} xG (-{diff:.1f} over) {in_team}")
                    else:
                        st.caption("No significant overperformers found")
                except Exception as e:
                    st.caption(f"Error: {e}")

            st.markdown("---")
            st.markdown("**🔥 Top xGI/90 Players** (Best captain/transfer targets)")
            try:
                top_xgi = engine.get_top_xgi_players(limit=10)
                if top_xgi:
                    xgi_data = []
                    for player, xgi in top_xgi:
                        team = teams_dict.get(player.team_id)
                        in_team = "✓" if player.id in my_player_ids else ""
                        # Handle players without xG data
                        p_xg90 = getattr(player, 'expected_goals_per_90', 0.0) or 0.0
                        p_xa90 = getattr(player, 'expected_assists_per_90', 0.0) or 0.0
                        xgi_data.append({
                            "Player": player.web_name,
                            "Team": team.short_name if team else "?",
                            "Pos": player.position_name,
                            "Price": f"£{player.price:.1f}m",
                            "xG/90": f"{p_xg90:.2f}",
                            "xA/90": f"{p_xa90:.2f}",
                            "xGI/90": f"{xgi:.2f}",
                            "Owned": in_team,
                        })
                    st.dataframe(xgi_data, use_container_width=True, hide_index=True)
            except Exception as e:
                st.caption(f"Error: {e}")

    st.markdown("---")

    # ===========================================
    # 1b. OWNERSHIP TRENDS (Price Changes)
    # ===========================================
    with st.expander("📈 Ownership Trends & Price Alerts", expanded=False):
        st.markdown("*Players rising/falling in ownership - indicates upcoming price changes*")

        try:
            from fpl_assistant.data.ownership import OwnershipTracker

            tracker = OwnershipTracker(db)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🔺 Rising Players** (potential price rises)")
                rising = tracker.get_rising_players(players, min_ownership=1.0, limit=5)
                if rising:
                    for trend in rising:
                        player = player_dict.get(trend.player_id)
                        if player:
                            team = teams_dict.get(player.team_id)
                            in_team = "✓" if trend.player_id in my_player_ids else ""
                            st.write(f"• {trend.player_name} ({team.short_name if team else '?'}) - {trend.current_ownership:.1f}% (+{trend.ownership_change_7d:.1f}%) {in_team}")
                else:
                    st.caption("No significant risers detected")

            with col2:
                st.markdown("**🔻 Falling Players** (potential price falls)")
                falling = tracker.get_falling_players(players, min_ownership=1.0, limit=5)
                if falling:
                    for trend in falling:
                        player = player_dict.get(trend.player_id)
                        if player:
                            team = teams_dict.get(player.team_id)
                            in_team = "⚠️ YOU OWN" if trend.player_id in my_player_ids else ""
                            st.write(f"• {trend.player_name} ({team.short_name if team else '?'}) - {trend.current_ownership:.1f}% ({trend.ownership_change_7d:.1f}%) {in_team}")
                else:
                    st.caption("No significant fallers detected")

            # Check if user owns any falling players
            my_falling = [trend for trend in falling if trend.player_id in my_player_ids] if falling else []
            if my_falling:
                st.warning(f"⚠️ **Alert:** You own {len(my_falling)} player(s) with falling ownership - consider transferring out before price drop!")

            st.markdown("---")
            st.markdown("**💎 Differential Candidates** (low ownership, high form)")
            differentials = tracker.get_differential_candidates(players, max_ownership=10.0, min_form=3.0, limit=5)
            if differentials:
                for player, trend in differentials:
                    team = teams_dict.get(player.team_id)
                    xp = projections.get(player.id, 0)
                    st.write(f"• {player.web_name} ({team.short_name if team else '?'}) - {player.selected_by_percent:.1f}% owned, Form: {player.form}, xP: {xp:.1f}")
            else:
                st.caption("No strong differentials found")

        except Exception as e:
            st.caption(f"Ownership tracking requires data snapshots. Error: {e}")

        # Detailed Price Predictions
        st.markdown("---")
        st.markdown("### �� Price Change Predictions")

        try:
            from fpl_assistant.predictions.prices import PricePredictor

            price_predictor = PricePredictor(db)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🔺 Rising Tonight/Soon**")
                rising = price_predictor.get_rising_players(players, min_probability=0.4, limit=5)
                if rising:
                    for pred in rising:
                        team = teams_dict.get(player_dict.get(pred.player_id, Player).team_id if player_dict.get(pred.player_id) else 0)
                        prob_text = f"{pred.probability*100:.0f}%"
                        time_text = pred.expected_time
                        st.write(f"• {pred.player_name} £{pred.current_price:.1f}m → £{pred.current_price + 0.1:.1f}m ({prob_text} - {time_text})")
                else:
                    st.caption("No imminent price rises detected")

            with col2:
                st.markdown("**🔻 Falling Tonight/Soon**")
                falling = price_predictor.get_falling_players(players, min_probability=0.4, limit=5)
                if falling:
                    for pred in falling:
                        team = teams_dict.get(player_dict.get(pred.player_id, Player).team_id if player_dict.get(pred.player_id) else 0)
                        prob_text = f"{pred.probability*100:.0f}%"
                        time_text = pred.expected_time
                        in_team = "⚠️" if pred.player_id in my_player_ids else ""
                        st.write(f"• {pred.player_name} £{pred.current_price:.1f}m → £{pred.current_price - 0.1:.1f}m ({prob_text} - {time_text}) {in_team}")
                else:
                    st.caption("No imminent price falls detected")

            st.caption("*Based on net transfers and ownership thresholds. Predictions update overnight.*")

        except Exception as e:
            st.caption(f"Price prediction not available: {e}")

    st.markdown("---")

    # ===========================================
    # 1B. YOUR OPTIMAL STARTING XI
    # ===========================================
    st.header("⚽ YOUR OPTIMAL STARTING XI")
    st.caption("Based on projections for GW" + str(gw) + " - this is who should start and where")

    # Get all squad players with projections
    squad_players = []
    for pick in picks:
        player = player_dict.get(pick["element"])
        if player:
            xp = projections.get(player.id, 0)
            squad_players.append({
                "player": player,
                "xp": xp,
                "current_position": pick.get("position", 15),  # 1-11 = starter, 12-15 = bench
                "is_captain": pick.get("is_captain", False),
                "is_vice_captain": pick.get("is_vice_captain", False),
            })

    # Group by position
    gks = [p for p in squad_players if p["player"].position == Position.GK]
    defs = [p for p in squad_players if p["player"].position == Position.DEF]
    mids = [p for p in squad_players if p["player"].position == Position.MID]
    fwds = [p for p in squad_players if p["player"].position == Position.FWD]

    # Sort each position by xP
    gks.sort(key=lambda x: -x["xp"])
    defs.sort(key=lambda x: -x["xp"])
    mids.sort(key=lambda x: -x["xp"])
    fwds.sort(key=lambda x: -x["xp"])

    # Pick optimal XI: 1 GK, best 4 DEF, best 4 MID, best 2 FWD (4-4-2 formation)
    # But we need to ensure valid formation (3-5 DEF, 2-5 MID, 1-3 FWD)
    optimal_xi = []
    optimal_xi.extend(gks[:1])  # 1 GK

    # Determine best formation based on xP
    remaining_spots = 10  # After GK
    best_defs = defs[:5]  # Max 5 DEF
    best_mids = mids[:5]  # Max 5 MID
    best_fwds = fwds[:3]  # Max 3 FWD

    # Score each formation
    formations = [
        (3, 5, 2, "3-5-2"),
        (3, 4, 3, "3-4-3"),
        (4, 4, 2, "4-4-2"),
        (4, 3, 3, "4-3-3"),
        (4, 5, 1, "4-5-1"),
        (5, 4, 1, "5-4-1"),
        (5, 3, 2, "5-3-2"),
    ]

    best_formation = None
    best_xi_xp = 0
    best_formation_name = "4-4-2"

    for n_def, n_mid, n_fwd, name in formations:
        if len(defs) < n_def or len(mids) < n_mid or len(fwds) < n_fwd:
            continue
        total_xp = sum(p["xp"] for p in defs[:n_def]) + sum(p["xp"] for p in mids[:n_mid]) + sum(p["xp"] for p in fwds[:n_fwd])
        if total_xp > best_xi_xp:
            best_xi_xp = total_xp
            best_formation = (n_def, n_mid, n_fwd)
            best_formation_name = name

    if best_formation:
        n_def, n_mid, n_fwd = best_formation
        optimal_xi.extend(defs[:n_def])
        optimal_xi.extend(mids[:n_mid])
        optimal_xi.extend(fwds[:n_fwd])
    else:
        # Default to 4-4-2
        optimal_xi.extend(defs[:4])
        optimal_xi.extend(mids[:4])
        optimal_xi.extend(fwds[:2])

    # Find bench players (not in optimal XI)
    xi_ids = {p["player"].id for p in optimal_xi}
    bench = [p for p in squad_players if p["player"].id not in xi_ids]
    bench.sort(key=lambda x: -x["xp"])

    # Calculate total XI xP
    total_xi_xp = sum(p["xp"] for p in optimal_xi)

    # Determine OPTIMAL captain - prioritize MID/FWD over DEF/GK for captain (higher ceiling)
    # Defenders shouldn't be captained even if they have higher xP - attackers have more upside
    attackers_in_xi = [p for p in optimal_xi if p["player"].position in [Position.MID, Position.FWD]]
    attackers_sorted = sorted(attackers_in_xi, key=lambda x: -x["xp"])

    if attackers_sorted:
        optimal_captain_id = attackers_sorted[0]["player"].id
    else:
        # Fallback to highest xP if no attackers (shouldn't happen)
        optimal_xi_sorted = sorted(optimal_xi, key=lambda x: -x["xp"])
        optimal_captain_id = optimal_xi_sorted[0]["player"].id if optimal_xi_sorted else None

    # Display formation
    st.markdown(f"**Recommended Formation: {best_formation_name}** | Total Starting XI: **{total_xi_xp:.1f} xP**")

    # Display as a pitch layout
    col1, col2, col3 = st.columns([1, 3, 1])

    # Helper function to get captain icon based on OPTIMAL picks
    def get_captain_icon(player_id):
        if player_id == optimal_captain_id:
            return " ©️"  # Recommended captain
        return ""

    with col2:
        # GK Row
        st.markdown("##### Goalkeeper")
        gk_cols = st.columns([2, 1, 2])
        with gk_cols[1]:
            gk = optimal_xi[0]
            status_icon = "🔴" if gk["player"].status != PlayerStatus.AVAILABLE else ""
            captain_icon = get_captain_icon(gk["player"].id)
            st.markdown(f"**{gk['player'].web_name}**{captain_icon}{status_icon}")
            st.caption(f"{gk['xp']:.1f} xP")

        # DEF Row
        st.markdown("##### Defenders")
        def_players = [p for p in optimal_xi if p["player"].position == Position.DEF]
        if def_players:
            def_cols = st.columns(len(def_players))
            for i, df in enumerate(def_players):
                with def_cols[i]:
                    status_icon = "🔴" if df["player"].status != PlayerStatus.AVAILABLE else ""
                    captain_icon = get_captain_icon(df["player"].id)
                    st.markdown(f"**{df['player'].web_name}**{captain_icon}{status_icon}")
                    st.caption(f"{df['xp']:.1f} xP")

        # MID Row
        st.markdown("##### Midfielders")
        mid_players = [p for p in optimal_xi if p["player"].position == Position.MID]
        if mid_players:
            mid_cols = st.columns(len(mid_players))
            for i, md in enumerate(mid_players):
                with mid_cols[i]:
                    status_icon = "🔴" if md["player"].status != PlayerStatus.AVAILABLE else ""
                    captain_icon = get_captain_icon(md["player"].id)
                    st.markdown(f"**{md['player'].web_name}**{captain_icon}{status_icon}")
                    st.caption(f"{md['xp']:.1f} xP")

        # FWD Row
        st.markdown("##### Forwards")
        fwd_players = [p for p in optimal_xi if p["player"].position == Position.FWD]
        if fwd_players:
            fwd_cols = st.columns(len(fwd_players))
            for i, fw in enumerate(fwd_players):
                with fwd_cols[i]:
                    status_icon = "🔴" if fw["player"].status != PlayerStatus.AVAILABLE else ""
                    captain_icon = get_captain_icon(fw["player"].id)
                    st.markdown(f"**{fw['player'].web_name}**{captain_icon}{status_icon}")
                    st.caption(f"{fw['xp']:.1f} xP")

    # Show bench
    st.markdown("##### Bench")
    bench_cols = st.columns(4)
    for i, bp in enumerate(bench[:4]):
        with bench_cols[i]:
            pos_short = {Position.GK: "GK", Position.DEF: "DEF", Position.MID: "MID", Position.FWD: "FWD"}
            status_icon = "🔴" if bp["player"].status != PlayerStatus.AVAILABLE else ""
            st.markdown(f"{pos_short.get(bp['player'].position, 'UNK')} - **{bp['player'].web_name}** {status_icon}")
            st.caption(f"{bp['xp']:.1f} xP")

    # Check if current lineup differs from optimal
    current_starters = [p["player"].id for p in squad_players if p["current_position"] <= 11]
    optimal_starters = [p["player"].id for p in optimal_xi]

    should_change = set(current_starters) != set(optimal_starters)
    if should_change:
        st.warning("⚠️ **Your current lineup differs from optimal!** Consider making changes before deadline.")

        # Show specific changes
        current_not_optimal = set(current_starters) - set(optimal_starters)
        optimal_not_current = set(optimal_starters) - set(current_starters)

        if current_not_optimal and optimal_not_current:
            changes = []
            for out_id in current_not_optimal:
                out_player = player_dict.get(out_id)
                if out_player:
                    changes.append(f"**Bench**: {out_player.web_name}")
            for in_id in optimal_not_current:
                in_player = player_dict.get(in_id)
                if in_player:
                    changes.append(f"**Start**: {in_player.web_name}")
            st.info("Suggested changes: " + ", ".join(changes))
    else:
        st.success("✅ Your current lineup is optimal!")

    st.markdown("---")

    # ===========================================
    # 2. TRANSFER RECOMMENDATIONS
    # ===========================================
    st.header("🔄 2. TRANSFERS")

    # Find the worst player in the team
    my_players_with_xp.sort(key=lambda x: x[1])  # Sort by lowest xP first

    # Find potential upgrades - track suggested players to avoid duplicates
    transfer_suggestions = []
    already_suggested_in: set[int] = set()  # Players already suggested to buy
    already_suggested_out: set[int] = set()  # Players already suggested to sell

    for p, xp, pick in my_players_with_xp[:7]:  # Check worst 7 players for more options
        if p.id in already_suggested_out:
            continue

        team = teams_dict.get(p.team_id)

        # Find best available replacement (not already suggested)
        best_replacement = None
        best_gain = 0

        for rp in players:
            if rp.id in my_player_ids:
                continue
            if rp.id in already_suggested_in:  # Skip already suggested players
                continue
            if rp.position != p.position:
                continue
            if rp.price > p.price + bank:
                continue
            if rp.status != PlayerStatus.AVAILABLE:
                continue

            # Check 3-per-team rule
            team_count = sum(1 for pid in my_player_ids if player_dict.get(pid) and player_dict[pid].team_id == rp.team_id)
            if rp.team_id == p.team_id:
                team_count -= 1  # We're removing this player
            if team_count >= 3:
                continue

            rp_xp = projections.get(rp.id, 0)
            gain = rp_xp - xp

            if gain > best_gain:
                best_gain = gain
                best_replacement = (rp, rp_xp)

        if best_replacement and best_gain > 1:
            rp, rp_xp = best_replacement
            # Determine if new player would be starter or bench
            # Compare their xP to other players at same position in squad
            same_pos_players = [(pl, projections.get(pl.id, 0)) for pl in players
                                if pl.id in my_player_ids and pl.position == p.position and pl.id != p.id]
            same_pos_players.append((rp, rp_xp))
            same_pos_players.sort(key=lambda x: -x[1])

            # Would they start? Check if in top N for their position
            pos_starter_count = {Position.GK: 1, Position.DEF: 4, Position.MID: 4, Position.FWD: 2}
            max_starters = pos_starter_count.get(rp.position, 3)
            player_rank = next((i for i, (pl, _) in enumerate(same_pos_players) if pl.id == rp.id), max_starters)
            would_start = player_rank < max_starters

            # CRITICAL FIX: Calculate REAL gain based on who actually plays
            # Case 1: Selling bench player, incoming would START → gain vs current starter
            # Case 2: Selling bench player, incoming would BENCH → gain = 0 (for this week)
            # Case 3: Selling starter, incoming would START → gain = incoming - outgoing
            # Case 4: Selling starter, incoming would BENCH → gain = 0 - outgoing (negative!)
            is_selling_bench_player = pick.get("position", 15) > 11
            real_gain = best_gain  # Default: incoming xP - outgoing xP

            if not would_start:
                # Incoming player would be on BENCH - no immediate xP gain this week
                if is_selling_bench_player:
                    # Bench to bench = 0 gain this week (better backup for auto-subs)
                    real_gain = 0
                else:
                    # Starter to bench = LOSE the outgoing player's points!
                    # This should rarely be recommended but let's be accurate
                    real_gain = -xp  # Losing xp from current starter
            elif is_selling_bench_player and would_start:
                # Bench player → Starter: the incoming player displaces the WORST current starter
                # Find current STARTERS at the incoming player's position (not all squad players)
                current_starters_at_pos = []
                for pl, pl_xp, pl_pick in my_players_with_xp:
                    if pl_pick.get("position", 15) <= 11:  # Only starters (positions 1-11)
                        if pl.position == rp.position and pl.id != p.id:
                            current_starters_at_pos.append((pl, projections.get(pl.id, 0)))

                # Sort ASCENDING - the LOWEST projected starter gets displaced
                current_starters_at_pos.sort(key=lambda x: x[1])

                if current_starters_at_pos:
                    # The WORST current starter at this position gets pushed to bench
                    displaced_player_xp = current_starters_at_pos[0][1]
                    # Real gain = incoming player's xP - displaced starter's xP
                    real_gain = rp_xp - displaced_player_xp
            # else: starter → starter = best_gain is already correct

            transfer_suggestions.append({
                "out": p,
                "out_xp": xp,
                "out_position": pick.get("position", 15),  # 1-11 = starter, 12-15 = bench
                "in": rp,
                "in_xp": rp_xp,
                "gain": real_gain,  # Use REAL gain, not naive calculation
                "naive_gain": best_gain,  # Keep for debugging
                "cost": rp.price - p.price,
                "would_start": would_start,
                "position_rank": player_rank + 1,
            })
            already_suggested_in.add(rp.id)
            already_suggested_out.add(p.id)

    # Sort by gain
    transfer_suggestions.sort(key=lambda x: -x["gain"])

    if transfer_suggestions:
        # Best single transfer
        best = transfer_suggestions[0]
        out_team = teams_dict.get(best["out"].team_id)
        in_team = teams_dict.get(best["in"].team_id)

        # Check if best transfer actually provides value
        if best["gain"] <= 0:
            st.info("👍 **No value transfers this week.** Your squad is well-optimized for the upcoming fixtures.")
            if best["gain"] == 0:
                st.caption("*Best available transfer is a bench upgrade with no immediate xP impact.*")
        else:
            # Get simulation data for transfer comparison
            try:
                from fpl_assistant.predictions.simulation import EventSimulator
                event_sim = EventSimulator()

                out_params = engine.get_simulation_params(best['out'], gw)
                in_params = engine.get_simulation_params(best['in'], gw)
                out_sim = event_sim.simulate_player(out_params, n_sims=5000)
                in_sim = event_sim.simulate_player(in_params, n_sims=5000)
                has_sim = True
            except Exception as e:
                has_sim = False
                logger.warning(f"Transfer simulation unavailable: {e}")

            # Show the recommended transfer
            st.subheader("Transfer 1 (Recommended):")
            col1, col2, col3 = st.columns([2, 1, 2])

            with col1:
                st.error(f"**SELL: {best['out'].web_name}**")
                st.caption(f"{out_team.short_name if out_team else '?'} | £{best['out'].price:.1f}m | {best['out_xp']:.1f} xP")
                # Show simulation metrics if available
                if has_sim:
                    st.caption(f"📊 P(haul): {out_sim.p_haul*100:.0f}% | Ceiling: {out_sim.percentile_90:.1f}")
                # Why sell
                sell_reasons = []
                if best['out'].form < 3:
                    sell_reasons.append(f"Poor form ({best['out'].form})")
                if best['out_xp'] < 3:
                    sell_reasons.append(f"Low projection ({best['out_xp']:.1f} xP)")
                if has_sim and out_sim.p_haul < 0.1:
                    sell_reasons.append(f"Low haul probability ({out_sim.p_haul*100:.0f}%)")
                # Add rotation risk as sell reason
                try:
                    from fpl_assistant.predictions.minutes import MinutesPredictor, RotationRisk
                    mp = MinutesPredictor(players)
                    out_pred = mp.predict_minutes(best['out'])
                    if out_pred.rotation_risk == RotationRisk.HIGH:
                        sell_reasons.append(f"High rotation risk (P(start)={out_pred.p_start*100:.0f}%)")
                    elif out_pred.rotation_risk == RotationRisk.MEDIUM and out_pred.p_start < 0.7:
                        sell_reasons.append(f"Rotation risk (P(start)={out_pred.p_start*100:.0f}%)")
                except:
                    pass
                st.caption(f"*Why: {', '.join(sell_reasons) if sell_reasons else 'Better options available'}*")

            with col2:
                st.markdown("### →")
                st.caption(f"+{best['gain']:.1f} xP")

            with col3:
                st.success(f"**BUY: {best['in'].web_name}**")
                # Show where they would fit in the team
                squad_role = "**STARTER**" if best.get('would_start', True) else "BENCH"
                pos_rank = best.get('position_rank', 1)
                st.caption(f"{in_team.short_name if in_team else '?'} | £{best['in'].price:.1f}m | {best['in_xp']:.1f} xP")
                # Show simulation metrics if available
                if has_sim:
                    st.caption(f"📊 P(haul): {in_sim.p_haul*100:.0f}% | Ceiling: {in_sim.percentile_90:.1f}")
                st.caption(f"📍 Role: {squad_role} (#{pos_rank} at {best['in'].position_name})")
                # Why buy
                buy_reasons = []
                if best['in'].form >= 5:
                    buy_reasons.append(f"Great form ({best['in'].form})")
                elif best['in'].form >= 3:
                    buy_reasons.append(f"Good form ({best['in'].form})")
                if best['gain'] > 0:
                    buy_reasons.append(f"Higher projection (+{best['gain']:.1f} xP)")
                if has_sim and in_sim.p_haul > out_sim.p_haul:
                    buy_reasons.append(f"Better haul chance (+{(in_sim.p_haul - out_sim.p_haul)*100:.0f}%)")
                # Add rotation risk info
                try:
                    from fpl_assistant.predictions.minutes import MinutesPredictor, RotationRisk
                    mp = MinutesPredictor(players)
                    in_pred = mp.predict_minutes(best['in'])
                    if in_pred.rotation_risk == RotationRisk.LOW:
                        buy_reasons.append(f"Nailed starter (P(start)={in_pred.p_start*100:.0f}%)")
                    elif in_pred.rotation_risk == RotationRisk.HIGH:
                        st.warning(f"⚠️ Rotation risk: P(start)={in_pred.p_start*100:.0f}%")
                except:
                    pass
                st.caption(f"*Why: {', '.join(buy_reasons)}*")

            # Show post-transfer XI preview if incoming player would start
            if best.get('would_start', False):
                # Calculate what the starting XI would look like after transfer
                incoming_pos = best['in'].position
                outgoing_id = best['out'].id
                incoming_player = best['in']
                incoming_xp = best['in_xp']

                # Get current starters at this position (excluding outgoing)
                current_starters = [(p, xp) for p, xp, pk in my_players_with_xp
                                    if pk.get("position", 15) <= 11 and p.position == incoming_pos and p.id != outgoing_id]
                # Add incoming player
                current_starters.append((incoming_player, incoming_xp))
                # Sort by xP
                current_starters.sort(key=lambda x: -x[1])

                # Get position limits based on typical formations
                pos_limits = {Position.GK: 1, Position.DEF: 4, Position.MID: 3, Position.FWD: 3}
                limit = pos_limits.get(incoming_pos, 3)

                # Show the new starting lineup at that position
                starters_after = current_starters[:limit]
                st.markdown(f"**After transfer, starting {incoming_pos.name}s:**")
                starter_names = [f"{p.web_name} ({xp:.1f})" for p, xp in starters_after]
                st.caption(" → ".join(starter_names))

            # ===========================================
            # HORIZON ANALYSIS - 5-Week Projection
            # ===========================================
            with st.expander("📅 5-Week Horizon Analysis", expanded=True):
                try:
                    # Calculate 5-GW projections for both players
                    horizon_gws = list(range(gw, min(gw + 5, 39)))  # Next 5 GWs (or until end of season)
                    out_5gw_total = 0.0
                    in_5gw_total = 0.0
                    out_fixtures = []
                    in_fixtures = []

                    for future_gw in horizon_gws:
                        out_xp_gw = engine.project_single_player(best['out'], future_gw)
                        in_xp_gw = engine.project_single_player(best['in'], future_gw)
                        out_5gw_total += out_xp_gw
                        in_5gw_total += in_xp_gw

                        # Get fixture info for display
                        gw_fixtures = [f for f in fixtures if f.gameweek == future_gw]
                        out_fix = next((f for f in gw_fixtures if f.home_team_id == best['out'].team_id or f.away_team_id == best['out'].team_id), None)
                        in_fix = next((f for f in gw_fixtures if f.home_team_id == best['in'].team_id or f.away_team_id == best['in'].team_id), None)

                        if out_fix:
                            is_home = out_fix.home_team_id == best['out'].team_id
                            opp_id = out_fix.away_team_id if is_home else out_fix.home_team_id
                            opp = teams_dict.get(opp_id)
                            fdr = out_fix.away_difficulty if is_home else out_fix.home_difficulty
                            venue = "H" if is_home else "A"
                            out_fixtures.append(f"{opp.short_name if opp else '?'}({venue})")
                        else:
                            out_fixtures.append("BGW")

                        if in_fix:
                            is_home = in_fix.home_team_id == best['in'].team_id
                            opp_id = in_fix.away_team_id if is_home else in_fix.home_team_id
                            opp = teams_dict.get(opp_id)
                            fdr = in_fix.away_difficulty if is_home else in_fix.home_difficulty
                            venue = "H" if is_home else "A"
                            in_fixtures.append(f"{opp.short_name if opp else '?'}({venue})")
                        else:
                            in_fixtures.append("BGW")

                    horizon_gain = in_5gw_total - out_5gw_total

                    # Display horizon comparison
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**{best['out'].web_name}** (SELL)")
                        st.caption(f"5-GW Total: {out_5gw_total:.1f} xP")
                        st.caption(f"Fixtures: {' | '.join(out_fixtures)}")
                    with col2:
                        st.markdown(f"**{best['in'].web_name}** (BUY)")
                        st.caption(f"5-GW Total: {in_5gw_total:.1f} xP")
                        st.caption(f"Fixtures: {' | '.join(in_fixtures)}")

                    st.markdown("---")
                    st.metric("5-Week Horizon Gain", f"+{horizon_gain:.1f} xP", delta=f"{horizon_gain - best['gain']:.1f} vs this week only")

                    # Recommendation based on horizon
                    if horizon_gain >= 5:
                        st.success(f"**DO IT NOW** - Strong 5-week value (+{horizon_gain:.1f} xP). Great fixture swing!")
                    elif horizon_gain >= 2:
                        st.info(f"**GOOD TRANSFER** - Solid horizon value (+{horizon_gain:.1f} xP)")
                    elif horizon_gain >= 0:
                        st.warning(f"**MARGINAL** - Only +{horizon_gain:.1f} xP over 5 weeks. Consider waiting for better opportunity.")
                    else:
                        st.error(f"**WAIT** - Negative horizon value ({horizon_gain:.1f} xP). This transfer hurts long-term!")

                except Exception as e:
                    st.caption(f"Horizon analysis unavailable: {e}")

            # ===========================================
            # MONTE CARLO TRANSFER ANALYSIS
            # ===========================================
            with st.expander("🎲 Monte Carlo Transfer Analysis (probability this transfer pays off)", expanded=False):
                try:
                    from fpl_assistant.predictions.uncertainty import MonteCarloSimulator

                    mc_sim = MonteCarloSimulator(players=players, projections=projections)

                    # Compare team scenarios: current team vs team after transfer
                    # Get current starting XI player IDs
                    current_xi_ids = [p.id for p, xp, pk in my_players_with_xp if pk.get("position", 15) <= 11]
                    current_captain_id = my_players_with_xp[0][0].id if my_players_with_xp else None

                    if current_xi_ids and current_captain_id:
                        # Create "after transfer" scenario
                        after_xi_ids = [pid for pid in current_xi_ids if pid != best['out'].id]
                        after_xi_ids.append(best['in'].id)

                        # Compare scenarios (5000 sims for speed)
                        scenarios = [
                            (current_xi_ids, current_captain_id, "Current Team"),
                            (after_xi_ids, current_captain_id, "After Transfer"),
                        ]
                        mc_results = mc_sim.compare_team_scenarios(scenarios, n_simulations=5000)

                        if mc_results:
                            current_result = next((r for r in mc_results if r.scenario_name == "Current Team"), None)
                            after_result = next((r for r in mc_results if r.scenario_name == "After Transfer"), None)

                            if current_result and after_result:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("**Current Team**")
                                    st.metric("Expected Points", f"{current_result.mean_total_points:.1f}")
                                    st.caption(f"Range: {current_result.p10_total:.1f} - {current_result.p90_total:.1f}")
                                    st.caption(f"Wins: {current_result.win_rate_vs_others:.0f}% of simulations")

                                with col2:
                                    st.markdown("**After Transfer**")
                                    st.metric("Expected Points", f"{after_result.mean_total_points:.1f}",
                                             delta=f"{after_result.mean_total_points - current_result.mean_total_points:+.1f}")
                                    st.caption(f"Range: {after_result.p10_total:.1f} - {after_result.p90_total:.1f}")
                                    st.caption(f"Wins: {after_result.win_rate_vs_others:.0f}% of simulations")

                                # Calculate probability transfer improves team
                                p_improves = after_result.win_rate_vs_others

                                st.markdown("---")
                                if p_improves >= 60:
                                    st.success(f"**{p_improves:.0f}% chance this transfer improves your gameweek** - Go for it!")
                                elif p_improves >= 50:
                                    st.info(f"**{p_improves:.0f}% chance this transfer improves your gameweek** - Slight edge")
                                elif p_improves >= 40:
                                    st.warning(f"**{p_improves:.0f}% chance this transfer improves your gameweek** - Marginal, consider waiting")
                                else:
                                    st.error(f"**{p_improves:.0f}% chance this transfer improves your gameweek** - Not worth it!")

                except Exception as e:
                    st.caption(f"Monte Carlo analysis unavailable: {e}")

        # Should you take hits?
        if len(transfer_suggestions) >= 2:
            st.markdown("---")

            # Use actual free transfers from API
            free_transfers = actual_free_transfers

            # Calculate optimal number of transfers considering hits
            st.subheader("🔄 Should you take hits for a better team?")

            # CRITICAL FIX: Only consider transfers where the player would actually START!
            # There's no point taking a -4 hit for a bench player who won't score
            starter_transfers = [t for t in transfer_suggestions if t.get("would_start", False)]

            # If no starter transfers beyond the first, no point evaluating hits
            if len(starter_transfers) < 2:
                st.info("👍 **No worthwhile hits available.** Additional transfers would only upgrade bench players.")
            else:
                # Evaluate taking 1, 2, 3, etc transfers (STARTERS ONLY)
                best_total_gain = starter_transfers[0]["gain"]  # 1 transfer with no hit
                best_num_transfers = 1

                for num_transfers in range(2, min(len(starter_transfers) + 1, 5)):
                    # Transfers beyond free ones cost -4 each
                    hits_needed = max(0, num_transfers - free_transfers)
                    hit_cost = hits_needed * 4

                    # Total gain from these transfers (STARTERS ONLY)
                    total_gain = sum(t["gain"] for t in starter_transfers[:num_transfers])
                    net_gain = total_gain - hit_cost

                    if net_gain > best_total_gain:
                        best_total_gain = net_gain
                        best_num_transfers = num_transfers

                if best_num_transfers > free_transfers:
                    hits_needed = best_num_transfers - free_transfers
                    hit_cost = hits_needed * 4
                    total_gain = sum(t["gain"] for t in starter_transfers[:best_num_transfers])
                    profit = total_gain - hit_cost

                    # Calculate probability of hit paying off using Monte Carlo
                    try:
                        from fpl_assistant.predictions.uncertainty import MonteCarloSimulator
                        import random

                        mc_sim = MonteCarloSimulator(players=players, projections=projections)
                        n_sims = 5000
                        hit_wins = 0

                        # Get distributions for the transfer pair
                        out_player = starter_transfers[1]["out"]
                        in_player = starter_transfers[1]["in"]
                        out_dist = mc_sim.uncertainty_model.estimate_distribution(out_player, starter_transfers[1]["out_xp"])
                        in_dist = mc_sim.uncertainty_model.estimate_distribution(in_player, starter_transfers[1]["in_xp"])

                        # Simulate: does incoming player beat outgoing + 4 hit cost?
                        for _ in range(n_sims):
                            out_pts = max(0, random.gauss(out_dist.mean, out_dist.std_dev))
                            in_pts = max(0, random.gauss(in_dist.mean, in_dist.std_dev))
                            # Add haul chance for attackers
                            if in_player.position.value >= 3:  # MID/FWD
                                if random.random() < 0.05:
                                    in_pts = max(in_pts, in_dist.mean * 2 + random.gauss(2, 1))
                            if in_pts > out_pts + 4:  # Hit pays off if gain > 4
                                hit_wins += 1

                        hit_prob = hit_wins / n_sims
                        prob_text = f" | P(hit pays off): {hit_prob*100:.0f}%"
                    except:
                        prob_text = ""

                    st.success(f"✅ **YES - Take a -{hit_cost} hit for {best_num_transfers} transfers!**")
                    st.caption(f"Combined xP gain: {total_gain:.1f} | Hit cost: -{hit_cost} | **Net profit: +{profit:.1f} xP**{prob_text}")
                    st.caption("*Only recommending hits for players who would START in your optimal XI*")

                    # Show each additional transfer (beyond the first one) - STARTERS ONLY
                    for i in range(1, best_num_transfers):
                        transfer = starter_transfers[i]
                        out_team = teams_dict.get(transfer["out"].team_id)
                        in_team = teams_dict.get(transfer["in"].team_id)

                        st.markdown(f"**Transfer {i + 1} (Additional Hit - STARTER):**")
                        col1, col2, col3 = st.columns([2, 1, 2])

                        with col1:
                            st.error(f"**SELL: {transfer['out'].web_name}**")
                            st.caption(f"{out_team.short_name if out_team else '?'} | £{transfer['out'].price:.1f}m | {transfer['out_xp']:.1f} xP")
                            # Why sell
                            sell_reasons = []
                            if transfer['out'].form < 3:
                                sell_reasons.append(f"Poor form ({transfer['out'].form})")
                            if transfer['out_xp'] < 3:
                                sell_reasons.append(f"Low projection ({transfer['out_xp']:.1f} xP)")
                            st.caption(f"*Why: {', '.join(sell_reasons) if sell_reasons else 'Better options available'}*")

                        with col2:
                            st.markdown("### →")
                            st.caption(f"+{transfer['gain']:.1f} xP")

                        with col3:
                            st.success(f"**BUY: {transfer['in'].web_name}**")
                            st.caption(f"{in_team.short_name if in_team else '?'} | £{transfer['in'].price:.1f}m | {transfer['in_xp']:.1f} xP")
                            # Why buy
                            buy_reasons = []
                            if transfer['in'].form >= 5:
                                buy_reasons.append(f"Great form ({transfer['in'].form})")
                            elif transfer['in'].form >= 3:
                                buy_reasons.append(f"Good form ({transfer['in'].form})")
                            buy_reasons.append(f"Higher projection (+{transfer['gain']:.1f} xP)")
                            st.caption(f"*Why: {', '.join(buy_reasons)}*")

                else:
                    # Check if any hit would be worth it for STARTERS
                    if len(starter_transfers) >= 2:
                        second = starter_transfers[1]
                        potential_gain = starter_transfers[0]["gain"] + second["gain"] - 4  # -4 for one hit

                        if potential_gain > starter_transfers[0]["gain"]:
                            st.info(f"📊 Taking a -4 hit would gain {potential_gain:.1f} xP net (marginal improvement)")
                        else:
                            st.warning(f"❌ **NO - Not worth taking hits.** Best 2nd starting transfer only gains {second['gain']:.1f} xP (need >4 to break even)")
                    else:
                        st.warning("❌ **NO - Not worth taking hits.** No additional starter upgrades available.")
    else:
        st.success("✅ Your team looks good! No urgent transfers needed.")

    # Hit Calculator - manual comparison tool
    with st.expander("🧮 Hit Calculator - Compare Any Two Players", expanded=False):
        st.markdown("*Manually calculate if a -4 hit is worth it for any transfer*")

        from fpl_assistant.predictions.transfers import calculate_hit_value

        col1, col2 = st.columns(2)

        # Get player lists for dropdowns
        available_to_sell = [p for p in players if p.id in my_player_ids]
        available_to_buy = [p for p in players if p.id not in my_player_ids and p.status == PlayerStatus.AVAILABLE]

        available_to_sell.sort(key=lambda p: p.web_name)
        available_to_buy.sort(key=lambda p: -projections.get(p.id, 0))

        with col1:
            st.markdown("**Player OUT**")
            out_options = {f"{p.web_name} ({p.position_name}, £{p.price:.1f}m)": p.id for p in available_to_sell}
            selected_out = st.selectbox("Sell", list(out_options.keys()), key="hit_calc_out")
            out_id = out_options.get(selected_out)

        with col2:
            st.markdown("**Player IN**")
            # Filter to same position if out player selected
            out_player = player_dict.get(out_id)
            if out_player:
                available_to_buy_filtered = [p for p in available_to_buy if p.position == out_player.position]
            else:
                available_to_buy_filtered = available_to_buy

            in_options = {f"{p.web_name} ({p.position_name}, £{p.price:.1f}m)": p.id for p in available_to_buy_filtered[:50]}
            selected_in = st.selectbox("Buy", list(in_options.keys()), key="hit_calc_in")
            in_id = in_options.get(selected_in)

        weeks_horizon = st.slider("Planning horizon (weeks)", 1, 8, 5, help="How many weeks to project the value over")

        if out_id and in_id and st.button("📊 Analyze Hit", key="analyze_hit"):
            out_player = player_dict.get(out_id)
            in_player = player_dict.get(in_id)

            if out_player and in_player:
                # Calculate multi-week projections
                out_total_xp = 0
                in_total_xp = 0

                for w in range(weeks_horizon):
                    future_gw = gw + w
                    try:
                        out_total_xp += engine.project_single_player(out_player, future_gw)
                        in_total_xp += engine.project_single_player(in_player, future_gw)
                    except:
                        out_total_xp += out_player.points_per_game
                        in_total_xp += in_player.points_per_game

                # Get hit analysis
                analysis = calculate_hit_value(
                    out_player, in_player, out_total_xp, in_total_xp, weeks_horizon
                )

                st.markdown("---")
                st.markdown("### Hit Analysis Results")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(f"{out_player.web_name} Total xP", f"{out_total_xp:.1f}")
                with col2:
                    st.metric(f"{in_player.web_name} Total xP", f"{in_total_xp:.1f}")
                with col3:
                    color = "normal" if analysis.net_gain > 0 else "inverse"
                    st.metric("Net Gain (after -4)", f"{analysis.net_gain:.1f}", delta_color=color)

                # Recommendation
                st.markdown("---")
                if analysis.recommendation.value == "TAKE HIT":
                    st.success(f"✅ **{analysis.recommendation.value}** - {analysis.explanation}")
                    if analysis.break_even_weeks:
                        st.caption(f"Break-even: {analysis.break_even_weeks:.1f} weeks | Confidence: {analysis.confidence}")
                elif analysis.recommendation.value == "AVOID HIT":
                    st.error(f"❌ **{analysis.recommendation.value}** - {analysis.explanation}")
                else:
                    st.warning(f"⚠️ **{analysis.recommendation.value}** - {analysis.explanation}")
                    if analysis.break_even_weeks:
                        st.caption(f"Break-even: {analysis.break_even_weeks:.1f} weeks | Confidence: {analysis.confidence}")

    st.markdown("---")

    # ===========================================
    # 3. STARTING XI & BENCH ORDER
    # ===========================================
    st.header("📝 3. YOUR STARTING XI")

    # Sort all players by position then xP
    my_players_with_xp.sort(key=lambda x: (-x[1]))  # Highest xP first

    # Pick best XI by position
    gk = [x for x in my_players_with_xp if x[0].position.value == 1]
    defs = [x for x in my_players_with_xp if x[0].position.value == 2]
    mids = [x for x in my_players_with_xp if x[0].position.value == 3]
    fwds = [x for x in my_players_with_xp if x[0].position.value == 4]

    # Standard 3-4-3 or adjust based on best players
    starting_xi = []
    bench = []

    # GK: 1 starter
    if gk:
        starting_xi.append(gk[0])
        bench.extend(gk[1:])

    # Try different formations to maximize points
    formations = [
        (3, 4, 3),
        (3, 5, 2),
        (4, 4, 2),
        (4, 3, 3),
        (5, 4, 1),
        (5, 3, 2),
    ]

    best_formation = None
    best_total = 0

    for n_def, n_mid, n_fwd in formations:
        if len(defs) >= n_def and len(mids) >= n_mid and len(fwds) >= n_fwd:
            total = sum(x[1] for x in defs[:n_def]) + sum(x[1] for x in mids[:n_mid]) + sum(x[1] for x in fwds[:n_fwd])
            if total > best_total:
                best_total = total
                best_formation = (n_def, n_mid, n_fwd)

    if best_formation:
        n_def, n_mid, n_fwd = best_formation
        starting_xi.extend(defs[:n_def])
        starting_xi.extend(mids[:n_mid])
        starting_xi.extend(fwds[:n_fwd])
        bench.extend(defs[n_def:])
        bench.extend(mids[n_mid:])
        bench.extend(fwds[n_fwd:])

    # Display starting XI
    st.markdown(f"**Formation: {best_formation[0]}-{best_formation[1]}-{best_formation[2]}**")

    xi_data = []
    total_xp = 0
    for p, xp, pick in starting_xi:
        team = teams_dict.get(p.team_id)
        captain_mark = ""
        if my_players_with_xp and p.id == my_players_with_xp[0][0].id:
            captain_mark = " 👑"
            total_xp += xp * 2  # Captain gets double
        else:
            total_xp += xp

        xi_data.append({
            "Pos": p.position_name,
            "Player": f"{p.web_name}{captain_mark}",
            "Team": team.short_name if team else "?",
            "xP": f"{xp:.1f}",
        })

    st.dataframe(xi_data, use_container_width=True, hide_index=True)
    st.metric("Projected Points", f"{total_xp:.1f}")

    # Bench order
    st.subheader("Bench Order:")
    bench.sort(key=lambda x: -x[1])  # Best first
    for i, (p, xp, pick) in enumerate(bench, 1):
        team = teams_dict.get(p.team_id)
        st.write(f"{i}. {p.web_name} ({team.short_name if team else '?'}) - {xp:.1f} xP")

    st.markdown("---")

    # ===========================================
    # 4. CHIP ADVICE
    # ===========================================
    st.header("🎴 4. CHIP ADVICE")

    # Check for upcoming blanks/doubles
    upcoming_gws = [db.get_gameweek(gw + i) for i in range(6)]
    upcoming_gws = [g for g in upcoming_gws if g]

    double_gws = [g for g in upcoming_gws if g.is_double]
    blank_gws = [g for g in upcoming_gws if g.is_blank]

    # Show blank/double alerts (these are important regardless of chip analysis)
    if double_gws:
        st.success(f"🔥 **Double Gameweek {double_gws[0].id} coming!** Consider Bench Boost or Triple Captain")
    if blank_gws:
        st.warning(f"⚠️ **Blank Gameweek {blank_gws[0].id} coming!** Consider Free Hit")

    # Detailed chip analysis - this runs the optimizer and gives actual recommendations
    with st.expander("📊 Detailed Chip Analysis", expanded=True):
        try:
            from fpl_assistant.optimizer.chips import ChipOptimizer
            from fpl_assistant.data.models import ChipType, Squad, SquadPlayer

            # Build squad from picks
            squad_players = []
            for i, pick in enumerate(picks):
                player = player_dict.get(pick["element"])
                if player:
                    squad_players.append(SquadPlayer(
                        player_id=player.id,
                        position=pick["position"],
                        purchase_price=player.price,
                        selling_price=player.price,
                    ))

            my_squad = Squad(
                players=squad_players,
                bank=bank,
                free_transfers=actual_free_transfers,
                total_value=sum(p.selling_price for p in squad_players),
                chips=[],  # TODO: Get actual chip status
            )

            # Build projections by gameweek
            projections_by_gw = {}
            for future_gw in range(gw, gw + 6):
                projections_by_gw[future_gw] = {}
                for p in players:
                    try:
                        projections_by_gw[future_gw][p.id] = engine.project_single_player(p, future_gw)
                    except:
                        projections_by_gw[future_gw][p.id] = p.form * 2

            all_gameweeks = [db.get_gameweek(i) for i in range(gw, gw + 6)]
            all_gameweeks = [g for g in all_gameweeks if g]

            chip_optimizer = ChipOptimizer(
                players=players,
                gameweeks=all_gameweeks,
                projections_by_gw=projections_by_gw,
            )

            # Get chip plan
            available_chips = [
                ChipType.BENCH_BOOST,
                ChipType.TRIPLE_CAPTAIN,
                ChipType.FREE_HIT,
                ChipType.WILDCARD,
            ]

            chip_plan = chip_optimizer.create_chip_timing_plan(
                squad=my_squad,
                start_gameweek=gw,
                horizon=6,
                available_chips=available_chips,
            )

            # 2025/26 Rule: Display deadline warnings for first-half chips
            if chip_plan.deadline_warnings:
                for warning in chip_plan.deadline_warnings:
                    if warning.urgency == "critical":
                        st.error(warning.message)
                    elif warning.urgency == "warning":
                        st.warning(warning.message)
                    else:
                        st.info(warning.message)
                st.markdown("---")

            # Display best timing for each chip
            st.markdown("**🎯 Optimal Chip Timing (Next 6 GWs)**")

            col1, col2 = st.columns(2)

            with col1:
                if chip_plan.best_bb_gw:
                    bb_rec = chip_plan.recommendations_by_gw.get(chip_plan.best_bb_gw)
                    bb_value = next((cv for cv in bb_rec.chip_values if cv.chip == ChipType.BENCH_BOOST), None) if bb_rec else None
                    if bb_value:
                        st.markdown(f"**Bench Boost:** GW{chip_plan.best_bb_gw} (+{bb_value.estimated_value:.1f} pts)")
                        st.caption(bb_value.reasoning)
                else:
                    st.markdown("**Bench Boost:** No clear opportunity")

                if chip_plan.best_tc_gw:
                    tc_rec = chip_plan.recommendations_by_gw.get(chip_plan.best_tc_gw)
                    tc_value = next((cv for cv in tc_rec.chip_values if cv.chip == ChipType.TRIPLE_CAPTAIN), None) if tc_rec else None
                    if tc_value:
                        st.markdown(f"**Triple Captain:** GW{chip_plan.best_tc_gw} (+{tc_value.estimated_value:.1f} pts)")
                        st.caption(tc_value.reasoning)
                else:
                    st.markdown("**Triple Captain:** No clear opportunity")

            with col2:
                if chip_plan.best_fh_gw:
                    fh_rec = chip_plan.recommendations_by_gw.get(chip_plan.best_fh_gw)
                    fh_value = next((cv for cv in fh_rec.chip_values if cv.chip == ChipType.FREE_HIT), None) if fh_rec else None
                    if fh_value:
                        st.markdown(f"**Free Hit:** GW{chip_plan.best_fh_gw} (+{fh_value.estimated_value:.1f} pts)")
                        st.caption(fh_value.reasoning)
                else:
                    st.markdown("**Free Hit:** No clear opportunity")

                if chip_plan.best_wc_gw:
                    wc_rec = chip_plan.recommendations_by_gw.get(chip_plan.best_wc_gw)
                    wc_value = next((cv for cv in wc_rec.chip_values if cv.chip == ChipType.WILDCARD), None) if wc_rec else None
                    if wc_value:
                        st.markdown(f"**Wildcard:** GW{chip_plan.best_wc_gw} (+{wc_value.estimated_value:.1f} pts)")
                        st.caption(wc_value.reasoning)
                else:
                    st.markdown("**Wildcard:** No urgent need")

            # This week's recommendation
            current_rec = chip_plan.recommendations_by_gw.get(gw)
            if current_rec:
                st.markdown("---")
                if current_rec.recommended_chip:
                    st.success(f"**This Week:** Use {current_rec.recommended_chip.value.upper()} - {current_rec.reasoning}")
                else:
                    st.info(f"**This Week:** {current_rec.reasoning}")

                if current_rec.save_for_later:
                    st.caption("💡 " + " | ".join(current_rec.save_for_later[:2]))

        except Exception as e:
            st.caption(f"Detailed chip analysis not available: {e}")

    st.markdown("---")

    # ===========================================
    # 5. COMPARE MY TEAM VS BEST SQUAD
    # ===========================================
    st.header("🔄 5. SQUAD COMPARISON - Do I Need a Wildcard?")
    st.caption("Compares your current squad to the mathematically optimal squad with the same budget")

    with st.expander("My Team vs Optimal Squad (5-week projection)", expanded=True):
        try:
            from fpl_assistant.optimizer.model import FPLOptimizer
            from fpl_assistant.predictions.enhanced_weights import (
                EnhancedSignalCalculator,
                convert_signals_to_projection,
                get_enhanced_weights,
            )
            from fpl_assistant.predictions.weight_optimizer import WeightOptimizer

            # Calculate total budget from current team
            my_team_value = sum(player_dict[pick["element"]].price for pick in picks if pick["element"] in player_dict)
            total_budget = my_team_value + bank

            # Get current team player IDs
            my_team_ids = {pick["element"] for pick in picks}

            # Use the SAME projection logic as Best Squad builder (Enhanced Signals + Price Floors)
            horizon = 5
            multi_gw_projections = {}

            # Initialize enhanced signal calculator (same as Best Squad)
            signal_calculator = EnhancedSignalCalculator(players, teams, fixtures)
            optimized_weights = WeightOptimizer.load_optimized_weights()
            weights_to_use = optimized_weights if optimized_weights else get_enhanced_weights()

            # Filter to available players with minimum minutes
            min_minutes = 180
            eligible_players = [p for p in players if p.status == PlayerStatus.AVAILABLE and p.minutes >= min_minutes]

            for p in eligible_players:
                total_xp = 0.0

                for i in range(horizon):
                    try:
                        # Use enhanced signals (same as Best Squad)
                        signals = signal_calculator.calculate_signals(p, gw + i)
                        enhanced_xp = convert_signals_to_projection(signals, weights_to_use, horizon=1)
                        total_xp += enhanced_xp
                    except:
                        total_xp += p.form * 2

                # Apply position-specific price floors with team strength (SAME as Best Squad)
                team_attack = engine._team_attack_strength.get(p.team_id, 1.0)
                team_defense = engine._team_defense_strength.get(p.team_id, 1.0)

                if p.position == Position.FWD:
                    base_ppg = 0.35 * p.price + 1.25
                    price_based_ppg = base_ppg * min(1.3, max(0.7, team_attack))
                elif p.position == Position.MID:
                    base_ppg = 0.40 * p.price + 0.8
                    team_factor = (team_attack * 0.8 + team_defense * 0.2)
                    price_based_ppg = base_ppg * min(1.3, max(0.7, team_factor))
                elif p.position == Position.DEF:
                    base_ppg = 0.35 * p.price + 1.5
                    team_factor = (team_defense * 0.7 + team_attack * 0.3)
                    price_based_ppg = base_ppg * min(1.3, max(0.7, team_factor))
                else:  # GK
                    base_ppg = 0.50 * p.price + 1.25
                    price_based_ppg = base_ppg * min(1.3, max(0.7, team_defense))

                price_based_xp = price_based_ppg * horizon
                total_xp = max(total_xp, price_based_xp)

                multi_gw_projections[p.id] = total_xp

            # Run optimizer to get best squad with same budget
            optimizer = FPLOptimizer(solver_time_limit=30)

            # Build players dict for optimizer (only eligible players)
            players_dict_opt = {p.id: p for p in eligible_players}

            # Run optimization (builds from scratch when current_squad=None)
            week_plan = optimizer.optimize_single_week(
                players=players_dict_opt,
                projections=multi_gw_projections,
                current_squad=None,  # Fresh optimization
                budget=total_budget,
            )

            if week_plan:
                # Get best squad player IDs from the WeekPlan (starting_xi + bench_order)
                best_squad_ids = set(week_plan.starting_xi + week_plan.bench_order)

                # Calculate comparison
                players_to_keep = my_team_ids & best_squad_ids
                players_to_sell = my_team_ids - best_squad_ids
                players_to_buy = best_squad_ids - my_team_ids

                # Calculate xP totals
                my_team_xp = sum(multi_gw_projections.get(pid, 0) for pid in my_team_ids)
                best_squad_xp = sum(multi_gw_projections.get(pid, 0) for pid in best_squad_ids)
                xp_gain = best_squad_xp - my_team_xp

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Your Team xP", f"{my_team_xp:.1f}", help=f"Over {horizon} GWs")
                with col2:
                    st.metric("Best Squad xP", f"{best_squad_xp:.1f}", help=f"Over {horizon} GWs")
                with col3:
                    delta_color = "normal" if xp_gain > 0 else "inverse"
                    st.metric("Potential Gain", f"+{xp_gain:.1f}" if xp_gain > 0 else f"{xp_gain:.1f}")
                with col4:
                    st.metric("Changes Needed", len(players_to_sell))

                st.markdown("---")

                # Recommendation based on number of changes
                if len(players_to_sell) == 0:
                    st.success("✅ **Your team IS the optimal squad!** No changes needed.")
                elif len(players_to_sell) <= 2:
                    st.info(f"🔄 **{len(players_to_sell)} transfers needed** - Use free transfers over next few weeks")
                elif len(players_to_sell) <= 4:
                    st.warning(f"⚠️ **{len(players_to_sell)} transfers needed** - Consider taking hits or saving WC")
                else:
                    st.error(f"🔥 **{len(players_to_sell)} transfers needed** - WILDCARD recommended!")

                # Show players to keep
                if players_to_keep:
                    st.markdown(f"### ✅ Keep ({len(players_to_keep)} players)")
                    keep_data = []
                    for pid in players_to_keep:
                        p = player_dict.get(pid)
                        if p:
                            team = teams_dict.get(p.team_id)
                            keep_data.append({
                                "Pos": p.position_name,
                                "Player": p.web_name,
                                "Team": team.short_name if team else "?",
                                "Price": f"£{p.price:.1f}m",
                                "xP (5GW)": f"{multi_gw_projections.get(pid, 0):.1f}",
                            })
                    keep_data.sort(key=lambda x: -float(x["xP (5GW)"]))
                    st.dataframe(keep_data, use_container_width=True, hide_index=True)

                # Show players to sell
                if players_to_sell:
                    st.markdown(f"### 🔴 Sell ({len(players_to_sell)} players)")
                    sell_data = []
                    for pid in players_to_sell:
                        p = player_dict.get(pid)
                        if p:
                            team = teams_dict.get(p.team_id)
                            sell_data.append({
                                "Pos": p.position_name,
                                "Player": p.web_name,
                                "Team": team.short_name if team else "?",
                                "Price": f"£{p.price:.1f}m",
                                "xP (5GW)": f"{multi_gw_projections.get(pid, 0):.1f}",
                            })
                    sell_data.sort(key=lambda x: float(x["xP (5GW)"]))  # Worst first
                    st.dataframe(sell_data, use_container_width=True, hide_index=True)

                # Show players to buy
                if players_to_buy:
                    st.markdown(f"### 🟢 Buy ({len(players_to_buy)} players)")
                    buy_data = []
                    for pid in players_to_buy:
                        p = player_dict.get(pid)
                        if p:
                            team = teams_dict.get(p.team_id)
                            buy_data.append({
                                "Pos": p.position_name,
                                "Player": p.web_name,
                                "Team": team.short_name if team else "?",
                                "Price": f"£{p.price:.1f}m",
                                "xP (5GW)": f"{multi_gw_projections.get(pid, 0):.1f}",
                            })
                    buy_data.sort(key=lambda x: -float(x["xP (5GW)"]))  # Best first
                    st.dataframe(buy_data, use_container_width=True, hide_index=True)

                # Position-by-position comparison
                st.markdown("---")
                st.markdown("### 📊 Position-by-Position Comparison")

                for pos_name, pos_val in [("GK", 1), ("DEF", 2), ("MID", 3), ("FWD", 4)]:
                    my_pos = [player_dict[pid] for pid in my_team_ids if player_dict.get(pid) and player_dict[pid].position.value == pos_val]
                    best_pos = [player_dict[pid] for pid in best_squad_ids if player_dict.get(pid) and player_dict[pid].position.value == pos_val]

                    my_names = sorted([p.web_name for p in my_pos])
                    best_names = sorted([p.web_name for p in best_pos])

                    if my_names == best_names:
                        st.markdown(f"**{pos_name}:** ✅ Same")
                    else:
                        my_only = set(p.web_name for p in my_pos) - set(p.web_name for p in best_pos)
                        best_only = set(p.web_name for p in best_pos) - set(p.web_name for p in my_pos)
                        if my_only or best_only:
                            changes = []
                            if my_only:
                                changes.append(f"OUT: {', '.join(my_only)}")
                            if best_only:
                                changes.append(f"IN: {', '.join(best_only)}")
                            st.markdown(f"**{pos_name}:** {' → '.join(changes)}")

            else:
                st.warning("Could not generate optimal squad for comparison")

        except Exception as e:
            st.caption(f"Squad comparison not available: {e}")
            import traceback
            st.caption(traceback.format_exc())

    st.markdown("---")

    # ===========================================
    # 6. LOOK AHEAD
    # ===========================================
    st.header("🔮 6. PLANNING AHEAD")

    # Show next 5 GW fixtures for top players
    st.markdown("**Best fixtures next 5 weeks:**")

    # Calculate multi-week projections
    multi_week = []
    for p in players:
        if p.status != PlayerStatus.AVAILABLE:
            continue
        total = 0
        for i in range(5):
            try:
                total += engine.project_single_player(p, gw + i)
            except:
                total += p.form * 2

        multi_week.append((p, total))

    multi_week.sort(key=lambda x: -x[1])

    # Show top 10 for each position
    for pos_name, pos_val in [("MID", 3), ("FWD", 4), ("DEF", 2)]:
        st.markdown(f"**{pos_name}:**")
        pos_players = [(p, t) for p, t in multi_week if p.position.value == pos_val][:5]
        for p, total in pos_players:
            team = teams_dict.get(p.team_id)
            in_team = "✓" if p.id in my_player_ids else ""
            st.write(f"• {p.web_name} ({team.short_name if team else '?'}) - {total:.1f} xP over 5 GWs {in_team}")

    st.markdown("---")

    # ===========================================
    # 6. MODEL PERFORMANCE - Backtest & Accuracy
    # ===========================================
    st.header("📊 7. MODEL ACCURACY")
    st.caption("*How accurate are our predictions? Run a backtest to find out.*")

    with st.expander("🔬 Backtest & Weight Optimization", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            num_gws = st.slider("Gameweeks to test", 2, 8, 4, help="More GWs = more reliable but slower")

        with col2:
            if st.button("🚀 Run Backtest", type="primary"):
                with st.spinner(f"Running backtest on last {num_gws} completed gameweeks..."):
                    try:
                        from fpl_assistant.predictions.backtest import run_backtest

                        result = run_backtest(num_gameweeks=num_gws, record_to_adaptive=True)
                        st.session_state.backtest_result = result
                        st.success("Backtest complete!")
                    except Exception as e:
                        st.error(f"Backtest failed: {e}")

        # Display backtest results if available
        if "backtest_result" in st.session_state:
            result = st.session_state.backtest_result

            st.markdown("### Results")

            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                # Correlation - higher is better (target: >0.40)
                corr_color = "green" if result.correlation > 0.40 else "orange" if result.correlation > 0.30 else "red"
                st.metric("Correlation", f"{result.correlation:.3f}",
                          help="Higher = better ranking ability. Target: >0.40, Elite: >0.45")
            with col2:
                # MAE - lower is better (target: <2.0)
                mae_color = "green" if result.mean_absolute_error < 2.0 else "orange" if result.mean_absolute_error < 2.5 else "red"
                st.metric("MAE", f"{result.mean_absolute_error:.2f}",
                          help="Mean Absolute Error. Lower = better. Target: <2.0")
            with col3:
                st.metric("Captain Top-5", f"{result.captain_accuracy*100:.0f}%",
                          help="% of time our #1 captain pick was in actual top 5 scorers")
            with col4:
                st.metric("Top-10 Hit Rate", f"{result.top_10_hit_rate*100:.0f}%",
                          help="% of top 10 predicted players in actual top 10")

            # Position breakdown
            st.markdown("**MAE by Position:**")
            pos_cols = st.columns(4)
            for i, (pos, mae) in enumerate(result.mae_by_position.items()):
                with pos_cols[i % 4]:
                    st.caption(f"{pos}: {mae:.2f}")

            # Interpretation
            st.markdown("---")
            st.markdown("**How to Read:**")
            if result.correlation >= 0.42:
                st.success("**Excellent** - Model is performing at elite level. Trust the projections.")
            elif result.correlation >= 0.35:
                st.info("**Good** - Model has solid predictive power. Consider as primary input.")
            elif result.correlation >= 0.25:
                st.warning("**Fair** - Model has some signal but consider other factors too.")
            else:
                st.error("**Poor** - Model may need weight optimization. Click optimize below.")

            # Weight optimization button
            st.markdown("---")
            if st.button("🔧 Optimize Weights"):
                with st.spinner("Running weight optimization (may take 1-2 minutes)..."):
                    try:
                        from fpl_assistant.predictions.adaptive import AdaptiveWeightManager

                        manager = AdaptiveWeightManager()
                        optimization_result = manager.run_weight_optimization(
                            gameweeks=[gw - i for i in range(1, min(6, gw))],
                            iterations=20,
                        )

                        if optimization_result:
                            st.success(f"Weights optimized! New correlation: {optimization_result.correlation:.3f}")
                            st.caption("New weights saved. Re-run backtest to verify improvement.")
                        else:
                            st.info("Current weights are already optimal (no improvement found).")
                    except Exception as e:
                        st.error(f"Optimization failed: {e}")

    st.markdown("---")

    # ===========================================
    # 8. xG REGRESSION WATCH
    # ===========================================
    st.header("📈 8. xG REGRESSION WATCH")
    st.caption("*Players over/under-performing their expected stats - due for regression*")

    with st.expander("🎯 Regression Analysis", expanded=False):
        try:
            from fpl_assistant.predictions.regression import (
                RegressionAnalyzer,
                RegressionType,
            )

            analyzer = RegressionAnalyzer(players)
            buy_targets = analyzer.get_buy_targets(10)
            sell_targets = analyzer.get_sell_targets(10)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🟢 Buy Targets (Due a Haul)")
                st.markdown("*Underperforming xG - likely to score soon*")

                if buy_targets:
                    import pandas as pd
                    buy_data = []
                    for candidate in buy_targets:
                        p = candidate.player
                        team = teams_dict.get(p.team_id)
                        buy_data.append({
                            "Player": p.web_name,
                            "Team": team.short_name if team else "?",
                            "Price": f"£{p.price:.1f}m",
                            "Goals": candidate.goals,
                            "xG": f"{candidate.expected_goals:.1f}",
                            "Diff": f"{candidate.goals_diff:+.1f}",
                        })
                    st.dataframe(pd.DataFrame(buy_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No clear buy targets based on xG regression")

            with col2:
                st.markdown("### 🔴 Sell Targets (Due to Blank)")
                st.markdown("*Overperforming xG - returns may decline*")

                if sell_targets:
                    import pandas as pd
                    sell_data = []
                    for candidate in sell_targets:
                        p = candidate.player
                        team = teams_dict.get(p.team_id)
                        sell_data.append({
                            "Player": p.web_name,
                            "Team": team.short_name if team else "?",
                            "Price": f"£{p.price:.1f}m",
                            "Goals": candidate.goals,
                            "xG": f"{candidate.expected_goals:.1f}",
                            "Diff": f"{candidate.goals_diff:+.1f}",
                        })
                    st.dataframe(pd.DataFrame(sell_data), use_container_width=True, hide_index=True)
                else:
                    st.info("No clear sell targets based on xG regression")

            st.markdown("---")
            st.markdown("""
            **How to use:**
            - **Buy targets** have fewer goals than expected (xG) - they're "due" for goals
            - **Sell targets** have more goals than expected - their scoring rate may decline
            - Negative diff = underperforming, Positive diff = overperforming
            """)

        except Exception as e:
            st.error(f"Failed to run regression analysis: {e}")


def show_best_squad():
    """Best possible squad for season start or wildcard."""
    st.title("🏆 Best Squad Builder")
    st.markdown("*Use this for season start or wildcard - maximizes total expected points*")

    db = get_db()
    if db.get_player_count() == 0:
        st.warning("No data loaded. Click 'Update' in the sidebar first.")
        return

    players = db.get_all_players()
    teams = db.get_all_teams()
    fixtures = db.get_all_fixtures()
    teams_dict = {t.id: t for t in teams}
    players_dict = {p.id: p for p in players}

    # Detect if it's season start (most players have 0 form)
    available_players = [p for p in players if p.status == PlayerStatus.AVAILABLE]
    avg_form = sum(p.form for p in available_players) / max(1, len(available_players))
    is_season_start = avg_form < 1.0

    if is_season_start:
        st.info("📅 **Season Start Mode**: Using price, ownership & fixtures (no form data yet)")
    else:
        st.info("📊 **In-Season Mode**: Using form, xG, ICT, and projections")

    # Budget settings
    col1, col2, col3 = st.columns(3)
    with col1:
        budget = st.slider("Budget (£m)", 95.0, 105.0, 100.0, 0.5)
    with col2:
        horizon = st.slider("Optimize for X gameweeks", 1, 10, 5)
    with col3:
        min_minutes = st.slider("Min season minutes", 0, 900, 180, 90,
                                help="Filter out players with too few minutes (avoids inflated xGI per 90)")

    from fpl_assistant.predictions import ProjectionEngine, EventSimulator, create_simulation_params
    from fpl_assistant.optimizer.model import FPLOptimizer
    from fpl_assistant.data.models import Position

    # Import enhanced signal system
    try:
        from fpl_assistant.predictions.enhanced_weights import (
            EnhancedSignalCalculator,
            EnhancedWeightConfig,
            convert_signals_to_projection,
            get_enhanced_weights,
        )
        from fpl_assistant.predictions.weight_optimizer import WeightOptimizer
        has_enhanced = True
    except ImportError:
        has_enhanced = False

    engine = ProjectionEngine(players, teams, fixtures)
    simulator = EventSimulator(seed=42)  # Reproducible results
    current_gw = db.get_current_gameweek()
    gw = current_gw.id if current_gw else 1

    # Projection method selection
    st.markdown("### 🎯 Projection Method")
    col_method1, col_method2 = st.columns(2)

    with col_method1:
        use_simulation = st.checkbox(
            "🎲 Monte Carlo Simulation",
            value=True,
            help="Run simulations per player for P(haul), ceiling stats."
        )

    with col_method2:
        use_enhanced_signals = st.checkbox(
            "📊 Enhanced Multi-Signal Model",
            value=has_enhanced,
            help="Use 10 predictive signals (form, xG, fixtures, team momentum, etc.)",
            disabled=not has_enhanced,
        )

    # Show weight configuration if enhanced mode
    if use_enhanced_signals and has_enhanced:
        with st.expander("⚙️ Signal Weights Configuration"):
            st.markdown("*Adjust how much each signal influences projections:*")

            # Try to load optimized weights
            optimized = WeightOptimizer.load_optimized_weights()
            default_weights = optimized if optimized else get_enhanced_weights()

            col_w1, col_w2 = st.columns(2)

            with col_w1:
                recent_form_w = st.slider("Recent Form (last 5 GWs)", 0.05, 0.40, default_weights.recent_form_weight, 0.01)
                rolling_xg_w = st.slider("Rolling xG/xA", 0.10, 0.45, default_weights.rolling_xg_weight, 0.01)
                fixture_diff_w = st.slider("Fixture Difficulty", 0.05, 0.30, default_weights.fixture_difficulty_weight, 0.01)
                team_momentum_w = st.slider("Team Momentum", 0.02, 0.20, default_weights.team_momentum_weight, 0.01)
                opp_weakness_w = st.slider("Opposition Weakness", 0.02, 0.15, default_weights.opposition_weakness_weight, 0.01)

            with col_w2:
                season_form_w = st.slider("Season Form", 0.02, 0.20, default_weights.season_form_weight, 0.01)
                ict_w = st.slider("ICT Index", 0.01, 0.15, default_weights.ict_index_weight, 0.01)
                home_away_w = st.slider("Home/Away", 0.01, 0.12, default_weights.home_away_weight, 0.01)
                mins_cert_w = st.slider("Minutes Certainty", 0.02, 0.12, default_weights.minutes_certainty_weight, 0.01)
                ownership_w = st.slider("Ownership Trend", 0.01, 0.10, default_weights.ownership_trend_weight, 0.01)

            custom_weights = EnhancedWeightConfig(
                recent_form_weight=recent_form_w,
                rolling_xg_weight=rolling_xg_w,
                fixture_difficulty_weight=fixture_diff_w,
                season_form_weight=season_form_w,
                ict_index_weight=ict_w,
                team_momentum_weight=team_momentum_w,
                home_away_weight=home_away_w,
                opposition_weakness_weight=opp_weakness_w,
                minutes_certainty_weight=mins_cert_w,
                ownership_trend_weight=ownership_w,
            ).normalize()

            # Show total and normalized weights
            st.caption(f"Weights normalized to sum to 1.0")

            # Add optimization button
            st.markdown("---")
            st.markdown("**🔬 Weight Optimization**")
            opt_iterations = st.slider("Optimization iterations", 10, 100, 30, 10,
                                       help="More iterations = better weights but slower")

            if st.button("🔬 Run Weight Optimization", help="Find optimal weights through simulated backtesting"):
                with st.spinner(f"Running {opt_iterations} optimization iterations..."):
                    # Create a simple scoring function based on current data
                    # Since we don't have full historical data, we'll optimize based on
                    # how well the projections match expected outcomes

                    best_score = 0.0
                    best_weights = custom_weights
                    optimization_log = []

                    import random

                    for iteration in range(opt_iterations):
                        # Generate random weight variation
                        test_weights = EnhancedWeightConfig(
                            recent_form_weight=random.uniform(0.15, 0.35),
                            rolling_xg_weight=random.uniform(0.18, 0.38),
                            fixture_difficulty_weight=random.uniform(0.10, 0.25),
                            season_form_weight=random.uniform(0.03, 0.15),
                            ict_index_weight=random.uniform(0.02, 0.12),
                            team_momentum_weight=random.uniform(0.04, 0.15),
                            home_away_weight=random.uniform(0.02, 0.10),
                            opposition_weakness_weight=random.uniform(0.03, 0.12),
                            minutes_certainty_weight=random.uniform(0.03, 0.10),
                            ownership_trend_weight=random.uniform(0.01, 0.08),
                        ).normalize()

                        # Score based on how well premium players rank
                        # (A good weighting should rank high-owned premiums higher)
                        test_calculator = EnhancedSignalCalculator(players, teams, fixtures)
                        score = 0.0
                        premium_scores = []
                        budget_scores = []

                        for p in players[:200]:  # Sample players
                            if p.status != PlayerStatus.AVAILABLE:
                                continue
                            try:
                                signals = test_calculator.calculate_signals(p, gw)
                                proj = convert_signals_to_projection(signals, test_weights, horizon=1)

                                # Premium players (high ownership, high price) should project higher
                                if p.price >= 10.0 and p.selected_by_percent > 15:
                                    premium_scores.append(proj)
                                elif p.price <= 5.0:
                                    budget_scores.append(proj)
                            except:
                                pass

                        # Good weights: premiums >> budget
                        if premium_scores and budget_scores:
                            avg_premium = sum(premium_scores) / len(premium_scores)
                            avg_budget = sum(budget_scores) / len(budget_scores)
                            ratio = avg_premium / max(0.1, avg_budget)

                            # Also reward variance (spread between best and worst)
                            all_scores = premium_scores + budget_scores
                            spread = max(all_scores) - min(all_scores) if all_scores else 0

                            score = ratio * 0.7 + (spread / 5) * 0.3

                        optimization_log.append({
                            "iteration": iteration + 1,
                            "score": score,
                            "ratio": ratio if premium_scores and budget_scores else 0,
                        })

                        if score > best_score:
                            best_score = score
                            best_weights = test_weights

                    # Show results
                    st.success(f"✅ Optimization complete! Best score: {best_score:.3f}")

                    col_opt1, col_opt2 = st.columns(2)
                    with col_opt1:
                        st.markdown("**Optimized Weights:**")
                        st.write(f"- Recent Form: {best_weights.recent_form_weight:.2f}")
                        st.write(f"- Rolling xG/xA: {best_weights.rolling_xg_weight:.2f}")
                        st.write(f"- Fixture Difficulty: {best_weights.fixture_difficulty_weight:.2f}")
                        st.write(f"- Team Momentum: {best_weights.team_momentum_weight:.2f}")
                        st.write(f"- Opposition Weakness: {best_weights.opposition_weakness_weight:.2f}")

                    with col_opt2:
                        st.markdown("**Secondary Weights:**")
                        st.write(f"- Season Form: {best_weights.season_form_weight:.2f}")
                        st.write(f"- ICT Index: {best_weights.ict_index_weight:.2f}")
                        st.write(f"- Home/Away: {best_weights.home_away_weight:.2f}")
                        st.write(f"- Minutes Certainty: {best_weights.minutes_certainty_weight:.2f}")
                        st.write(f"- Ownership Trend: {best_weights.ownership_trend_weight:.2f}")

                    st.info("👆 Copy these weights to the sliders above and click 'Build Best Squad'")

                    # Save optimized weights
                    try:
                        from fpl_assistant.predictions.weight_optimizer import WeightOptimizer
                        import json
                        from pathlib import Path
                        from datetime import datetime

                        cache_path = Path("data/optimized_weights.json")
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        data = {
                            "best_weights": best_weights.to_dict(),
                            "best_score": best_score,
                            "optimized_at": datetime.now().isoformat(),
                            "iterations": opt_iterations,
                        }
                        with open(cache_path, "w") as f:
                            json.dump(data, f, indent=2)
                        st.caption("💾 Weights saved to data/optimized_weights.json")
                    except Exception as e:
                        pass

    else:
        custom_weights = None

    if st.button("🚀 Build Best Squad", type="primary"):
        spinner_text = "Running enhanced multi-signal projections..." if use_enhanced_signals else (
            "Running Monte Carlo simulations..." if use_simulation else "Optimizing squad..."
        )
        with st.spinner(spinner_text):
            # Calculate multi-week projections for all available players
            projections = {}
            player_stats = {}  # For display
            simulation_results = {}  # Store simulation data
            signal_data = {}  # Store enhanced signals for display

            # Initialize enhanced signal calculator if enabled
            if use_enhanced_signals and has_enhanced:
                signal_calculator = EnhancedSignalCalculator(players, teams, fixtures)
                weights_to_use = custom_weights if custom_weights else get_enhanced_weights()
            else:
                signal_calculator = None
                weights_to_use = None

            # Pre-filter players to reduce simulation load
            eligible_players = []
            for p in players:
                if p.status != PlayerStatus.AVAILABLE:
                    continue
                if not is_season_start and p.minutes < min_minutes:
                    continue
                eligible_players.append(p)

            progress_bar = st.progress(0, text="Calculating projections...")
            total_players = len(eligible_players)

            for idx, p in enumerate(eligible_players):
                # Update progress
                if idx % 50 == 0:
                    progress_bar.progress(
                        idx / total_players,
                        text=f"Projecting {p.web_name}... ({idx}/{total_players})"
                    )

                total_xp = 0.0
                total_p_haul = 0.0
                total_ceiling = 0.0

                for i in range(horizon):
                    try:
                        # PRIMARY: Use enhanced signals if enabled
                        if signal_calculator and weights_to_use:
                            signals = signal_calculator.calculate_signals(p, gw + i)
                            enhanced_xp = convert_signals_to_projection(signals, weights_to_use, horizon=1)

                            # Store signal data for first GW only (for display)
                            if i == 0:
                                signal_data[p.id] = signals

                            total_xp += enhanced_xp
                        else:
                            # Fallback to standard engine
                            base_xp = engine.project_single_player(p, gw + i)
                            total_xp += base_xp

                        # Add simulation overlay if enabled
                        if use_simulation:
                            try:
                                sim_params = engine.get_simulation_params(p, gw + i)
                                sim_result = simulator.simulate_player(sim_params, n_sims=2000)

                                total_p_haul += sim_result.p_haul
                                total_ceiling += sim_result.percentile_90

                                if p.id not in simulation_results:
                                    simulation_results[p.id] = sim_result
                            except Exception:
                                pass  # Simulation optional

                    except Exception:
                        # Fallback to base fallback
                        total_xp += 3.0

                # Position-specific price-based FLOOR (ensures premiums aren't undervalued)
                # IMPORTANT: Scale by team strength so weak teams get lower floors
                team_attack = engine._team_attack_strength.get(p.team_id, 1.0)
                team_defense = engine._team_defense_strength.get(p.team_id, 1.0)

                if p.position == Position.FWD:
                    # FWDs benefit from team attack strength
                    base_ppg = 0.35 * p.price + 1.25
                    price_based_ppg = base_ppg * min(1.3, max(0.7, team_attack))
                elif p.position == Position.MID:
                    # MIDs benefit from both attack (goals/assists) and slight defense (CS point)
                    base_ppg = 0.40 * p.price + 0.8
                    team_factor = (team_attack * 0.8 + team_defense * 0.2)
                    price_based_ppg = base_ppg * min(1.3, max(0.7, team_factor))
                elif p.position == Position.DEF:
                    # DEFs benefit mainly from team defense (clean sheets)
                    # Crystal Palace has weak defense → lower CS probability → lower floor
                    base_ppg = 0.35 * p.price + 1.5
                    team_factor = (team_defense * 0.7 + team_attack * 0.3)  # Some attacking DEFs
                    price_based_ppg = base_ppg * min(1.3, max(0.7, team_factor))
                else:  # GK
                    # GKs depend heavily on team defense
                    base_ppg = 0.50 * p.price + 1.25
                    price_based_ppg = base_ppg * min(1.3, max(0.7, team_defense))

                price_based_xp = price_based_ppg * horizon

                # Season start: boost high-ownership template players
                if is_season_start:
                    if p.selected_by_percent > 20:
                        ownership_factor = 1.15
                    elif p.selected_by_percent > 10:
                        ownership_factor = 1.08
                    elif p.selected_by_percent > 5:
                        ownership_factor = 1.03
                    else:
                        ownership_factor = 1.0
                    price_based_xp *= ownership_factor

                # Use higher of simulation/projection or price floor
                total_xp = max(total_xp, price_based_xp)

                projections[p.id] = total_xp
                player_stats[p.id] = {
                    "player": p,
                    "xp": total_xp,
                    "xp_per_m": total_xp / p.price if p.price > 0 else 0,
                    "p_haul": total_p_haul / horizon if use_simulation else None,
                    "ceiling": total_ceiling / horizon if use_simulation else None,
                }

            progress_bar.progress(1.0, text="Optimization complete!")

            if len(projections) < 15:
                st.error(f"Not enough players pass filters ({len(projections)} found, need 15+). Try reducing min minutes.")
                return

            # Use MILP optimizer to maximize TOTAL expected points (not value per pound)
            optimizer = FPLOptimizer(solver_time_limit=30)

            # Build from scratch (no current squad) = Wildcard/season start scenario
            week_plan = optimizer.optimize_single_week(
                players=players_dict,
                projections=projections,
                current_squad=None,  # No existing squad - build from scratch
                budget=budget,
                chip=None,  # No chip for this optimization
                allow_hits=False,
            )

            if week_plan is None:
                st.error("Optimization failed. Try adjusting budget or filters.")
                return

            # Extract selected players
            selected = []
            for pid in week_plan.starting_xi + week_plan.bench_order:
                p = players_dict.get(pid)
                if p:
                    xp = projections.get(pid, 0)
                    selected.append((p, xp))

            if len(selected) < 15:
                st.error(f"Only selected {len(selected)} players (need 15). Try adjusting filters.")
                return

            # Calculate total cost
            total_cost = sum(p.price for p, _ in selected)

            # Display results
            st.success(f"**Squad built!** Total cost: £{total_cost:.1f}m | Left in bank: £{budget - total_cost:.1f}m")

            st.markdown("---")

            # Sort by position for display
            selected.sort(key=lambda x: (x[0].position.value, -x[1]))

            total_xp = sum(xp for p, xp in selected)
            squad_data = []

            for p, xp in selected:
                team = teams_dict.get(p.team_id)
                is_starter = p.id in week_plan.starting_xi
                is_captain = p.id == week_plan.captain_id
                is_vice = p.id == week_plan.vice_captain_id

                role = ""
                if is_captain:
                    role = "👑 (C)"
                elif is_vice:
                    role = "⭐ (VC)"
                elif not is_starter:
                    role = "📋 Bench"

                # Get simulation stats if available
                stats = player_stats.get(p.id, {})
                p_haul = stats.get("p_haul")
                ceiling = stats.get("ceiling")

                row = {
                    "Pos": p.position_name,
                    "Player": p.web_name,
                    "Team": team.short_name if team else "?",
                    "Price": f"£{p.price:.1f}m",
                    f"xP ({horizon} GWs)": f"{xp:.1f}",
                }

                # Add simulation columns if available
                if use_simulation and p_haul is not None:
                    row["P(haul)"] = f"{p_haul*100:.0f}%"
                    row["90th %ile"] = f"{ceiling:.1f}"

                row["Ownership"] = f"{p.selected_by_percent:.1f}%"
                row["Role"] = role

                squad_data.append(row)

            st.dataframe(squad_data, use_container_width=True, hide_index=True)

            # Show signal breakdown if enhanced mode was used
            if use_enhanced_signals and signal_data:
                with st.expander("📊 Signal Breakdown - Why These Players?"):
                    st.markdown("*Each signal is scored 0-10. Higher = stronger signal.*")

                    signal_rows = []
                    for p, xp in selected[:11]:  # Starting XI only
                        signals = signal_data.get(p.id)
                        if signals:
                            signal_rows.append({
                                "Player": p.web_name,
                                "Form (5GW)": f"{signals.recent_form:.1f}",
                                "xG/xA": f"{signals.rolling_xg:.1f}",
                                "Fixture": f"{signals.fixture_difficulty:.1f}",
                                "Team Form": f"{signals.team_momentum:.1f}",
                                "Opp Weak": f"{signals.opposition_weakness:.1f}",
                                "Mins Cert": f"{signals.minutes_certainty:.1f}",
                                "Weighted": f"{signals.weighted_score(weights_to_use):.1f}",
                            })

                    if signal_rows:
                        st.dataframe(signal_rows, use_container_width=True, hide_index=True)

                        st.markdown("""
                        **Signal Legend:**
                        - **Form (5GW)**: Recent points performance (last 5 gameweeks)
                        - **xG/xA**: Expected goals + assists per 90 minutes
                        - **Fixture**: Difficulty of upcoming fixture (10 = very easy)
                        - **Team Form**: Team's recent results momentum
                        - **Opp Weak**: Opposition defensive vulnerability
                        - **Mins Cert**: Likelihood of playing 60+ minutes
                        """)

            # Show key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"Total xP ({horizon} GWs)", f"{total_xp:.1f}")
            with col2:
                captain = players_dict.get(week_plan.captain_id)
                if captain:
                    team = teams_dict.get(captain.team_id)
                    st.metric("Captain", f"{captain.web_name} ({team.short_name if team else '?'})")

            with col3:
                # Captain's xP doubled
                captain_xp = projections.get(week_plan.captain_id, 0)
                starting_xi_xp = sum(projections.get(pid, 0) for pid in week_plan.starting_xi)
                total_with_captain = starting_xi_xp + captain_xp  # Captain counted twice
                st.metric("Starting XI + Captain", f"{total_with_captain:.1f}")

            # Monte Carlo Simulation Analysis
            if use_simulation and simulation_results:
                with st.expander("🎲 Monte Carlo Simulation Analysis", expanded=True):
                    st.markdown("### Captain Candidates by Ceiling Score")
                    st.markdown("*Ceiling score = 60% expected + 30% 90th percentile + 10% P(haul)*")

                    # Get simulation results for starters
                    captain_candidates = []
                    for pid in week_plan.starting_xi:
                        if pid in simulation_results:
                            p = players_dict.get(pid)
                            sim = simulation_results[pid]
                            if p:
                                captain_candidates.append({
                                    "Player": p.web_name,
                                    "Team": teams_dict.get(p.team_id).short_name if teams_dict.get(p.team_id) else "?",
                                    "xP": f"{sim.expected_points:.1f}",
                                    "P(haul)": f"{sim.p_haul*100:.0f}%",
                                    "P(blank)": f"{sim.p_blank*100:.0f}%",
                                    "90th %ile": f"{sim.percentile_90:.1f}",
                                    "Ceiling Score": f"{sim.ceiling_score:.1f}",
                                    "_ceiling": sim.ceiling_score,
                                })

                    if captain_candidates:
                        # Sort by ceiling score
                        captain_candidates.sort(key=lambda x: -x["_ceiling"])

                        # Remove internal sort key
                        for c in captain_candidates:
                            del c["_ceiling"]

                        st.dataframe(captain_candidates[:8], use_container_width=True, hide_index=True)

                        # Recommendation
                        best = captain_candidates[0]
                        st.success(f"**Recommended Captain:** {best['Player']} - {best['P(haul)']} haul probability, {best['90th %ile']} ceiling")

            # Show why premium players weren't selected (diagnostic)
            selected_ids = {p.id for p, _ in selected}

            # Formation breakdown
            from fpl_assistant.data.models import Position
            formation_counts = {Position.GK: 0, Position.DEF: 0, Position.MID: 0, Position.FWD: 0}
            starters_by_pos = {Position.GK: [], Position.DEF: [], Position.MID: [], Position.FWD: []}
            bench_by_pos = {Position.GK: [], Position.DEF: [], Position.MID: [], Position.FWD: []}

            for p, xp in selected:
                if p.id in week_plan.starting_xi:
                    formation_counts[p.position] += 1
                    starters_by_pos[p.position].append((p, xp))
                else:
                    bench_by_pos[p.position].append((p, xp))

            formation_str = f"{formation_counts[Position.DEF]}-{formation_counts[Position.MID]}-{formation_counts[Position.FWD]}"
            st.info(f"**Formation: {formation_str}** (1 GK + {formation_counts[Position.DEF]} DEF + {formation_counts[Position.MID]} MID + {formation_counts[Position.FWD]} FWD)")

            with st.expander("🔍 Formation Analysis - Why are forwards benched?"):
                st.markdown("### Starting XI vs Bench Comparison")

                # Show benched forwards vs lowest starting mids
                benched_fwds = bench_by_pos[Position.FWD]
                starting_mids = starters_by_pos[Position.MID]

                if benched_fwds and starting_mids:
                    starting_mids.sort(key=lambda x: x[1])  # Lowest xP first
                    benched_fwds.sort(key=lambda x: -x[1])  # Highest xP first

                    st.markdown("**Benched Forwards:**")
                    for p, xp in benched_fwds:
                        team = teams_dict.get(p.team_id)
                        st.write(f"• {p.web_name} ({team.short_name if team else '?'}) - {xp:.1f} xP")

                    st.markdown("**Lowest Starting Midfielders:**")
                    for p, xp in starting_mids[:2]:  # Show 2 lowest
                        team = teams_dict.get(p.team_id)
                        st.write(f"• {p.web_name} ({team.short_name if team else '?'}) - {xp:.1f} xP")

                    # Calculate if swapping would help
                    if benched_fwds and starting_mids:
                        best_benched_fwd = benched_fwds[0]
                        worst_starting_mid = starting_mids[0]
                        xp_diff = best_benched_fwd[1] - worst_starting_mid[1]

                        if xp_diff > 0:
                            st.warning(f"⚠️ Your benched forward ({best_benched_fwd[0].web_name}) has HIGHER xP than lowest mid ({worst_starting_mid[0].web_name}) by {xp_diff:.1f}. This might indicate a bug.")
                        else:
                            st.success(f"✅ Formation is optimal: {worst_starting_mid[0].web_name} ({worst_starting_mid[1]:.1f} xP) > {best_benched_fwd[0].web_name} ({best_benched_fwd[1]:.1f} xP)")

                st.markdown("---")
                st.markdown("**Why midfielders often outscore forwards in FPL:**")
                st.write("• MIDs get **5 pts/goal**, FWDs get **4 pts/goal**")
                st.write("• MIDs get **1 pt for clean sheets**, FWDs get **0**")
                st.write("• Premium MIDs (Salah, Palmer) have similar xGI to forwards")
                st.write("• Playing 5 MIDs (4-5-1 or 3-5-2) is often mathematically optimal!")

            with st.expander("📊 Top Projections Not Selected (debug)"):
                # Find highest projected players not in squad
                not_selected = [(players_dict[pid], xp) for pid, xp in projections.items()
                                if pid not in selected_ids and pid in players_dict]
                not_selected.sort(key=lambda x: -x[1])

                st.markdown("**Highest xP players NOT in your squad:**")
                for p, xp in not_selected[:10]:
                    team = teams_dict.get(p.team_id)
                    status = "⚠️ " + p.status.name if p.status != PlayerStatus.AVAILABLE else ""
                    st.write(f"• {p.web_name} ({team.short_name if team else '?'}) - £{p.price:.1f}m - {xp:.1f} xP - {p.selected_by_percent:.1f}% owned {status}")

                st.markdown("---")
                st.markdown("**Why weren't they selected?**")
                st.write("The optimizer maximizes TOTAL team xP subject to:")
                st.write("• £100m budget constraint")
                st.write("• Max 3 players per team")
                st.write("• Position limits (2 GK, 5 DEF, 5 MID, 3 FWD)")
                st.write("")
                st.write("Sometimes 3x £7m players outscore 1x £15m + 2x £4m players!")
            with col3:
                # Calculate xP of starting XI only
                starting_xp = sum(projections.get(pid, 0) for pid in week_plan.starting_xi)
                captain_bonus = projections.get(week_plan.captain_id, 0)
                st.metric("Starting XI + Captain", f"{starting_xp + captain_bonus:.1f}")

            # Export option
            st.markdown("---")
            if st.button("📥 Export Squad"):
                import csv
                import io
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(["Pos", "Player", "Team", "Price", f"xP ({horizon} GWs)", "Ownership"])
                for row in squad_data:
                    writer.writerow([row["Pos"], row["Player"], row["Team"], row["Price"],
                                   row[f"xP ({horizon} GWs)"], row["Ownership"]])
                st.download_button(
                    label="Download CSV",
                    data=output.getvalue(),
                    file_name="best_squad.csv",
                    mime="text/csv"
                )


def show_my_team():
    """My Team page - fetch and display user's actual FPL team."""
    st.title("👤 My Team")

    db = get_db()
    if db.get_player_count() == 0:
        st.warning("No data loaded. Click 'Update' in the sidebar first.")
        return

    manager_id = get_manager_id()

    # Show any previous errors
    if "fetch_error" in st.session_state:
        st.error(f"Failed to fetch team: {st.session_state.fetch_error}")
        with st.expander("Error details"):
            st.code(st.session_state.get("fetch_error_detail", "No details"))

    # Fetch team from API
    if st.button("🔄 Fetch My Team from FPL", type="primary"):
        fetch_my_team(manager_id)

    # Display stored team if exists
    if "my_team" in st.session_state and st.session_state.my_team:
        display_my_team()
    else:
        st.info("Click the button above to fetch your team from FPL.")


def fetch_my_team(manager_id: int):
    """Fetch team from FPL API."""
    import httpx

    # Clear any previous errors
    if "fetch_error" in st.session_state:
        del st.session_state.fetch_error

    try:
        with st.spinner("Fetching your team..."):
            # Add cache-busting headers to get fresh data
            import time
            headers = {
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "User-Agent": "FPL-Assistant/1.0",
            }
            with httpx.Client(timeout=30.0, headers=headers) as client:
                # Get current gameweek from bootstrap
                bootstrap_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
                bootstrap_resp = client.get(bootstrap_url)
                bootstrap_resp.raise_for_status()
                bootstrap = bootstrap_resp.json()
                events = bootstrap.get("events", [])

                # Find current and next gameweek
                current_gw = None
                next_gw = None
                for e in events:
                    if e.get("is_current"):
                        current_gw = e["id"]
                    if e.get("is_next"):
                        next_gw = e["id"]

                if not current_gw and not next_gw:
                    # Fallback: find latest finished gameweek
                    finished = [e for e in events if e.get("finished")]
                    if finished:
                        current_gw = finished[-1]["id"]
                    else:
                        current_gw = 1

                # Determine gameweeks
                target_gw = next_gw or current_gw or 1
                last_completed_gw = current_gw if current_gw else (next_gw - 1 if next_gw else 1)

                st.info(f"Fetching squad for GW{target_gw}...")

                # Step 1: Try to get picks for TARGET (upcoming) gameweek directly
                # This should have the most up-to-date squad including recent transfers
                picks_url = f"https://fantasy.premierleague.com/api/entry/{manager_id}/event/{target_gw}/picks/"
                response = client.get(picks_url)

                picks_source = "target_gw"
                all_transfers = []
                unique_events = []
                last_3_transfers = []
                recent_transfers = []

                if response.status_code == 200:
                    # Success! Use target GW picks directly
                    picks_data = response.json()
                    picks = picks_data.get("picks", [])
                    st.info(f"Got GW{target_gw} picks directly from API")
                else:
                    # Fall back to last completed GW + apply transfers
                    st.info(f"GW{target_gw} picks not available (status {response.status_code}), falling back to GW{last_completed_gw}...")
                    picks_source = "fallback"

                    picks_url = f"https://fantasy.premierleague.com/api/entry/{manager_id}/event/{last_completed_gw}/picks/"
                    response = client.get(picks_url)
                    response.raise_for_status()
                    picks_data = response.json()
                    picks = picks_data.get("picks", [])

                    # Get transfers to apply
                    transfers_url = f"https://fantasy.premierleague.com/api/entry/{manager_id}/transfers/"
                    transfers_resp = client.get(transfers_url)
                    transfers_resp.raise_for_status()
                    all_transfers = transfers_resp.json()

                    last_3_transfers = all_transfers[-3:] if all_transfers else []
                    unique_events = sorted(set(t.get("event") for t in all_transfers))
                    recent_transfers = [t for t in all_transfers if t.get("event") == target_gw]

                # Step 2: Apply transfers if using fallback
                if recent_transfers:
                    st.info(f"Applying {len(recent_transfers)} transfer(s) for GW{target_gw}...")
                    pick_elements = {p["element"]: p for p in picks}

                    for transfer in recent_transfers:
                        out_id = transfer.get("element_out")
                        in_id = transfer.get("element_in")

                        if out_id in pick_elements:
                            # Replace the outgoing player with incoming
                            old_pick = pick_elements[out_id]
                            new_pick = old_pick.copy()
                            new_pick["element"] = in_id
                            pick_elements[in_id] = new_pick
                            del pick_elements[out_id]
                            st.write(f"DEBUG: Applied transfer: {out_id} → {in_id}")

                    # Rebuild picks list
                    picks = list(pick_elements.values())

                # Update picks_data with modified picks
                picks_data["picks"] = picks

                # Get manager info
                entry_url = f"https://fantasy.premierleague.com/api/entry/{manager_id}/"
                entry_response = client.get(entry_url)
                entry_response.raise_for_status()
                entry_data = entry_response.json()

                # Debug info - store in session so it persists after rerun
                pick_ids = [p.get("element") for p in picks]
                st.session_state.fetch_debug = {
                    "picks_source": picks_source,  # "target_gw" or "fallback"
                    "pick_ids": pick_ids,
                    "transfers_for_gw": [(t['element_out'], t['element_in']) for t in recent_transfers] if recent_transfers else [],
                    "all_transfers_count": len(all_transfers),
                    "target_gw": target_gw,
                    "base_gw": last_completed_gw,
                    "last_3_transfers": [(t.get('event'), t.get('element_out'), t.get('element_in')) for t in last_3_transfers],
                    "unique_events": unique_events,
                }

                # Store in session
                st.session_state.my_team = {
                    "picks": picks_data.get("picks", []),
                    "entry_history": picks_data.get("entry_history", {}),
                    "manager": entry_data,
                    "gameweek": target_gw,
                }

                transfers_applied = len(recent_transfers) if recent_transfers else 0
                st.success(f"Loaded team for GW{target_gw}! ({len(pick_ids)} players, {transfers_applied} transfers applied)")
                st.rerun()

    except Exception as e:
        # Store error in session state so it persists after rerun
        st.session_state.fetch_error = str(e)
        import traceback
        st.session_state.fetch_error_detail = traceback.format_exc()


def display_my_team():
    """Display the user's team."""
    db = get_db()
    team_data = st.session_state.my_team
    picks = team_data["picks"]
    entry_history = team_data.get("entry_history", {})
    manager = team_data.get("manager", {})

    # Manager info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Team", manager.get("name", "Unknown"))
    with col2:
        st.metric("Overall Rank", f"{manager.get('summary_overall_rank', 0):,}")
    with col3:
        st.metric("Total Points", manager.get("summary_overall_points", 0))
    with col4:
        st.metric("Bank", f"£{entry_history.get('bank', 0) / 10:.1f}m")

    st.markdown("---")

    # Get player details
    all_players = db.get_all_players()
    player_dict = {p.id: p for p in all_players}
    teams_dict = {t.id: t for t in db.get_all_teams()}
    fixtures = db.get_all_fixtures()

    # Generate projections for the squad
    from fpl_assistant.predictions import ProjectionEngine
    engine = ProjectionEngine(all_players, list(teams_dict.values()), fixtures)
    current_gw = db.get_current_gameweek()
    gw = current_gw.id if current_gw else 1

    projections = {}
    for pick in picks:
        player = player_dict.get(pick["element"])
        if player:
            try:
                projections[player.id] = engine.project_single_player(player, gw)
            except:
                projections[player.id] = player.form * 2

    # Starting XI
    st.subheader("Starting XI")
    starting = [p for p in picks if p["position"] <= 11]
    bench = [p for p in picks if p["position"] > 11]

    xi_data = []
    total_xp = 0
    for pick in sorted(starting, key=lambda x: x["position"]):
        player = player_dict.get(pick["element"])
        if player:
            team = teams_dict.get(player.team_id)
            captain = "©" if pick.get("is_captain") else ("(VC)" if pick.get("is_vice_captain") else "")
            xp = projections.get(player.id, 0)
            # Captain gets double points
            if pick.get("is_captain"):
                total_xp += xp * 2
            else:
                total_xp += xp
            xi_data.append({
                "Pos": player.position_name,
                "Player": f"{player.web_name} {captain}",
                "Team": team.short_name if team else "?",
                "Price": f"£{player.price:.1f}m",
                "xP": f"{xp:.1f}",
                "Form": player.form,
                "Status": "✅" if player.status == PlayerStatus.AVAILABLE else "⚠️",
            })

    st.dataframe(xi_data, use_container_width=True, hide_index=True)
    st.metric("Projected Starting XI Total", f"{total_xp:.1f} xP")

    # Bench
    st.subheader("Bench")
    bench_data = []
    for pick in sorted(bench, key=lambda x: x["position"]):
        player = player_dict.get(pick["element"])
        if player:
            team = teams_dict.get(player.team_id)
            xp = projections.get(player.id, 0)
            bench_data.append({
                "Order": pick["position"] - 11,
                "Pos": player.position_name,
                "Player": player.web_name,
                "Team": team.short_name if team else "?",
                "xP": f"{xp:.1f}",
                "Form": player.form,
            })

    st.dataframe(bench_data, use_container_width=True, hide_index=True)


def show_scout_tips():
    """Scout Tips page - community recommendations and team intelligence."""
    st.title("🔍 Scout Tips & Intelligence")

    db = get_db()
    if db.get_player_count() == 0:
        st.warning("No data loaded. Click 'Update' in the sidebar first.")
        return

    from fpl_assistant.predictions import fetch_scout_picks, fetch_differentials, fetch_intelligence_report

    players = db.get_all_players()
    teams = db.get_all_teams()
    teams_dict = {t.id: t for t in teams}

    current_gw = db.get_current_gameweek()
    gw = current_gw.id if current_gw else 1

    # ===========================================
    # DEADLINE INTELLIGENCE - Most important for decisions!
    # ===========================================
    st.subheader("🚨 Deadline Intelligence")
    st.caption("Injury news, rotation risk, and lineup intelligence from multiple sources")

    with st.spinner("Gathering team intelligence..."):
        try:
            intel_report = fetch_intelligence_report(players, gw, include_web=True)

            # Show key headlines
            if intel_report.key_headlines:
                for headline in intel_report.key_headlines:
                    st.markdown(headline)
            else:
                st.success("✅ No major injury concerns detected")

            # Show flagged players table
            if intel_report.flagged_players:
                st.markdown("**⚠️ Players Requiring Attention:**")
                flagged_data = []
                for intel in intel_report.flagged_players[:15]:
                    team = teams_dict.get(int(intel.team_name)) if intel.team_name.isdigit() else None
                    team_name = team.short_name if team else intel.team_name

                    # Risk emoji
                    risk_emoji = {
                        "critical": "⛔",
                        "high": "🔴",
                        "medium": "🟡",
                        "low": "🟢",
                    }.get(intel.risk_level, "⚪")

                    flagged_data.append({
                        "Risk": f"{risk_emoji} {intel.risk_level.upper()}",
                        "Player": intel.player_name,
                        "Team": team_name,
                        "Chance": f"{intel.chance_of_playing}%" if intel.chance_of_playing is not None else "-",
                        "Status": intel.status.value,
                        "Reason": intel.flag_reason[:50] + "..." if len(intel.flag_reason) > 50 else intel.flag_reason,
                    })

                st.dataframe(flagged_data, use_container_width=True, hide_index=True)

                # Expandable detail for each flagged player
                with st.expander("📋 Detailed News Items", expanded=False):
                    for intel in intel_report.flagged_players[:10]:
                        if intel.news_items:
                            st.markdown(f"**{intel.player_name}**")
                            for news in intel.news_items:
                                st.caption(f"• [{news.source}] {news.summary[:150]}")
                            st.markdown("---")

            st.caption(f"Sources: {', '.join(intel_report.sources_checked)}")

        except Exception as e:
            st.warning(f"Could not fetch intelligence: {e}")
            logger.error(f"Intelligence fetch error: {e}")

    st.markdown("---")

    # Fetch scout data
    if st.button("🔄 Fetch Latest Scout Tips", type="primary"):
        with st.spinner("Fetching scout recommendations..."):
            try:
                report = fetch_scout_picks(gw)
                st.session_state.scout_report = report
                st.success(f"Loaded tips from {len(report.sources)} sources")
            except Exception as e:
                st.error(f"Failed to fetch scout tips: {e}")

    if "scout_report" in st.session_state:
        report = st.session_state.scout_report

        # Show sources used
        st.info(f"**Data Sources:** {', '.join(report.sources)}")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎯 Top Form Players")
            picks_data = []
            for pick in report.picks[:10]:
                picks_data.append({
                    "Player": pick.player_name,
                    "Team": pick.team,
                    "Pos": pick.position,
                    "Why": pick.reason,
                    "Source": pick.source,
                    "Confidence": f"{pick.confidence * 100:.0f}%",
                    "Captain?": "👑" if pick.is_captain else "",
                })
            st.dataframe(picks_data, use_container_width=True, hide_index=True)

        with col2:
            st.subheader("👑 Captain Picks")
            captain_data = []
            for pick in report.captain_picks[:5]:
                captain_data.append({
                    "Player": pick.player_name,
                    "Team": pick.team,
                    "Why": pick.reason,
                    "Source": pick.source,
                    "Confidence": f"{pick.confidence * 100:.0f}%",
                })
            st.dataframe(captain_data, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Differentials
        st.subheader("💎 Differentials (Low Ownership, High Potential)")
        with st.spinner("Finding differentials..."):
            try:
                diffs = fetch_differentials(gw, max_ownership=10.0)
                diff_data = []
                for pick in diffs[:10]:
                    diff_data.append({
                        "Player": pick.player_name,
                        "Team": pick.team,
                        "Pos": pick.position,
                        "Why": pick.reason,
                        "Source": pick.source,
                        "Confidence": f"{pick.confidence * 100:.0f}%",
                    })
                st.dataframe(diff_data, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"Failed to fetch differentials: {e}")

        st.markdown("---")
        st.caption(f"Sources: {', '.join(report.sources)}")
    else:
        st.info("Click the button above to fetch the latest scout tips.")


def show_players():
    """Players page - browse and filter players."""
    st.title("👥 Players Database")

    db = get_db()

    if db.get_player_count() == 0:
        st.warning("No data loaded. Click 'Update' in the sidebar first.")
        return

    # Filters
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        position = st.selectbox("Position", ["All", "GK", "DEF", "MID", "FWD"])

    teams = db.get_all_teams()
    team_names = ["All"] + [t.name for t in teams]

    with col2:
        team = st.selectbox("Team", team_names)

    with col3:
        sort_by = st.selectbox("Sort by", ["Projected", "Points", "Price", "Form", "Ownership"])

    with col4:
        limit = st.slider("Show", 10, 100, 30)

    # Get players
    from fpl_assistant.data import Position as Pos
    from fpl_assistant.predictions import ProjectionEngine

    players = db.get_all_players()
    fixtures = db.get_all_fixtures()

    # Generate projections
    engine = ProjectionEngine(players, teams, fixtures)
    current_gw = db.get_current_gameweek()
    gw = current_gw.id if current_gw else 1

    player_projections = {}
    for p in players:
        try:
            player_projections[p.id] = engine.project_single_player(p, gw)
        except:
            player_projections[p.id] = p.form * 2

    if position != "All":
        pos_map = {"GK": Pos.GK, "DEF": Pos.DEF, "MID": Pos.MID, "FWD": Pos.FWD}
        pos = pos_map.get(position)
        players = [p for p in players if p.position == pos]

    if team != "All":
        team_obj = next((t for t in teams if t.name == team), None)
        if team_obj:
            players = [p for p in players if p.team_id == team_obj.id]

    # Sort
    sort_map = {
        "Projected": lambda p: -player_projections.get(p.id, 0),
        "Price": lambda p: -p.price,
        "Points": lambda p: -p.total_points,
        "Form": lambda p: -p.form,
        "Ownership": lambda p: -p.selected_by_percent,
    }
    players.sort(key=sort_map.get(sort_by, sort_map["Projected"]))
    players = players[:limit]

    # Display
    teams_dict = {t.id: t for t in teams}

    data = []
    for p in players:
        team_obj = teams_dict.get(p.team_id)
        xp = player_projections.get(p.id, 0)
        data.append({
            "Player": p.web_name,
            "Team": team_obj.short_name if team_obj else "?",
            "Pos": p.position_name,
            "Price": f"£{p.price:.1f}m",
            "xP": f"{xp:.1f}",
            "Points": p.total_points,
            "Form": p.form,
            "Own%": f"{p.selected_by_percent:.1f}%",
            "Status": "✅" if p.status == PlayerStatus.AVAILABLE else "⚠️",
        })

    st.dataframe(data, use_container_width=True, hide_index=True)


def show_backtest():
    """Backtest page - test predictions against actual results."""
    st.title("📊 Prediction Backtest")
    st.markdown("*Test how accurate our predictions are against actual results*")

    from fpl_assistant.predictions.backtest import run_backtest, print_backtest_report

    # Settings
    col1, col2 = st.columns(2)
    with col1:
        num_gws = st.slider("Number of gameweeks to test", 1, 10, 3)
    with col2:
        st.info("Tests predictions against actual FPL points from completed gameweeks")

    if st.button("🧪 Run Backtest", type="primary"):
        with st.spinner(f"Running backtest on {num_gws} gameweeks... This may take a minute..."):
            try:
                result = run_backtest(num_gameweeks=num_gws)

                if not result.gameweeks_tested:
                    st.warning("No completed gameweeks found to test")
                    return

                st.success(f"Tested {len(result.gameweeks_tested)} gameweeks: {result.gameweeks_tested}")

                # Key metrics
                st.markdown("---")
                st.subheader("Accuracy Metrics")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Absolute Error", f"{result.mean_absolute_error:.2f} pts")
                with col2:
                    st.metric("RMSE", f"{result.root_mean_square_error:.2f} pts")
                with col3:
                    st.metric("Correlation", f"{result.correlation:.3f}")
                with col4:
                    st.metric("Predictions Tested", result.total_predictions)

                st.markdown("---")
                st.subheader("Position Breakdown")
                pos_data = []
                for pos, mae in result.mae_by_position.items():
                    pos_data.append({"Position": pos, "MAE (pts)": f"{mae:.2f}"})
                st.dataframe(pos_data, use_container_width=True, hide_index=True)

                st.markdown("---")
                st.subheader("Top Pick Accuracy")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    hit_pct = result.top_10_hit_rate * 100
                    st.metric("Top 10 Hit Rate", f"{hit_pct:.1f}%",
                              help="How often our top 10 predicted players appear in actual top 10")
                with col2:
                    cap_pct = result.captain_accuracy * 100
                    st.metric("Captain in Top 5", f"{cap_pct:.1f}%",
                              help="How often our #1 captain pick was in the actual top 5 scorers")
                with col3:
                    cap_top3 = getattr(result, 'captain_top_3_rate', 0) * 100
                    st.metric("Captain in Top 3", f"{cap_top3:.1f}%",
                              help="How often our #1 captain pick was in the actual top 3 scorers")
                with col4:
                    cap_exact = getattr(result, 'captain_exact_hit', 0) * 100
                    st.metric("Captain Exact #1", f"{cap_exact:.1f}%",
                              help="How often our #1 pick was THE top scorer (very difficult!)")

                st.markdown("---")
                st.subheader("Sample Predictions")

                if result.player_results:
                    # Sort by error
                    sorted_results = sorted(
                        result.player_results,
                        key=lambda x: abs(x.actual_points - x.predicted_points),
                    )

                    # Best predictions
                    st.markdown("**Best Predictions (smallest error):**")
                    best_data = []
                    for r in sorted_results[:10]:
                        error = r.actual_points - r.predicted_points
                        best_data.append({
                            "Player": r.player_name,
                            "GW": r.gameweek,
                            "Predicted": f"{r.predicted_points:.1f}",
                            "Actual": r.actual_points,
                            "Error": f"{error:+.1f}",
                        })
                    st.dataframe(best_data, use_container_width=True, hide_index=True)

                    # Worst predictions
                    st.markdown("**Worst Predictions (largest error):**")
                    worst_data = []
                    for r in sorted_results[-10:]:
                        error = r.actual_points - r.predicted_points
                        worst_data.append({
                            "Player": r.player_name,
                            "GW": r.gameweek,
                            "Predicted": f"{r.predicted_points:.1f}",
                            "Actual": r.actual_points,
                            "Error": f"{error:+.1f}",
                        })
                    st.dataframe(worst_data, use_container_width=True, hide_index=True)

                # Interpretation
                st.markdown("---")
                st.subheader("What This Means")

                if result.mean_absolute_error < 2:
                    st.success("✅ **Excellent accuracy** - predictions are very close to actual points")
                elif result.mean_absolute_error < 3:
                    st.info("👍 **Good accuracy** - predictions are reasonably reliable")
                elif result.mean_absolute_error < 4:
                    st.warning("⚠️ **Moderate accuracy** - use with caution, consider other factors")
                else:
                    st.error("❌ **Poor accuracy** - predictions need improvement")

                if result.correlation > 0.5:
                    st.success(f"✅ **Strong correlation ({result.correlation:.2f})** - rankings are reliable")
                elif result.correlation > 0.3:
                    st.info(f"👍 **Moderate correlation ({result.correlation:.2f})** - rankings somewhat useful")
                else:
                    st.warning(f"⚠️ **Weak correlation ({result.correlation:.2f})** - rankings less reliable")

                # Adaptive learning section
                st.markdown("---")
                st.subheader("🧠 Adaptive Learning")
                st.info("Results have been recorded to the adaptive learning system. The algorithm will adjust weights based on accumulated backtest data.")

            except Exception as e:
                st.error(f"Backtest failed: {e}")
                import traceback
                with st.expander("Error details"):
                    st.code(traceback.format_exc())

    # Show adaptive weights section
    st.markdown("---")
    st.subheader("⚙️ Algorithm Weights")

    try:
        from fpl_assistant.predictions.adaptive import get_adaptive_manager

        manager = get_adaptive_manager()
        summary = manager.get_performance_summary()
        weights = summary["current_weights"]

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Current Weights:**")
            st.caption(f"Form: {weights['form_weight']*100:.0f}%")
            st.caption(f"ICT Index: {weights['ict_weight']*100:.0f}%")
            st.caption(f"Fixture Difficulty: {weights['fdr_weight']*100:.0f}%")
            st.caption(f"Consistency: {weights['consistency_weight']*100:.0f}%")
            st.caption(f"Team Strength: {weights['team_strength_weight']*100:.0f}%")

        with col2:
            st.markdown("**Performance History:**")
            if weights.get("correlation"):
                st.caption(f"Best Correlation: {weights['correlation']:.3f}")
                st.caption(f"Best MAE: {weights['mae']:.2f} pts")
                st.caption(f"Captain Accuracy: {weights['captain_accuracy']*100:.0f}%")
                st.caption(f"Last Updated: {weights.get('tested_at', 'Never')[:10] if weights.get('tested_at') else 'Never'}")
            else:
                st.caption("No optimization data yet - run backtests to improve!")

        st.caption(f"Historical results stored: {summary['num_historical_results']}")

    except Exception as e:
        st.warning(f"Could not load adaptive weights: {e}")


def show_performance_tracking():
    """Performance tracking page - track if the tool is improving your scores."""
    st.title("📈 Performance Tracking")
    st.markdown("*Track whether following the tool's advice is improving your FPL scores*")

    from fpl_assistant.analysis.performance_tracker import (
        PerformanceTracker,
        get_performance_tracker,
        get_auto_performance_summary,
    )

    tracker = get_performance_tracker()

    # Three tabs: Auto stats | Record manually | View history
    tab1, tab2, tab3 = st.tabs(["🤖 Auto Stats (Live)", "📝 Record Manually", "📊 Detailed Report"])

    with tab1:
        st.subheader("Live Performance from FPL API")
        st.markdown("*Auto-calculated from your FPL account - no manual entry needed!*")

        manager_id = get_manager_id()

        if st.button("🔄 Load Latest Stats from FPL", type="primary"):
            with st.spinner("Fetching your FPL data..."):
                auto_summary = get_auto_performance_summary(manager_id)

                if "error" in auto_summary:
                    st.error(f"Failed to load: {auto_summary['error']}")
                else:
                    st.session_state["auto_performance"] = auto_summary
                    st.success("Stats loaded successfully!")

        # Display auto-loaded stats
        if "auto_performance" in st.session_state:
            auto = st.session_state["auto_performance"]

            st.markdown("---")
            st.markdown("### Season Overview")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Gameweeks Played", auto["gameweeks_tracked"])
            with col2:
                st.metric("Total Points", f"{auto['total_points']:,}")
            with col3:
                st.metric("Avg per Week", f"{auto['avg_points_per_week']:.1f}")
            with col4:
                rank_change = auto["rank_improvement"]
                st.metric("Rank Change",
                          f"{rank_change:+,}" if rank_change != 0 else "0",
                          delta_color="normal" if rank_change > 0 else "inverse")

            st.markdown("---")
            st.markdown("### Rank Progression")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Starting Rank", f"{auto['starting_rank']:,}")
            with col2:
                st.metric("Current Rank", f"{auto['current_rank']:,}")
            with col3:
                st.metric("Best Rank", f"{auto['best_rank']:,}")

            st.markdown("---")
            st.markdown("### Transfer Activity")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Transfers", auto["total_transfers"])
            with col2:
                st.metric("Hits Taken", auto["total_hits"])
            with col3:
                st.metric("Hit Cost", f"-{auto['total_hit_cost']} pts")

            # Weekly points chart
            st.markdown("---")
            st.markdown("### Weekly Points")

            if auto.get("weekly_points"):
                import pandas as pd

                weekly_df = pd.DataFrame({
                    "Gameweek": list(range(1, len(auto["weekly_points"]) + 1)),
                    "Points": auto["weekly_points"],
                    "Rank": auto["weekly_ranks"]
                })

                st.line_chart(weekly_df.set_index("Gameweek")["Points"])

                # Show best/worst weeks
                best_week = weekly_df.loc[weekly_df["Points"].idxmax()]
                worst_week = weekly_df.loc[weekly_df["Points"].idxmin()]

                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"🏆 Best Week: GW{int(best_week['Gameweek'])} - {int(best_week['Points'])} pts")
                with col2:
                    st.error(f"📉 Worst Week: GW{int(worst_week['Gameweek'])} - {int(worst_week['Points'])} pts")

            st.markdown("---")
            st.markdown("### Performance Analysis")

            avg = auto["avg_points_per_week"]
            if avg >= 55:
                st.success(f"🌟 **Elite Performance** - {avg:.1f} pts/week puts you in top tier!")
            elif avg >= 50:
                st.success(f"✅ **Strong Performance** - {avg:.1f} pts/week is above average")
            elif avg >= 45:
                st.info(f"👍 **Decent Performance** - {avg:.1f} pts/week is around average")
            else:
                st.warning(f"⚠️ **Below Average** - {avg:.1f} pts/week needs improvement")

            if auto["rank_improvement"] > 100000:
                st.success(f"📈 **Rank Climbing!** Improved {auto['rank_improvement']:,} places this season")
            elif auto["rank_improvement"] < -100000:
                st.warning(f"📉 **Rank Dropping** - Lost {abs(auto['rank_improvement']):,} places")
        else:
            st.info("Click 'Load Latest Stats' to see your auto-calculated performance.")

    with tab2:
        st.subheader("Record Gameweek Manually")
        st.markdown("*Use this to track tool predictions vs actual results for accuracy analysis.*")

        db = get_db()
        current_gw = db.get_current_gameweek()
        default_gw = (current_gw.id - 1) if current_gw and current_gw.id > 1 else 1

        col1, col2 = st.columns(2)

        with col1:
            gw_number = st.number_input("Gameweek", min_value=1, max_value=38, value=default_gw)
            actual_points = st.number_input("Your Actual Points", min_value=0, max_value=200, value=50)
            predicted_points = st.number_input("Tool's Predicted Points", min_value=0.0, max_value=200.0, value=50.0)
            captain_name = st.text_input("Who did you captain?", value="")

        with col2:
            captain_points = st.number_input("Captain's Points (before 2x)", min_value=0, max_value=50, value=6)
            captain_was_recommended = st.checkbox("Was this the tool's recommendation?", value=True)
            best_captain_points = st.number_input("Best Captain Points (in your team)", min_value=0, max_value=50, value=10)
            overall_rank = st.number_input("Your Overall Rank", min_value=1, max_value=10000000, value=100000)

        st.markdown("---")
        st.subheader("Transfer Info (Optional)")

        col3, col4 = st.columns(2)
        with col3:
            transfers_made = st.number_input("Transfers Made", min_value=0, max_value=10, value=1)
            hits_taken = st.number_input("Hits Taken (-4s)", min_value=0, max_value=5, value=0)
        with col4:
            transfer_net_gain = st.number_input("Net Points from Transfers", min_value=-50, max_value=50, value=0,
                                                help="Points gained from transfers minus what sold players scored")

        if st.button("💾 Save Gameweek", type="primary"):
            if not captain_name:
                st.error("Please enter your captain's name")
            else:
                try:
                    tracker.record_gameweek(
                        gameweek=gw_number,
                        actual_points=actual_points,
                        predicted_points=predicted_points,
                        captain_name=captain_name,
                        captain_actual_points=captain_points,
                        captain_was_recommended=captain_was_recommended,
                        best_captain_points=best_captain_points,
                        overall_rank=overall_rank,
                        transfers_made=transfers_made,
                        hits_taken=hits_taken,
                        transfer_net_gain=transfer_net_gain,
                    )
                    st.success(f"GW{gw_number} recorded! Actual: {actual_points} pts, Predicted: {predicted_points} pts")
                except Exception as e:
                    st.error(f"Failed to record: {e}")

    with tab3:
        st.subheader("Detailed Performance Report")

        summary = tracker.get_summary()

        if summary.gameweeks_tracked == 0:
            st.info("No gameweeks recorded yet. Use the 'Record Gameweek' tab after each gameweek to track your performance.")
            return

        # Overview metrics
        st.markdown("### Overview")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Weeks Tracked", summary.gameweeks_tracked)
        with col2:
            st.metric("Total Points", summary.total_actual_points)
        with col3:
            st.metric("Avg per Week", f"{summary.avg_points_per_week:.1f}")
        with col4:
            rank_delta = summary.rank_improvement
            st.metric("Rank Change", f"{rank_delta:+,}" if rank_delta != 0 else "0",
                      delta_color="normal" if rank_delta > 0 else "inverse")

        # Prediction accuracy
        st.markdown("---")
        st.markdown("### Prediction Accuracy")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Avg Prediction Error", f"±{summary.avg_prediction_error:.1f} pts")
        with col2:
            beat_pct = (summary.weeks_beat_prediction / summary.gameweeks_tracked * 100) if summary.gameweeks_tracked > 0 else 0
            st.metric("Beat Prediction", f"{summary.weeks_beat_prediction}/{summary.gameweeks_tracked}",
                      help="Weeks where you scored more than predicted")
        with col3:
            variance = summary.total_actual_points - summary.total_predicted_points
            st.metric("Total Variance", f"{variance:+.0f} pts",
                      help="Actual - Predicted (positive = exceeding predictions)")

        # Captain performance
        st.markdown("---")
        st.markdown("### Captain Performance")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Captain Accuracy", f"{summary.captain_accuracy_rate*100:.0f}%",
                      help="% of weeks captain was in top picks")
        with col2:
            st.metric("Points Captured", f"{summary.captain_points_captured*100:.0f}%",
                      help="% of maximum captain points achieved")
        with col3:
            st.metric("Avg Captain Points", f"{summary.avg_captain_points:.1f}")

        # Rank progression
        st.markdown("---")
        st.markdown("### Rank Progression")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Starting Rank", f"{summary.starting_rank:,}")
        with col2:
            st.metric("Current Rank", f"{summary.current_rank:,}")
        with col3:
            st.metric("Best Rank", f"{summary.best_rank:,}")

        # Transfer stats
        st.markdown("---")
        st.markdown("### Transfer Stats")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Transfers", summary.total_transfers)
        with col2:
            st.metric("Hits Taken", f"{summary.total_hits} (-{summary.total_hit_cost} pts)")
        with col3:
            st.metric("Transfer Value", f"{summary.total_transfer_value:+d} pts",
                      help="Net points gained from all transfers")

        # Weekly trend
        st.markdown("---")
        st.markdown("### Weekly Trend")

        trend_data = tracker.get_weekly_trend()
        if trend_data:
            import pandas as pd

            df = pd.DataFrame(trend_data)
            df.columns = ["Gameweek", "Actual", "Predicted", "Rank", "Captain %"]

            # Show chart
            chart_df = df[["Gameweek", "Actual", "Predicted"]].set_index("Gameweek")
            st.line_chart(chart_df)

            # Show table
            st.dataframe(df, use_container_width=True, hide_index=True)

        # Interpretation
        st.markdown("---")
        st.markdown("### What This Means")

        if summary.avg_prediction_error < 5:
            st.success("✅ **Predictions are accurate** - tool is well calibrated")
        elif summary.avg_prediction_error < 10:
            st.info("👍 **Predictions are reasonable** - some variance is normal in FPL")
        else:
            st.warning("⚠️ **High variance** - FPL is inherently unpredictable, but consider running backtests")

        if summary.captain_accuracy_rate > 0.5:
            st.success("✅ **Captain picks are strong** - over 50% in top options")
        elif summary.captain_accuracy_rate > 0.3:
            st.info("👍 **Captain picks are decent** - room for improvement")
        else:
            st.warning("⚠️ **Captain picks need work** - review the captain selection logic")

        if summary.rank_improvement > 0:
            st.success(f"✅ **Rank improving** - gained {summary.rank_improvement:,} places!")
        elif summary.rank_improvement < -10000:
            st.warning(f"⚠️ **Rank dropping** - lost {abs(summary.rank_improvement):,} places. Review strategy.")
        else:
            st.info("👍 **Rank stable** - maintaining position")

        # Full report
        st.markdown("---")
        with st.expander("📄 Full Text Report"):
            st.code(tracker.print_report())


def show_fixture_ticker():
    """Show fixture difficulty ticker for all teams."""
    st.title("📆 Fixture Ticker")
    st.markdown("*Plan your transfers by targeting teams with favorable fixture runs.*")

    db = get_db()
    if db.get_player_count() == 0:
        st.warning("No data loaded. Click 'Update' in the sidebar first.")
        return

    players = db.get_all_players()
    teams = db.get_all_teams()
    fixtures = db.get_all_fixtures()
    gameweeks = [db.get_gameweek(i) for i in range(1, 39) if db.get_gameweek(i)]
    current_gw = db.get_current_gameweek()
    start_gw = current_gw.id if current_gw else 1

    from fpl_assistant.analysis.fixture_ticker import FixtureTicker, get_fdr_color, get_fdr_emoji

    ticker = FixtureTicker(teams, fixtures, gameweeks)

    # Settings
    col1, col2 = st.columns(2)
    with col1:
        num_weeks = st.slider("Weeks to show", 4, 10, 6)
    with col2:
        start_gw = st.number_input("Starting GW", min_value=1, max_value=38, value=start_gw)

    # Get all team runs
    runs = ticker.get_all_team_runs(start_gw, num_weeks)

    st.markdown("---")

    # Summary metrics
    st.subheader("Best Fixture Runs")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🎯 Best for Attacking (FWD/MID)**")
        best_attack = ticker.get_best_attack_fixtures(start_gw, num_weeks, 5)
        for run in best_attack:
            fdr_str = " ".join([get_fdr_emoji(f.fdr) if not f.is_blank else "⬜" for f in run.fixtures[:6]])
            st.markdown(f"**{run.team_short_name}** ({run.avg_fdr:.1f} avg): {fdr_str}")

    with col2:
        st.markdown("**🛡️ Best for Clean Sheets (GK/DEF)**")
        best_defense = ticker.get_best_defense_fixtures(start_gw, num_weeks, 5)
        for run in best_defense:
            fdr_str = " ".join([get_fdr_emoji(f.fdr) if not f.is_blank else "⬜" for f in run.fixtures[:6]])
            st.markdown(f"**{run.team_short_name}** ({run.avg_fdr:.1f} avg): {fdr_str}")

    # Blank/Double alerts
    blanks = ticker.get_teams_with_blanks(start_gw, num_weeks)
    doubles = ticker.get_teams_with_doubles(start_gw, num_weeks)

    if blanks or doubles:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if blanks:
                st.warning(f"⚠️ **{len(blanks)} teams have BLANK gameweeks**")
                for run in blanks:
                    blank_gws = [f.gameweek for f in run.fixtures if f.is_blank]
                    st.write(f"• {run.team_name}: GW {', GW '.join(map(str, blank_gws))}")
        with col2:
            if doubles:
                st.success(f"🔥 **{len(doubles)} teams have DOUBLE gameweeks**")
                for run in doubles:
                    double_gws = [f.gameweek for f in run.fixtures if f.is_double]
                    st.write(f"• {run.team_name}: GW {', GW '.join(map(str, set(double_gws)))}")

    # Full fixture grid
    st.markdown("---")
    st.subheader("Full Fixture Grid")
    st.markdown("*🟢 Easy | 🟨 Medium | 🔴 Hard | ⬜ Blank*")

    # Build data for display
    import pandas as pd

    data = []
    for run in runs:
        row = {"Team": run.team_short_name, "Avg FDR": f"{run.avg_fdr:.1f}"}
        for i, fix in enumerate(run.fixtures[:num_weeks]):
            gw_num = start_gw + i
            if fix.is_blank:
                row[f"GW{gw_num}"] = "BLANK"
            else:
                venue = "H" if fix.is_home else "a"
                double_marker = "x2" if fix.is_double else ""
                row[f"GW{gw_num}"] = f"{fix.opponent_name}({venue}) {double_marker}"
        data.append(row)

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Legend
    with st.expander("FDR Legend"):
        st.markdown("""
        - **FDR 1-2** 🟢🟩: Easy fixtures - good for attacking returns and clean sheets
        - **FDR 3** 🟨: Medium fixtures - balanced risk/reward
        - **FDR 4-5** 🟧🔴: Hard fixtures - difficult for points
        - **(H)** = Home, **(a)** = Away
        - **x2** = Double gameweek (two fixtures)
        """)


def show_rival_analysis():
    """Show mini-league rival tracking and analysis."""
    st.title("🎯 Mini-League Rival Analysis")
    st.markdown("*Track your rivals and find differential picks to climb the league.*")

    db = get_db()
    if db.get_player_count() == 0:
        st.warning("No data loaded. Click 'Update' in the sidebar first.")
        return

    manager_id = get_manager_id()

    # Get league ID from user
    league_id = st.text_input(
        "Enter your mini-league ID",
        help="Find this in the URL when viewing your league on the FPL website: fantasy.premierleague.com/leagues/[ID]/standings"
    )

    if not league_id:
        st.info("Enter a mini-league ID to analyze your rivals.")
        return

    try:
        league_id = int(league_id)
    except ValueError:
        st.error("League ID must be a number")
        return

    # Fetch league data
    from fpl_assistant.api import SyncFPLClient
    from fpl_assistant.predictions.rivals import (
        RivalTracker, parse_league_standings, parse_rival_team,
        RivalStrategy
    )

    client = SyncFPLClient()

    try:
        with st.spinner("Fetching league standings..."):
            league_data = client.get_classic_league(league_id)
            league_name, standings = parse_league_standings(league_data)

        st.success(f"Loaded **{league_name}** ({len(standings)} managers)")

        # Find your position
        your_entry = next((e for e in standings if e.manager_id == manager_id), None)

        if not your_entry:
            st.warning("Your team wasn't found in this league. Check the league ID.")
            return

        st.markdown("---")
        st.subheader(f"Your Position: #{your_entry.rank}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Points", your_entry.total_points)
        with col2:
            st.metric("GW Points", your_entry.gameweek_points)
        with col3:
            gap_to_first = standings[0].total_points - your_entry.total_points if standings else 0
            st.metric("Gap to 1st", f"-{gap_to_first}" if gap_to_first > 0 else "Leading!")

        # Get current GW
        current_gw = db.get_current_gameweek()
        gw = current_gw.id if current_gw else 1

        # Fetch rival teams (top 5 + those near you)
        rivals_to_fetch = []
        for entry in standings[:5]:
            if entry.manager_id != manager_id:
                rivals_to_fetch.append(entry)

        # Also add rivals within 50 points
        for entry in standings:
            if entry.manager_id != manager_id:
                gap = abs(your_entry.total_points - entry.total_points)
                if gap <= 50 and entry not in rivals_to_fetch:
                    rivals_to_fetch.append(entry)

        rivals_to_fetch = rivals_to_fetch[:10]  # Limit to 10

        # Fetch each rival's team
        st.markdown("---")
        st.subheader("Rival Analysis")

        rival_teams = {}
        with st.spinner("Fetching rival teams..."):
            for rival in rivals_to_fetch:
                try:
                    picks_data = client.get_entry_picks(rival.manager_id, gw)
                    rival_team = parse_rival_team(picks_data)
                    rival_team.manager_id = rival.manager_id
                    rival_teams[rival.manager_id] = rival_team
                except Exception:
                    pass  # Skip if can't fetch

        # Get your team
        if "my_team" in st.session_state and st.session_state.my_team:
            my_team = st.session_state.my_team
            picks = my_team.get("picks", [])
            your_player_ids = [p["element"] for p in picks]
        else:
            st.warning("Load your team first (from Weekly Advice page)")
            return

        # Analyze rivals
        players = db.get_all_players()
        player_dict = {p.id: p for p in players}

        tracker = RivalTracker(player_dict)

        # Show rival comparisons
        for rival in rivals_to_fetch:
            if rival.manager_id not in rival_teams:
                continue

            rival_team = rival_teams[rival.manager_id]
            analysis = tracker.analyze_rival(rival, rival_team, your_player_ids, your_entry.total_points)

            with st.expander(f"**{rival.team_name}** (#{rival.rank}) - {rival.manager_name}", expanded=rival.rank <= 3):
                col1, col2 = st.columns(2)
                with col1:
                    gap_str = f"+{analysis.points_gap}" if analysis.points_gap > 0 else str(analysis.points_gap)
                    st.metric("Points Gap", gap_str)
                    st.metric("Overlap", f"{analysis.overlap_percentage:.0f}%")

                with col2:
                    st.metric("Common Players", len(analysis.common_players))
                    if analysis.recommended_strategy == RivalStrategy.MATCH:
                        st.warning(f"Strategy: **{analysis.recommended_strategy.value}**")
                    elif analysis.recommended_strategy == RivalStrategy.DIFFERENTIATE:
                        st.success(f"Strategy: **{analysis.recommended_strategy.value}**")
                    else:
                        st.info(f"Strategy: **{analysis.recommended_strategy.value}**")

                # Show their differentials (players they have that you don't)
                if analysis.their_differentials:
                    st.markdown("**They have (you don't):**")
                    diff_names = [player_dict[pid].web_name for pid in analysis.their_differentials[:5] if pid in player_dict]
                    st.write(", ".join(diff_names))

                # Show your differentials (players you have that they don't)
                if analysis.your_differentials:
                    st.markdown("**You have (they don't):**")
                    diff_names = [player_dict[pid].web_name for pid in analysis.your_differentials[:5] if pid in player_dict]
                    st.write(", ".join(diff_names))

        # Differential targets
        st.markdown("---")
        st.subheader("Differential Pick Targets")
        st.markdown("*High xP players your rivals don't own - great for gaining ground*")

        from fpl_assistant.predictions import ProjectionEngine
        engine = ProjectionEngine(players, db.get_all_teams(), db.get_all_fixtures())
        projections = {p.id: engine.project_single_player(p, gw) for p in players}

        tracker_with_projs = RivalTracker(player_dict, projections)
        differentials = tracker_with_projs._find_differential_targets(
            your_player_ids, rival_teams, manager_id
        )

        if differentials:
            import pandas as pd
            diff_data = []
            for diff in differentials[:10]:
                diff_data.append({
                    "Player": diff.player.web_name,
                    "Price": f"£{diff.player.price:.1f}m",
                    "xP": f"{diff.projected_points:.1f}",
                    "Rival Own%": f"{diff.ownership_in_league:.0f}%",
                    "Score": f"{diff.differential_score:.1f}",
                })
            st.dataframe(pd.DataFrame(diff_data), use_container_width=True, hide_index=True)
        else:
            st.info("No clear differential targets found")

    except Exception as e:
        st.error(f"Failed to fetch league data: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
