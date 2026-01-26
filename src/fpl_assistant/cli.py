"""
FPL Assistant Command Line Interface.

Built with Typer for a modern, type-safe CLI experience.
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .api import CachedFPLClient, SyncFPLClient, get_cache
from .data import (
    Database,
    Team,
    get_database,
    identify_blank_double_gameweeks,
    process_bootstrap_static,
    process_fixtures,
    update_gameweeks_with_blank_double,
)

app = typer.Typer(
    name="fpl-assistant",
    help="Fantasy Premier League Assistant - Optimize your FPL team with AI",
    add_completion=True,
    rich_markup_mode="rich",
)

console = Console()


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path("data")


@app.callback()
def main_callback() -> None:
    """
    FPL Assistant - Your AI-powered Fantasy Premier League companion.

    Use the commands below to manage your team, get recommendations,
    and optimize your transfers.
    """
    pass


@app.command()
def update(
    force: bool = typer.Option(
        False, "--force", "-f", help="Force refresh, bypass cache"
    ),
) -> None:
    """
    Fetch latest data from FPL API.

    Updates players, fixtures, and your team data.
    """
    console.print(Panel("[bold blue]FPL Data Update[/bold blue]", style="blue"))

    # Initialize clients
    cache = get_cache()
    raw_client = SyncFPLClient()
    client = CachedFPLClient(raw_client, cache)
    db = get_database(get_data_dir() / "fpl.db")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Fetch bootstrap-static
            task = progress.add_task("Fetching player data...", total=None)
            bootstrap = client.get_bootstrap_static(force_refresh=force)
            players, teams, gameweeks = process_bootstrap_static(bootstrap)
            progress.update(task, description=f"[green]Loaded {len(players)} players")

            # Fetch fixtures
            progress.update(task, description="Fetching fixtures...")
            fixtures_data = client.get_fixtures(force_refresh=force)
            fixtures = process_fixtures(fixtures_data)
            progress.update(task, description=f"[green]Loaded {len(fixtures)} fixtures")

            # Identify blank/double gameweeks
            progress.update(task, description="Analyzing gameweeks...")
            blank_double = identify_blank_double_gameweeks(fixtures)
            gameweeks = update_gameweeks_with_blank_double(gameweeks, blank_double)

            # Save to database
            progress.update(task, description="Saving to database...")
            db.upsert_players(players)
            db.upsert_teams(teams)
            db.upsert_fixtures(fixtures)
            for gw in gameweeks:
                db.upsert_gameweek(gw)

            progress.update(task, description="[green]Data saved!")

        # Show summary
        console.print()
        current_gw = next((gw for gw in gameweeks if gw.is_current), None)
        next_gw = next((gw for gw in gameweeks if gw.is_next), None)

        summary_table = Table(title="Update Summary", show_header=False)
        summary_table.add_column("Item", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Players", str(len(players)))
        summary_table.add_row("Teams", str(len(teams)))
        summary_table.add_row("Fixtures", str(len(fixtures)))

        if current_gw:
            summary_table.add_row("Current Gameweek", current_gw.name)
        if next_gw:
            summary_table.add_row("Next Deadline", str(next_gw.deadline))

        # Show blank/double info
        upcoming_blank = [gw for gw in gameweeks if gw.is_blank and not gw.finished]
        upcoming_double = [gw for gw in gameweeks if gw.is_double and not gw.finished]

        if upcoming_blank:
            blank_gws = ", ".join([f"GW{gw.id}" for gw in upcoming_blank[:3]])
            summary_table.add_row("Upcoming Blanks", blank_gws)
        if upcoming_double:
            double_gws = ", ".join([f"GW{gw.id}" for gw in upcoming_double[:3]])
            summary_table.add_row("Upcoming Doubles", double_gws)

        console.print(summary_table)
        console.print("\n[green]Update complete![/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    finally:
        raw_client.close()


@app.command()
def status() -> None:
    """
    Show current squad and game status.

    Displays your team, bank balance, free transfers, and available chips.
    """
    db = get_database(get_data_dir() / "fpl.db")

    # Check if we have data
    player_count = db.get_player_count()
    if player_count == 0:
        console.print(
            "[yellow]No data loaded. Run 'fpl-assistant update' first.[/yellow]"
        )
        raise typer.Exit(1)

    # Get current gameweek info
    current_gw = db.get_current_gameweek()
    next_gw = db.get_next_gameweek()

    # Header
    console.print(Panel("[bold]FPL Assistant Status[/bold]", style="green"))

    # Game status table
    status_table = Table(title="Game Status", show_header=False)
    status_table.add_column("Item", style="cyan")
    status_table.add_column("Value", style="white")

    status_table.add_row("Players in DB", str(player_count))

    if current_gw:
        status_table.add_row("Current Gameweek", current_gw.name)
        if current_gw.is_blank:
            status_table.add_row("  Blank Teams", f"{len(current_gw.blank_teams)} teams")
        if current_gw.is_double:
            status_table.add_row("  Double Teams", f"{len(current_gw.double_teams)} teams")

    if next_gw:
        status_table.add_row("Next Gameweek", next_gw.name)
        status_table.add_row("Deadline", str(next_gw.deadline))

    console.print(status_table)

    # User squad info
    squad = db.get_user_squad()
    if squad:
        console.print()
        squad_table = Table(title="Your Squad")
        squad_table.add_column("#", style="dim")
        squad_table.add_column("Player", style="white")
        squad_table.add_column("Pos", style="cyan")
        squad_table.add_column("Price", style="green")
        squad_table.add_column("", style="yellow")

        for sp in sorted(squad.players, key=lambda x: x.position):
            player = db.get_player(sp.player_id)
            if player:
                pos_label = "C" if sp.is_captain else ("V" if sp.is_vice_captain else "")
                bench = "BENCH" if not sp.is_starter else ""
                squad_table.add_row(
                    str(sp.position),
                    player.web_name,
                    player.position_name,
                    f"£{sp.selling_price:.1f}m",
                    pos_label or bench,
                )

        console.print(squad_table)
        console.print(f"\n[cyan]Bank:[/cyan] £{squad.bank:.1f}m")
        console.print(f"[cyan]Free Transfers:[/cyan] {squad.free_transfers}")
        console.print(f"[cyan]Available Chips:[/cyan] {', '.join(c.value for c in squad.get_available_chips()) or 'None'}")
    else:
        console.print(
            "\n[yellow]No squad data. Configure FPL credentials to fetch your team.[/yellow]"
        )


@app.command()
def players(
    position: str = typer.Option(
        None, "--position", "-p", help="Filter by position (GK, DEF, MID, FWD)"
    ),
    team: str = typer.Option(None, "--team", "-t", help="Filter by team name"),
    limit: int = typer.Option(20, "--limit", "-n", help="Number of players to show"),
    sort: str = typer.Option(
        "price", "--sort", "-s", help="Sort by: price, points, form, ownership"
    ),
) -> None:
    """
    List players from the database.

    Filter and sort players by various criteria.
    """
    from .data import Position as Pos

    db = get_database(get_data_dir() / "fpl.db")

    # Check if we have data
    if db.get_player_count() == 0:
        console.print(
            "[yellow]No data loaded. Run 'fpl-assistant update' first.[/yellow]"
        )
        raise typer.Exit(1)

    # Get players
    if position:
        pos_map = {"GK": Pos.GK, "DEF": Pos.DEF, "MID": Pos.MID, "FWD": Pos.FWD}
        pos = pos_map.get(position.upper())
        if not pos:
            console.print(f"[red]Invalid position: {position}[/red]")
            raise typer.Exit(1)
        all_players = db.get_players_by_position(pos)
    else:
        all_players = db.get_all_players()

    # Filter by team
    if team:
        teams = db.get_all_teams()
        team_id = None
        for t in teams:
            if team.lower() in t.name.lower() or team.lower() == t.short_name.lower():
                team_id = t.id
                break
        if team_id:
            all_players = [p for p in all_players if p.team_id == team_id]
        else:
            console.print(f"[red]Team not found: {team}[/red]")
            raise typer.Exit(1)

    # Sort
    sort_map = {
        "price": lambda p: -p.price,
        "points": lambda p: -p.total_points,
        "form": lambda p: -p.form,
        "ownership": lambda p: -p.selected_by_percent,
    }
    sort_fn = sort_map.get(sort.lower(), sort_map["price"])
    all_players.sort(key=sort_fn)

    # Limit
    all_players = all_players[:limit]

    # Get teams for display
    teams = {t.id: t for t in db.get_all_teams()}

    # Display
    table = Table(title=f"Players (sorted by {sort})")
    table.add_column("Name", style="white")
    table.add_column("Pos", style="cyan")
    table.add_column("Team", style="yellow")
    table.add_column("Price", style="green")
    table.add_column("Pts", style="magenta")
    table.add_column("Form", style="blue")
    table.add_column("Own%", style="dim")

    for player in all_players:
        team_name = teams.get(player.team_id, Team(id=0, name="?", short_name="?")).short_name
        table.add_row(
            player.web_name,
            player.position_name,
            team_name,
            f"£{player.price:.1f}m",
            str(player.total_points),
            f"{player.form:.1f}",
            f"{player.selected_by_percent:.1f}%",
        )

    console.print(table)


@app.command()
def optimize(
    horizon: int = typer.Argument(5, help="Number of gameweeks to plan ahead (1-10)"),
    allow_hits: bool = typer.Option(True, "--hits/--no-hits", help="Allow point hits"),
    fast: bool = typer.Option(False, "--fast", help="Use faster solver settings"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
) -> None:
    """
    Run optimization for upcoming gameweeks.

    Generates optimal transfer and lineup recommendations.
    """
    from .data.models import ChipStatus, ChipType, PlayerProjection, Squad, SquadPlayer
    from .optimizer import FPLSolver

    if horizon < 1 or horizon > 10:
        console.print("[red]Horizon must be between 1 and 10 gameweeks[/red]")
        raise typer.Exit(1)

    db = get_database(get_data_dir() / "fpl.db")

    # Check if we have data
    if db.get_player_count() == 0:
        console.print(
            "[yellow]No data loaded. Run 'fpl-assistant update' first.[/yellow]"
        )
        raise typer.Exit(1)

    console.print(
        Panel(f"[bold blue]Optimizing {horizon} Gameweeks[/bold blue]", style="blue")
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading data...", total=None)

        # Load players and gameweeks
        players = db.get_all_players()
        gameweeks = [db.get_current_gameweek(), db.get_next_gameweek()]
        gameweeks = [gw for gw in gameweeks if gw is not None]

        # Get or create sample squad for demo (no auth yet)
        squad = db.get_user_squad()
        if not squad:
            progress.update(task, description="[yellow]No squad - using sample[/yellow]")
            # Create a simple sample squad for testing
            sorted_players = sorted(players, key=lambda p: -p.total_points)
            gk_players = [p for p in sorted_players if p.position.value == 1][:2]
            def_players = [p for p in sorted_players if p.position.value == 2][:5]
            mid_players = [p for p in sorted_players if p.position.value == 3][:5]
            fwd_players = [p for p in sorted_players if p.position.value == 4][:3]

            sample_players = gk_players + def_players + mid_players + fwd_players
            squad_players = []
            for i, p in enumerate(sample_players):
                squad_players.append(SquadPlayer(
                    player_id=p.id,
                    position=i + 1,
                    is_captain=(i == 2),
                    is_vice_captain=(i == 7),
                    purchase_price=p.price,
                    selling_price=p.price,
                ))

            squad = Squad(
                players=squad_players,
                bank=0.0,
                free_transfers=1,
                total_value=sum(p.price for p in sample_players),
                chips=[
                    ChipStatus(chip_type=ChipType.WILDCARD),
                    ChipStatus(chip_type=ChipType.FREE_HIT),
                    ChipStatus(chip_type=ChipType.BENCH_BOOST),
                    ChipStatus(chip_type=ChipType.TRIPLE_CAPTAIN),
                ],
            )

        progress.update(task, description="Generating projections...")

        # Get projections using the data-driven engine
        from .predictions import ProjectionEngine

        teams = db.get_all_teams()
        fixtures = [db.get_fixtures_by_gameweek(gw.id) for gw in gameweeks]
        all_fixtures = [f for gw_fixtures in fixtures for f in gw_fixtures]

        # Use projection engine for data-driven predictions
        engine = ProjectionEngine(players, teams, all_fixtures, gameweeks)
        current_gw = gameweeks[0].id if gameweeks else 1

        progress.update(task, description="Running optimizer...")

        projections = engine.project_all_players(current_gw, current_gw + horizon - 1)

        # Create solver and run optimization
        solver = FPLSolver(
            time_limit=15 if fast else 60,
            gap_tolerance=0.05 if fast else 0.01,
        )

        result = solver.optimize_horizon(
            players=players,
            projections=projections,
            current_squad=squad,
            gameweeks=gameweeks,
            horizon=horizon,
            allow_hits=allow_hits,
        )

        progress.update(task, description="[green]Optimization complete[/green]")

    # Display results
    if not result.success:
        console.print(f"[red]Optimization failed: {result.message}[/red]")
        raise typer.Exit(1)

    console.print()
    console.print(f"[green]{result.message}[/green]")

    plan = result.plan
    if plan and hasattr(plan, 'week_plans'):
        # Multi-week plan
        summary_table = Table(title="Optimization Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Weeks Planned", str(len(plan.week_plans)))
        summary_table.add_row("Total Expected Points", f"{plan.total_expected_points:.1f}")
        summary_table.add_row("Total Hits", str(plan.total_hits))
        summary_table.add_row("Net Expected Points", f"{plan.net_expected_points:.1f}")
        console.print(summary_table)

        # Show each week's plan
        teams = {t.id: t for t in db.get_all_teams()}
        player_dict = {p.id: p for p in players}

        for wp in plan.week_plans:
            console.print()
            week_table = Table(title=f"Gameweek {wp.gameweek}")
            week_table.add_column("Info", style="cyan")
            week_table.add_column("Details", style="white")

            week_table.add_row("Expected Points", f"{wp.expected_points:.1f}")
            if wp.hit_cost > 0:
                week_table.add_row("Hit Cost", f"-{wp.hit_cost}")
            if wp.chip_used:
                week_table.add_row("Chip", wp.chip_used.value)

            captain = player_dict.get(wp.captain_id)
            vice = player_dict.get(wp.vice_captain_id)
            if captain:
                week_table.add_row("Captain", captain.web_name)
            if vice:
                week_table.add_row("Vice Captain", vice.web_name)

            if wp.transfers:
                transfers_str = []
                for t in wp.transfers:
                    out_p = player_dict.get(t.player_out_id)
                    in_p = player_dict.get(t.player_in_id)
                    out_name = out_p.web_name if out_p else "?"
                    in_name = in_p.web_name if in_p else "?"
                    transfers_str.append(f"{out_name} → {in_name}")
                week_table.add_row("Transfers", ", ".join(transfers_str))

            console.print(week_table)

            if verbose:
                # Show starting XI
                xi_table = Table(title="Starting XI")
                xi_table.add_column("Pos", style="cyan")
                xi_table.add_column("Player", style="white")
                xi_table.add_column("Team", style="yellow")
                xi_table.add_column("Proj", style="green")

                for pid in wp.starting_xi:
                    p = player_dict.get(pid)
                    if p:
                        team = teams.get(p.team_id)
                        proj = p.form * 2
                        xi_table.add_row(
                            p.position_name,
                            p.web_name,
                            team.short_name if team else "?",
                            f"{proj:.1f}",
                        )
                console.print(xi_table)

    console.print("\n[dim]Use 'fpl-assistant suggest' for transfer recommendations.[/dim]")


@app.command()
def suggest() -> None:
    """
    Display current recommendations.

    Shows the latest optimization results including transfers, lineup, and captain.
    """
    console.print(Panel("[bold]Weekly Recommendations[/bold]", style="green"))

    # TODO: Load latest plan from database
    console.print(
        "[yellow]No recommendations available. Run 'fpl-assistant optimize' first.[/yellow]"
    )


@app.command()
def explain() -> None:
    """
    Get AI explanation of recommendations.

    Uses LLM to provide detailed reasoning for the suggested plan.
    """
    console.print("[bold blue]Generating AI explanation...[/bold blue]")

    # TODO: Implement LLM integration (Epic 6)
    console.print("[yellow]LLM integration not yet implemented (Epic 6)[/yellow]")


@app.command()
def chat() -> None:
    """
    Interactive Q&A with the AI assistant.

    Ask questions about your team, players, or strategy.
    """
    console.print(
        Panel(
            "[bold]FPL Assistant Chat[/bold]\n\n"
            "Ask me anything about FPL strategy, player comparisons, or your team.\n"
            "Type 'exit' or 'quit' to leave.",
            style="blue",
        )
    )

    # TODO: Implement chat loop with LLM (Epic 6)
    console.print("[yellow]Chat feature not yet implemented (Epic 6)[/yellow]")


@app.command()
def compare(
    player1: str = typer.Argument(..., help="First player name"),
    player2: str = typer.Argument(..., help="Second player name"),
) -> None:
    """
    Compare two players.

    Shows stats, fixtures, and projections side by side.
    """
    db = get_database(get_data_dir() / "fpl.db")

    if db.get_player_count() == 0:
        console.print(
            "[yellow]No data loaded. Run 'fpl-assistant update' first.[/yellow]"
        )
        raise typer.Exit(1)

    # Find players by name (partial match)
    all_players = db.get_all_players()
    teams = {t.id: t for t in db.get_all_teams()}

    def find_player(name: str):
        name_lower = name.lower()
        matches = [p for p in all_players if name_lower in p.web_name.lower()]
        if not matches:
            matches = [p for p in all_players if name_lower in p.name.lower()]
        return matches[0] if matches else None

    p1 = find_player(player1)
    p2 = find_player(player2)

    if not p1:
        console.print(f"[red]Player not found: {player1}[/red]")
        raise typer.Exit(1)
    if not p2:
        console.print(f"[red]Player not found: {player2}[/red]")
        raise typer.Exit(1)

    # Create comparison table
    table = Table(title=f"Comparison: {p1.web_name} vs {p2.web_name}")
    table.add_column("Stat", style="cyan")
    table.add_column(p1.web_name, style="white")
    table.add_column(p2.web_name, style="white")

    team1 = teams.get(p1.team_id)
    team2 = teams.get(p2.team_id)

    table.add_row("Team", team1.name if team1 else "?", team2.name if team2 else "?")
    table.add_row("Position", p1.position_name, p2.position_name)
    table.add_row("Price", f"£{p1.price:.1f}m", f"£{p2.price:.1f}m")
    table.add_row("Total Points", str(p1.total_points), str(p2.total_points))
    table.add_row("Form", f"{p1.form:.1f}", f"{p2.form:.1f}")
    table.add_row("PPG", f"{p1.points_per_game:.1f}", f"{p2.points_per_game:.1f}")
    table.add_row("ICT Index", f"{p1.ict_index:.1f}", f"{p2.ict_index:.1f}")
    table.add_row("Ownership", f"{p1.selected_by_percent:.1f}%", f"{p2.selected_by_percent:.1f}%")
    table.add_row("Goals", str(p1.goals_scored), str(p2.goals_scored))
    table.add_row("Assists", str(p1.assists), str(p2.assists))
    table.add_row("Minutes", str(p1.minutes), str(p2.minutes))
    table.add_row("Status", p1.status.value, p2.status.value)

    if p1.news or p2.news:
        table.add_row("News", p1.news or "-", p2.news or "-")

    console.print(table)


@app.command()
def login(
    email: str = typer.Option(None, "--email", "-e", help="FPL account email"),
    manager_id: int = typer.Option(None, "--id", "-i", help="Your FPL manager ID"),
) -> None:
    """
    Login to FPL and fetch your team.

    You can find your manager ID in your FPL URL after /entry/
    Example: fantasy.premierleague.com/entry/123456 -> manager ID is 123456
    """
    from .api.auth import FPLAuthenticator

    console.print(Panel("[bold blue]FPL Login[/bold blue]", style="blue"))

    # Get credentials
    if not email:
        email = typer.prompt("FPL Email")
    password = typer.prompt("FPL Password", hide_input=True)

    if not manager_id:
        console.print("[dim]Find your manager ID in your FPL URL: fantasy.premierleague.com/entry/[your-id][/dim]")
        manager_id = typer.prompt("Manager ID", type=int)

    auth = FPLAuthenticator()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Logging in...", total=None)

        try:
            cookies = auth.login_sync(email, password)
            auth.set_manager_id(manager_id)
            progress.update(task, description="[green]Login successful!")

            # Fetch team
            progress.update(task, description="Fetching your team...")
            from .data import process_my_team

            client = SyncFPLClient()
            client.set_cookies(cookies)

            try:
                my_team_data = client.get_my_team(manager_id)
                squad = process_my_team(my_team_data)

                # Save to database
                db = get_database(get_data_dir() / "fpl.db")
                db.save_user_squad(squad)

                progress.update(task, description="[green]Team loaded!")

                console.print()
                console.print(f"[green]✓ Logged in as manager {manager_id}[/green]")
                console.print(f"[green]✓ Loaded {len(squad.players)} players[/green]")
                console.print(f"[green]✓ Bank: £{squad.bank:.1f}m[/green]")
                console.print(f"[green]✓ Free Transfers: {squad.free_transfers}[/green]")

            finally:
                client.close()

        except Exception as e:
            progress.update(task, description=f"[red]Login failed")
            console.print(f"\n[red]Error: {e}[/red]")
            console.print("[dim]Make sure your email and password are correct.[/dim]")
            raise typer.Exit(1)


@app.command()
def scout() -> None:
    """
    Fetch scout tips and community recommendations.

    Collects data from form, transfers, and ownership for better projections.
    """
    from .predictions import fetch_scout_picks, fetch_differentials

    console.print(Panel("[bold blue]Scout Tips[/bold blue]", style="blue"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching scout data...", total=None)

        try:
            report = fetch_scout_picks()
            progress.update(task, description="[green]Data loaded!")
        except Exception as e:
            progress.update(task, description=f"[red]Error: {e}")
            raise typer.Exit(1)

    console.print()
    console.print(f"[bold]Gameweek {report.gameweek} Scout Report[/bold]")
    console.print(f"Sources: {', '.join(report.sources)}")

    # Top form picks
    if report.picks:
        console.print()
        picks_table = Table(title="Top Form Picks")
        picks_table.add_column("Player", style="white")
        picks_table.add_column("Pos", style="cyan")
        picks_table.add_column("Team", style="yellow")
        picks_table.add_column("Reason", style="dim")

        for pick in report.picks[:8]:
            picks_table.add_row(
                pick.player_name,
                pick.position,
                pick.team,
                pick.reason,
            )
        console.print(picks_table)

    # Captain picks
    if report.captain_picks:
        console.print()
        captain_table = Table(title="Captain Picks")
        captain_table.add_column("Player", style="white")
        captain_table.add_column("Pos", style="cyan")
        captain_table.add_column("Source", style="yellow")
        captain_table.add_column("Reason", style="dim")

        for pick in report.captain_picks[:5]:
            captain_table.add_row(
                pick.player_name,
                pick.position,
                pick.source,
                pick.reason,
            )
        console.print(captain_table)

    # Differentials
    console.print()
    diff_picks = fetch_differentials()
    if diff_picks:
        diff_table = Table(title="Differentials (<10% ownership)")
        diff_table.add_column("Player", style="white")
        diff_table.add_column("Pos", style="cyan")
        diff_table.add_column("Team", style="yellow")
        diff_table.add_column("Reason", style="dim")

        for pick in diff_picks[:5]:
            diff_table.add_row(
                pick.player_name,
                pick.position,
                pick.team,
                pick.reason,
            )
        console.print(diff_table)


@app.command()
def backtest(
    weeks: int = typer.Option(3, "--weeks", "-w", help="Number of gameweeks to test"),
) -> None:
    """
    Backtest predictions against actual results.

    Tests the projection model against recent completed gameweeks.
    """
    from .predictions import run_backtest, print_backtest_report

    console.print(Panel("[bold blue]Prediction Backtest[/bold blue]", style="blue"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Testing last {weeks} gameweeks...", total=None)

        try:
            result = run_backtest()
            progress.update(task, description="[green]Backtest complete!")
        except Exception as e:
            progress.update(task, description=f"[red]Error: {e}")
            console.print(f"\n[red]{e}[/red]")
            raise typer.Exit(1)

    console.print()

    # Summary metrics
    summary_table = Table(title="Backtest Results")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Gameweeks Tested", str(result.gameweeks_tested))
    summary_table.add_row("Total Predictions", str(result.total_predictions))
    summary_table.add_row("Mean Absolute Error", f"{result.mean_absolute_error:.2f} pts")
    summary_table.add_row("RMSE", f"{result.root_mean_square_error:.2f} pts")
    summary_table.add_row("Correlation", f"{result.correlation:.3f}")
    summary_table.add_row("Top 10 Hit Rate", f"{result.top_10_hit_rate * 100:.1f}%")
    summary_table.add_row("Captain Accuracy", f"{result.captain_accuracy * 100:.1f}%")

    console.print(summary_table)

    # Position breakdown
    if result.mae_by_position:
        console.print()
        pos_table = Table(title="Error by Position")
        pos_table.add_column("Position", style="cyan")
        pos_table.add_column("MAE", style="yellow")

        for pos, mae in result.mae_by_position.items():
            pos_table.add_row(pos, f"{mae:.2f} pts")
        console.print(pos_table)

    # Sample predictions
    if result.player_results:
        console.print()
        console.print("[bold]Sample Predictions[/bold]")

        sorted_results = sorted(
            result.player_results,
            key=lambda x: abs(x.actual_points - x.predicted_points),
        )

        # Best
        console.print("\n[green]Most accurate:[/green]")
        for r in sorted_results[:3]:
            error = r.actual_points - r.predicted_points
            console.print(
                f"  {r.player_name}: Predicted {r.predicted_points:.1f}, "
                f"Actual {r.actual_points} (error: {error:+.1f})"
            )

        # Worst
        console.print("\n[red]Least accurate:[/red]")
        for r in sorted_results[-3:]:
            error = r.actual_points - r.predicted_points
            console.print(
                f"  {r.player_name}: Predicted {r.predicted_points:.1f}, "
                f"Actual {r.actual_points} (error: {error:+.1f})"
            )


@app.command()
def version() -> None:
    """Show version information."""
    console.print("[bold]FPL Assistant[/bold] v0.1.0")
    console.print(
        "A local Fantasy Premier League assistant using optimization and LLM integration"
    )


if __name__ == "__main__":
    app()
