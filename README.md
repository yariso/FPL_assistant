# FPL Assistant

A Fantasy Premier League assistant using mathematical optimization to provide weekly squad recommendations, transfer suggestions, and strategic advice.

## Features

- **Multi-week optimization** - Plan transfers across 4-6 gameweeks to maximize expected points
- **xG-based projections** - Player projections powered by expected goals, assists, and underlying stats
- **Chip strategy** - Optimal timing for Wildcard, Free Hit, Bench Boost, Triple Captain
- **Blank/Double GW handling** - Automatically accounts for fixture congestion
- **Risk analysis** - Effective ownership and differential tracking
- **Set piece integration** - Penalty, corner, and FK taker boosts in projections
- **Post-GW review** - Luck vs skill analysis for each gameweek
- **Rival tracking** - Mini-league analysis with differential recommendations
- **Dual interface** - Both Streamlit web UI and CLI available

## Installation

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)

### Setup

1. Clone or download this repository

2. Install uv (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source ~/.local/bin/env
   ```

3. Install dependencies:
   ```bash
   cd fpl-assistant
   uv sync
   ```

4. Copy the environment template and configure:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

5. Initialize the database:
   ```bash
   uv run python scripts/setup_db.py
   ```

## Configuration

Edit the `.env` file with your settings:

### FPL Credentials
```
FPL_EMAIL=your.email@example.com
FPL_PASSWORD=your_fpl_password
FPL_MANAGER_ID=1234567
```

### Optional Settings
See `.env.example` for all available configuration options.

## Usage

### Command Line Interface

```bash
# Fetch latest FPL data
uv run fpl-assistant update

# Show your current squad
uv run fpl-assistant status

# Run optimization for next 5 gameweeks
uv run fpl-assistant optimize 5

# Compare two players
uv run fpl-assistant compare "Salah" "Saka"

# Backtest prediction accuracy
uv run fpl-assistant backtest
```

### Web Interface (Streamlit)

```bash
uv run streamlit run src/fpl_assistant/app.py
```

Then open http://localhost:8501 in your browser.

## Project Structure

```
fpl-assistant/
├── src/fpl_assistant/
│   ├── api/          # FPL API client
│   ├── data/         # Data models and storage
│   ├── optimizer/    # PuLP optimization engine
│   ├── predictions/  # Points projection system
│   ├── analysis/     # Post-GW, rival, and differential analysis
│   └── app.py        # Streamlit web UI
├── data/             # SQLite database and projections
├── tests/            # Test suite
└── scripts/          # Utility scripts
```

## External Data Sources

For best results, consider subscribing to a projection service:

| Source | Cost | Description |
|--------|------|-------------|
| [FPL Review](https://fplreview.com) | ~£5/month | CSV exports, 6-week projections |
| [Fantasy Football Hub](https://fantasyfootballhub.co.uk) | ~£3/month | AI predictions |
| [FPL Form](https://fplform.com) | Free | Basic projections |

Place CSV files in `data/projections/` for the app to import.

## Development

### Running Tests
```bash
uv run pytest
```

### Type Checking
```bash
uv run mypy src/fpl_assistant
```

### Linting
```bash
uv run ruff check src/fpl_assistant
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [FPL API](https://fantasy.premierleague.com/api/bootstrap-static/) - Unofficial API documentation
- [PuLP](https://coin-or.github.io/pulp/) - Linear programming library
- [Streamlit](https://streamlit.io/) - Web UI framework
