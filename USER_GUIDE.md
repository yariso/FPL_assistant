# FPL Assistant User Guide

A powerful Fantasy Premier League assistant that uses data analysis and optimization to help you make better decisions.

**Live App:** [fplassistantjh.streamlit.app](https://fplassistantjh.streamlit.app)

---

## Getting Started

### 1. Enter Your Manager ID

In the sidebar under **"Your Team"**, enter your FPL Manager ID.

**How to find your Manager ID:**
1. Go to [fantasy.premierleague.com](https://fantasy.premierleague.com)
2. Click on "Points" or "Pick Team"
3. Look at the URL - it will be: `fantasy.premierleague.com/entry/XXXXXXX/event/...`
4. The number (XXXXXXX) is your Manager ID

No password or login required - the app only reads public data.

### 2. Load Data

Click **"Update"** in the sidebar to fetch the latest player data from the FPL API.

### 3. Load Your Team

On the Weekly Advice page, click **"Load My Team"** to fetch your current squad.

---

## Features

### Weekly Advice (Main Page)

Your one-stop shop for weekly decisions. Shows:

- **Deadline Countdown** - Time until the next deadline
- **Captain Pick** - Who to captain based on expected points and fixture difficulty
- **Transfer Suggestions** - Who to sell and who to buy
- **Team News** - Injury/suspension alerts for your players
- **Chip Timing** - When to use Bench Boost, Triple Captain, Free Hit, Wildcard

#### Captain Analysis
Uses Monte Carlo simulation (1000+ scenarios) to find the optimal captain. Shows:
- Expected points for top options
- Ceiling (best case) and floor (worst case)
- Effective Ownership impact

#### Transfer Suggestions
Analyzes your squad and recommends:
- Players to sell (poor form, bad fixtures, injury risk)
- Players to buy (good form, easy fixtures, value)
- Whether a -4 hit is worth it

---

### Best Squad

Shows the mathematically optimal 15-man squad within budget constraints using linear programming.

**Options:**
- **Budget** - Set your available budget (default: 100.0m)
- **Planning Horizon** - How many gameweeks to optimize for (1-10)

The optimizer considers:
- Expected points over the horizon
- Fixture difficulty
- Form and consistency
- Price changes

---

### My Team

View your current FPL squad with detailed analysis:
- Player values and selling prices
- Form ratings
- Fixture difficulty for upcoming games
- Expected points projections

---

### Scout Tips

Quick-fire recommendations based on the latest data:
- **Differentials** - Low-ownership players with high expected points
- **Value Picks** - Best points-per-million options
- **Form Players** - Hot players over the last 4 weeks
- **Premium Picks** - Best expensive options (8.0m+)

---

### Fixtures

Visual fixture ticker showing difficulty ratings for all 20 teams.

**Color coding:**
- **Green (1-2)** - Easy fixtures (target these teams' attackers)
- **Yellow (3)** - Medium difficulty
- **Red (4-5)** - Hard fixtures (avoid or bench)

**Features:**
- View 1-10 gameweeks ahead
- Sort by average difficulty
- Identify fixture swings for transfer planning
- See blank and double gameweeks

---

### Rivals (Mini-League Analysis)

Track your mini-league rivals to inform differential strategy.

**How to use:**
1. Enter your mini-league ID (find it in the league URL on FPL website)
2. Click "Analyze Rivals"

**Shows:**
- Points gap to rivals
- Players you share with rivals
- Their differentials (they own, you don't)
- Your differentials (you own, they don't)
- Strategy suggestion (match leaders or differentiate)

**Late-season strategy:**
- Behind? Pick differentials to catch up
- Ahead? Match your rivals to protect lead

---

### Players

Full player database with filtering and sorting:

**Filters:**
- Position (GK, DEF, MID, FWD)
- Team
- Price range
- Minimum minutes played

**Columns:**
- Current price and price change
- Form (last 4 GWs)
- Total points
- Goals, assists, clean sheets
- xG and xA (expected goals/assists)
- ICT Index (Influence, Creativity, Threat)

---

### Backtest

Test the accuracy of the tool's predictions against historical data.

See how well the projections matched actual points scored in past gameweeks.

---

### Performance

Track your FPL season performance:

**Auto Stats (Live):**
- Total points and average per week
- Rank progression
- Transfer activity and hit costs
- Best/worst gameweeks
- Weekly points chart

**Manual Recording:**
- Log predictions vs actual results
- Track captain success rate
- Measure tool accuracy over time

---

## Understanding the Metrics

### Expected Points (xP)
Projected points based on:
- Fixture difficulty (FDR)
- Recent form
- xG and xA data
- Historical performance vs similar opponents
- Minutes probability

### Fixture Difficulty Rating (FDR)
Scale of 1-5:
- 1 = Very easy (e.g., top team vs bottom team at home)
- 5 = Very hard (e.g., away at Man City)

### Effective Ownership (EO)
How much a player's performance affects your rank:
- Captain = 2x ownership impact
- High EO = safe pick (won't lose ground if they score)
- Low EO = differential (gain ground if they score, lose if they blank)

### xG / xA
Expected Goals and Expected Assists - statistical measure of chance quality:
- xG > Goals = Player is unlucky, likely to score more
- xG < Goals = Player is overperforming, may regress

---

## Tips for Success

### Weekly Routine
1. **Monday/Tuesday:** Review your team, check injury news
2. **Wednesday/Thursday:** Make transfers, analyze captain options
3. **Friday:** Final check before deadline, set lineup
4. **After deadline:** Check your rivals' moves

### Transfer Strategy
- Don't chase last week's points
- Plan 2-3 weeks ahead using fixture ticker
- Avoid -4 hits unless the numbers clearly support it
- Build team value early season with rising players

### Captain Strategy
- Usually pick your best player with the best fixture
- Consider effective ownership in mini-leagues
- Differential captains only when chasing or significantly behind

### Chip Strategy
- **Bench Boost:** Double gameweeks with 15 good fixtures
- **Triple Captain:** Big DGW or dream fixture
- **Free Hit:** Blank gameweeks or fixture swings
- **Wildcard:** Major squad restructure (usually GW8-ish and GW20-ish)

---

## Troubleshooting

### "No data loaded"
Click "Update" in the sidebar to fetch data from the FPL API.

### Team not loading
- Check your Manager ID is correct
- The FPL API may be slow during peak times - try again in a few minutes

### Data seems outdated
Click "Force" in the sidebar to clear cache and fetch fresh data.

### App is slow
The app runs calculations on-demand. Complex pages (Best Squad optimizer) take longer.

---

## Privacy

- Your Manager ID is only used to fetch your public FPL data
- No login credentials required
- No data is stored permanently
- All data comes from the official FPL API

---

## Feedback & Issues

Report bugs or suggest features: [GitHub Issues](https://github.com/yariso/FPL_assistant/issues)

---

*Built with Python, Streamlit, and PuLP optimization*
