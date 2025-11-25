# FEH Team Generator

A web-based team builder for Fire Emblem Heroes that generates optimal teams based on unit synergies calculated from historical competitive data (CSV format).

## Features

- **Synergy-Based Team Building**: Analyzes CSV data to find units that work well together
- **Interactive Editing**: Drag-and-drop units between teams, remove units, and regenerate
- **Smart Regeneration**: After removing units, remaining units become "seeds" and the removed unit is excluded from that specific team
- **CSV Upload**: Upload your own competitive data for custom synergy calculations
- **Constraint Support**: Seed units, must-use units, forbidden pairs, and more

## Project Structure

```
feh-sds-builder/
├── backend/
│   ├── main.py              # FastAPI server
│   └── team_builder.py      # Core team building logic
├── static/
│   ├── index.html           # Frontend UI
│   ├── app.js               # Frontend logic
│   └── style.css            # Styles
├── uploads/                 # User-uploaded CSVs (created automatically)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration
├── render.yaml              # Render.com deployment config
└── README.md                # This file
```

## Quick Start (Local Development)

### Prerequisites
- Python 3.11+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/feh-sds-builder.git
   cd feh-sds-builder
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the server**
   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

5. **Open in browser**
   ```
   http://localhost:8000
   ```

## Usage

### 1. Input Your Units
Paste your available units (one per line) in the "Available Units" text area.

### 2. Optional: Configure Seeds and Must-Use Units
- **Seed Units**: JSON array specifying which units must be on which team: `[[team1_units], [team2_units], [team3_units], [team4_units]]`
- **Must Use Units**: Units that must be placed somewhere (one per line)

### 3. Optional: Upload CSV Data
Upload a CSV file containing historical team data for synergy calculations. Format:
- Every 4 rows = 1 brigade (4 teams)
- Columns: Region, Player, blank, Captain, blank, Unit1, blank, Unit2, blank, Unit3, blank, Unit4, blank, Unit5

### 4. Generate Teams
Click "Generate Teams" to create 4 teams of 5 units each based on synergies.

### 5. Edit and Regenerate
- **Drag units** between teams to swap them
- **Remove units** using the "Remove" button
- **Click "Re-Run Builder"** to fill empty slots with new units
  - Removed units are **excluded** from their original team
  - Remaining units become **seeds** (locked in place)
  - Removed units can be placed on different teams

### 6. Undo Changes
Click "Undo" to revert the last change (up to 30 steps).

## Deployment

### Deploy to Render.com

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Connect to Render**
   - Go to [Render Dashboard](https://dashboard.render.com/)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will automatically detect the `render.yaml` configuration

3. **Deploy**
   - Render will build using the Dockerfile
   - Your app will be live at `https://your-app-name.onrender.com`

### Docker Deployment

```bash
# Build image
docker build -t feh-team-generator .

# Run container
docker run -p 8000:8000 feh-team-generator

# Access at http://localhost:8000
```

## API Endpoints

### `POST /generate`
Generate initial teams from scratch.

**Request Body:**
```json
{
  "available_units": ["Unit A", "Unit B", ...],
  "seed_units": [[...], [...], [...], [...]],
  "must_use_units": ["Unit X", "Unit Y"],
  "forbidden_pairs": [["Unit A", "Unit B"]],
  "required_pairs": [["Unit C", "Unit D"]],
  "csv_filename": "optional_uploaded_file.csv"
}
```

### `POST /regenerate`
Regenerate teams after user edits.

**Request Body:**
```json
{
  "edited_teams": [[...], [...], [...], [...]],
  "banned_assignments": [{"unit": "Unit A", "team": 0}, ...],
  "all_available_units": ["Unit A", "Unit B", ...],
  "must_use_units": ["Unit X"],
  "csv_filename": "optional_file.csv"
}
```

### `POST /upload-csv`
Upload CSV file for synergy calculations.

**Form Data:**
- `file`: CSV file

### `GET /health`
Health check endpoint.

## How It Works

### Synergy Calculation
1. **Load CSV Data**: Parse historical team compositions
2. **Count Co-occurrences**: Track which units appear together
3. **Calculate Synergy Scores**: Use Jaccard-like similarity with rarity weighting
4. **Build Teams**: Greedily place units to maximize global synergy

### Regeneration Logic
1. **Preserve Seeds**: Remaining units after removal stay in place
2. **Apply Exclusions**: Removed units cannot rejoin their original team
3. **Fill Empty Slots**: Place best-synergy units from available pool
4. **Allow Cross-Team Placement**: Removed units can join other teams

## Configuration

### Environment Variables
- `PORT`: Server port (default: 8000)

### CSV Format
Your CSV should contain brigade data where every 4 rows represent one player's 4 teams:
```
Region, Player, blank, Captain, blank, Unit1, blank, Unit2, blank, Unit3, blank, Unit4, blank, Unit5
```

Skip the first 3 rows if they contain headers.

## Development

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest
```

### Code Structure
- **`backend/main.py`**: FastAPI routes and request handling
- **`backend/team_builder.py`**: Core team building algorithm
- **`static/app.js`**: Frontend JavaScript (drag-drop, API calls)
- **`static/index.html`**: UI structure
- **`static/style.css`**: Styling

## Troubleshooting

### Issue: Teams aren't regenerating correctly
- Check that `banned_assignments` are being passed correctly
- Verify `edited_teams` reflect current team state

### Issue: CSV upload not working
- Ensure CSV follows the expected format
- Check that `skip_header_rows` is set correctly (default: 3)

### Issue: Drag-and-drop not working
- Check browser console for JavaScript errors
- Ensure `app.js` is loaded correctly

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - feel free to use this project for any purpose.

## Credits

Original team building algorithm by LessThan3door
Web interface and deployment setup by the community
