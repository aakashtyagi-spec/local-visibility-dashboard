# Local Visibility Intelligence Dashboard

A Streamlit-based SEO visibility analysis tool for local businesses using Google Places API and Google Gemini AI.

## Features

- 🔍 Automatic business discovery via Google Places
- 🏷️ Smart category detection with deterministic mapping
- 📊 Search ranking simulation across multiple queries
- 🏪 Competitor analysis (nearby businesses)
- 💬 AI-powered review sentiment analysis
- ✅ Profile completeness scoring
- 🎯 Overall visibility score (0-100)
- 🤖 AI-generated optimization recommendations

## Tech Stack

- **Frontend:** Streamlit
- **APIs:** Google Places API, Google Gemini 2.5 Flash Lite
- **Visualization:** Plotly
- **HTTP Client:** httpx

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd <repo-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `.env` file:
```bash
cp .env.example .env
```

4. Add your API keys to `.env`:
```
GOOGLE_API_KEY=your_google_places_api_key
GEMINI_KEY=your_gemini_api_key
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## How It Works

1. **Business Discovery:** Searches Google Places for the business
2. **Category Detection:** Uses deterministic type mapping (80+ categories) with AI fallback
3. **Query Generation:** Creates 5 high-intent local search queries
4. **Ranking Analysis:** Checks business position for each query
5. **Competitor Analysis:** Finds top 5 nearby competitors
6. **Sentiment Analysis:** Analyzes reviews using Gemini AI
7. **Scoring:** Calculates visibility score based on:
   - Ranking performance (35%)
   - Review authority (25%)
   - Sentiment health (15%)
   - Profile completeness (25%)

## API Keys

### Google Places API
- Get key from: https://console.cloud.google.com/
- Enable: Places API, Places API (New)
- Free tier: $200 credit/month

### Google Gemini API
- Get key from: https://aistudio.google.com/app/apikey
- Model: gemini-2.5-flash-lite (free tier)

## License

MIT
