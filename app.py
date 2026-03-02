import streamlit as st
import httpx
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import re
import time
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_KEY")

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')

st.set_page_config(page_title="Local Visibility Intelligence Dashboard", layout="wide")

# Deterministic type mapping
TYPE_MAP = {
    "hospital": "Hospital",
    "doctor": "Clinic",
    "health": "Healthcare",
    "dentist": "Dental Clinic",
    "pharmacy": "Pharmacy",
    "restaurant": "Restaurant",
    "cafe": "Cafe",
    "coffee_shop": "Cafe",
    "gym": "Gym",
    "school": "School",
    "university": "Education",
    "store": "Retail Store",
    "beauty_salon": "Salon",
    "hair_care": "Salon",
    "lodging": "Hotel",
    "bank": "Bank",
    "lawyer": "Legal Services",
    "accounting": "Accounting",
    "car_dealer": "Car Dealership",
    "car_repair": "Auto Repair",
    "clothing_store": "Clothing Store",
    "electronics_store": "Electronics Store",
    "furniture_store": "Furniture Store",
    "hardware_store": "Hardware Store",
    "home_goods_store": "Home Goods Store",
    "jewelry_store": "Jewelry Store",
    "shoe_store": "Shoe Store",
    "shopping_mall": "Shopping Mall",
    "supermarket": "Supermarket",
    "bakery": "Bakery",
    "bar": "Bar",
    "meal_delivery": "Food Delivery",
    "meal_takeaway": "Takeaway",
    "night_club": "Night Club",
    "spa": "Spa",
    "tourist_attraction": "Tourist Attraction",
    "travel_agency": "Travel Agency",
    "real_estate_agency": "Real Estate",
    "insurance_agency": "Insurance",
    "moving_company": "Moving Company",
    "painter": "Painting Service",
    "plumber": "Plumbing Service",
    "electrician": "Electrical Service",
    "roofing_contractor": "Roofing Service",
    "locksmith": "Locksmith",
    "laundry": "Laundry Service",
    "veterinary_care": "Veterinary Clinic",
    "pet_store": "Pet Store",
    "florist": "Florist",
    "book_store": "Book Store",
    "library": "Library",
    "movie_theater": "Movie Theater",
    "museum": "Museum",
    "art_gallery": "Art Gallery",
    "stadium": "Stadium",
    "park": "Park",
    "amusement_park": "Amusement Park",
    "aquarium": "Aquarium",
    "zoo": "Zoo",
    "bowling_alley": "Bowling Alley",
    "casino": "Casino",
    "church": "Church",
    "hindu_temple": "Temple",
    "mosque": "Mosque",
    "synagogue": "Synagogue",
    "atm": "ATM",
    "post_office": "Post Office",
    "fire_station": "Fire Station",
    "police": "Police Station",
    "courthouse": "Courthouse",
    "embassy": "Embassy",
    "local_government_office": "Government Office",
    "gas_station": "Gas Station",
    "parking": "Parking",
    "car_wash": "Car Wash",
    "car_rental": "Car Rental",
    "taxi_stand": "Taxi Stand",
    "transit_station": "Transit Station",
    "train_station": "Train Station",
    "bus_station": "Bus Station",
    "airport": "Airport",
    "campground": "Campground",
    "rv_park": "RV Park",
    "storage": "Storage Facility"
}


@dataclass
class BusinessData:
    place_id: str
    name: str
    rating: float
    user_ratings_total: int
    types: List[str]
    website: Optional[str]
    opening_hours: Optional[Dict]
    photos: List[Dict]
    reviews: List[Dict]
    lat: float
    lng: float
    formatted_address: str


def google_api_call_with_retry(url: str, params: Dict, max_retries: int = 3) -> Tuple[Optional[Dict], Optional[str]]:
    """Make Google API call with retry logic"""
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(url, params=params)
                data = response.json()
                
                status = data.get("status")
                if status == "OK":
                    return data, None
                elif status in ["ZERO_RESULTS", "NOT_FOUND"]:
                    return data, f"No results found (Status: {status})"
                else:
                    error_msg = f"API Error: {status}"
                    if "error_message" in data:
                        error_msg += f" - {data['error_message']}"
                    return data, error_msg
                    
        except Exception as e:
            if attempt == max_retries - 1:
                return None, f"Request failed after {max_retries} attempts: {str(e)}"
            time.sleep(1)
    
    return None, "Request failed"


def fetch_place_by_text(business_name: str, city: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[Dict]]:
    """Find place using Text Search API"""
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{business_name} {city}",
        "key": GOOGLE_API_KEY,
        "region": "IN",
        "locationbias": "country:IN"
    }
    
    data, error = google_api_call_with_retry(url, params)
    
    if error:
        return None, error, data
    
    if data and data.get("results"):
        return data["results"][0], None, data
    
    return None, "No results found", data


def fetch_place_details(place_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[Dict]]:
    """Get detailed place information"""
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,rating,user_ratings_total,types,website,opening_hours,photos,reviews,geometry,formatted_address",
        "key": GOOGLE_API_KEY
    }
    
    data, error = google_api_call_with_retry(url, params)
    
    if error:
        return None, error, data
    
    if data and data.get("result"):
        return data["result"], None, data
    
    return None, "No details found", data


def fetch_nearby_competitors(lat: float, lng: float, category: str, target_place_id: str, radius: int = 3000) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[Dict]]:
    """Find nearby competitors"""
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "keyword": category,
        "key": GOOGLE_API_KEY
    }
    
    data, error = google_api_call_with_retry(url, params)
    
    if error:
        return [], error, data
    
    if data and data.get("results"):
        competitors = [r for r in data["results"] if r.get("place_id") != target_place_id][:5]
        return competitors, None, data
    
    return [], None, data


def search_ranking_for_query(query: str, target_place_id: str) -> Tuple[int, Optional[str], Optional[Dict]]:
    """Simulate ranking by searching and finding position"""
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": query,
        "key": GOOGLE_API_KEY,
        "region": "IN",
        "locationbias": "country:IN"
    }
    
    data, error = google_api_call_with_retry(url, params)
    
    if error:
        return 21, error, data
    
    if data and data.get("results"):
        results = data["results"]
        for idx, result in enumerate(results[:20], 1):
            if result.get("place_id") == target_place_id:
                return idx, None, data
    
    return 21, None, data


def safe_json_parse(content: str) -> Optional[Dict]:
    """Safely parse JSON from OpenAI response"""
    try:
        return json.loads(content)
    except:
        json_match = re.search(r'\{.*\}|\[.*\]', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
    return None


def detect_category(place_types: List[str]) -> Tuple[str, List[str]]:
    """Detect business category using deterministic mapping first, then Gemini fallback"""
    
    # First try deterministic mapping
    for place_type in place_types:
        place_type_lower = place_type.lower()
        if place_type_lower in TYPE_MAP:
            return TYPE_MAP[place_type_lower], place_types
    
    # Fallback to Gemini only if no match
    prompt = f"""Given these Google Place types: {', '.join(place_types)}

Return a single, clean, human-readable business category.

Examples:
- restaurant, food, establishment → Restaurant
- cafe, coffee_shop → Cafe
- hair_care, beauty_salon → Salon
- gym, health → Gym
- doctor, health → Clinic
- store, clothing_store → Retail Store

Return only the category name, nothing else."""

    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip(), place_types
    except Exception as e:
        st.warning(f"Gemini category detection failed: {str(e)}")
        return place_types[0].replace("_", " ").title() if place_types else "Business", place_types


def generate_search_queries(category: str, city: str) -> List[str]:
    """Generate high-intent local search queries"""
    prompt = f"""Generate 5 high-intent local search queries for finding a {category} business in {city}, India.

IMPORTANT: The business category is "{category}". Generate queries specifically for this category only.

These should be queries real customers would use to find this type of business.

Format: Return as JSON array of strings.

Example for a Cafe in Mumbai:
["best cafe in Mumbai", "top rated cafe near me", "coffee shop Mumbai", "cafe with wifi Mumbai", "popular cafe in Mumbai"]

Category: {category}
City: {city}

Return only the JSON array with queries specifically for {category}."""

    try:
        response = gemini_model.generate_content(prompt)
        content = response.text.strip()
        queries = safe_json_parse(content)
        
        if queries and isinstance(queries, list):
            return queries[:5]
    except Exception as e:
        st.warning(f"Gemini query generation failed: {str(e)}")
    
    return [
        f"best {category} in {city}",
        f"top rated {category} near me",
        f"{category} {city}",
        f"popular {category} in {city}",
        f"{category} near me"
    ]


def analyze_sentiment(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze review sentiment using Gemini"""
    if not reviews:
        return {
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "praise_keywords": [],
            "complaint_keywords": []
        }
    
    review_texts = [r.get("text", "")[:500] for r in reviews[:10]]
    combined = "\n---\n".join(review_texts)
    
    prompt = f"""Analyze these customer reviews and return sentiment analysis.

Reviews:
{combined}

Return JSON with:
{{
  "positive": <percentage 0-100>,
  "neutral": <percentage 0-100>,
  "negative": <percentage 0-100>,
  "praise_keywords": [<top 3 positive keywords>],
  "complaint_keywords": [<top 3 negative keywords>]
}}

Percentages must sum to 100."""

    try:
        response = gemini_model.generate_content(prompt)
        content = response.text.strip()
        sentiment = safe_json_parse(content)
        
        if sentiment:
            return sentiment
    except Exception as e:
        st.warning(f"Gemini sentiment analysis failed: {str(e)}")
    
    return {
        "positive": 60,
        "neutral": 30,
        "negative": 10,
        "praise_keywords": ["quality", "service", "friendly"],
        "complaint_keywords": ["wait time", "price", "parking"]
    }


def calculate_profile_score(business: BusinessData) -> int:
    """Calculate profile completeness score"""
    score = 0
    
    if business.website:
        score += 15
    
    if business.opening_hours:
        score += 15
    
    if len(business.photos) > 20:
        score += 20
    
    if business.user_ratings_total > 75:
        score += 15
    
    if business.rating >= 4.2:
        score += 15
    
    score += 20
    
    return min(score, 100)


def calculate_ranking_score(ranks: List[int]) -> float:
    """Calculate ranking score from positions"""
    if not ranks:
        return 0.0
    
    avg_rank = sum(ranks) / len(ranks)
    
    if avg_rank <= 3:
        return 100.0
    elif avg_rank <= 10:
        return 100 - ((avg_rank - 3) * 7)
    elif avg_rank <= 20:
        return 51 - ((avg_rank - 10) * 5)
    else:
        return 0.0


def calculate_visibility_score(ranking_score: float, review_count: int, sentiment: Dict, profile_score: int) -> float:
    """Calculate final visibility score"""
    review_authority = min((review_count / 100) * 100, 100)
    
    sentiment_health = sentiment.get("positive", 0) - (sentiment.get("negative", 0) * 0.5)
    sentiment_health = max(0, min(sentiment_health, 100))
    
    visibility = (
        0.35 * ranking_score +
        0.25 * review_authority +
        0.15 * sentiment_health +
        0.25 * profile_score
    )
    
    return round(visibility, 1)


def generate_recommendations(business_name: str, category: str, city: str, visibility_score: float, 
                            profile_score: int, sentiment: Dict, avg_rank: float) -> Dict[str, Any]:
    """Generate AI recommendations"""
    prompt = f"""Business: {business_name}
Category: {category}
City: {city}
Visibility Score: {visibility_score}/100
Profile Score: {profile_score}/100
Average Ranking: {avg_rank}
Sentiment: {sentiment.get('positive', 0)}% positive, {sentiment.get('negative', 0)}% negative

Generate:
1. 5 specific optimization suggestions
2. An improved business description (2-3 sentences)
3. 3 example review reply templates
4. 3 Google post ideas

Return as JSON:
{{
  "suggestions": [<5 strings>],
  "description": "<improved description>",
  "review_replies": [<3 templates>],
  "post_ideas": [<3 ideas>]
}}"""

    try:
        response = gemini_model.generate_content(prompt)
        content = response.text.strip()
        recommendations = safe_json_parse(content)
        
        if recommendations:
            return recommendations
    except Exception as e:
        st.warning(f"Gemini recommendations failed: {str(e)}")
    
    return {
        "suggestions": [
            "Increase review count by asking satisfied customers",
            "Add more high-quality photos",
            "Update business hours regularly",
            "Respond to all reviews within 24 hours",
            "Add detailed business description"
        ],
        "description": f"Welcome to {business_name}, your trusted {category} in {city}. We pride ourselves on exceptional service and customer satisfaction.",
        "review_replies": [
            "Thank you for your wonderful feedback! We're thrilled you enjoyed your experience.",
            "We appreciate your review and are glad we could serve you well.",
            "Thanks for taking the time to share your thoughts. We look forward to seeing you again!"
        ],
        "post_ideas": [
            "Share a behind-the-scenes look at your team",
            "Highlight a customer success story",
            "Announce a special promotion or event"
        ]
    }


def main_analysis(business_name: str, city: str):
    """Main analysis pipeline"""
    
    debug_data = {}
    
    with st.spinner("🔍 Finding your business..."):
        place_basic, error, raw_data = fetch_place_by_text(business_name, city)
        debug_data["text_search"] = raw_data
        
        if error:
            st.error(f"❌ {error}")
            if raw_data:
                with st.expander("🔍 Debug: API Response"):
                    st.json(raw_data)
            return
        
        if not place_basic:
            st.error("❌ Business not found. Please check the name and city.")
            return
        
        place_id = place_basic["place_id"]
    
    with st.spinner("📊 Fetching business details..."):
        place_details, error, raw_data = fetch_place_details(place_id)
        debug_data["place_details"] = raw_data
        
        if error:
            st.error(f"❌ {error}")
            if raw_data:
                with st.expander("🔍 Debug: API Response"):
                    st.json(raw_data)
            return
        
        if not place_details:
            st.error("❌ Could not fetch business details.")
            return
        
        business = BusinessData(
            place_id=place_id,
            name=place_details.get("name", business_name),
            rating=place_details.get("rating", 0.0),
            user_ratings_total=place_details.get("user_ratings_total", 0),
            types=place_details.get("types", []),
            website=place_details.get("website"),
            opening_hours=place_details.get("opening_hours"),
            photos=place_details.get("photos", []),
            reviews=place_details.get("reviews", []),
            lat=place_details["geometry"]["location"]["lat"],
            lng=place_details["geometry"]["location"]["lng"],
            formatted_address=place_details.get("formatted_address", "")
        )
    
    with st.spinner("🏷️ Detecting business category..."):
        category, raw_types = detect_category(business.types)
        
        # Display detected types for debugging
        st.info(f"📋 **Detected Google Types:** {', '.join(raw_types[:5])}")
        st.success(f"🏷️ **Classified as:** {category}")
    
    with st.spinner("🔎 Generating search queries..."):
        queries = generate_search_queries(category, city)
    
    with st.spinner("📈 Analyzing search rankings..."):
        ranks = []
        for query in queries:
            rank, error, raw_data = search_ranking_for_query(query, place_id)
            ranks.append(rank)
            if error:
                st.warning(f"Ranking check failed for '{query}': {error}")
    
    with st.spinner("🏪 Finding competitors..."):
        competitors, error, raw_data = fetch_nearby_competitors(business.lat, business.lng, category, place_id)
        debug_data["nearby_search"] = raw_data
        if error:
            st.warning(f"Competitor search: {error}")
    
    with st.spinner("💬 Analyzing reviews..."):
        sentiment = analyze_sentiment(business.reviews)
    
    with st.spinner("✅ Calculating scores..."):
        profile_score = calculate_profile_score(business)
        ranking_score = calculate_ranking_score(ranks)
        visibility_score = calculate_visibility_score(
            ranking_score, 
            business.user_ratings_total, 
            sentiment, 
            profile_score
        )
    
    with st.spinner("🤖 Generating AI recommendations..."):
        avg_rank = sum(ranks) / len(ranks) if ranks else 21
        recommendations = generate_recommendations(
            business.name, category, city, visibility_score, 
            profile_score, sentiment, avg_rank
        )
    
    st.success("✅ Analysis complete!")
    
    st.markdown("---")
    st.header(f"📊 Visibility Report: {business.name}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        color = "normal" if visibility_score >= 70 else ("inverse" if visibility_score < 50 else "off")
        st.metric("🎯 Visibility Score", f"{visibility_score}/100", delta=None)
    
    with col2:
        st.metric("⭐ Rating", f"{business.rating}/5.0")
    
    with col3:
        st.metric("💬 Reviews", business.user_ratings_total)
    
    with col4:
        st.metric("🏷️ Category", category)
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🔍 Search Query Performance")
        st.write("**Generated Queries:**")
        for i, query in enumerate(queries, 1):
            rank = ranks[i-1]
            rank_text = f"#{rank}" if rank <= 20 else "Not in top 20"
            emoji = "🟢" if rank <= 5 else ("🟡" if rank <= 10 else "🔴")
            st.write(f"{emoji} `{query}` → **{rank_text}**")
        
        st.metric("📊 Ranking Score", f"{ranking_score:.1f}/100")
    
    with col_right:
        st.subheader("✅ Profile Completeness")
        st.progress(profile_score / 100)
        st.metric("Profile Score", f"{profile_score}/100")
        
        checks = []
        checks.append(("Website", "✅" if business.website else "❌"))
        checks.append(("Opening Hours", "✅" if business.opening_hours else "❌"))
        checks.append(("20+ Photos", "✅" if len(business.photos) > 20 else "❌"))
        checks.append(("75+ Reviews", "✅" if business.user_ratings_total > 75 else "❌"))
        checks.append(("Rating ≥ 4.2", "✅" if business.rating >= 4.2 else "❌"))
        
        for check, status in checks:
            st.write(f"{status} {check}")
    
    st.markdown("---")
    
    col_sent, col_comp = st.columns(2)
    
    with col_sent:
        st.subheader("💬 Review Sentiment Analysis")
        
        if business.reviews:
            fig = go.Figure(data=[go.Pie(
                labels=['Positive', 'Neutral', 'Negative'],
                values=[sentiment['positive'], sentiment['neutral'], sentiment['negative']],
                marker=dict(colors=['#00CC66', '#FFD700', '#FF6B6B']),
                hole=0.4
            )])
            fig.update_layout(height=300, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
            
            st.write("**Top Praise Keywords:**")
            for kw in sentiment.get('praise_keywords', []):
                st.write(f"✅ {kw}")
            
            st.write("**Top Complaint Keywords:**")
            for kw in sentiment.get('complaint_keywords', []):
                st.write(f"⚠️ {kw}")
        else:
            st.info("No reviews available for analysis")
    
    with col_comp:
        st.subheader("🏪 Nearby Competitors")
        
        if competitors:
            comp_data = []
            for comp in competitors:
                comp_data.append({
                    "Name": comp.get("name", "Unknown"),
                    "Rating": comp.get("rating", 0.0),
                    "Reviews": comp.get("user_ratings_total", 0)
                })
            
            st.dataframe(comp_data, use_container_width=True, hide_index=True)
        else:
            st.info("No competitors found nearby")
    
    st.markdown("---")
    
    st.subheader("🤖 AI-Generated Recommendations")
    
    with st.expander("💡 Optimization Suggestions", expanded=True):
        for i, suggestion in enumerate(recommendations.get('suggestions', []), 1):
            st.write(f"{i}. {suggestion}")
    
    with st.expander("📝 Improved Business Description"):
        st.write(recommendations.get('description', ''))
    
    with st.expander("💬 Example Review Replies"):
        for i, reply in enumerate(recommendations.get('review_replies', []), 1):
            st.write(f"**Template {i}:**")
            st.write(f"_{reply}_")
            st.write("")
    
    with st.expander("📱 Google Post Ideas"):
        for i, idea in enumerate(recommendations.get('post_ideas', []), 1):
            st.write(f"{i}. {idea}")
    
    st.markdown("---")
    
    with st.expander("🔍 Debug Information"):
        st.write("**API Call Status:**")
        for key, data in debug_data.items():
            if data:
                st.write(f"**{key}:** Status = {data.get('status', 'N/A')}")
                if data.get('status') != 'OK':
                    st.json(data)


st.title("� Local Visibility Intelligence Dashboard")
st.markdown("**Analyze your local SEO performance for any business category**")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    business_name = st.text_input("🏢 Business Name", placeholder="e.g., Blue Bottle Coffee")

with col2:
    city = st.text_input("📍 City", placeholder="e.g., Mumbai")

if st.button("🚀 Analyze Visibility", type="primary", use_container_width=True):
    if not business_name or not city:
        st.error("⚠️ Please enter both business name and city")
    elif not GOOGLE_API_KEY or not GEMINI_API_KEY:
        st.error("⚠️ API keys not configured. Please set GOOGLE_API_KEY and GEMINI_KEY in .env file")
    else:
        main_analysis(business_name, city)

st.markdown("---")
st.caption("Powered by Google Places API & Google Gemini 2.5 Flash Lite (Free Tier)")
