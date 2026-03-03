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
    "hospital": "Hospital", "doctor": "Clinic", "health": "Healthcare", "dentist": "Dental Clinic",
    "pharmacy": "Pharmacy", "restaurant": "Restaurant", "cafe": "Cafe", "coffee_shop": "Cafe",
    "gym": "Gym", "school": "School", "university": "Education", "store": "Retail Store",
    "beauty_salon": "Salon", "hair_care": "Salon", "lodging": "Hotel", "bank": "Bank",
    "lawyer": "Legal Services", "accounting": "Accounting", "car_dealer": "Car Dealership",
    "car_repair": "Auto Repair", "clothing_store": "Clothing Store", "electronics_store": "Electronics Store",
    "furniture_store": "Furniture Store", "hardware_store": "Hardware Store", "home_goods_store": "Home Goods Store",
    "jewelry_store": "Jewelry Store", "shoe_store": "Shoe Store", "shopping_mall": "Shopping Mall",
    "supermarket": "Supermarket", "bakery": "Bakery", "bar": "Bar", "meal_delivery": "Food Delivery",
    "meal_takeaway": "Takeaway", "night_club": "Night Club", "spa": "Spa", "tourist_attraction": "Tourist Attraction",
    "travel_agency": "Travel Agency", "real_estate_agency": "Real Estate", "insurance_agency": "Insurance",
    "moving_company": "Moving Company", "painter": "Painting Service", "plumber": "Plumbing Service",
    "electrician": "Electrical Service", "roofing_contractor": "Roofing Service", "locksmith": "Locksmith",
    "laundry": "Laundry Service", "veterinary_care": "Veterinary Clinic", "pet_store": "Pet Store",
    "florist": "Florist", "book_store": "Book Store", "library": "Library", "movie_theater": "Movie Theater",
    "museum": "Museum", "art_gallery": "Art Gallery", "stadium": "Stadium", "park": "Park",
    "amusement_park": "Amusement Park", "aquarium": "Aquarium", "zoo": "Zoo", "bowling_alley": "Bowling Alley",
    "casino": "Casino", "church": "Church", "hindu_temple": "Temple", "mosque": "Mosque",
    "synagogue": "Synagogue", "atm": "ATM", "post_office": "Post Office", "fire_station": "Fire Station",
    "police": "Police Station", "courthouse": "Courthouse", "embassy": "Embassy",
    "local_government_office": "Government Office", "gas_station": "Gas Station", "parking": "Parking",
    "car_wash": "Car Wash", "car_rental": "Car Rental", "taxi_stand": "Taxi Stand",
    "transit_station": "Transit Station", "train_station": "Train Station", "bus_station": "Bus Station",
    "airport": "Airport", "campground": "Campground", "rv_park": "RV Park", "storage": "Storage Facility"
}

# Rule-based keyword enrichment map
RULE_BASED_KEYWORD_MAP = {
    "misthan": "Sweet Shop", "sweets": "Sweet Shop", "mithai": "Sweet Shop", "halwai": "Sweet Shop",
    "confectionery": "Sweet Shop", "nursing": "Nursing Home", "clinic": "Clinic", "hospital": "Hospital",
    "diagnostic": "Diagnostic Center", "pathology": "Pathology Lab", "dental": "Dental Clinic",
    "eye": "Eye Clinic", "skin": "Skin Clinic", "physiotherapy": "Physiotherapy Center",
    "pharmacy": "Pharmacy", "medical": "Medical Store", "restaurant": "Restaurant", "dhaba": "Dhaba",
    "cafe": "Cafe", "coffee": "Cafe", "bakery": "Bakery", "pizza": "Pizza Place",
    "burger": "Burger Joint", "biryani": "Biryani Restaurant", "chinese": "Chinese Restaurant",
    "salon": "Salon", "parlour": "Beauty Parlor", "spa": "Spa", "gym": "Gym",
    "fitness": "Fitness Center", "yoga": "Yoga Center", "school": "School", "college": "College",
    "institute": "Institute", "coaching": "Coaching Center", "tuition": "Tuition Center",
    "hotel": "Hotel", "guest": "Guest House", "lodge": "Lodge", "resort": "Resort",
    "jewel": "Jewellery Shop", "gold": "Jewellery Shop", "electronics": "Electronics Store",
    "mobile": "Mobile Shop", "computer": "Computer Store", "furniture": "Furniture Store",
    "cloth": "Clothing Store", "garment": "Garment Shop", "boutique": "Boutique",
    "tailor": "Tailor Shop", "shoe": "Shoe Store", "footwear": "Footwear Shop",
    "book": "Book Store", "stationery": "Stationery Shop", "gift": "Gift Shop",
    "toy": "Toy Store", "hardware": "Hardware Store", "paint": "Paint Shop",
    "plumber": "Plumbing Service", "electrician": "Electrical Service", "carpenter": "Carpentry Service",
    "mechanic": "Mechanic Shop", "garage": "Auto Garage", "service": "Service Center",
    "repair": "Repair Shop", "laundry": "Laundry Service", "dry": "Dry Cleaning",
    "pet": "Pet Shop", "veterinary": "Veterinary Clinic", "temple": "Temple",
    "mandir": "Temple", "mosque": "Mosque", "masjid": "Mosque", "church": "Church",
    "gurudwara": "Gurudwara", "bank": "Bank", "atm": "ATM", "insurance": "Insurance Agency",
    "real estate": "Real Estate Agency", "property": "Property Dealer", "travel": "Travel Agency",
    "tour": "Tour Operator"
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


def search_businesses(business_name: str, city: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Search for businesses and return top 5 results"""
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": f"{business_name} {city}",
        "key": GOOGLE_API_KEY,
        "region": "IN",
        "locationbias": "country:IN"
    }
    data, error = google_api_call_with_retry(url, params)
    if error:
        return [], error
    if data and data.get("results"):
        return data["results"][:5], None
    return [], "No results found"


def fetch_place_details(place_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Get detailed place information"""
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "name,rating,user_ratings_total,types,website,opening_hours,photos,reviews,geometry,formatted_address",
        "key": GOOGLE_API_KEY
    }
    data, error = google_api_call_with_retry(url, params)
    if error:
        return None, error
    if data and data.get("result"):
        return data["result"], None
    return None, "No details found"


def get_local_radius_rank(lat: float, lng: float, keyword: str, target_place_id: str, radius: int = 3000) -> int:
    """Get business rank in local radius using Nearby Search"""
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "keyword": keyword,
        "key": GOOGLE_API_KEY
    }
    data, error = google_api_call_with_retry(url, params)
    if error or not data or not data.get("results"):
        return 21
    results = data["results"]
    for idx, result in enumerate(results[:20], 1):
        if result.get("place_id") == target_place_id:
            return idx
    return 21


def extract_locality(formatted_address: str) -> Optional[str]:
    """Extract locality from formatted address"""
    parts = formatted_address.split(",")
    if len(parts) >= 2:
        locality = parts[0].strip()
        return locality
    return None


def search_ranking_for_query(query: str, target_place_id: str) -> int:
    """Simulate ranking by searching and finding position"""
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {
        "query": query,
        "key": GOOGLE_API_KEY,
        "region": "IN",
        "locationbias": "country:IN"
    }
    data, error = google_api_call_with_retry(url, params)
    if error or not data or not data.get("results"):
        return 21
    results = data["results"]
    for idx, result in enumerate(results[:20], 1):
        if result.get("place_id") == target_place_id:
            return idx
    return 21


def fetch_nearby_competitors(lat: float, lng: float, keyword: str, target_place_id: str, radius: int = 3000) -> List[Dict[str, Any]]:
    """Find nearby competitors"""
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "keyword": keyword,
        "key": GOOGLE_API_KEY
    }
    data, error = google_api_call_with_retry(url, params)
    if error or not data or not data.get("results"):
        return []
    competitors = [r for r in data["results"] if r.get("place_id") != target_place_id][:5]
    return competitors


def safe_json_parse(content: str) -> Optional[Dict]:
    """Safely parse JSON from response"""
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


def detect_category(place_types: List[str]) -> str:
    """Detect business category using deterministic mapping"""
    for place_type in place_types:
        place_type_lower = place_type.lower()
        if place_type_lower in TYPE_MAP:
            return TYPE_MAP[place_type_lower]
    return place_types[0].replace("_", " ").title() if place_types else "Business"


def extract_search_keyword(business_name: str, place_types: List[str], reviews: List[Dict[str, Any]]) -> str:
    """Extract niche keyword using business name, types, and reviews"""
    name_tokens = business_name.lower().split()
    for token in name_tokens:
        for key, value in RULE_BASED_KEYWORD_MAP.items():
            if key in token:
                return value
    for place_type in place_types:
        place_type_lower = place_type.lower()
        for key, value in RULE_BASED_KEYWORD_MAP.items():
            if key in place_type_lower:
                return value
    if reviews:
        review_texts = " ".join([r.get("text", "")[:200] for r in reviews[:5]]).lower()
        for key, value in RULE_BASED_KEYWORD_MAP.items():
            if key in review_texts:
                return value
    for place_type in place_types:
        place_type_lower = place_type.lower()
        if place_type_lower in TYPE_MAP:
            return TYPE_MAP[place_type_lower]
    return place_types[0].replace("_", " ").title() if place_types else "Business"


def analyze_sentiment(reviews: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze review sentiment using Gemini"""
    if not reviews:
        return {"positive": 0, "neutral": 0, "negative": 0, "praise_keywords": [], "complaint_keywords": []}
    review_texts = [r.get("text", "")[:500] for r in reviews[:10]]
    combined = "\n---\n".join(review_texts)
    prompt = f"""Analyze these customer reviews and return sentiment analysis.
Reviews:
{combined}
Return JSON with:
{{"positive": <percentage 0-100>, "neutral": <percentage 0-100>, "negative": <percentage 0-100>, "praise_keywords": [<top 3 positive keywords>], "complaint_keywords": [<top 3 negative keywords>]}}
Percentages must sum to 100."""
    try:
        response = gemini_model.generate_content(prompt)
        content = response.text.strip()
        sentiment = safe_json_parse(content)
        if sentiment:
            return sentiment
    except:
        pass
    return {"positive": 60, "neutral": 30, "negative": 10, "praise_keywords": ["quality", "service", "friendly"], "complaint_keywords": ["wait time", "price", "parking"]}


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


def calculate_ranking_score(local_rank: int, locality_rank: int, city_rank: int) -> float:
    """Calculate weighted ranking score"""
    def rank_to_score(rank: int) -> float:
        if rank <= 3:
            return 100.0
        elif rank <= 10:
            return 100 - ((rank - 3) * 7)
        elif rank <= 20:
            return 51 - ((rank - 10) * 5)
        else:
            return 0.0
    local_score = rank_to_score(local_rank)
    locality_score = rank_to_score(locality_rank)
    city_score = rank_to_score(city_rank)
    return (0.70 * local_score) + (0.20 * locality_score) + (0.10 * city_score)


def calculate_visibility_score(ranking_score: float, review_count: int, sentiment: Dict, profile_score: int) -> float:
    """Calculate final visibility score"""
    review_authority = min((review_count / 100) * 100, 100)
    sentiment_health = sentiment.get("positive", 0) - (sentiment.get("negative", 0) * 0.5)
    sentiment_health = max(0, min(sentiment_health, 100))
    visibility = (0.35 * ranking_score + 0.25 * review_authority + 0.15 * sentiment_health + 0.25 * profile_score)
    return round(visibility, 1)


def generate_recommendations(business_name: str, category: str, city: str, visibility_score: float, profile_score: int, sentiment: Dict, local_rank: int) -> Dict[str, Any]:
    """Generate AI recommendations"""
    prompt = f"""Business: {business_name}
Category: {category}
City: {city}
Visibility Score: {visibility_score}/100
Profile Score: {profile_score}/100
Local Area Rank: {local_rank}
Sentiment: {sentiment.get('positive', 0)}% positive, {sentiment.get('negative', 0)}% negative
Generate:
1. 5 specific optimization suggestions
2. An improved business description (2-3 sentences)
3. 3 example review reply templates
4. 3 Google post ideas
Return as JSON:
{{"suggestions": [<5 strings>], "description": "<improved description>", "review_replies": [<3 templates>], "post_ideas": [<3 ideas>]}}"""
    try:
        response = gemini_model.generate_content(prompt)
        content = response.text.strip()
        recommendations = safe_json_parse(content)
        if recommendations:
            return recommendations
    except:
        pass
    return {
        "suggestions": ["Increase review count by asking satisfied customers", "Add more high-quality photos", "Update business hours regularly", "Respond to all reviews within 24 hours", "Add detailed business description"],
        "description": f"Welcome to {business_name}, your trusted {category} in {city}. We pride ourselves on exceptional service and customer satisfaction.",
        "review_replies": ["Thank you for your wonderful feedback! We're thrilled you enjoyed your experience.", "We appreciate your review and are glad we could serve you well.", "Thanks for taking the time to share your thoughts. We look forward to seeing you again!"],
        "post_ideas": ["Share a behind-the-scenes look at your team", "Highlight a customer success story", "Announce a special promotion or event"]
    }


def main_analysis(selected_place_id: str, city: str):
    """Main analysis pipeline"""
    with st.spinner("📊 Fetching business details..."):
        place_details, error = fetch_place_details(selected_place_id)
        if error:
            st.error(f"❌ {error}")
            return
        if not place_details:
            st.error("❌ Could not fetch business details.")
            return
        business = BusinessData(
            place_id=selected_place_id,
            name=place_details.get("name", ""),
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
        category = detect_category(business.types)
        st.info(f"📋 **Detected Google Types:** {', '.join(business.types[:5])}")
        st.success(f"🏷️ **Classified as:** {category}")
    
    with st.spinner("🔎 Extracting search keyword..."):
        search_keyword = extract_search_keyword(business.name, business.types, business.reviews)
        st.success(f"🔑 **Search Keyword:** {search_keyword}")
    
    with st.spinner("📈 Analyzing local area ranking..."):
        local_rank = get_local_radius_rank(business.lat, business.lng, search_keyword, selected_place_id)
    
    with st.spinner("📍 Analyzing locality ranking..."):
        locality = extract_locality(business.formatted_address)
        if locality:
            locality_query = f"best {search_keyword} in {locality}"
            locality_rank = search_ranking_for_query(locality_query, selected_place_id)
        else:
            locality_rank = 21
    
    with st.spinner("🌆 Analyzing city ranking..."):
        city_query = f"best {search_keyword} in {city}"
        city_rank = search_ranking_for_query(city_query, selected_place_id)
    
    with st.spinner("🏪 Finding competitors..."):
        competitors = fetch_nearby_competitors(business.lat, business.lng, search_keyword, selected_place_id)
    
    with st.spinner("💬 Analyzing reviews..."):
        sentiment = analyze_sentiment(business.reviews)
    
    with st.spinner("✅ Calculating scores..."):
        profile_score = calculate_profile_score(business)
        ranking_score = calculate_ranking_score(local_rank, locality_rank, city_rank)
        visibility_score = calculate_visibility_score(ranking_score, business.user_ratings_total, sentiment, profile_score)
    
    with st.spinner("🤖 Generating AI recommendations..."):
        recommendations = generate_recommendations(business.name, category, city, visibility_score, profile_score, sentiment, local_rank)
    
    st.success("✅ Analysis complete!")
    st.markdown("---")
    st.header(f"📊 Visibility Report: {business.name}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Visibility Score", f"{visibility_score}/100")
    with col2:
        st.metric("⭐ Rating", f"{business.rating}/5.0")
    with col3:
        st.metric("💬 Reviews", business.user_ratings_total)
    with col4:
        st.metric("🏷️ Category", category)
    
    st.markdown("---")
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("🔍 Ranking Performance")
        st.write(f"📍 **Local Area Rank (3km radius):** #{local_rank}")
        st.write(f"🏘️ **Locality Rank ({locality if locality else 'N/A'}):** #{locality_rank}")
        st.write(f"🌆 **City Rank ({city}):** #{city_rank}")
        st.metric("📊 Overall Ranking Score", f"{ranking_score:.1f}/100")
    
    with col_right:
        st.subheader("✅ Profile Completeness")
        st.progress(profile_score / 100)
        st.metric("Profile Score", f"{profile_score}/100")
        checks = [
            ("Website", "✅" if business.website else "❌"),
            ("Opening Hours", "✅" if business.opening_hours else "❌"),
            ("20+ Photos", "✅" if len(business.photos) > 20 else "❌"),
            ("75+ Reviews", "✅" if business.user_ratings_total > 75 else "❌"),
            ("Rating ≥ 4.2", "✅" if business.rating >= 4.2 else "❌")
        ]
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


st.title("🔍 Local Visibility Intelligence Dashboard")
st.markdown("**Analyze your local SEO performance for any business category**")
st.markdown("---")

if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'selected_place_id' not in st.session_state:
    st.session_state.selected_place_id = None

col1, col2 = st.columns(2)
with col1:
    business_name = st.text_input("🏢 Business Name", placeholder="e.g., Blue Bottle Coffee")
with col2:
    city = st.text_input("📍 City", placeholder="e.g., Mumbai")

if st.button("🔍 Search Businesses", type="primary", use_container_width=True):
    if not business_name or not city:
        st.error("⚠️ Please enter both business name and city")
    elif not GOOGLE_API_KEY or not GEMINI_API_KEY:
        st.error("⚠️ API keys not configured. Please set GOOGLE_API_KEY and GEMINI_KEY in .env file")
    else:
        with st.spinner("🔍 Searching for businesses..."):
            results, error = search_businesses(business_name, city)
            if error:
                st.error(f"❌ {error}")
            elif not results:
                st.warning("No businesses found. Try a different search.")
            else:
                st.session_state.search_results = results
                st.session_state.selected_place_id = None

if st.session_state.search_results:
    st.markdown("---")
    st.subheader("📋 Select Your Business")
    
    for idx, result in enumerate(st.session_state.search_results):
        col_radio, col_info = st.columns([1, 9])
        with col_radio:
            if st.radio("", [idx], key=f"radio_{idx}", label_visibility="collapsed"):
                st.session_state.selected_place_id = result.get("place_id")
        with col_info:
            st.write(f"**{result.get('name', 'Unknown')}**")
            st.write(f"⭐ {result.get('rating', 'N/A')} | 📍 {result.get('formatted_address', 'N/A')}")
        st.markdown("---")
    
    if st.session_state.selected_place_id:
        if st.button("🚀 Analyze Selected Business", type="primary", use_container_width=True):
            main_analysis(st.session_state.selected_place_id, city)

st.markdown("---")
st.caption("Powered by Google Places API & Google Gemini 2.5 Flash Lite (Free Tier)")
