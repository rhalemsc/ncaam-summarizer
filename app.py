import streamlit as st
import pandas as pd
import requests
import json
import re
from cohere import Client

API_KEY = st.secrets["COHERE_API_KEY"]

st.set_page_config(page_title="NCAAM Summarizer", page_icon="🏀")

# -----------------------------
# FUNCTIONS
# -----------------------------
def get_teams():
    url = "http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams?limit=400"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    teams = data['sports'][0]['leagues'][0]['teams']
    df_teams = pd.json_normalize(teams)
    return df_teams[['team.id', 'team.displayName']]

def get_games(target_team_id):
    """Return a dataframe of completed games for the team with added result/opponent/score columns."""
    url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/teams/{target_team_id}/schedule"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    events = data.get('events', [])
    df_events = pd.json_normalize(events)

    if df_events.empty:
        return df_events

    # Keep only completed games (safe checks)
    df_events = df_events[
        df_events["competitions"].apply(
            lambda comps: (
                isinstance(comps, list)
                and len(comps) > 0
                and comps[0].get("status", {}).get("type", {}).get("completed") is True
            )
        )
    ].copy()

    if df_events.empty:
        return df_events

    # Add result (Win/Loss), opponent name, and score string
    def parse_row(row):
        comps = row["competitions"]
        if not isinstance(comps, list) or len(comps) == 0:
            return pd.Series({"result": None, "opponent_name": None, "score_str": None})

        competitors = comps[0].get("competitors", [])
        # find our team and opponent
        our_score = None
        opp_score = None
        opponent_name = None
        our_id = str(target_team_id)
        our_winner = None

        for c in competitors:
            team_obj = c.get("team", {})
            team_id = str(team_obj.get("id", ""))
            score_val = c.get("score", {}).get("value")
            if team_id == our_id:
                our_score = int(score_val) if score_val is not None else 0
                our_winner = c.get("winner")
            else:
                opp_score = int(score_val) if score_val is not None else 0
                opponent_name = team_obj.get("displayName")

        # score string (display as ourScore–oppScore)
        if our_score is None: our_score = 0
        if opp_score is None: opp_score = 0
        score_str = f"{our_score}–{opp_score}"

        result = "Win" if our_winner else "Loss"
        return pd.Series({"result": result, "opponent_name": opponent_name, "score_str": score_str})

    parsed = df_events.apply(parse_row, axis=1)
    df_events = pd.concat([df_events, parsed], axis=1)

    # Add formatted date and option_name (base text, used for other logic if needed)
    df_events['display_date'] = pd.to_datetime(df_events["date"]).dt.strftime("%Y-%m-%d")
    # keep original option_name too if exists, but we'll build a nicer label later
    df_events['option_name'] = df_events.get('option_name', df_events['display_date'] + ": " + df_events.get('name', ''))

    # Ensure id column is string
    df_events['id'] = df_events['id'].astype(str)

    return df_events

def load_game_from_espn(game_id: str):
    url = f"http://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary?event={game_id}"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

def safe_dump(section, name):
    if section is None:
        return f"\n=== {name} (MISSING) ===\n"
    compact = json.dumps(section, separators=(',', ':'))
    return f"\n=== {name} ===\n{compact}\n"

def render_card(title, content, bg_color="#f0f0f0", is_html=False):
    """Render a full-width card. If is_html=True, content is injected as raw HTML."""
    inner = content if is_html else f"<pre style='white-space:pre-wrap'>{st.Markdown(str(content))}</pre>"
    # Note: when is_html is False above we just show content via Markdown fallback; we won't use that path much.
    html = f"""
    <div style="
        background-color: {bg_color};
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    ">
        <h3 style="margin-top:0;">{title}</h3>
        {content}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def split_sections(html_text):
    """Extract sections from model HTML output (<h2>Title</h2> ...)."""
    pattern = r"<h2>(.*?)<\/h2>\s*(.*?)(?=<h2>|$)"
    matches = re.findall(pattern, html_text, flags=re.DOTALL)
    return {title.strip(): body.strip() for title, body in matches}

# -----------------------------
# MAIN APP
# -----------------------------
st.title("College Basketball Summarizer")

# Load teams and present team selector
df_teams = get_teams()
team_names = ["Select a team..."] + df_teams["team.displayName"].sort_values().tolist()
selected_team_name = st.selectbox("Select a team:", team_names, index=0)

# When a team is chosen, load that team's completed games
if selected_team_name != "Select a team...":
    team_id = df_teams.loc[df_teams["team.displayName"] == selected_team_name, "team.id"].iloc[0]

    @st.cache_data
    def load_team_events_cached(team_id):
        return get_games(team_id)

    df_events = load_team_events_cached(team_id)

    if df_events is None or df_events.empty:
        st.warning("No completed games found for this team.")
    else:
        # Build display map: id -> pretty label "YYYY-MM-DD • Opponent • SCORE • 🟢 Win/🔴 Loss"
        display_map = {}
        options = ["none"]  # first option selects nothing
        display_map["none"] = "Select a game..."
        for _, row in df_events.iterrows():
            gid = str(row["id"])
            badge = "🟢 Win" if row["result"] == "Win" else "🔴 Loss"
            label = f"{row['display_date']} • {row['opponent_name']} • {row['score_str']} • {badge}"
            display_map[gid] = label
            options.append(gid)

        # Selectbox uses the gid values, but shows the pretty label via format_func
        selected_game_id = st.selectbox(
            "Select a game:",
            options,
            index=0,
            format_func=lambda gid: display_map.get(gid, "Select a game...")
        )

        if selected_game_id != "none":
            # show chosen label
            st.write(f"Selected: **{display_map[selected_game_id]}**")

            # Generate summary button
            if st.button("Generate Game Summary"):
                # Load game JSON
                data = load_game_from_espn(selected_game_id)

                # Remove text/article fields if present
                for field in ["article", "news", "videos"]:
                    if field in data:
                        del data[field]

                # Build clean JSON text to send to model
                clean_text = ""
                for field in ["header", "boxscore", "leaders", "gameInfo", "plays", "scoring"]:
                    clean_text += safe_dump(data.get(field), field.upper())
                clean_text = re.sub(r'"href"\s*:\s*"[^"]*"\s*(,)?', "", clean_text)

                # Put your exact prompt here — keep the HTML output requirement
                prompt = f"""
                    You are an ex-college basketball coach with over 30 years of experience coaching Division 1 basketball. During your tenure as a coach, you compiled a winning record of 902 wins to 371 losses, won 3 national championships, and went undefeated one year with a perfect 32-0 record and a national championship. 

                                        Below is structured JSON game data (header, boxscore, leaders, plays, etc.) The 'article' field was removed entirely. Ignore all recap text that might have existed originally. Use ONLY the structured stats, box scores, plays, scoring data, and leaders. Ignore any links to external articles. When writing your response, think about how your prior role as a successful college basketball coach would influence your interpretation of the data. 

                                        ------------------- BEGIN GAME DATA ------------------- 
                                        {clean_text} 
                                        ------------------- END GAME DATA --------------------- 

                                        TASK: Analyze {selected_team_name}'s performance and produce the following sections in clean HTML. Only talk about {selected_team_name} even if they lost the game. Center your narrative on {selected_team_name}. 

                                        Each section has explicit content instructions: 

                                        # Game Summary
                                        - A concise narrative of how the game unfolded. 
                                        - Focus on flow, momentum swings, and what decided the game. 
                                        - Call out a specific moment or play which may have shifted the momentum in the game from one team to the other. 
                                        - Who won the jump ball is not important. 
                                        - Utilizing the play-by-play data in <plays>, determine important game trends or shifts in momentum 
                                        - Use ONLY stats and play data; do NOT invent events. 

                                        # The Good 
                                        - Bullet points. 
                                        - Identify players who performed well. 
                                        - Highlight positive trends, efficiency, hustle plays, defensive success, shooting, rebounding, etc. 
                                        - Use stats to justify claims. 

                                        # The Bad 
                                        - Bullet points. 
                                        - Identify players who struggled. 
                                        - Highlight negative trends: turnovers, poor shooting, foul trouble, defensive breakdowns, etc. 
                                        - Use stats to justify claims. 

                                        # The Mixed 
                                        - Bullet points. 
                                        - Call out players or aspects of the team that showed both strengths and weaknesses. 
                                        - Use stats to illustrate nuance. 

                                        # Interesting Stats 
                                        - Bullet points. 
                                        - Pull specific numerical facts from the JSON: shooting splits, runs, rebounding margins, leader stats, etc. 
                                        - Must be factual, drawn directly from the data. 

                                        # Key Players 
                                        - Bullet points. 
                                        - Based on leaders, boxscore performance, and play impact. 
                                        - Should include a brief justification for why each player stands out. 
                                        # Game Notes 
                                        - Bullet points. 
                                        - These are coaching notes: things the team or specific players must improve moving forward. 
                                        - Speak as the head coach: focus on adjustments, discipline, execution, decision-making, effort, etc. 

                                        # RULES: 
                                        - DO NOT invent stats. 
                                        - Do not use any article/recap text. 
                                        - Ground everything in the JSON provided. 
                                        - If data is missing, say so. 

                                        # OUTPUT FORMAT (CRITICAL): 
                                        - Return ONLY valid HTML (no code fences or backticks). 
                                        - Use clean, semantic HTML tags such as: 
                                        - <div>, <p>, <h2>, <h3>, <ul>, <li>, <strong>, <em> 

                                        STRUCTURE REQUIREMENTS: 
                                        - <h2>Game Summary</h2> <div> ... summary content ... </div> 
                                        - <h2>The Good</h2> <ul><li>...</li></ul> 
                                        - <h2>The Mixed</h2> <ul><li>...</li></ul> 
                                        - <h2>The Bad</h2> <ul><li>...</li></ul> 
                                        - <h2>Interesting Stats</h2> <ul><li>...</li></ul> 
                                        - <h2>Key Players</h2> <ul><li>...</li></ul> 
                                        - <h2>Game Notes</h2> <ul><li>...</li></ul> 

                                        IMPORTANT: 
                                        - Bullet lists MUST be valid <ul><li>...</li></ul> lists. 
                                        - No Markdown is allowed. 
                                        - No surrounding code blocks. 
                                        - Do not add CSS or styling; just structural HTML. 
                                        - Do not add extra sections. 
                                        - Keep structure exactly as above. 
                                        - Do not explain your formatting.
"""

                # Spinner
                spinner = st.empty()
                spinner.markdown("""
                <div style="
                    background-color: #d4edda;
                    color: #155724;
                    padding: 12px;
                    border-radius: 8px;
                    font-weight: bold;
                    text-align: center;
                ">
                    Writing your summary, please wait...
                </div>
                """, unsafe_allow_html=True)

                # Call Cohere
                co = Client(API_KEY)
                response = co.chat(
                    model="command-a-03-2025",
                    message=prompt,
                    temperature=0.2,
                    max_tokens=2500
                )

                spinner.empty()

                # Parse returned HTML into sections and render
                sections = split_sections(response.text)

                # Game Summary
                render_card("Game Summary", sections.get("Game Summary", ""), bg_color="#e2e3e5", is_html=True)

                # Tabs for Good / Mixed / Bad (full-width cards inside)
                tabs = st.tabs(["The Good", "The Mixed", "The Bad"])
                with tabs[0]:
                    render_card("The Good", sections.get("The Good", ""), bg_color="#d4edda", is_html=True)
                with tabs[1]:
                    render_card("The Mixed", sections.get("The Mixed", ""), bg_color="#fff3cd", is_html=True)
                with tabs[2]:
                    render_card("The Bad", sections.get("The Bad", ""), bg_color="#f8d7da", is_html=True)

                # Other sections
                render_card("Interesting Stats", sections.get("Interesting Stats", ""), bg_color="#f0f0f0", is_html=True)
                render_card("Key Players", sections.get("Key Players", ""), bg_color="#f0f0f0", is_html=True)
                render_card("Game Notes", sections.get("Game Notes", ""), bg_color="#f0f0f0", is_html=True)

