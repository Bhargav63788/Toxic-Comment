"""
dashboard/app.py

Professional live monitoring dashboard:
  - Left: embedded YouTube video player + title/description
  - Right: live comment feed with Toxic / Non-Toxic / All filter buttons
  - Top: live fairness metrics + per-identity toxicity chart

Run:
  streamlit run dashboard/app.py
"""

import time
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    from googleapiclient.discovery import build
except ImportError:
    build = None

st.set_page_config(page_title="Toxic Comment & Fairness Monitor", layout="wide")

CSV_PATH = "dashboard/live_predictions.csv"
REFRESH_SECONDS = 5

# ---------------- Sidebar controls ----------------
st.sidebar.title("⚙️ Settings")
video_id_input = st.sidebar.text_input("YouTube Video ID", value="")
api_key_input = st.sidebar.text_input("YouTube API Key (for title/description)", value="", type="password")
st.sidebar.caption("Leave API key blank to skip fetching the video title/description.")

st.title("🎯 Real-Time Toxic Comment Detection — Fairness Monitor")


@st.cache_data(ttl=300)
def fetch_video_meta(video_id, api_key):
    if build is None:
        return None
    if not video_id or not api_key:
        return None
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        resp = youtube.videos().list(part="snippet,statistics", id=video_id).execute()
        items = resp.get("items", [])
        if not items:
            return None
        snippet = items[0]["snippet"]
        stats = items[0].get("statistics", {})
        return {
            "title": snippet.get("title"),
            "description": snippet.get("description", ""),
            "channel": snippet.get("channelTitle"),
            "views": stats.get("viewCount"),
            "likes": stats.get("likeCount"),
        }
    except Exception:
        return None


def load_data():
    try:
        df = pd.read_csv(CSV_PATH)
        df["identity_flags"] = df["identity_flags"].fillna("")
        return df
    except FileNotFoundError:
        return pd.DataFrame()


# ---------------- Filter state ----------------
if "filter_mode" not in st.session_state:
    st.session_state.filter_mode = "All"

placeholder = st.empty()

while True:
    df = load_data()

    with placeholder.container():
        if df.empty:
            st.info("Waiting for live data... run realtime_scorer.py in another terminal first.")
            time.sleep(REFRESH_SECONDS)
            continue

        detected_video_id = video_id_input or df["video_id"].iloc[-1]

        left_col, right_col = st.columns([1, 1.1])

        # ---------------- LEFT: video + description ----------------
        with left_col:
            st.video(f"https://www.youtube.com/watch?v={detected_video_id}")

            meta = fetch_video_meta(detected_video_id, api_key_input)
            if meta:
                st.subheader(meta["title"])
                st.caption(f"Channel: {meta['channel']}  |  Views: {meta.get('views', 'N/A')}  |  Likes: {meta.get('likes', 'N/A')}")
                with st.expander("Description"):
                    st.write(meta["description"])
            else:
                st.caption(f"Video ID: {detected_video_id}  (add an API key in the sidebar to show title/description)")

            st.markdown("---")
            st.subheader("📊 Live Fairness Summary")

            c1, c2, c3 = st.columns(3)
            c1.metric("Total scored", len(df))
            c2.metric("Toxicity rate", f"{df['is_toxic'].mean()*100:.1f}%")
            c3.metric("Identity mentions", int((df["identity_flags"] != "").sum()))

            exploded = df.assign(identity_flags=df["identity_flags"].str.split(",")).explode("identity_flags")
            exploded = exploded[exploded["identity_flags"] != ""]

            if not exploded.empty:
                group_rates = exploded.groupby("identity_flags")["is_toxic"].mean().sort_values(ascending=False)
                overall_rate = df["is_toxic"].mean()

                fig, ax = plt.subplots(figsize=(6, 3))
                ax.bar(group_rates.index, group_rates.values, color="#e15759")
                ax.axhline(overall_rate, color="black", linestyle="--", linewidth=1, label="Overall rate")
                ax.set_ylabel("Toxicity rate")
                plt.xticks(rotation=30, ha="right", fontsize=8)
                ax.legend(fontsize=8)
                st.pyplot(fig)
            else:
                st.caption("No identity-tagged comments yet.")

        # ---------------- RIGHT: comment feed with filter buttons ----------------
        with right_col:
            st.subheader("💬 Live Comments")

            b1, b2, b3 = st.columns(3)
            if b1.button("All", use_container_width=True):
                st.session_state.filter_mode = "All"
            if b2.button("🔴 Toxic only", use_container_width=True):
                st.session_state.filter_mode = "Toxic"
            if b3.button("🟢 Clean only", use_container_width=True):
                st.session_state.filter_mode = "Clean"

            st.caption(f"Showing: **{st.session_state.filter_mode}**")

            view_df = df.sort_values("scored_at", ascending=False)
            if st.session_state.filter_mode == "Toxic":
                view_df = view_df[view_df["is_toxic"] == True]
            elif st.session_state.filter_mode == "Clean":
                view_df = view_df[view_df["is_toxic"] == False]

            st.caption(f"{len(view_df)} comments")

            for _, row in view_df.head(40).iterrows():
                is_toxic = row["is_toxic"]
                border_color = "#e15759" if is_toxic else "#59a14f"
                label = "🔴 TOXIC" if is_toxic else "🟢 CLEAN"
                identity_badge = f"  ·  🏷 {row['identity_flags']}" if row["identity_flags"] else ""

                st.markdown(
                    f"""
                    <div style="border-left: 4px solid {border_color}; padding: 8px 12px; margin-bottom: 8px; background-color: rgba(128,128,128,0.05); border-radius: 4px;">
                        <b>{row['author']}</b>
                        <span style="float:right; font-size:0.8em; color:{border_color};">{label} ({row['toxicity_prob']:.2f})</span>
                        <br>{row['text']}
                        <br><span style="font-size:0.75em; color:gray;">{row['scored_at']}{identity_badge}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    time.sleep(REFRESH_SECONDS)