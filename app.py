import streamlit as st
import pandas as pd
import plotly.express as px
from scipy.stats import poisson
import google.generativeai as genai
import json

# KLÍČ PRO AI 
API_KEY = "AIzaSyC-N4wVuOLei2Agn6J1z4kTNKqs_m9jR1w"

# KONFIGURACE STRÁNKY 
st.set_page_config(
    page_title="Football Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; }
    
    .css-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #e0e0e0;
    }
    
    h1 { color: #1b5e20; font-family: 'Arial Black', sans-serif; text-shadow: 1px 1px 2px rgba(0,0,0,0.1); }
    h2, h3, h4 { color: #2c3e50; font-weight: 600; }
    
    div.stButton > button {
        background: linear-gradient(to right, #1b5e20, #2e7d32);
        color: white; font-weight: bold; border: none; padding: 12px 24px;
        border-radius: 8px; width: 100%; transition: all 0.3s; box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 8px rgba(0,0,0,0.25); }
    /* Přesné skóre - zelený */
    .score-box {
        background-color: white; border: 2px solid #4caf50; border-radius: 10px;
        padding: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: transform 0.2s;
    }
    .score-box:hover { transform: scale(1.05); border-color: #1b5e20; }
    .score-val { font-size: 24px; font-weight: bold; color: #2c3e50; }
    .prob-val { font-size: 16px; font-weight: bold; color: #1b5e20; }
    /* modrý */
    .score-box-3y {
        background-color: white; border: 2px solid #1565c0; border-radius: 10px;
        padding: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: transform 0.2s;
    }
    .score-box-3y:hover { transform: scale(1.05); border-color: #0d47a1; }
    .prob-val-3y { font-size: 16px; font-weight: bold; color: #1565c0; }
    /* ai - růžový */
    .ai-score-box {
        background-color: white; border: 2px solid #e91e63; border-radius: 10px;
        padding: 10px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); transition: transform 0.2s;
        margin-top: 15px;
    }
    .ai-score-box:hover { transform: scale(1.05); border-color: #880e4f; }
    .ai-prob-val { font-size: 16px; font-weight: bold; color: #e91e63; }
    
    .ai-analysis-card {
        background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        border-left: 5px solid #e91e63;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>⚽ FOOTBALL PREDICTOR ⚽</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #555; font-size: 18px;'>Pokročilá analýza a predikce výsledků</p>", unsafe_allow_html=True)
st.divider()

# VÝBĚR LIGY
st.sidebar.header("🏆 Nastavení Soutěže")
league_choice = st.sidebar.selectbox("Vyber ligu:", ("🇨🇿 Česk Liga (Chance Liga)", "🇬🇧 Premier League", "🇪🇸 La Liga"))
files = {"🇨🇿 Česk Liga (Chance Liga)": "data.csv", "🇬🇧 Premier League": "premier_league.csv", "🇪🇸 La Liga": "laliga.csv"}
selected_file = files[league_choice]
# načítání csv
@st.cache_data
def load_data(filename):
    try:
        df = pd.read_csv(filename, sep=None, engine='python', encoding='utf-8-sig')
        df.columns = df.columns.str.strip()
        if 'Goal_Scored' in df.columns:
            df.rename(columns={'Goal_Scored': 'Goals_Scored'}, inplace=True)
        # Převod na čísla
        df[['Matches', 'Goals_Scored', 'Goals_Conceded']] = df[['Matches', 'Goals_Scored', 'Goals_Conceded']].apply(pd.to_numeric, errors='coerce')
        return df
    except UnicodeDecodeError:
        # Pokud utf-8 selže, zkusí cp1250
        try:
            df = pd.read_csv(filename, sep=None, engine='python', encoding='cp1250')
            df.columns = df.columns.str.strip()
            if 'Goal_Scored' in df.columns:
                df.rename(columns={'Goal_Scored': 'Goals_Scored'}, inplace=True)
            df[['Matches', 'Goals_Scored', 'Goals_Conceded']] = df[['Matches', 'Goals_Scored', 'Goals_Conceded']].apply(pd.to_numeric, errors='coerce')
            return df
        except Exception as e:
            st.error(f"Chyba načítání: {e}")
            return None
    except Exception as e:
        st.error(f"Chyba načítání: {e}")
        return None

df = load_data(selected_file)

if df is None:
    st.error(f"❌ Chyba: Soubor **{selected_file}** nebyl nalezen!")
    st.stop()

# Výpočet ligového průměru gólů a síly útoku a obrany
try:
    league_avg_goals = df['Goals_Scored'].mean() / df['Matches'].mean()
    df['Attack'] = (df['Goals_Scored'] / df['Matches']) / league_avg_goals
    df['Defense'] = (df['Goals_Conceded'] / df['Matches']) / league_avg_goals
except Exception as e:
    st.error(f"Chyba ve výpočtech statistik: {e}")
    st.stop()

# Výběr týmů
st.sidebar.markdown("---")
st.sidebar.header("⚔️ Výběr Týmů")
teams = sorted(df['Team'].unique())
home_team = st.sidebar.selectbox("Domácí tým", teams, index=0)
away_team = st.sidebar.selectbox("Hostující tým", teams, index=1 if len(teams) > 1 else 0)
home_advantage = st.sidebar.slider("Výhoda domácího hřiště", 1.0, 1.3, 1.15, 0.01)

# Pomocná funkce pro poisson
def compute_poisson(lambda_h, lambda_a):
    max_goals = 8
    prob_h = [poisson.pmf(i, lambda_h) for i in range(max_goals)]
    prob_a = [poisson.pmf(i, lambda_a) for i in range(max_goals)]
    win_h, draw, win_a = 0, 0, 0
    scores = []
    for h in range(max_goals):
        for a in range(max_goals):
            p = prob_h[h] * prob_a[a]
            scores.append((f"{h}:{a}", p))
            if h > a:
                win_h += p
            elif h == a:
                draw += p
            else:
                win_a += p
    scores.sort(key=lambda x: x[1], reverse=True)
    return win_h, draw, win_a, scores[:5]

# Pomocná funkce pro zobrazení sekce
def render_prediction_section(title, home_team, away_team, lambda_home, lambda_away, score_box_class, prob_val_class):
    st.markdown(f"## {title}")

    win_home, draw, win_away, top5 = compute_poisson(lambda_home, lambda_away)

    col1, col2 = st.columns([1, 1.6])

    with col1:
        # Očekávané góly
        st.markdown(f"""
        <div class="css-card">
            <h4 style="margin-top:0; text-align: center;">📊 Očekávané góly (xG)</h4>
            <hr style="margin: 10px 0; border: 0; border-top: 1px solid #eee;">
            <div style="display:flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-weight:bold; color:#333; font-size: 14px;">🏠 {home_team}</span>
                <span style="background-color:#e8f5e9; padding: 5px 10px; border-radius:5px; color:#1b5e20; font-weight:bold;">{lambda_home:.2f}</span>
            </div>
            <div style="display:flex; justify-content: space-between; align-items: center;">
                <span style="font-weight:bold; color:#333; font-size: 14px;">✈️ {away_team}</span>
                <span style="background-color:#ffebee; padding: 5px 10px; border-radius:5px; color:#c62828; font-weight:bold;">{lambda_away:.2f}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        # Tip algoritmu
        diff = lambda_home - lambda_away
        if diff > 0.4:
            tip_text = f"Vítěz: {home_team}"
            tip_color = "#d4edda"
            text_color = "#155724"
        elif diff < -0.4:
            tip_text = f"Vítěz: {away_team}"
            tip_color = "#f8d7da"
            text_color = "#721c24"
        else:
            tip_text = "Remíza / Vyrovnaný zápas"
            tip_color = "#fff3cd"
            text_color = "#856404"

        st.markdown(f"""
        <div class="css-card" style="background-color: {tip_color}; border-color: {tip_color}; padding: 30px 20px;">
            <h4 style="margin:0; text-align: center; color: {text_color};">🎯 Tip Algoritmu</h4>
            <p style="text-align: center; font-size: 18px; font-weight: bold; margin: 15px 0 0 0; color: {text_color};">{tip_text}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Graf
        st.markdown('<h4 style="text-align: center; margin-bottom: 0px; color: #2c3e50;">📈 Pravděpodobnost výsledku</h4>', unsafe_allow_html=True)
        labels = [f"Výhra {home_team}", "Remíza", f"Výhra {away_team}"]
        values = [win_home, draw, win_away]
        colors = ['#4caf50', '#9e9e9e', '#ef5350']

        fig = px.pie(names=labels, values=values, hole=0.6, color_discrete_sequence=colors)
        fig.update_traces(
            textinfo='percent+label',
            textfont_size=13,
            textposition='outside',
            hoverinfo='none',
            hovertemplate=None
        )
        fig.update_layout(
            showlegend=False,
            margin=dict(t=30, b=30, l=150, r=150),
            height=320,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🎯 Nejpravděpodobnější přesná skóre")
    cols = st.columns(5)
    for i, (score, prob) in enumerate(top5):
        with cols[i]:
            st.markdown(f'<div class="{score_box_class}"><div class="score-val">{score}</div><div class="{prob_val_class}">{prob*100:.1f} %</div></div>', unsafe_allow_html=True)

# Hlavní část
if st.sidebar.button("🚀 ANALYZOVAT ZÁPAS"):
    if home_team == away_team:
        st.warning("⚠️ Tým nemůže hrát sám se sebou!")
    else:
        # Aktuální
        # Načtení statistik a výpočet lambdy
        home_stats = df[df['Team'] == home_team].iloc[0]
        away_stats = df[df['Team'] == away_team].iloc[0]
        # Očekávaný počet gólů
        lambda_home = home_stats['Attack'] * away_stats['Defense'] * league_avg_goals * home_advantage
        lambda_away = away_stats['Attack'] * home_stats['Defense'] * league_avg_goals

        render_prediction_section(
            "Predikce na základě aktuální sezóny (Poisson)",
            home_team, away_team, lambda_home, lambda_away,
            "score-box", "prob-val"
        )

        # Dlouhodobá
        files_3y = {
            "🇨🇿 Česk Liga (Chance Liga)": "data3.csv",
            "🇬🇧 Premier League": "premier_league3.csv",
            "🇪🇸 La Liga": "laliga3.csv"
        }
        df_3y = load_data(files_3y[league_choice])
        if df_3y is not None:
            try:
                league_avg_3y = df_3y['Goals_Scored'].mean() / df_3y['Matches'].mean()
                h_stats_3y = df_3y[df_3y['Team'] == home_team].iloc[0]
                a_stats_3y = df_3y[df_3y['Team'] == away_team].iloc[0]

                l_home_3y = (h_stats_3y['Goals_Scored'] / h_stats_3y['Matches'] / league_avg_3y) * \
                            (a_stats_3y['Goals_Conceded'] / a_stats_3y['Matches'] / league_avg_3y) * \
                            league_avg_3y * home_advantage

                l_away_3y = (a_stats_3y['Goals_Scored'] / a_stats_3y['Matches'] / league_avg_3y) * \
                            (h_stats_3y['Goals_Conceded'] / h_stats_3y['Matches'] / league_avg_3y) * \
                            league_avg_3y

                st.markdown("<br>", unsafe_allow_html=True)
                render_prediction_section(
                    "Predikce na základě dat za poslední 3 roky (Poissonův model)",
                    home_team, away_team, l_home_3y, l_away_3y,
                    "score-box-3y", "prob-val-3y"
                )

            except Exception as e:
                st.warning(f"Nepodařilo se vypočítat dlouhodobou predikci: {e}")
        else:
            st.warning("Soubory pro 3letou historii nejsou nahrány.")

        # AI
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("## Expertní AI Predikce (Gemini)")
        
        if API_KEY != "ZDE_VLOZ_SVUJ_KLIC" and API_KEY.strip() != "":
            with st.spinner("AI studuje týmy, vymýšlí skóre a sepisuje analýzu..."):
                try:
                    genai.configure(api_key=API_KEY)
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    # Prompt
                    ai_prompt = f"""
                        Jsi zkušený fotbalový analytik s mnohaletou praxí. Analyzuj nadcházející zápas: {home_team} (domácí) vs {away_team} (hosté).

                        Při analýze zohledni:
                        - Historickou výkonnost obou týmů
                        - Výhodu domácího prostředí
                        - Typický herní styl každého týmu

                        DŮLEŽITÉ POKYNY:
                        - Odpověz VÝHRADNĚ ve formátu JSON níže, bez jakéhokoliv textu před ani po něm
                        - Všechna procenta zaokrouhli na jedno desetinné místo
                        - Součet prob_home + prob_draw + prob_away musí být přesně 100
                        - Analýza musí být konkrétní a věcná, ne obecná

                        ODPOVĚZ POUZE V TOMTO JSON FORMÁTU:
                    {{
                        "prob_home": <číslo_procento_výhry_domácích>,
                        "prob_draw": <číslo_procento_remízy>,
                        "prob_away": <číslo_procento_výhry_hostů>,
                        "tip_text": "<text: napiš buď 'Vítěz: {home_team}', 'Vítěz: {away_team}' nebo 'Remíza / Vyrovnaný zápas'>",
                        "top_scores": [
                            {{"score": "<skóre_1>", "prob": <procento_1>}},
                            {{"score": "<skóre_2>", "prob": <procento_2>}},
                            {{"score": "<skóre_3>", "prob": <procento_3>}},
                            {{"score": "<skóre_4>", "prob": <procento_4>}},
                            {{"score": "<skóre_5>", "prob": <procento_5>}}
                        ],
                        "analysis": "<Stručná, odborná analýza zápasu. Proč to tak dopadne? (Max 4 věty)>"
                    }}
                    """
                    # Odeslání požadavku AI a zpracování odpovědi 
                    response = model.generate_content(ai_prompt)
                    # Odstranění nepotřebných AI znaků
                    clean_response = response.text.replace('```json', '').replace('```', '').strip()
                    # Převod JSON na python 
                    ai_data = json.loads(clean_response)

                    col_ai1, col_ai2 = st.columns([1, 1.6])
                    
                    with col_ai1:
                        # Slovní analýza od AI
                        st.markdown(f"""
                        <div class="ai-analysis-card">
                            <h4 style="margin-top: 0; margin-bottom: 10px; color: #e91e63;">🧠 Slovní analýza (Gemini)</h4>
                            <p style="font-size: 15px; line-height: 1.5; color: #333; margin-bottom: 0;">{ai_data['analysis']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        if "Remíza" in ai_data['tip_text']:
                            ai_tip_color = "#fff3cd"
                            ai_text_color = "#856404"
                        elif home_team in ai_data['tip_text']:
                            ai_tip_color = "#d4edda"
                            ai_text_color = "#155724"
                        else:
                            ai_tip_color = "#f8d7da"
                            ai_text_color = "#721c24"

                        st.markdown(f"""
                        <div class="css-card" style="background-color: {ai_tip_color}; border-color: {ai_tip_color}; padding: 30px 20px;">
                            <h4 style="margin:0; text-align: center; color: {ai_text_color};">💡 Expertní Tip AI</h4>
                            <p style="text-align: center; font-size: 18px; font-weight: bold; margin: 15px 0 0 0; color: {ai_text_color};">{ai_data['tip_text']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_ai2:
                        # Graf AI
                        st.markdown('<h4 style="text-align: center; margin-bottom: 0px; color: #2c3e50;">📈 Pravděpodobnost podle AI</h4>', unsafe_allow_html=True)
                        ai_labels = [f"Výhra {home_team}", "Remíza", f"Výhra {away_team}"]
                        ai_values = [ai_data['prob_home'], ai_data['prob_draw'], ai_data['prob_away']]
                        ai_colors = ['#4caf50', '#9e9e9e', '#ef5350']
                        
                        fig_ai = px.pie(names=ai_labels, values=ai_values, hole=0.6, color_discrete_sequence=ai_colors)
                        fig_ai.update_traces(
                            textinfo='percent+label',
                            textfont_size=13,
                            textposition='outside',
                            hoverinfo='none',
                            hovertemplate=None
                        )
                        fig_ai.update_layout(
                            showlegend=False,
                            margin=dict(t=30, b=30, l=150, r=150),
                            height=320,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        st.plotly_chart(fig_ai, use_container_width=True)
                        
                    st.markdown("#### 🎯 Nejpravděpodobnější přesná skóre (Gemini AI)")
                    cols_ai = st.columns(5)
                    for i, score_data in enumerate(ai_data['top_scores']):
                        with cols_ai[i]:
                            st.markdown(f"""
                            <div class="ai-score-box">
                                <div class="score-val">{score_data['score']}</div>
                                <div class="ai-prob-val">{score_data['prob']:.1f} %</div>
                            </div>
                            """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Při zpracování AI dat došlo k chybě. Detail chyby: {e}")
        else:
            st.warning("⚠️ Pro zobrazení AI predikce musíš vložit svůj API klíč přímo do kódu.")

else:
    st.info("👈 Vyber si ligu a týmy v levém menu a klikni na 'ANALYZOVAT ZÁPAS'.")