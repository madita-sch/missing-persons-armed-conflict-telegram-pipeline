from dash import Dash, html, dcc, Input, Output, State, dash_table, callback_context, no_update
import psycopg2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─────────────────────────────────────────────
# DB CONNECTION
# ─────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(
        dbname="missing_db",
        user="postgres",
        password="Masch464hhu!?",
        host="localhost",
        port="5432"
    )

# ─────────────────────────────────────────────
# DATA FUNCTIONS
# ─────────────────────────────────────────────
def search_cases(q):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT case_id, name_ar, name_en, location_ar, location_en, age
        FROM cases
        WHERE name_ar ILIKE %s OR name_en ILIKE %s
    """, (f"%{q}%", f"%{q}%"))
    rows = cur.fetchall()
    cols = ["case_id", "name_ar", "name_en", "location_ar", "location_en", "age"]
    cur.close()
    conn.close()
    return pd.DataFrame(rows, columns=cols)


def get_case(case_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM cases WHERE case_id=%s", (case_id,))
    case = cur.fetchone()
    cur.execute("""
        SELECT message_id, posted_at, text_clean, text_raw, views, forwards, reactions
        FROM messages
        WHERE case_id=%s ORDER BY posted_at
    """, (case_id,))
    messages = cur.fetchall()
    cur.close()
    conn.close()
    return case, messages


def get_kpis():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM cases")
    total_cases = cur.fetchone()[0]
    cur.execute("""
        SELECT COUNT(*) FROM cases
        WHERE location_en IS NULL OR location_en = ''
           OR location_ar IS NULL OR location_ar = ''
    """)
    missing_locations = cur.fetchone()[0]
    cur.execute("""
        SELECT COUNT(DISTINCT c.case_id)
        FROM cases c
        JOIN messages m ON c.case_id = m.case_id
        WHERE m.posted_at >= NOW() - INTERVAL '30 days'
    """)
    active_cases = cur.fetchone()[0]
    cur.close()
    conn.close()
    return total_cases, active_cases, missing_locations


def get_analytics_data():
    conn = get_conn()
    cur = conn.cursor()

    # Cases by location
    cur.execute("""
        SELECT COALESCE(location_en, 'Unknown') AS loc, COUNT(*) AS cnt
        FROM cases
        GROUP BY loc
        ORDER BY cnt DESC
        LIMIT 10
    """)
    loc_rows = cur.fetchall()

    # Cases by age group
    cur.execute("""
        SELECT
            CASE
                WHEN age < 18 THEN 'Under 18'
                WHEN age BETWEEN 18 AND 30 THEN '18–30'
                WHEN age BETWEEN 31 AND 50 THEN '31–50'
                ELSE '50+'
            END AS age_group,
            COUNT(*) AS cnt
        FROM cases
        WHERE age IS NOT NULL
        GROUP BY age_group
    """)
    age_rows = cur.fetchall()

    # Messages over time
    cur.execute("""
        SELECT DATE_TRUNC('week', posted_at) AS week, COUNT(*) AS cnt
        FROM messages
        GROUP BY week
        ORDER BY week
    """)
    msg_rows = cur.fetchall()

    cur.close()
    conn.close()
    return loc_rows, age_rows, msg_rows


# ─────────────────────────────────────────────
# STYLING CONSTANTS
# ─────────────────────────────────────────────
SIDEBAR_BG   = "#0F172A"
ACCENT       = "#3B82F6"
ACCENT_GREEN = "#10B981"
ACCENT_RED   = "#EF4444"
ACCENT_AMB   = "#F59E0B"
PAGE_BG      = "#F1F5F9"
CARD_BG      = "#FFFFFF"
TEXT_DARK    = "#1E293B"
TEXT_MED     = "#64748B"

FONT = "'IBM Plex Sans', 'Helvetica Neue', sans-serif"
GOOGLE_FONT = html.Link(
    rel="stylesheet",
    href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap"
)

def card(children, style=None):
    base = {
        "backgroundColor": CARD_BG,
        "borderRadius": "12px",
        "padding": "24px",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
        "fontFamily": FONT,
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)


def kpi_card(title, value, color, icon):
    return html.Div([
        html.Div([
            html.Span(icon, style={"fontSize": "28px"}),
            html.Div([
                html.Div(str(value), style={
                    "fontSize": "32px", "fontWeight": "700",
                    "color": TEXT_DARK, "lineHeight": "1"
                }),
                html.Div(title, style={
                    "fontSize": "13px", "color": TEXT_MED,
                    "marginTop": "4px", "fontWeight": "500"
                }),
            ])
        ], style={"display": "flex", "alignItems": "center", "gap": "16px"})
    ], style={
        "backgroundColor": CARD_BG,
        "borderRadius": "12px",
        "padding": "20px 24px",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
        "borderTop": f"4px solid {color}",
        "flex": "1",
        "fontFamily": FONT,
    })


# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
NAV_ITEM_STYLE = {
    "padding": "12px 16px",
    "cursor": "pointer",
    "borderRadius": "8px",
    "color": "#94A3B8",
    "fontSize": "14px",
    "fontWeight": "500",
    "display": "flex",
    "alignItems": "center",
    "gap": "10px",
    "transition": "all 0.15s",
    "fontFamily": FONT,
}

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "Missing Persons Dashboard"

app.layout = html.Div([

    GOOGLE_FONT,

    # ── SIDEBAR ──────────────────────────────────
    html.Div([
        html.Div([
            html.Div("🔍", style={"fontSize": "24px"}),
            html.Div([
                html.Div("MISSING", style={
                    "fontSize": "14px", "fontWeight": "700",
                    "color": "white", "letterSpacing": "3px"
                }),
                html.Div("PERSONS DB", style={
                    "fontSize": "10px", "color": "#475569",
                    "letterSpacing": "2px", "marginTop": "1px"
                }),
            ])
        ], style={"display": "flex", "alignItems": "center",
                  "gap": "12px", "marginBottom": "36px"}),

        html.Div([
            html.Div(id="nav-cases-btn", children=["📁  Cases"],
                     n_clicks=0, style=NAV_ITEM_STYLE),
            html.Div(id="nav-analytics-btn", children=["📊  Analytics"],
                     n_clicks=0, style=NAV_ITEM_STYLE),
        ]),

        # Version stamp at bottom
        html.Div("v1.0", style={
            "position": "absolute", "bottom": "24px", "left": "20px",
            "color": "#334155", "fontSize": "11px", "fontFamily": FONT
        })
    ], style={
        "width": "220px",
        "minHeight": "100vh",
        "position": "fixed",
        "top": 0, "left": 0,
        "backgroundColor": SIDEBAR_BG,
        "padding": "28px 20px",
        "boxSizing": "border-box",
        "zIndex": "100",
    }),

    # ── MAIN AREA ────────────────────────────────
    html.Div([
        html.Div(id="page-content")
    ], style={
        "marginLeft": "220px",
        "minHeight": "100vh",
        "backgroundColor": PAGE_BG,
        "padding": "32px 36px",
        "boxSizing": "border-box",
        "fontFamily": FONT,
    }),

    # ── HIDDEN STORES ────────────────────────────
    # These always-present components prevent "ID not found" errors
    dcc.Store(id="active-page", data="cases"),
    dcc.Store(id="selected-case-id", data=None),

], style={"margin": "0", "padding": "0", "fontFamily": FONT})


# ─────────────────────────────────────────────
# NAV → PAGE ROUTING
# ─────────────────────────────────────────────
@app.callback(
    Output("active-page", "data"),
    Input("nav-cases-btn", "n_clicks"),
    Input("nav-analytics-btn", "n_clicks"),
    prevent_initial_call=True
)
def update_active_page(cases_clicks, analytics_clicks):
    triggered = callback_context.triggered[0]["prop_id"]
    if "nav-cases-btn" in triggered:
        return "cases"
    if "nav-analytics-btn" in triggered:
        return "analytics"
    return no_update


@app.callback(
    Output("page-content", "children"),
    Input("active-page", "data")
)
def render_page(page):

    # ── CASES PAGE ───────────────────────────────
    if page == "cases":
        try:
            total, active, missing = get_kpis()
        except Exception:
            total, active, missing = "—", "—", "—"

        return html.Div([

            # Header
            html.Div([
                html.H1("Cases", style={
                    "margin": "0", "fontSize": "26px",
                    "fontWeight": "700", "color": TEXT_DARK
                }),
                html.P("Search and explore missing persons cases",
                       style={"margin": "4px 0 0", "color": TEXT_MED, "fontSize": "14px"})
            ], style={"marginBottom": "28px"}),

            # KPI Row
            html.Div([
                kpi_card("Total Cases",         total,   ACCENT,       "📁"),
                kpi_card("Active (30 days)",    active,  ACCENT_GREEN, "🟢"),
                kpi_card("Missing Locations",   missing, ACCENT_RED,   "📍"),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "28px"}),

            # Search bar
            card([
                html.Div("Search Cases", style={
                    "fontSize": "15px", "fontWeight": "600",
                    "color": TEXT_DARK, "marginBottom": "16px"
                }),
                html.Div([
                    dcc.Input(
                        id="search-input",
                        type="text",
                        placeholder="Search by name (Arabic or English)…",
                        debounce=False,
                        style={
                            "flex": "1",
                            "padding": "10px 14px",
                            "borderRadius": "8px",
                            "border": "1px solid #E2E8F0",
                            "fontSize": "14px",
                            "outline": "none",
                            "fontFamily": FONT,
                        }
                    ),
                    html.Button("Search", id="search-btn", n_clicks=0, style={
                        "padding": "10px 24px",
                        "backgroundColor": ACCENT,
                        "color": "white",
                        "border": "none",
                        "borderRadius": "8px",
                        "cursor": "pointer",
                        "fontSize": "14px",
                        "fontWeight": "600",
                        "fontFamily": FONT,
                    })
                ], style={"display": "flex", "gap": "12px", "alignItems": "center"}),

                # Results always in layout
                html.Div(id="search-output", style={"marginTop": "20px"}),

            ], style={"marginBottom": "24px"}),

            # Case detail always in layout
            html.Div(id="case-detail"),

        ])

    # ── ANALYTICS PAGE ───────────────────────────
    if page == "analytics":
        try:
            loc_rows, age_rows, msg_rows = get_analytics_data()
        except Exception:
            loc_rows, age_rows, msg_rows = [], [], []

        # Location bar chart
        if loc_rows:
            df_loc = pd.DataFrame(loc_rows, columns=["Location", "Cases"])
            fig_loc = px.bar(
                df_loc, x="Cases", y="Location", orientation="h",
                color="Cases", color_continuous_scale=["#BFDBFE", "#1D4ED8"],
                title="Top 10 Locations"
            )
            fig_loc.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                title_font_size=15, showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis=dict(autorange="reversed")
            )
            loc_chart = dcc.Graph(figure=fig_loc, config={"displayModeBar": False})
        else:
            loc_chart = html.Div("No location data", style={"color": TEXT_MED})

        # Age donut chart
        if age_rows:
            df_age = pd.DataFrame(age_rows, columns=["Age Group", "Count"])
            fig_age = px.pie(
                df_age, names="Age Group", values="Count",
                hole=0.55, title="Age Distribution",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            fig_age.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                title_font_size=15,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            age_chart = dcc.Graph(figure=fig_age, config={"displayModeBar": False})
        else:
            age_chart = html.Div("No age data", style={"color": TEXT_MED})

        # Messages timeline
        if msg_rows:
            df_msg = pd.DataFrame(msg_rows, columns=["Week", "Messages"])
            df_msg["Week"] = pd.to_datetime(df_msg["Week"])
            fig_msg = px.area(
                df_msg, x="Week", y="Messages",
                title="Message Activity Over Time",
                color_discrete_sequence=[ACCENT]
            )
            fig_msg.update_traces(fillcolor="rgba(59,130,246,0.12)")
            fig_msg.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                title_font_size=15,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            msg_chart = dcc.Graph(figure=fig_msg, config={"displayModeBar": False})
        else:
            msg_chart = html.Div("No message data", style={"color": TEXT_MED})

        return html.Div([

            html.Div([
                html.H1("Analytics", style={
                    "margin": "0", "fontSize": "26px",
                    "fontWeight": "700", "color": TEXT_DARK
                }),
                html.P("Trends and breakdowns across all cases",
                       style={"margin": "4px 0 0", "color": TEXT_MED, "fontSize": "14px"})
            ], style={"marginBottom": "28px"}),

            # Top row: 2 charts side by side
            html.Div([
                card([loc_chart], style={"flex": "1"}),
                card([age_chart], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),

            # Full-width timeline
            card([msg_chart]),
        ])

    # Default: load cases
    return html.Div("Select a section from the sidebar.")


# ─────────────────────────────────────────────
# SEARCH CALLBACK
# ─────────────────────────────────────────────
@app.callback(
    Output("search-output", "children"),
    Input("search-btn", "n_clicks"),
    State("search-input", "value"),
    prevent_initial_call=True
)
def run_search(n_clicks, query):
    if not query or not query.strip():
        return html.Div("⚠️ Please enter a search term.",
                        style={"color": ACCENT_AMB, "fontSize": "14px"})
    try:
        df = search_cases(query.strip())
    except Exception as e:
        return html.Div(f"❌ Database error: {e}",
                        style={"color": ACCENT_RED, "fontSize": "14px"})

    if df.empty:
        return html.Div("No cases found for that query.",
                        style={"color": TEXT_MED, "fontSize": "14px"})

    return html.Div([
        html.Div(f"Found {len(df)} case(s)", style={
            "fontSize": "13px", "fontWeight": "600",
            "color": TEXT_MED, "marginBottom": "12px"
        }),
        dash_table.DataTable(
            id="results-table",
            data=df.to_dict("records"),
            columns=[{"name": c.replace("_", " ").title(), "id": c} for c in df.columns],
            page_size=8,
            row_selectable="single",
            selected_rows=[],
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#F8FAFC",
                "fontWeight": "600",
                "fontSize": "12px",
                "color": TEXT_MED,
                "border": "none",
                "borderBottom": "2px solid #E2E8F0",
                "fontFamily": FONT,
            },
            style_cell={
                "fontSize": "13px",
                "padding": "10px 14px",
                "border": "none",
                "borderBottom": "1px solid #F1F5F9",
                "color": TEXT_DARK,
                "fontFamily": FONT,
            },
            style_data_conditional=[{
                "if": {"state": "selected"},
                "backgroundColor": "#EFF6FF",
                "border": "none",
            }],
        )
    ])


# ─────────────────────────────────────────────
# CASE DETAIL CALLBACK
# ─────────────────────────────────────────────
@app.callback(
    Output("case-detail", "children"),
    Input("results-table", "selected_rows"),
    State("results-table", "data"),
    prevent_initial_call=True
)
def show_case(selected_rows, table_data):
    if not selected_rows or table_data is None:
        return ""

    case_id = table_data[selected_rows[0]]["case_id"]

    try:
        case, messages = get_case(case_id)
    except Exception as e:
        return html.Div(f"❌ Error loading case: {e}",
                        style={"color": ACCENT_RED, "marginTop": "16px"})

    if not case:
        return html.Div("Case not found.", style={"color": TEXT_MED})

    # Safe field access
    def safe(val):
        return val if val else "—"

    info_rows = [
        ("Case ID",   safe(case[0])),
        ("Name (AR)", safe(case[1])),
        ("Name (EN)", safe(case[2])),
        ("Location",  safe(case[3])),
        ("Age",       safe(case[7])),
    ]

    profile = card([
        html.Div("📁 Case Profile", style={
            "fontSize": "15px", "fontWeight": "700",
            "color": TEXT_DARK, "marginBottom": "16px"
        }),
        html.Div([
            html.Div([
                html.Div(label, style={
                    "fontSize": "12px", "fontWeight": "600",
                    "color": TEXT_MED, "textTransform": "uppercase",
                    "letterSpacing": "0.5px", "marginBottom": "2px"
                }),
                html.Div(value, style={
                    "fontSize": "15px", "color": TEXT_DARK, "fontWeight": "500"
                }),
            ], style={"marginBottom": "14px"})
            for label, value in info_rows
        ])
    ], style={"marginTop": "24px", "marginBottom": "16px"})

    # Messages chart
    if messages:
        df_m = pd.DataFrame(messages, columns=[
            "message_id", "posted_at", "text_clean", "text_raw",
            "views", "forwards", "reactions"
        ])
        df_m["posted_at"] = pd.to_datetime(df_m["posted_at"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_m["posted_at"], y=df_m["views"],
            mode="lines+markers", name="Views",
            line=dict(color=ACCENT, width=2),
            marker=dict(size=5)
        ))
        fig.add_trace(go.Scatter(
            x=df_m["posted_at"], y=df_m["forwards"],
            mode="lines+markers", name="Forwards",
            line=dict(color=ACCENT_GREEN, width=2),
            marker=dict(size=5)
        ))
        fig.update_layout(
            title="Message Engagement Over Time",
            plot_bgcolor="white", paper_bgcolor="white",
            title_font_size=15,
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(gridcolor="#F1F5F9"),
            yaxis=dict(gridcolor="#F1F5F9"),
        )
        msg_section = card([
            html.Div(f"📊 Messages ({len(df_m)} total)", style={
                "fontSize": "15px", "fontWeight": "700",
                "color": TEXT_DARK, "marginBottom": "12px"
            }),
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
        ], style={"marginBottom": "16px"})
    else:
        msg_section = card([
            html.Div("No messages for this case.",
                     style={"color": TEXT_MED, "fontSize": "14px"})
        ], style={"marginBottom": "16px"})

    return html.Div([profile, msg_section])


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)