# Import libraries
from dash import Dash, html, dcc, Input, Output, State, dash_table, callback_context, no_update
import psycopg2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Connect to Database
def get_conn():
    return psycopg2.connect(
        dbname="missing_persons",
        user="postgres",
        password="Masch464hhu!?",
        host="localhost",
        port="5432"
    )

# Define database query functions
def get_all_cases():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT case_id, name_ar, name_en, location_ar, location_en, age, verified
        FROM cases
        ORDER BY case_id
    """)
    rows = cur.fetchall()
    cols = ["case_id", "name_ar", "name_en", "location_ar", "location_en", "age", "verified"]
    cur.close()
    conn.close()
    df = pd.DataFrame(rows, columns=cols)
    df['name_ar'] = df['name_ar'].fillna('Unknown name').replace('', 'Unknown name')
    df['name_en'] = df['name_en'].fillna('Unknown name').replace('', 'Unknown name')
    df['verified'] = df['verified'].apply(lambda x: '✅ Verified' if x else '⏳ Pending')
    return df


def search_cases(q):
    conn = get_conn()
    cur = conn.cursor()
    q = q.replace("*", "%")
    cur.execute("""
        SELECT case_id, name_ar, name_en, location_ar, location_en, age, verified
        FROM cases
        WHERE name_ar ILIKE %s OR name_en ILIKE %s
           OR location_ar ILIKE %s OR location_en ILIKE %s
    """, (f"%{q}%", f"%{q}%", f"%{q}%", f"%{q}%"))
    rows = cur.fetchall()
    cols = ["case_id", "name_ar", "name_en", "location_ar", "location_en", "age", "verified"]
    cur.close()
    conn.close()
    df = pd.DataFrame(rows, columns=cols)
    df['name_ar'] = df['name_ar'].fillna('Unknown name').replace('', 'Unknown name')
    df['name_en'] = df['name_en'].fillna('Unknown name').replace('', 'Unknown name')
    df['verified'] = df['verified'].apply(lambda x: '✅ Verified' if x else '⏳ Pending')
    return df


def get_case(case_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM cases WHERE case_id=%s", (case_id,))
    case = cur.fetchone()
    cur.execute("""
        SELECT 
            m.message_id,
            m.posted_at,
            m.text_clean,
            m.text_raw,
            COALESCE(t.text_clean_en, '') AS text_clean_en,
            m.views,
            m.forwards,
            m.reactions
        FROM messages m
        LEFT JOIN message_translations t ON m.message_id = t.message_id
        WHERE m.case_id=%s
        ORDER BY m.posted_at
    """, (case_id,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    messages = []
    for r in rows:
        message_id = r[0]
        link = f"https://t.me/GAZA20249/{message_id}"
        messages.append({
            "message_id": message_id,
            "posted_at": r[1],
            "text_clean": r[2],
            "text_raw": r[3],
            "text_clean_en": r[4],
            "views": r[5],
            "forwards": r[6],
            "reactions": r[7],
            "link": link
        })
    return case, messages


def verify_case_in_db(case_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        UPDATE cases
        SET verified = TRUE,
            verified_at = NOW()
        WHERE case_id = %s
    """, (case_id,))
    conn.commit()
    cur.close()
    conn.close()


def get_kpis():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM cases")
    total_cases = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM messages")
    messages = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM cases WHERE verified = TRUE")
    verified_cases = cur.fetchone()[0]
    cur.close()
    conn.close()
    return total_cases, verified_cases, messages


def get_analytics_data():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT COALESCE(location_en, 'Unknown') AS loc, COUNT(*) AS cnt
        FROM cases
        GROUP BY loc
        ORDER BY cnt DESC
        LIMIT 10
    """)
    loc_rows = cur.fetchall()
    cur.execute("""
        SELECT DATE_TRUNC('week', posted_at) AS week, COUNT(*) AS cnt
        FROM messages
        WHERE is_missing = TRUE
        GROUP BY week
        ORDER BY week
    """)
    msg_rows = cur.fetchall()
    cur.execute("""
        SELECT DATE(posted_at) AS day, COUNT(*) AS cnt
        FROM messages
        WHERE is_missing = TRUE
        GROUP BY day
        ORDER BY day
    """)
    cases_by_day_rows = cur.fetchall()
    cur.execute("SELECT COUNT(*) FROM cases")
    total_cases = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM messages")
    total_messages = cur.fetchone()[0]
    cur.close()
    conn.close()
    return loc_rows, msg_rows, cases_by_day_rows, total_cases, total_messages


# Styling constants
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


def build_cases_table(df, table_id="results-table"):
    name_mapping = {
        "case_id":     "Case ID",
        "name_ar":     "Name (Ar)",
        "name_en":     "Name (En)",
        "location_ar": "Location (Ar)",
        "location_en": "Location (En)",
        "age":         "Age",
        "verified":    "Status",
    }
    columns = [{"name": name_mapping.get(c, c.replace("_", " ").title()), "id": c} for c in df.columns]
    return dash_table.DataTable(
        id=table_id,
        data=df.to_dict("records"),
        columns=columns,
        page_size=15,
        row_selectable="single",
        selected_rows=[],
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#F8FAFC",
            "fontWeight": "600",
            "fontSize": "12px",
            "color": TEXT_MED,
            "border": "1px solid #E2E8F0",
            "borderBottom": "2px solid #E2E8F0",
            "fontFamily": FONT,
        },
        style_cell={
            "fontSize": "13px",
            "padding": "10px 14px",
            "border": "1px solid #E2E8F0",
            "borderBottom": "1px solid #F1F5F9",
            "color": TEXT_DARK,
            "fontFamily": FONT,
            "textAlign": "center",
        },
        style_data_conditional=[
            {
                "if": {"state": "selected"},
                "backgroundColor": "#EFF6FF",
                "border": "none",
            },
            {
                "if": {"column_id": "case_id"},
                "fontWeight": "700",
            },
            {
                "if": {"column_id": "name_en"},
                "backgroundColor": "#F3F4F6",
            },
            {
                "if": {"column_id": "location_en"},
                "backgroundColor": "#F3F4F6",
            },
            {
                "if": {
                    "column_id": "name_ar",
                    "filter_query": "{name_ar} = 'Unknown name'"
                },
                "fontStyle": "italic",
            },
            {
                "if": {
                    "column_id": "name_en",
                    "filter_query": "{name_en} = 'Unknown name'"
                },
                "fontStyle": "italic",
            },
            {
                "if": {
                    "column_id": "verified",
                    "filter_query": '{verified} = "✅ Verified"'
                },
                "color": ACCENT_GREEN,
                "fontWeight": "600",
            },
            {
                "if": {
                    "column_id": "verified",
                    "filter_query": '{verified} = "⏳ Pending"'
                },
                "color": ACCENT_AMB,
                "fontWeight": "600",
            },
        ],
        style_cell_conditional=[
            {"if": {"column_id": "name_ar"},
             "maxWidth": "180px", "textAlign": "center", "direction": "rtl",
             "whiteSpace": "normal", "height": "auto"},
            {"if": {"column_id": "name_en"},
             "maxWidth": "180px", "whiteSpace": "normal", "height": "auto"},
            {"if": {"column_id": "location_ar"},
             "maxWidth": "180px", "whiteSpace": "normal", "height": "auto"},
            {"if": {"column_id": "location_en"},
             "maxWidth": "180px", "whiteSpace": "normal", "height": "auto"},
            {"if": {"column_id": "verified"},
             "maxWidth": "100px"},
        ],
    )


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


def get_main_layout():
    return html.Div([
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
        dcc.Store(id="active-page", data="cases"),
        dcc.Store(id="selected-case-id", data=None),
        dcc.Store(id="search-performed", data=False),
    ], style={"margin": "0", "padding": "0", "fontFamily": FONT})


app.layout = html.Div([
    GOOGLE_FONT,
    dcc.Store(id="logged_in", data=False),
    dcc.Store(id="login-error-store", data=""),
    html.Div(id="app-content"),
])


@app.callback(
    [Output("logged_in", "data"), Output("login-error-store", "data")],
    Input("login-btn", "n_clicks"),
    State("password-input", "value"),
    prevent_initial_call=True
)
def login(n_clicks, password):
    if password == "missing_db26":
        return True, ""
    return False, "Incorrect password. Try again."


@app.callback(
    Output("app-content", "children"),
    Input("logged_in", "data"),
    Input("login-error-store", "data")
)
def render_app_content(logged_in, error_msg):
    if not logged_in:
        return html.Div([
            html.Div([
                html.H1("Missing Persons Database", style={
                    "textAlign": "center", "marginBottom": "20px",
                    "color": TEXT_DARK, "fontFamily": FONT
                }),
                html.Div("Enter Password:", style={
                    "textAlign": "center", "marginBottom": "10px",
                    "fontSize": "16px", "fontFamily": FONT
                }),
                dcc.Input(id="password-input", type="password", placeholder="Password", style={
                    "width": "200px", "padding": "10px", "fontSize": "16px",
                    "textAlign": "center", "margin": "0 auto", "display": "block", "fontFamily": FONT
                }),
                html.Button("Login", id="login-btn", n_clicks=0, style={
                    "marginTop": "20px", "padding": "10px 20px", "fontSize": "16px",
                    "backgroundColor": ACCENT, "color": "white", "border": "none",
                    "borderRadius": "5px", "cursor": "pointer", "display": "block",
                    "margin": "20px auto", "fontFamily": FONT
                }),
                html.Div(error_msg, style={
                    "textAlign": "center", "color": ACCENT_RED,
                    "marginTop": "10px", "fontFamily": FONT
                })
            ], style={
                "position": "absolute", "top": "50%", "left": "50%",
                "transform": "translate(-50%, -50%)", "textAlign": "center"
            })
        ], style={"height": "100vh", "backgroundColor": PAGE_BG, "fontFamily": FONT})
    return get_main_layout()


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

    if page == "cases":
        try:
            total, verified, messages = get_kpis()
        except Exception:
            total, verified, messages = "—", "—", "—"

        try:
            df_all = get_all_cases()
            all_cases_table = html.Div([
                html.Div(f"{len(df_all)} total case(s)", style={
                    "fontSize": "13px", "fontWeight": "600",
                    "color": TEXT_MED, "marginBottom": "12px"
                }),
                build_cases_table(df_all, table_id="results-table"),
            ])
        except Exception as e:
            all_cases_table = html.Div(
                f"❌ Could not load cases: {e}",
                style={"color": ACCENT_RED, "fontSize": "14px"}
            )

        return html.Div([
            html.Div([
                html.Div([
                    html.H1("Cases", style={
                        "margin": "0", "fontSize": "26px",
                        "fontWeight": "700", "color": TEXT_DARK
                    }),
                    html.P(
                        "Browse and search missing persons cases from Telegram channels",
                        style={"margin": "4px 0 0", "color": TEXT_MED, "fontSize": "14px"}
                    )
                ]),
                html.Div([
                    dcc.Input(
                        id="search-input",
                        type="text",
                        placeholder="Search by name or location (e.g. Ahm*d, Gaza)",
                        style={
                            "padding": "10px 14px",
                            "borderRadius": "8px",
                            "border": "1px solid #E2E8F0",
                            "fontSize": "14px",
                            "width": "240px",
                            "fontFamily": FONT,
                        }
                    ),
                    html.Button(
                        "Search",
                        id="search-btn",
                        n_clicks=0,
                        style={
                            "marginLeft": "10px",
                            "padding": "10px 16px",
                            "backgroundColor": ACCENT,
                            "color": "white",
                            "border": "none",
                            "borderRadius": "8px",
                            "cursor": "pointer",
                            "fontWeight": "600",
                            "fontSize": "14px",
                        }
                    )
                ], style={"display": "flex", "alignItems": "center"})
            ], style={
                "display": "flex",
                "justifyContent": "space-between",
                "alignItems": "flex-start",
                "marginBottom": "28px"
            }),

            html.Div([
                kpi_card("Total Cases",     total,    ACCENT,       "📁"),
                kpi_card("Verified Cases",  verified, ACCENT_GREEN, "✅"),
                kpi_card("Messages",        messages, ACCENT_AMB,   "💬"),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "28px"}),

            html.Div(id="all-cases-section", children=[
                card([
                    html.Div("All Cases", style={
                        "fontSize": "15px", "fontWeight": "600",
                        "color": TEXT_DARK, "marginBottom": "8px"
                    }),
                    html.Div(
                        "Select a case to see the full profile and verify it below.",
                        style={"fontSize": "13px", "color": TEXT_MED, "marginBottom": "16px"}
                    ),
                    all_cases_table,
                ], style={"marginBottom": "24px"})
            ]),

            html.Div(id="search-output", style={"marginBottom": "24px"}),
            html.Div(id="case-detail"),

            # Hidden store for current case ID used by verify callback
            dcc.Store(id="current-case-id", data=None),
        ])

    if page == "analytics":
        try:
            loc_rows, msg_rows, cases_by_day_rows, total_cases, total_messages = get_analytics_data()
        except Exception:
            loc_rows, msg_rows, cases_by_day_rows, total_cases, total_messages = [], [], [], "—", "—"

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

        if cases_by_day_rows:
            df_cbd = pd.DataFrame(cases_by_day_rows, columns=["Date", "Cases"])
            fig_cbd = px.bar(
                df_cbd, x="Date", y="Cases",
                title="Cases by Day",
                color_discrete_sequence=[ACCENT_GREEN]
            )
            fig_cbd.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                title_font_size=15,
                margin=dict(l=0, r=0, t=40, b=0),
                xaxis=dict(gridcolor="#F1F5F9"),
                yaxis=dict(gridcolor="#F1F5F9"),
            )
            cbd_chart = dcc.Graph(figure=fig_cbd, config={"displayModeBar": False})
        else:
            cbd_chart = html.Div("No cases-by-day data available.",
                                 style={"color": TEXT_MED, "fontSize": "13px"})

        return html.Div([
            html.Div([
                html.H1("Analytics", style={
                    "margin": "0", "fontSize": "26px",
                    "fontWeight": "700", "color": TEXT_DARK
                }),
                html.P("Trends and breakdowns across all cases",
                       style={"margin": "4px 0 0", "color": TEXT_MED, "fontSize": "14px"})
            ], style={"marginBottom": "28px"}),
            html.Div([
                kpi_card("Total Cases",    total_cases,    ACCENT,       "📁"),
                kpi_card("Total Messages", total_messages, ACCENT_GREEN, "💬"),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "24px"}),
            html.Div([
                card([loc_chart], style={"flex": "1"}),
                card([cbd_chart], style={"flex": "1"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "16px"}),
            card([msg_chart]),
        ])

    return html.Div("Select a section from the sidebar.")


@app.callback(
    [Output("search-output", "children"), Output("search-performed", "data")],
    Input("search-btn", "n_clicks"),
    State("search-input", "value"),
    prevent_initial_call=True
)
def run_search(n_clicks, query):
    if not query or not query.strip():
        return "", False
    try:
        df = search_cases(query.strip())
    except Exception as e:
        return html.Div(f"❌ Database error: {e}",
                        style={"color": ACCENT_RED, "fontSize": "14px"}), True
    if df.empty:
        return html.Div("No cases found for that query.",
                        style={"color": TEXT_MED, "fontSize": "14px"}), True
    return html.Div([
        html.Div(f"Found {len(df)} case(s) — select a row to view details and verify below", style={
            "fontSize": "13px", "fontWeight": "600",
            "color": TEXT_MED, "marginBottom": "12px"
        }),
        build_cases_table(df, table_id="results-table"),
    ]), True


@app.callback(
    Output("all-cases-section", "style"),
    Input("search-performed", "data")
)
def toggle_all_cases_visibility(search_performed):
    if search_performed:
        return {"display": "none"}
    return {}


@app.callback(
    [Output("case-detail", "children"),
     Output("current-case-id", "data")],
    Input("results-table", "selected_rows"),
    State("results-table", "data"),
    prevent_initial_call=True
)
def show_case(selected_rows, table_data):
    if not selected_rows or table_data is None:
        return "", None

    case_id = table_data[selected_rows[0]]["case_id"]

    try:
        case, messages = get_case(case_id)
    except Exception as e:
        return html.Div(f"❌ Error loading case: {e}",
                        style={"color": ACCENT_RED, "marginTop": "16px"}), None

    if not case:
        return html.Div("Case not found.", style={"color": TEXT_MED}), None

    def safe(val):
        return val if val else "—"

    # Check current verification status
    is_verified = bool(case[12]) if len(case) > 12 and case[12] is not None else False
    
    info_rows = [
        ("Case ID",       safe(case[0])),
        ("Name (AR)",     safe(case[1])),
        ("Name (EN)",     safe(case[2])),
        ("Location (AR)", safe(case[3])),
        ("Location (EN)", safe(case[4])),
        ("Age",           safe(case[7])),
        ("Status",        "✅ Verified" if is_verified else "⏳ Pending review"),
    ]

    verify_section = html.Div([
        html.Div(
            "Human-in-the-loop verification: review the extracted information and associated "
            "messages below, then mark this case as verified if the data is correct.",
            style={
                "fontSize": "13px", "color": TEXT_MED,
                "marginBottom": "12px", "fontStyle": "italic"
            }
        ),
        html.Button(
            "✅ Verify Case" if not is_verified else "✅ Already Verified",
            id="verify-btn",
            n_clicks=0,
            disabled=is_verified,
            style={
                "padding": "10px 20px",
                "backgroundColor": ACCENT_GREEN if not is_verified else "#94A3B8",
                "color": "white",
                "border": "none",
                "borderRadius": "8px",
                "cursor": "pointer" if not is_verified else "default",
                "fontWeight": "600",
                "fontSize": "14px",
                "fontFamily": FONT,
            }
        ),
        html.Div(id="verify-status", style={
            "marginTop": "10px",
            "fontSize": "13px",
            "color": TEXT_MED,
            "fontFamily": FONT,
        })
    ], style={"marginTop": "20px", "paddingTop": "16px",
              "borderTop": "1px solid #E2E8F0"})

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
        ]),
        verify_section,
    ], style={"marginTop": "24px", "marginBottom": "16px"})

    if messages:
        df_m = pd.DataFrame(messages)
        df_m["posted_at"] = pd.to_datetime(df_m["posted_at"]).dt.strftime("%Y-%m-%d")

        def make_link(row):
            link = row.get("link")
            if pd.notna(link):
                link = str(link).strip()
                if link:
                    return f"[Open ↗]({link})"
            return "—"

        df_m["Telegram Link"] = df_m.apply(make_link, axis=1)
        df_display = df_m[[
            "posted_at", "text_raw", "text_clean_en",
            "views", "forwards", "reactions", "Telegram Link"
        ]].rename(columns={
            "posted_at":      "Date",
            "text_raw":       "Arabic Text",
            "text_clean_en":  "English Translation",
            "views":          "Views",
            "forwards":       "Forwards",
            "reactions":      "Reactions",
        })

        msg_section = card([
            html.Div(f"💬 Related Messages ({len(df_m)} total)", style={
                "fontSize": "15px", "fontWeight": "700",
                "color": TEXT_DARK, "marginBottom": "12px"
            }),
            dash_table.DataTable(
                data=df_display.to_dict("records"),
                columns=[
                    {"name": "Date",               "id": "Date"},
                    {"name": "Arabic Text",         "id": "Arabic Text"},
                    {"name": "English Translation", "id": "English Translation"},
                    {"name": "Views",               "id": "Views"},
                    {"name": "Forwards",            "id": "Forwards"},
                    {"name": "Reactions",           "id": "Reactions"},
                    {"name": "Telegram Link",       "id": "Telegram Link",
                     "presentation": "markdown"},
                ],
                page_size=10,
                style_table={"overflowX": "auto"},
                style_header={
                    "backgroundColor": "#F8FAFC",
                    "fontWeight": "600",
                    "fontSize": "12px",
                    "color": TEXT_MED,
                    "border": "1px solid #E2E8F0",
                    "borderBottom": "2px solid #E2E8F0",
                    "fontFamily": FONT,
                },
                style_cell={
                    "fontSize": "13px",
                    "padding": "10px 14px",
                    "border": "1px solid #E2E8F0",
                    "borderBottom": "1px solid #F1F5F9",
                    "color": TEXT_DARK,
                    "fontFamily": FONT,
                    "whiteSpace": "normal",
                    "height": "auto",
                    "maxWidth": "300px",
                    "overflow": "hidden",
                    "textOverflow": "ellipsis",
                    "textAlign": "center",
                },
                style_cell_conditional=[
                    {"if": {"column_id": "Arabic Text"},
                     "textAlign": "right", "direction": "rtl", "maxWidth": "280px"},
                    {"if": {"column_id": "English Translation"},
                     "maxWidth": "280px"},
                    {"if": {"column_id": "Date"},      "minWidth": "130px"},
                    {"if": {"column_id": "Views"},     "textAlign": "center", "maxWidth": "70px"},
                    {"if": {"column_id": "Forwards"},  "textAlign": "center", "maxWidth": "80px"},
                    {"if": {"column_id": "Reactions"}, "textAlign": "center", "maxWidth": "80px"},
                    {"if": {"column_id": "Telegram Link"},
                     "textAlign": "center", "maxWidth": "100px"},
                ],
                tooltip_data=[
                    {
                        "Arabic Text":         {"value": str(row["Arabic Text"]),         "type": "markdown"},
                        "English Translation": {"value": str(row["English Translation"]), "type": "markdown"},
                    }
                    for row in df_display.to_dict("records")
                ],
                tooltip_duration=None,
            )
        ], style={"marginBottom": "16px"})
    else:
        msg_section = card([
            html.Div("No messages for this case.",
                     style={"color": TEXT_MED, "fontSize": "14px"})
        ], style={"marginBottom": "16px"})

    return html.Div([profile, msg_section]), case_id


@app.callback(
    Output("verify-status", "children"),
    Input("verify-btn", "n_clicks"),
    State("current-case-id", "data"),
    prevent_initial_call=True
)
def verify_case(n_clicks, case_id):
    if not case_id:
        return "No case selected."
    try:
        verify_case_in_db(case_id)
        return "✅ Case successfully verified and saved to database."
    except Exception as e:
        return f"❌ Error verifying case: {e}"


if __name__ == "__main__":
    app.run(debug=True)
