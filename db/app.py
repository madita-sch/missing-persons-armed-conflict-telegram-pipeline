from dash import Dash, html, dcc, Input, Output, State, dash_table
import psycopg2
import pandas as pd
import plotly.express as px

# -----------------------
# DB connection
# -----------------------
def get_conn():
    return psycopg2.connect(
        dbname="missing_db",
        user="postgres",
        password="YOUR_PASSWORD",
        host="localhost",
        port="5432"
    )

# -----------------------
# SEARCH CASES
# -----------------------
def search_cases(q):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        SELECT case_id, name_ar, name_en, location_ar, location_en, age
        FROM cases
        WHERE name_ar ILIKE %s OR name_en ILIKE %s
    """, (f"%{q}%", f"%{q}%"))

    rows = cur.fetchall()
    cols = ["case_id","name_ar","name_en","location_ar","location_en","age"]

    cur.close()
    conn.close()

    return pd.DataFrame(rows, columns=cols)

# -----------------------
# GET FULL CASE PROFILE
# -----------------------
def get_case(case_id):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM cases WHERE case_id=%s", (case_id,))
    case = cur.fetchone()

    cur.execute("""
        SELECT message_id, posted_at, text_clean, text_raw, views, forwards, reactions
        FROM messages
        WHERE case_id=%s
        ORDER BY posted_at
    """, (case_id,))
    messages = cur.fetchall()

    cur.execute("""
        SELECT kind, value_ar, value_en
        FROM extracted_entities
        WHERE case_id=%s
    """, (case_id,))
    entities = cur.fetchall()

    cur.close()
    conn.close()

    return case, messages, entities


# -----------------------
# APP
# -----------------------
app = Dash(__name__)

app.layout = html.Div([
    html.H1("🕵️ Investigation Dashboard"),

    dcc.Input(
        id="search-input",
        type="text",
        placeholder="Search missing person name",
        style={"width": "300px"}
    ),

    html.Button("Search", id="search-btn"),

    html.Div(id="search-output"),

    html.Hr(),

    html.Div(id="case-detail")
])


# -----------------------
# SEARCH CALLBACK
# -----------------------
@app.callback(
    Output("search-output", "children"),
    Input("search-btn", "n_clicks"),
    State("search-input", "value")
)
def run_search(n_clicks, query):
    if not n_clicks:
        return ""

    if not query:
        return html.Div("Enter a search term")

    df = search_cases(query)

    if df.empty:
        return html.Div("❌ No cases found")

    return html.Div([
        html.Div(f"✅ Found {len(df)} cases"),

        dash_table.DataTable(
            id="results-table",
            data=df.to_dict("records"),
            columns=[{"name": i, "id": i} for i in df.columns],
            page_size=5,
            row_selectable="single"
        )
    ])


# -----------------------
# CASE DETAIL CALLBACK
# -----------------------
@app.callback(
    Output("case-detail", "children"),
    Input("results-table", "selected_rows"),
    State("results-table", "data")
)
def show_case(selected_rows, table_data):
    if not selected_rows:
        return ""

    selected_index = selected_rows[0]
    case_id = table_data[selected_index]["case_id"]

    case, messages, entities = get_case(case_id)

    # -----------------------
    # Case Info
    # -----------------------
    case_info = html.Div([
        html.H3("📁 Case Profile"),
        html.P(f"Case ID: {case[0]}"),
        html.P(f"Name (AR): {case[1]}"),
        html.P(f"Name (EN): {case[2]}"),
        html.P(f"Location: {case[3]}"),
        html.P(f"Age: {case[7]}")
    ])

    # -----------------------
    # Messages
    # -----------------------
    messages_section = html.Div("No messages")

    if messages:
        df = pd.DataFrame(messages, columns=[
            "message_id","posted_at","text_clean","text_raw","views","forwards","reactions"
        ])

        df["posted_at"] = pd.to_datetime(df["posted_at"])

        fig = px.line(df, x="posted_at", y="views", title="Views Over Time")

        messages_section = html.Div([
            html.H3("📊 Messages Timeline"),
            dcc.Graph(figure=fig),

            dash_table.DataTable(
                data=df.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df.columns],
                page_size=5
            )
        ])

    # -----------------------
    # Entities
    # -----------------------
    entities_section = html.Div("No entities")

    if entities:
        df_e = pd.DataFrame(entities, columns=["kind","value_ar","value_en"])

        fig_bar = px.bar(
            df_e["kind"].value_counts().reset_index(),
            x="index",
            y="kind",
            title="Entity Distribution"
        )

        entities_section = html.Div([
            html.H3("🧩 Extracted Entities"),

            dash_table.DataTable(
                data=df_e.to_dict("records"),
                columns=[{"name": i, "id": i} for i in df_e.columns],
                page_size=5
            ),

            dcc.Graph(figure=fig_bar)
        ])

    return html.Div([
        case_info,
        html.Hr(),
        messages_section,
        html.Hr(),
        entities_section
    ])


# -----------------------
# RUN
# -----------------------
if __name__ == "__main__":
    app.run(debug=True)