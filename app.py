import streamlit as st
import psycopg2
import psycopg2.extras
import pandas as pd
import plotly.express as px
import os
import json
import re

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DB Explorer",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DANGEROUS_KEYWORDS = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|COPY|EXECUTE|"
    r"DO\s*\$|CALL|SET\s+ROLE|SET\s+SESSION\s+AUTHORIZATION)\b",
    re.IGNORECASE,
)


def is_read_only_query(sql: str) -> bool:
    """Extra client-side check. The DB session is already read-only."""
    cleaned = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
    return not DANGEROUS_KEYWORDS.search(cleaned)


def get_connection(host, port, dbname, user, password):
    """Create a READ-ONLY connection to PostgreSQL."""
    conn = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        options="-c default_transaction_read_only=on",
        connect_timeout=10,
    )
    conn.set_session(readonly=True, autocommit=True)
    return conn


@st.cache_data(ttl=300, show_spinner=False)
def fetch_schemas(_conn_id, host, port, dbname, user, password):
    conn = get_connection(host, port, dbname, user, password)
    cur = conn.cursor()
    cur.execute(
        "SELECT schema_name FROM information_schema.schemata "
        "WHERE schema_name NOT IN ('pg_catalog','information_schema','pg_toast') "
        "ORDER BY schema_name"
    )
    schemas = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return schemas


@st.cache_data(ttl=300, show_spinner=False)
def fetch_tables(_conn_id, host, port, dbname, user, password, schema):
    conn = get_connection(host, port, dbname, user, password)
    cur = conn.cursor()
    cur.execute(
        "SELECT table_name, table_type FROM information_schema.tables "
        "WHERE table_schema = %s ORDER BY table_name",
        (schema,),
    )
    tables = cur.fetchall()
    cur.close()
    conn.close()
    return tables


@st.cache_data(ttl=300, show_spinner=False)
def fetch_columns(_conn_id, host, port, dbname, user, password, schema, table):
    conn = get_connection(host, port, dbname, user, password)
    cur = conn.cursor()
    cur.execute(
        "SELECT column_name, data_type, is_nullable, column_default "
        "FROM information_schema.columns "
        "WHERE table_schema = %s AND table_name = %s "
        "ORDER BY ordinal_position",
        (schema, table),
    )
    cols = cur.fetchall()
    cur.close()
    conn.close()
    return cols


@st.cache_data(ttl=300, show_spinner=False)
def fetch_row_count(_conn_id, host, port, dbname, user, password, schema, table):
    conn = get_connection(host, port, dbname, user, password)
    cur = conn.cursor()
    cur.execute(f'SELECT count(*) FROM "{schema}"."{table}"')
    count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return count


def run_query(host, port, dbname, user, password, sql, limit=500):
    conn = get_connection(host, port, dbname, user, password)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(sql)
    rows = cur.fetchmany(limit)
    columns = [desc[0] for desc in cur.description] if cur.description else []
    cur.close()
    conn.close()
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame()


def get_schema_summary(host, port, dbname, user, password):
    """Build a compact schema description for the AI."""
    conn = get_connection(host, port, dbname, user, password)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT t.table_schema, t.table_name, c.column_name, c.data_type
        FROM information_schema.tables t
        JOIN information_schema.columns c
          ON t.table_schema = c.table_schema AND t.table_name = c.table_name
        WHERE t.table_schema NOT IN ('pg_catalog','information_schema','pg_toast')
          AND t.table_type = 'BASE TABLE'
        ORDER BY t.table_schema, t.table_name, c.ordinal_position
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    tables = {}
    for schema, table, col, dtype in rows:
        key = f"{schema}.{table}"
        tables.setdefault(key, []).append(f"  {col} ({dtype})")

    parts = []
    for tbl, cols in tables.items():
        parts.append(f"{tbl}:\n" + "\n".join(cols))
    return "\n\n".join(parts)


def ask_claude(question, schema_text, api_key):
    """Send a natural language question to Claude and get SQL back."""
    try:
        import anthropic
    except ImportError:
        return None, "anthropic package not installed. Run: pip install anthropic"

    client = anthropic.Anthropic(api_key=api_key)
    prompt = f"""You are a PostgreSQL expert. Given the database schema below, write a SQL SELECT query that answers the user's question.

RULES:
- ONLY write SELECT queries. Never write INSERT, UPDATE, DELETE, DROP, or any data-modifying statement.
- Return ONLY the SQL query, no explanation, no markdown fences.
- Use double quotes for identifiers if they contain special characters.
- Limit results to 500 rows unless the user asks for more.
- If the question is ambiguous, make reasonable assumptions.

DATABASE SCHEMA:
{schema_text}

USER QUESTION: {question}"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        sql = message.content[0].text.strip()
        sql = sql.removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
        return sql, None
    except Exception as e:
        return None, str(e)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key, default in {
    "connected": False,
    "host": "",
    "port": "5432",
    "dbname": "",
    "user": "",
    "password": "",
    "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Sidebar - Connection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Database Connection")

    if not st.session_state.connected:
        with st.form("connect_form"):
            host = st.text_input("Host / IP", value=st.session_state.host)
            port = st.text_input("Port", value=st.session_state.port)
            dbname = st.text_input("Database Name", value=st.session_state.dbname)
            user = st.text_input("Username", value=st.session_state.user)
            password = st.text_input("Password", type="password", value=st.session_state.password)
            submitted = st.form_submit_button("Connect (Read-Only)", type="primary")

        if submitted:
            with st.spinner("Connecting..."):
                try:
                    conn = get_connection(host, int(port), dbname, user, password)
                    conn.close()
                    st.session_state.connected = True
                    st.session_state.host = host
                    st.session_state.port = port
                    st.session_state.dbname = dbname
                    st.session_state.user = user
                    st.session_state.password = password
                    st.rerun()
                except Exception as e:
                    st.error(f"Connection failed: {e}")
    else:
        st.success(f"Connected to **{st.session_state.dbname}**")
        st.caption(f"{st.session_state.user}@{st.session_state.host}:{st.session_state.port}")
        st.caption("Mode: **READ-ONLY**")
        if st.button("Disconnect"):
            st.session_state.connected = False
            st.rerun()

    st.divider()
    st.header("AI Settings (Optional)")
    api_key = st.text_input(
        "Anthropic API Key",
        value=st.session_state.api_key,
        type="password",
        help="Get yours at console.anthropic.com. Enables natural language queries.",
    )
    st.session_state.api_key = api_key


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------
def conn_params():
    return dict(
        host=st.session_state.host,
        port=int(st.session_state.port),
        dbname=st.session_state.dbname,
        user=st.session_state.user,
        password=st.session_state.password,
    )


def conn_id():
    s = st.session_state
    return f"{s.host}:{s.port}/{s.dbname}/{s.user}"


if not st.session_state.connected:
    st.title("Database Explorer")
    st.info("Enter your PostgreSQL credentials in the sidebar to get started. The connection is **read-only** so you cannot accidentally modify any data.")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_schema, tab_sql, tab_nl, tab_analytics = st.tabs([
    "Schema Explorer",
    "SQL Query",
    "Ask in English",
    "Quick Analytics",
])

# --- Schema Explorer ---
with tab_schema:
    st.header("Schema Explorer")
    schemas = fetch_schemas(conn_id(), **conn_params())

    if not schemas:
        st.warning("No schemas found.")
    else:
        sel_schema = st.selectbox("Schema", schemas, index=schemas.index("public") if "public" in schemas else 0)
        tables = fetch_tables(conn_id(), **conn_params(), schema=sel_schema)

        if not tables:
            st.info("No tables in this schema.")
        else:
            st.markdown(f"**{len(tables)} tables** in `{sel_schema}`")

            for tbl_name, tbl_type in tables:
                with st.expander(f"{'VIEW' if tbl_type == 'VIEW' else 'TABLE'} &mdash; {tbl_name}"):
                    cols = fetch_columns(conn_id(), **conn_params(), schema=sel_schema, table=tbl_name)
                    col_df = pd.DataFrame(cols, columns=["Column", "Type", "Nullable", "Default"])
                    st.dataframe(col_df, use_container_width=True, hide_index=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"Row count", key=f"cnt_{sel_schema}_{tbl_name}"):
                            count = fetch_row_count(conn_id(), **conn_params(), schema=sel_schema, table=tbl_name)
                            st.metric("Total Rows", f"{count:,}")
                    with col2:
                        if st.button(f"Preview 10 rows", key=f"preview_{sel_schema}_{tbl_name}"):
                            df = run_query(**conn_params(), sql=f'SELECT * FROM "{sel_schema}"."{tbl_name}" LIMIT 10')
                            st.dataframe(df, use_container_width=True, hide_index=True)


# --- SQL Query ---
with tab_sql:
    st.header("SQL Query")
    st.caption("Write any SELECT query. The connection is read-only so INSERT/UPDATE/DELETE will be blocked.")

    sql = st.text_area(
        "SQL",
        height=150,
        placeholder="SELECT * FROM public.my_table LIMIT 100",
        key="sql_input",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        run_btn = st.button("Run Query", type="primary", key="run_sql")
    with col2:
        limit = st.number_input("Row limit", min_value=1, max_value=10000, value=500, step=100)

    if run_btn and sql.strip():
        if not is_read_only_query(sql):
            st.error("Only SELECT / read-only queries are allowed.")
        else:
            with st.spinner("Running..."):
                try:
                    df = run_query(**conn_params(), sql=sql, limit=limit)
                    if df.empty:
                        st.info("Query returned no rows.")
                    else:
                        st.success(f"Returned {len(df)} rows")
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        csv = df.to_csv(index=False)
                        st.download_button("Download CSV", csv, "query_results.csv", "text/csv")
                except Exception as e:
                    st.error(f"Query error: {e}")

# --- Natural Language ---
with tab_nl:
    st.header("Ask in English")

    if not st.session_state.api_key:
        st.warning("Enter your Anthropic API key in the sidebar to enable natural language queries. Get one at [console.anthropic.com](https://console.anthropic.com)")
        st.stop()

    question = st.text_input(
        "Your question",
        placeholder="e.g., Show me the top 10 customers by total order amount",
        key="nl_input",
    )

    if st.button("Ask", type="primary", key="ask_nl") and question.strip():
        with st.spinner("Understanding your question..."):
            schema_text = get_schema_summary(**conn_params())
            sql, error = ask_claude(question, schema_text, st.session_state.api_key)

        if error:
            st.error(f"AI error: {error}")
        elif sql:
            st.code(sql, language="sql")

            if not is_read_only_query(sql):
                st.error("The AI generated a non-read-only query. This has been blocked for safety.")
            else:
                with st.spinner("Running query..."):
                    try:
                        df = run_query(**conn_params(), sql=sql, limit=500)
                        if df.empty:
                            st.info("Query returned no rows.")
                        else:
                            st.success(f"Returned {len(df)} rows")
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            csv = df.to_csv(index=False)
                            st.download_button("Download CSV", csv, "nl_results.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Query error: {e}")

# --- Quick Analytics ---
with tab_analytics:
    st.header("Quick Analytics")
    st.caption("Run a query in the SQL or English tab first, then come here to visualize.")

    analytics_sql = st.text_area(
        "SQL for chart",
        height=100,
        placeholder="SELECT category, SUM(amount) as total FROM orders GROUP BY category",
        key="analytics_sql",
    )

    if st.button("Run & Visualize", type="primary", key="run_analytics") and analytics_sql.strip():
        if not is_read_only_query(analytics_sql):
            st.error("Only SELECT queries are allowed.")
        else:
            with st.spinner("Running..."):
                try:
                    df = run_query(**conn_params(), sql=analytics_sql, limit=5000)
                    if df.empty:
                        st.info("No data returned.")
                    else:
                        st.dataframe(df, use_container_width=True, hide_index=True)

                        st.subheader("Chart Settings")
                        cols = list(df.columns)
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            chart_type = st.selectbox("Chart type", ["Bar", "Line", "Scatter", "Pie", "Histogram"])
                        with c2:
                            x_col = st.selectbox("X axis", cols, index=0)
                        with c3:
                            y_col = st.selectbox("Y axis", cols, index=min(1, len(cols) - 1))

                        chart_map = {
                            "Bar": px.bar,
                            "Line": px.line,
                            "Scatter": px.scatter,
                            "Pie": px.pie,
                            "Histogram": px.histogram,
                        }
                        try:
                            if chart_type == "Pie":
                                fig = px.pie(df, names=x_col, values=y_col)
                            else:
                                fig = chart_map[chart_type](df, x=x_col, y=y_col)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Chart error: {e}")
                except Exception as e:
                    st.error(f"Query error: {e}")
