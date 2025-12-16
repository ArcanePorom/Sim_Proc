import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
from dash.dependencies import State
import plotly.graph_objects as go
import json
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, "sim_proc_combined_results.csv"))

# Coerce expected numeric columns (defensive) so Plotly does not treat them categorically, preventing visual offset.
_numeric_expect = ["dps", "priority_dps", "StdDev", "range_pct_dps", "crit", "haste", "vers", "mastery", "targets", "min_dps", "max_dps"]
for _col in _numeric_expect:
    if _col in df.columns:
        df[_col] = pd.to_numeric(df[_col], errors='coerce')

# Ensure both build identifiers exist (script mixes 'builds' for legend and 'build_id' for color map originally).
if 'builds' not in df.columns and 'build_id' in df.columns:
    df['builds'] = df['build_id']
if 'build_id' not in df.columns and 'builds' in df.columns:
    df['build_id'] = df['builds']

df = df.dropna(subset=["dps", "priority_dps", "StdDev", "range_pct_dps"])
features = ["dps", "priority_dps", "StdDev", "range_pct_dps"]
X_scaled = StandardScaler().fit_transform(df[features])
# KMeans: ensure n_clusters <= number of samples to avoid ValueError
n_samples = X_scaled.shape[0]
if n_samples < 2:
    raise ValueError(f"Not enough samples for clustering: {n_samples}")
_requested_clusters = 4
_n_clusters = _requested_clusters if n_samples >= _requested_clusters else n_samples
if _n_clusters < _requested_clusters:
    print(f"[KMeans] Reducing n_clusters from {_requested_clusters} to {_n_clusters} (n_samples={n_samples})")
df["cluster"] = KMeans(n_clusters=_n_clusters, random_state=42).fit_predict(X_scaled)
coords = PCA(n_components=2).fit_transform(X_scaled)
df["pc1"], df["pc2"] = coords[:, 0], coords[:, 1]

# Load talents early to derive token-based talent columns (more robust than positional after 'trinket_2').
def _tok(s: str) -> str | None:
    if not s:
        return None
    s = s.lower().replace(" ", "_")
    return "".join(c for c in s if c == "_" or c.isalpha())

with open(os.path.join(script_dir, "talents.json"), encoding="utf-8") as f:
    talents_raw = json.load(f)

_tokens = set()
for block in talents_raw:
    for node in (block.get("heroNodes", []) or []):
        ntype = node.get("type")
        if ntype == "choice":
            for e in node.get("entries", []) or []:
                tok = _tok(e.get("name"))
                if tok:
                    _tokens.add(tok)
        else:
            tok = _tok(node.get("name"))
            if tok:
                _tokens.add(tok)
    for node in (block.get("classNodes", []) or []) + (block.get("specNodes", []) or []):
        if node.get("type") == "choice":
            for e in node.get("entries", []) or []:
                tok = _tok(e.get("name"))
                if tok:
                    _tokens.add(tok)
        else:
            tok = _tok(node.get("name"))
            if tok:
                _tokens.add(tok)

# Fallback positional detection (retain existing behavior) then intersect with known tokens to avoid accidental metric inclusion.
trinket2_idx = list(df.columns).index('trinket_2') if 'trinket_2' in df.columns else None
_post_trinket = list(df.columns)[trinket2_idx+1:] if trinket2_idx is not None else []
exclude_dashboard = {'cluster', 'pc1', 'pc2'}
talent_fields = [c for c in _post_trinket if c not in exclude_dashboard and c in _tokens]

if not talent_fields:
    # If intersection produced empty (e.g., tokens naming mismatch), revert to original positional slice excluding dashboard fields.
    talent_fields = [t for t in _post_trinket if t not in exclude_dashboard]

# --- Talent consolidation logic ---
def _tok(s: str) -> str:
    if not s:
        return None
    s = s.lower().replace(" ", "_")
    return "".join(c for c in s if c == "_" or c.isalpha())

    # talents_raw already loaded above.

# Build sets for hero tree tokens and choice child tokens (to match CSV columns)
hero_tree_nodes = set()
choice_nodes = set()
for entry in talents_raw:
    # Hero nodes: non-choice -> use node name token; choice -> use child entry name tokens
    for node in entry.get("heroNodes", []) or []:
        ntype = node.get("type")
        if ntype == "choice":
            for e in node.get("entries", []) or []:
                nm = e.get("name") or node.get("name")
                tok = _tok(nm)
                if tok:
                    choice_nodes.add(tok)
        else:
            nm = node.get("name")
            tok = _tok(nm)
            if tok:
                hero_tree_nodes.add(tok)
    # Also check class/spec nodes for choices (CSV uses child entry tokens for choices)
    for node in (entry.get("classNodes", []) or []) + (entry.get("specNodes", []) or []):
        if node.get("type") == "choice":
            for e in node.get("entries", []) or []:
                nm = e.get("name") or node.get("name")
                tok = _tok(nm)
                if tok:
                    choice_nodes.add(tok)

# For consolidation, create a mapping from talent_fields to hero/choice
talent_field_type = {}
for t in talent_fields:
    if t in hero_tree_nodes:
        talent_field_type[t] = "hero"
    elif t in choice_nodes:
        talent_field_type[t] = "choice"
    else:
        talent_field_type[t] = "other"

def group_type(name):
    name = name.lower()
    if "spellslinger" in name:
        return "Spellslinger"
    elif "sunfury" in name:
        return "Sunfury"
    else:
        return "Other"
df["sim_type"] = df["build_id"].apply(group_type)

spellslinger_palette = ["#d16ba5", "#ba83ca", "#aa8fd8", "#8aa7ec"]
sunfury_palette = ["#e76f51", "#ea7a4f", "#f1854d", "#f5974b", "#f8a54a"]
file_color_map = {}
unique_file_ids = df["build_id"].unique()
for i, f in enumerate(unique_file_ids):
    f_low = f.lower()
    if "spellslinger" in f_low:
        file_color_map[f] = spellslinger_palette[i % len(spellslinger_palette)]
    elif "sunfury" in f_low:
        file_color_map[f] = sunfury_palette[i % len(sunfury_palette)]
    else:
        file_color_map[f] = "#999999"

# Separate color map for grouped 'builds' values (previous bug: using build_id map while coloring by 'builds').
builds_color_map = {}
_combined_palette = spellslinger_palette + sunfury_palette
unique_build_groups = df["builds"].unique()
for i, b in enumerate(unique_build_groups):
    builds_color_map[b] = _combined_palette[i % len(_combined_palette)]

def wrap_text(text, width=40):
    return "<br>".join([text[i:i+width] for i in range(0, len(text), width)])
def clean_decimal(val):
    val = round(val, 3)
    return f"{val}".rstrip("0").rstrip(".") if "." in f"{val}" else f"{val}"

exclude = {"cluster", "pc1", "pc2"}
numeric_options = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in exclude]
axis_options = [col for col in numeric_options if col not in talent_fields]

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Build Explorer"

app.layout = html.Div(
    style={
        "backgroundColor": "#1e1e2f",
        "color": "#f2b5d4",
        "fontFamily": "Segoe UI",
        "textAlign": "center",
        "padding": "2em",
        "margin": "0",
        "border": "none",
        "boxSizing": "border-box",
        "overflowX": "hidden"
    },
    children=[
        html.H1(id="graph-title", style={"color": "#ffc8dd"}),

        html.Div([
            html.Label("Sidebar Legend", style={"color": "#ffc8dd", "fontWeight": "bold", "marginRight": "1em"}),
            dcc.Dropdown(
                id="sidebar-legend",
                options=[
                    {"label": "Build", "value": "builds"},
                    {"label": "Talent", "value": "talent"},
                    {"label": "Spec", "value": "spec"},
                    {"label": "Race", "value": "race"},
                ],
                value="talent",
                multi=False,
                style={"width": "200px", "backgroundColor": "#2b2b3c", "color": "#fff", "marginBottom": "1em"}
            )
        ], style={"display": "flex", "justifyContent": "center", "alignItems": "center", "marginBottom": "1em"}),

        dbc.Collapse(
            id="filter-collapse",
            is_open=False,
            children=[
                html.Div([
                    html.Label("Race Filter", style={"color": "#ffc8dd", "fontWeight": "bold", "marginRight": "1em"}),
                    dcc.Dropdown(
                        id="race-filter",
                        options=[{"label": r, "value": r} for r in sorted(df["race"].dropna().unique())],
                        value=[],
                        multi=True,
                        placeholder="Select race",
                        style={"width": "200px", "marginRight": "2em", "backgroundColor": "#2b2b3c", "color": "#fff"}
                    ),
                    html.Label("Class Filter", style={"color": "#ffc8dd", "fontWeight": "bold", "marginRight": "1em"}),
                    dcc.Dropdown(
                        id="class-filter",
                        options=[{"label": c, "value": c} for c in sorted(df["class"].dropna().unique())],
                        value=[],
                        multi=True,
                        placeholder="Select class",
                        style={"width": "200px", "marginRight": "2em", "backgroundColor": "#2b2b3c", "color": "#fff"}
                    ),
                    html.Label("Spec Filter", style={"color": "#ffc8dd", "fontWeight": "bold", "marginRight": "1em"}),
                    dcc.Dropdown(
                        id="spec-filter",
                        options=[{"label": s, "value": s} for s in sorted(df["spec"].dropna().unique())],
                        value=[],
                        multi=True,
                        placeholder="Select spec",
                        style={"width": "200px", "marginRight": "2em", "backgroundColor": "#2b2b3c", "color": "#fff"}
                    ),
                    html.Label("Build Filter", style={"color": "#ffc8dd", "fontWeight": "bold", "marginRight": "1em"}),
                    dcc.Dropdown(
                        id="buildid-filter",
                        options=[{"label": b, "value": b} for b in sorted(df["builds"].dropna().unique())],
                        value=[],
                        multi=True,
                        placeholder="Select build id",
                        style={"width": "200px", "backgroundColor": "#2b2b3c", "color": "#fff"}
                    ),
                    html.Label("Talent Filter", style={"color": "#ffc8dd", "fontWeight": "bold", "marginLeft": "2em", "marginRight": "1em"}),
                    dcc.Dropdown(
                        id="talent-filter",
                        options=[{"label": t, "value": t} for t in talent_fields],
                        value=[],
                        multi=True,
                        placeholder="Select talent(s)",
                        style={"width": "200px", "marginLeft": "2em", "backgroundColor": "#2b2b3c", "color": "#fff"}
                    ),
                ], style={"display": "flex", "justifyContent": "center", "gap": "20px", "marginBottom": "1.5em"}),
            ]
        ),
        html.Button(
            id="toggle-filters",
            children="Show/Hide Filters",
            style={"marginBottom": "1em", "backgroundColor": "#2b2b3c", "color": "#ffc8dd", "border": "1px solid #444", "padding": "0.5em", "fontWeight": "bold"}
        ),
        html.Div([
            html.Label("Selected talent string:", style={"color": "#ffc8dd", "marginBottom": "0.3em"}),
            dcc.Input(
                id="talent-box",
                value="",
                type="text",
                style={
                    "width": "500px",
                    "backgroundColor": "#2b2b3c",
                    "color": "#ffffff",
                    "border": "1px solid #777",
                    "padding": "0.5em",
                    "fontSize": "1em",
                    "margin": "auto"
                }
            )
        ], style={"textAlign": "center", "paddingBottom": "1.5em"}),

        html.Div(id="main-graph-block"),

        html.Div([
            html.Div([
                html.Label("View Mode", style={"color": "#ffc8dd", "fontWeight": "bold", "marginBottom": "0.5em"}),
                dcc.RadioItems(
                    id="mode-selector",
                    options=[
                        {"label": "2D", "value": "2d"},
                        {"label": "3D", "value": "3d"},
                        {"label": "Heatmap", "value": "heatmap"},
                        {"label": "Best Profiles", "value": "best_profiles"}
                    ],
                    value="3d",
                    labelStyle={"display": "inline-block", "marginRight": "1em"},
                    style={
                        "backgroundColor": "#2c2c3c",
                        "color": "#f2b5d4",
                        "fontWeight": "bold",
                        "border": "1px solid #444",
                        "padding": "0.5em",
                        "width": "150px",
                        "margin": "auto"
                    }
                )
            ], style={"marginRight": "30px"}),
            html.Div([
                html.Label("Sample %", style={"color": "#ffc8dd", "fontWeight": "bold", "marginBottom": "0.5em"}),
                dcc.Slider(
                    id="sample-percent",
                    min=0,
                    max=100,
                    step=1,
                    value=10,
                    marks={0: "0%", 25: "25%", 50: "50%", 75: "75%", 100: "100%"},
                    tooltip={"placement": "bottom", "always_visible": False},
                    updatemode="drag"
                ),
                html.Div(id="sample-readout", style={"color": "#f2b5d4", "marginTop": "4px", "fontSize": "0.8em"})
            ], style={"width": "220px", "marginRight": "30px"}),
            html.Div([
                html.Label("Legend Talent Limit", style={"color": "#ffc8dd", "fontWeight": "bold", "marginBottom": "0.5em"}),
                dcc.Slider(
                    id="legend-talent-limit",
                    min=5,
                    max=50,
                    step=5,
                    value=50,
                    marks={5: "5", 10: "10", 20: "20", 30: "30", 40: "40", 50: "50"},
                    tooltip={"placement": "bottom", "always_visible": False},
                    updatemode="drag"
                ),
                html.Div("Tip: Use Talent Filter to force a talent into the legend.", style={"color": "#a9a9b3", "marginTop": "4px", "fontSize": "0.75em"})
            ], style={"width": "260px"}),

            html.Div([
                html.Label("X Axis", style={"color": "#ffc8dd", "fontWeight": "bold", "marginBottom": "0.5em"}),
                dcc.RadioItems(
                    id="x-metric",
                    options=[{"label": col, "value": col} for col in axis_options],
                    value="dps",
                    labelStyle={"display": "block", "marginBottom": "0.2em"},
                    style={
                        "backgroundColor": "#2c2c3c",
                        "color": "#f2b5d4",
                        "fontWeight": "bold",
                        "border": "1px solid #444",
                        "padding": "0.5em",
                        "width": "150px",
                        "margin": "auto"
                    }
                )
            ], style={"marginRight": "30px"}),

            html.Div([
                html.Label("Y Axis", style={"color": "#ffc8dd", "fontWeight": "bold", "marginBottom": "0.5em"}),
                dcc.RadioItems(
                    id="y-metric",
                    options=[{"label": col, "value": col} for col in axis_options],
                    value="priority_dps",
                    labelStyle={"display": "block", "marginBottom": "0.2em"},
                    style={
                        "backgroundColor": "#2c2c3c",
                        "color": "#f2b5d4",
                        "fontWeight": "bold",
                        "border": "1px solid #444",
                        "padding": "0.5em",
                        "width": "150px",
                        "margin": "auto"
                    }
                )
            ], style={"marginRight": "30px"}),

            html.Div(id="z-axis-block", children=[
                html.Label("Z Axis", style={"color": "#ffc8dd", "fontWeight": "bold", "marginBottom": "0.5em"}),
                dcc.RadioItems(
                    id="z-metric",
                    options=[{"label": col, "value": col} for col in axis_options],
                    value="targets",
                    labelStyle={"display": "block", "marginBottom": "0.2em"},
                    style={
                        "backgroundColor": "#2c2c3c",
                        "color": "#f2b5d4",
                        "fontWeight": "bold",
                        "border": "1px solid #444",
                        "padding": "0.5em",
                        "width": "150px",
                        "margin": "auto"
                    }
                )
            ])
        ], style={
            "display": "flex",
            "justifyContent": "center",
            "gap": "20px",
            "marginTop": "20px",
            "marginBottom": "40px"
        }),
        html.Div(
            id="copy-output",
            style={
                "textAlign": "center",
                "padding": "0.5em",
                "fontSize": "1.1em",
                "color": "#ffc8dd"
            }
        )
    ]
)

@app.callback(
    Output("main-graph-block", "children"),
    Input("mode-selector", "value")
)
def render_main_graph_block(mode):
    if mode == "heatmap":
        return html.Div([
            html.Label("Heatmap Type", style={"color": "#ffc8dd", "fontWeight": "bold", "marginRight": "1em"}),
            dcc.Dropdown(
                id="heatmap-type",
                options=[
                    {"label": "Talent vs Talent", "value": "talent_talent"},
                    {"label": "Talent vs Stat", "value": "talent_stat"},
                    {"label": "Stat vs Stat", "value": "stat_stat"}
                ],
                value="talent_talent",
                style={"width": "250px", "backgroundColor": "#2b2b3c", "color": "#fff", "marginBottom": "1em"}
            ),
            html.Label("Significance Threshold", style={"color": "#ffc8dd", "fontWeight": "bold", "marginLeft": "1em"}),
            dcc.Slider(
                id="significance-threshold",
                min=0,
                max=100,
                step=1,
                value=95,
                marks={i: f"{i}%" for i in range(0, 101, 10)},
                tooltip={"placement": "bottom", "always_visible": True},
                updatemode="drag",
                included=True
            ),
            dbc.Checkbox(
                id="flip-threshold",
                value=False,
                label="Flip Filter (keep talents below threshold)",
                style={"marginLeft": "1em", "color": "#ffc8dd", "fontWeight": "bold"}
            ),
            html.Label("Metric", style={"color": "#ffc8dd", "fontWeight": "bold", "marginLeft": "1em"}),
            dcc.Dropdown(
                id="heatmap-metric",
                options=[
                    {"label": "DPS", "value": "dps"},
                    {"label": "Priority DPS", "value": "priority_dps"},
                    {"label": "StdDev", "value": "StdDev"},
                    {"label": "Range % DPS", "value": "range_pct_dps"}
                ],
                value="dps",
                style={"width": "200px", "backgroundColor": "#2b2b3c", "color": "#fff", "marginBottom": "1em", "marginLeft": "1em"}
            ),
            dcc.Graph(id="heatmap-graph", config={"displayModeBar": False}, style={
                "width": "900px",
                "height": "700px",
                "margin": "auto"
            })
        ], style={"textAlign": "center", "marginTop": "2em"})
    elif mode == "best_profiles":
        # Get unique target counts for weight inputs
        unique_targets = sorted(df['targets'].unique())
        target_weight_inputs = []
        for target in unique_targets:
            target_weight_inputs.append(
                html.Div([
                    html.Label(f"{target} Targets:", style={"color": "#ffc8dd", "fontWeight": "bold", "marginRight": "10px"}),
                    dcc.Input(
                        id=f'target-{target}-weight',
                        type='number',
                        value=1.0,
                        min=0,
                        max=1,
                        step=0.1,
                        style={'width': '80px', 'backgroundColor': '#2b2b3c', 'color': '#fff', 'border': '1px solid #444'}
                    )
                ], style={'display': 'inline-block', 'marginRight': '20px', 'marginBottom': '10px'})
            )
        
        return html.Div([
            html.H2("Best Profiles", style={'color': '#ffc8dd', 'marginBottom': '20px'}),
            html.Div([
                html.H4("Weight Controls", style={'color': '#ffc8dd', 'marginBottom': '15px'}),
                html.Div([
                    html.Div([
                        html.Label("DPS Weight:", style={"color": "#ffc8dd", "fontWeight": "bold", "marginRight": "10px"}),
                        dcc.Input(
                            id='dps-weight',
                            type='number',
                            value=1.0,
                            min=0,
                            max=1,
                            step=0.1,
                            style={'width': '80px', 'backgroundColor': '#2b2b3c', 'color': '#fff', 'border': '1px solid #444'}
                        )
                    ], style={'display': 'inline-block', 'marginRight': '20px'}),
                    html.Div([
                        html.Label("Priority DPS Weight:", style={"color": "#ffc8dd", "fontWeight": "bold", "marginRight": "10px"}),
                        dcc.Input(
                            id='priority-dps-weight',
                            type='number',
                            value=1.0,
                            min=0,
                            max=1,
                            step=0.1,
                            style={'width': '80px', 'backgroundColor': '#2b2b3c', 'color': '#fff', 'border': '1px solid #444'}
                        )
                    ], style={'display': 'inline-block', 'marginRight': '20px'})
                ], style={'marginBottom': '20px'}),
                html.Div(target_weight_inputs, style={'marginBottom': '20px'}),
                html.Button('Update Rankings', id='update-rankings-btn', 
                           style={'backgroundColor': '#2b2b3c', 'color': '#ffc8dd', 'border': '1px solid #444', 'padding': '10px', 'fontWeight': 'bold'})
            ], style={'marginBottom': '30px'}),
            dash_table.DataTable(
                id='best-profiles-table',
                columns=[
                    {'name': 'Rank', 'id': 'rank'},
                    {'name': 'Talent String', 'id': 'talent_str'},
                    {'name': 'Trinket 1', 'id': 'trinket_1'},
                    {'name': 'Trinket 2', 'id': 'trinket_2'},
                    {'name': 'Crit', 'id': 'crit'},
                    {'name': 'Haste', 'id': 'haste'},
                    {'name': 'Vers', 'id': 'vers'},
                    {'name': 'Mastery', 'id': 'mastery'},
                    {'name': 'Weighted Score', 'id': 'weighted_score', 'type': 'numeric', 'format': {'specifier': '.3f'}}
                ],
                style_cell={'textAlign': 'left', 'backgroundColor': '#2b2b3c', 'color': '#fff'},
                style_header={'backgroundColor': '#1e1e2f', 'color': '#ffc8dd', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {
                        'if': {'row_index': 0},
                        'backgroundColor': '#FFD700',
                        'color': 'black',
                    }
                ],
                page_size=10
            )
        ], style={'textAlign': 'center'})
    else:
        return dcc.Graph(id="scatter", config={"displayModeBar": False}, style={
            "width": "1000px",
            "height": "750px",
            "margin": "auto"
        })

@app.callback(
    Output("filter-collapse", "is_open"),
    Input("toggle-filters", "n_clicks"),
    prevent_initial_call=True
)
def toggle_filter_collapse(n_clicks):
    return n_clicks % 2 == 1 if n_clicks else False

@app.callback(
    Output("scatter", "figure"),
    Input("x-metric", "value"),
    Input("y-metric", "value"),
    Input("z-metric", "value"),
    Input("mode-selector", "value"),
    Input("sample-percent", "value"),
    Input("race-filter", "value"),
    Input("class-filter", "value"),
    Input("spec-filter", "value"),
    Input("buildid-filter", "value"),
    Input("talent-filter", "value"),
    Input("sidebar-legend", "value"),
    Input("legend-talent-limit", "value")
)
def build_chart(x_metric, y_metric, z_metric, mode, sample_percent, race, cls, spec, buildid, talent, sidebar, legend_limit):
    df_copy = df.copy()
    # Apply filters
    if race:
        df_copy = df_copy[df_copy["race"].isin(race)]
    if cls:
        df_copy = df_copy[df_copy["class"].isin(cls)]
    if spec:
        df_copy = df_copy[df_copy["spec"].isin(spec)]
    if buildid:
        df_copy = df_copy[df_copy["builds"].isin(buildid)]
    # Talent filter: show builds where any selected talent is nonzero (OR logic)
    if talent:
        if isinstance(talent, list) and talent:
            mask = (df_copy[talent] != 0).any(axis=1)
            df_copy = df_copy[mask]
        elif isinstance(talent, str):
            df_copy = df_copy[df_copy[talent] != 0]
    
    # Reset index to ensure alignment after filtering
    df_copy = df_copy.reset_index(drop=True)

    # Percentage-based random sampling
    if sample_percent is None:
        sample_percent = 100
    sample_percent = max(0, min(100, sample_percent))
    if sample_percent == 0 and not df_copy.empty:
        # Keep at least one point to avoid empty figure
        df_copy = df_copy.sample(n=1, random_state=42)
    elif sample_percent < 100:
        n_keep = int(len(df_copy) * (sample_percent / 100.0))
        n_keep = max(1, n_keep)
        df_copy = df_copy.sample(n=n_keep, random_state=42)

    # Outlier logic removed per new specification
    
    # Create active_talents column for coloring
    def get_active_talents(row):
        return ','.join([t for t in talent_fields if row[t] != 0])
    df_copy['active_talents'] = df_copy.apply(get_active_talents, axis=1)
    df_copy["hover"] = df_copy.apply(lambda row: (
        "<b style='color:black;'>" + wrap_text(str(row["talent_str"])) + "</b><br><br>" +
        f"Race: <b>{row['race']}</b><br>Class: <b>{row['class']}</b><br>Spec: <b>{row['spec']}</b><br>" +
        f"Trinket 1: <b>{row['trinket_1']}</b><br>Trinket 2: <b>{row['trinket_2']}</b><br>" +
        f"{x_metric}: " + clean_decimal(row[x_metric]) + "<br>" +
        f"{y_metric}: " + clean_decimal(row[y_metric]) +
        (f"<br>{z_metric}: " + clean_decimal(row[z_metric]) if mode == "3d" else "")
    ), axis=1)

    hover_data = {"build_id": True, "talent_str": False, "race": True, "class": True, "spec": True}
    custom_data = ["talent_str", "race", "class", "spec"]
    # Correct color map selection; earlier version incorrectly mapped 'builds' legend using build_id keys.
    if sidebar == "builds":
        color_map = builds_color_map
    elif sidebar == "build_id":  # not an exposed option, defensive
        color_map = file_color_map
    else:
        color_map = None
    labels_map = {"builds": "Build", "talent": "Talent", "spec": "Spec", "race": "Race"}
    color_col = sidebar
    # If sidebar is 'talent', create a dedicated factor for legend
    if sidebar == "talent":
        if not df_copy.empty:
            # Only melt if we have filtered talent data
            if talent:  # Only process if specific talents are selected
                variable_talents = talent if isinstance(talent, list) else [talent]
            else:
                # Get only talents that have any non-zero values in the filtered data
                variable_talents = [t for t in talent_fields if t in df_copy.columns and df_copy[t].sum() > 0]
                # Limit number of legend entries with configurable cap
                if legend_limit and len(variable_talents) > int(legend_limit):
                    talent_counts = {t: df_copy[t].sum() for t in variable_talents}
                    variable_talents = sorted(variable_talents, key=lambda t: talent_counts[t], reverse=True)[:int(legend_limit)]
            
            if variable_talents:
                # Only include necessary columns for melting
                axis_cols = [x_metric, y_metric]
                if mode == "3d":
                    axis_cols.append(z_metric)
                # Preserve outlier flag through melt so highlight can still overlay (though we use base_outliers for overlay)
                id_vars = axis_cols + ['build_id', 'talent_str', 'race', 'class', 'spec']
                
                # Pre-filter to only rows that have at least one of our variable talents
                talent_mask = df_copy[variable_talents].any(axis=1)
                df_filtered = df_copy.loc[talent_mask, id_vars + variable_talents].copy()
                
                if not df_filtered.empty:
                    # Melt only the variable talents
                    melted = pd.melt(
                        df_filtered,
                        id_vars=id_vars,
                        value_vars=variable_talents,
                        var_name='talent_entry',
                        value_name='talent_value'
                    )
                    # Keep only non-zero talents
                    melted = melted[melted['talent_value'] != 0]
                    
                    if not melted.empty:
                        color_col = 'talent_entry'
                        # Simplified hover text
                        melted["hover"] = melted.apply(lambda row: (
                            f"<b>{row['talent_entry']}</b><br>" +
                            f"{x_metric}: {clean_decimal(row[x_metric])}<br>" +
                            f"{y_metric}: {clean_decimal(row[y_metric])}" +
                            (f"<br>{z_metric}: {clean_decimal(row[z_metric])}" if mode == "3d" else "")
                        ), axis=1)
                        df_copy = melted
                        
                        # ...existing plotting code continues...
    
    if mode == "3d":
        fig = px.scatter_3d(
            df_copy,
            x=x_metric, y=y_metric, z=z_metric,
            color=color_col,
            color_discrete_map=color_map,
            hover_data=hover_data,
            custom_data=custom_data,
            labels={color_col: labels_map.get(sidebar, color_col)},
            template="plotly_dark",
            opacity=0.65
        )
    else:
        fig = px.scatter(
            df_copy,
            x=x_metric, y=y_metric,
            color=color_col,
            color_discrete_map=color_map,
            hover_data=hover_data,
            custom_data=custom_data,
            labels={color_col: labels_map.get(sidebar, color_col)},
            template="plotly_dark",
            opacity=0.65
        )

    # No outlier overlay

    # Use per-point text for hover; previous code passed a Series directly as hovertemplate causing potential mismatches.
    # Correct per-trace hover text assignment.
    # Previous implementation applied the full dataframe's hover Series to every trace,
    # causing mismatches (e.g., wrong target counts appearing in tooltips).
    for tr in fig.data:
        if color_col in df_copy.columns and getattr(tr, 'name', None) is not None:
            subset = df_copy[df_copy[color_col] == tr.name]
        else:
            subset = df_copy
        tr.hovertemplate = "%{text}"
        tr.text = subset["hover"].tolist()
        # Ensure marker styling retained per trace
        if hasattr(tr, 'marker'):
            tr.marker.update(size=6, line=dict(width=0.5, color="#444"))

    # Debug: verify plotted coordinates match dataframe values per trace (helps investigate perceived 'offset' points).
    try:
        mismatch_total = 0
        checked_total = 0
        if color_col in df_copy.columns:
            for trace in fig.data:
                # trace.name may be None for single-trace plots
                trace_name = getattr(trace, 'name', None)
                if trace_name is None:
                    # Single trace: compare whole dataframe
                    x_vals = np.array(trace.x) if hasattr(trace, 'x') else np.array([])
                    y_vals = np.array(trace.y) if hasattr(trace, 'y') else np.array([])
                    z_vals = np.array(getattr(trace, 'z', [])) if mode == '3d' else None
                    df_subset = df_copy
                else:
                    df_subset = df_copy[df_copy[color_col] == trace_name]
                    x_vals = np.array(trace.x) if hasattr(trace, 'x') else np.array([])
                    y_vals = np.array(trace.y) if hasattr(trace, 'y') else np.array([])
                    z_vals = np.array(getattr(trace, 'z', [])) if mode == '3d' else None
                # Align lengths (Plotly may drop NAs; ensure we compare min length)
                n = min(len(df_subset), len(x_vals), len(y_vals))
                if n == 0:
                    continue
                checked_total += n
                x_df = df_subset[x_metric].to_numpy()[:n]
                y_df = df_subset[y_metric].to_numpy()[:n]
                mismatches = (np.abs(x_df - x_vals[:n]) > 1e-6) | (np.abs(y_df - y_vals[:n]) > 1e-6)
                if mode == '3d' and z_vals is not None:
                    z_df = df_subset[z_metric].to_numpy()[:n]
                    mismatches = mismatches | (np.abs(z_df - z_vals[:n]) > 1e-6)
                mismatch_total += int(mismatches.sum())
        if checked_total > 0:
            fig.add_annotation(
                text=f"Debug: coord mismatches {mismatch_total}/{checked_total}",
                xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False,
                font=dict(size=10, color="#ffc8dd"), bgcolor="#2b2b3c", opacity=0.7
            )
    except Exception as _dbg_err:
        fig.add_annotation(
            text=f"Debug check error: {_dbg_err}",
            xref="paper", yref="paper", x=0.01, y=0.99, showarrow=False,
            font=dict(size=10, color="#ffc8dd"), bgcolor="#662222", opacity=0.7
        )

    legend_title = labels_map.get(sidebar, sidebar.capitalize())
    fig.update_layout(
        legend_title_text=legend_title,
        font=dict(color="#f2b5d4"),
        plot_bgcolor="#1e1e2f",
        paper_bgcolor="#1e1e2f",
        margin=dict(l=30, r=30, t=60, b=60),
        height=750,
        autosize=True
    )
    # If sidebar is 'talent' and no builds, add annotation (redundant if already handled above, but ensures visibility)
    if sidebar == "talent" and (df_copy.empty or (isinstance(df_copy, pd.DataFrame) and 'talent_entry' in df_copy.columns and df_copy['talent_entry'].eq('None').all())):
        fig.add_annotation(
            text="No builds found for selected filters",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="#ffc8dd"),
            bgcolor="#2b2b3c"
        )
    # Add bulk legend toggle buttons for talent traces (Show All / Hide All)
    if sidebar == "talent" and 'talent_entry' in df_copy.columns and not df_copy.empty:
        # Determine indices of talent traces
        talent_trace_names = set(df_copy['talent_entry'].unique())
        trace_indices = [i for i, tr in enumerate(fig.data) if getattr(tr, 'name', None) in talent_trace_names]
        if trace_indices:
            # Build visibility arrays for toggle (args hides, args2 shows)
            hide_visibility = []
            show_visibility = []
            for i, tr in enumerate(fig.data):
                if i in trace_indices:
                    hide_visibility.append('legendonly')
                    show_visibility.append(True)
                else:
                    # Preserve non-talent traces visibility state
                    cur_vis = getattr(tr, 'visible', True)
                    hide_visibility.append(cur_vis)
                    show_visibility.append(cur_vis)
            fig.update_layout(
                updatemenus=[dict(
                    type='buttons',
                    direction='right',  # valid values: left/right/up/down
                    x=0.0,
                    y=1.12,
                    pad=dict(r=5, t=5),
                    showactive=True,
                    bgcolor='#2b2b3c',
                    bordercolor='#444444',
                    buttons=[
                        dict(label='Toggle Talents',
                             method='update',
                             args=[{'visible': hide_visibility}],
                             args2=[{'visible': show_visibility}],
                             execute=True
                        )
                    ]
                )]
            )
    return fig

@app.callback(
    Output("heatmap-graph", "figure"),
    Input("heatmap-type", "value"),
    Input("significance-threshold", "value"),
    Input("flip-threshold", "value"),
    Input("heatmap-metric", "value")
)
def build_heatmap(heatmap_type, threshold, flip, metric):
    # Isolated heatmap logic: do not use any shared variables from 2D/3D chart
    df_heatmap = df.copy()
    stat_fields = ["haste", "crit", "vers", "mastery"]
    # Detect talent fields for heatmap: all columns after 'trinket_2', excluding dashboard fields
    trinket2_idx = list(df_heatmap.columns).index('trinket_2') if 'trinket_2' in df_heatmap.columns else None
    heatmap_talent_fields = list(df_heatmap.columns)[trinket2_idx+1:] if trinket2_idx is not None else []
    exclude_dashboard = {'cluster', 'pc1', 'pc2'}
    heatmap_talent_fields = [t for t in heatmap_talent_fields if t not in exclude_dashboard]
    # --- Robust node aggregation using subTreeNodes entries ---
    import sys
    hero_tree_map = {}  # {tree_name: [non-choice hero node names]}
    hero_tree_nodes_set = set()
    choice_nodes_set = set()
    class_nodes_set = set()
    spec_nodes_set = set()
    all_node_names = set()
    for spec_block in talents_raw:
        if not isinstance(spec_block, dict):
            continue
        # Class nodes
        for node in spec_block.get("classNodes", []):
            for e in node.get("entries", []):
                name = e.get("name")
                if not name:
                    continue
                all_node_names.add(name)
                if node.get("type") == "choice":
                    choice_nodes_set.add(name)
                else:
                    class_nodes_set.add(name)
        # Spec nodes
        for node in spec_block.get("specNodes", []):
            for e in node.get("entries", []):
                name = e.get("name")
                if not name:
                    continue
                all_node_names.add(name)
                if node.get("type") == "choice":
                    choice_nodes_set.add(name)
                else:
                    spec_nodes_set.add(name)
        # Hero nodes
        hero_nodes_by_id = {node["id"]: node for node in spec_block.get("heroNodes", []) if "id" in node}
        hero_nodes_name_by_id = {}
        for node in spec_block.get("heroNodes", []):
            for e in node.get("entries", []):
                name = e.get("name")
                if not name:
                    continue
                all_node_names.add(name)
                if node.get("type") == "choice":
                    choice_nodes_set.add(name)
                else:
                    hero_tree_nodes_set.add(name)
                hero_nodes_name_by_id[node["id"]] = (name, node.get("type"))
        # Subtree aggregation for hero trees (using entries)
        for subtree in spec_block.get("subTreeNodes", []):
            if "entries" in subtree:
                for entry in subtree["entries"]:
                    tree_name = entry.get("name")
                    node_ids = entry.get("nodes", [])
                    for node_id in node_ids:
                        node = hero_nodes_by_id.get(node_id)
                        if not node:
                            continue
                        for e in node.get("entries", []):
                            name = e.get("name")
                            if not name:
                                continue
                            if node.get("type") == "choice":
                                choice_nodes_set.add(name)
                            else:
                                hero_tree_map.setdefault(tree_name, []).append(name)
                                hero_tree_nodes_set.add(name)
        # Legacy support: if subtree has top-level nodes (rare)
        if "nodes" in subtree and subtree.get("type") == "subtree":
            tree_name = subtree.get("name")
            node_ids = subtree.get("nodes", [])
            for node_id in node_ids:
                node = hero_nodes_by_id.get(node_id)
                if not node:
                    continue
                for e in node.get("entries", []):
                    name = e.get("name")
                    if not name:
                        continue
                    if node.get("type") == "choice":
                        choice_nodes_set.add(name)
                    else:
                        hero_tree_map.setdefault(tree_name, []).append(name)
                        hero_tree_nodes_set.add(name)
    # Only keep hero trees with nodes present in the dataframe
    hero_tree_map = {k: v for k, v in hero_tree_map.items() if any(n in heatmap_talent_fields for n in v)}
    # Print diagnostics
    # print("\n[HEATMAP DIAGNOSTICS]")
    # print("hero_tree_map:", hero_tree_map)
    # print("class_nodes_set:", sorted(list(class_nodes_set)))
    # print("spec_nodes_set:", sorted(list(spec_nodes_set)))
    # print("choice_nodes_set:", sorted(list(choice_nodes_set)))
    # print("hero_tree_nodes_set:", sorted(list(hero_tree_nodes_set)))
    # print("All node names:", sorted(list(all_node_names)))
    sys.stdout.flush()
    # Build the list of choice nodes present in the dataframe
    choice_selected = [t for t in heatmap_talent_fields if t in choice_nodes_set]
    # Only show regular talents (not hero tree non-choice nodes), hero tree consolidated entries, and choice nodes
    regular_talents = [t for t in heatmap_talent_fields if t not in hero_tree_nodes_set and t not in choice_nodes_set]
    # For heatmap, use: [regular_talents...] + [tree_name_1, tree_name_2, ...] + [choice_selected...]
    consolidated_talents = regular_talents + list(hero_tree_map.keys()) + choice_selected
    # Aggressive consolidation: strictly collapse all non-choice hero nodes into a single binary column per tree
    for tree_name, node_list in hero_tree_map.items():
        present_nodes = [n for n in node_list if n in df_heatmap.columns]
        df_heatmap[tree_name] = (df_heatmap[present_nodes].sum(axis=1) > 0).astype(int) if present_nodes else 0
    # Drop all individual hero node columns from heatmap logic
    df_heatmap = df_heatmap.drop(columns=[n for n in hero_tree_nodes_set if n in df_heatmap.columns], errors="ignore")
    # Drop any columns that are all zero (except hero tree names, choice nodes, and regular talents)
    keep_cols = set(regular_talents + list(hero_tree_map.keys()) + choice_selected + stat_fields + ["build_id", "talent_str", "race", "class", "spec"])
    drop_cols = [col for col in df_heatmap.columns if col not in keep_cols and df_heatmap[col].nunique() == 1 and df_heatmap[col].iloc[0] == 0]
    df_heatmap = df_heatmap.drop(columns=drop_cols, errors="ignore")
    # Sort hero tree names to top of talent list for heatmap axes
    sorted_hero_trees = sorted(hero_tree_map.keys())
    sorted_regular_talents = sorted(regular_talents)
    sorted_choice_nodes = sorted(choice_selected)
    consolidated_talents = sorted_hero_trees + sorted_regular_talents + sorted_choice_nodes
    # Calculate mean metric value for each talent (when active)
    metric_vals = {}
    for t in consolidated_talents:
        if t in df_heatmap.columns:
            mask = df_heatmap[t] != 0
            metric_vals[t] = df_heatmap.loc[mask, metric].mean() if mask.any() else np.nan
        else:
            metric_vals[t] = np.nan
    # Filter talents above or below threshold depending on flip
    ref_val = np.nanmax([v for v in metric_vals.values() if not np.isnan(v)])
    cutoff = (threshold / 100.0) * ref_val
    if flip:
        filtered_talents = [t for t in consolidated_talents if t in metric_vals and not np.isnan(metric_vals[t]) and metric_vals[t] <= cutoff]
    else:
        filtered_talents = [t for t in consolidated_talents if t in metric_vals and not np.isnan(metric_vals[t]) and metric_vals[t] >= cutoff]
    # Sort by impact (mean metric value, descending)
    significant_talents = sorted(filtered_talents, key=lambda t: metric_vals[t], reverse=True)
    # Custom magenta-pink-purple-blue colorscale
    custom_colorscale = [
        [0.0, '#1a1aff'],   # deep blue
        [0.2, '#6a4fb6'],  # purple
        [0.5, '#d16ba5'],  # pink
        [0.8, '#e75480'],  # magenta-pink
        [1.0, '#ff00cc']   # magenta
    ]
    # Talent vs Talent (3D surface for any metric)
    if heatmap_type == "talent_talent" and metric in ["dps", "priority_dps", "StdDev", "range_pct_dps"]:
        import plotly.graph_objects as go
        mat = np.zeros((len(significant_talents), len(significant_talents)), dtype=float)
        global_mean_val = df_heatmap[metric].mean()
        for i, t1 in enumerate(significant_talents):
            mask1 = df_heatmap[t1] != 0
            for j, t2 in enumerate(significant_talents):
                mask2 = df_heatmap[t2] != 0
                mask = mask1 & mask2
                if mask.any():
                    mat[i, j] = df_heatmap.loc[mask, metric].mean()
                else:
                    mat[i, j] = global_mean_val
        min_val = np.nanmin(mat)
        max_val = np.nanmax(mat)
        mat_pct = ((mat - min_val) / (max_val - min_val) * 100) if max_val > min_val else mat
        fig = go.Figure(data=[go.Surface(
            z=mat,
            x=np.arange(len(significant_talents)),
            y=np.arange(len(significant_talents)),
            surfacecolor=mat_pct,
            colorscale=custom_colorscale,
            colorbar=dict(title=f"% of Max {metric}", tickvals=[0, 25, 50, 75, 100]),
            cmin=0,
            cmax=100,
            showscale=True
        )])
        fig.update_layout(
            title=f"Talent Synergy (Avg {metric} Surface, threshold ≥ {threshold}%)",
            scene=dict(
                xaxis=dict(title="Talent", tickvals=np.arange(len(significant_talents)), ticktext=significant_talents, color="#ffc8dd", backgroundcolor="rgba(30,30,47,0.85)"),
                yaxis=dict(title="Talent", tickvals=np.arange(len(significant_talents)), ticktext=significant_talents, color="#ffc8dd", backgroundcolor="rgba(30,30,47,0.85)"),
                zaxis=dict(title=f"Avg {metric}", color="#ffc8dd", backgroundcolor="rgba(30,30,47,0.85)")
            ),
            font=dict(color="#f2b5d4"),
            margin=dict(l=30, r=30, t=60, b=60),
            height=700,
            paper_bgcolor="#1e1e2f",
            plot_bgcolor="#1e1e2f"
        )
        return fig
    # Talent vs Stat (fully duplicated logic)
    elif heatmap_type == "talent_stat" and metric in ["dps", "priority_dps", "StdDev", "range_pct_dps"]:
        import plotly.graph_objects as go
        mat = np.zeros((len(significant_talents), len(stat_fields)), dtype=float)
        global_mean_val = df_heatmap[metric].mean()
        for i, t1 in enumerate(significant_talents):
            mask1 = df_heatmap[t1] != 0
            for j, s in enumerate(stat_fields):
                mask2 = df_heatmap[s] != 0 if s in df_heatmap.columns else pd.Series([False]*len(df_heatmap))
                mask = mask1 & mask2 if not mask2.empty else mask1
                if mask.any():
                    mat[i, j] = df_heatmap.loc[mask, metric].mean()
                else:
                    mat[i, j] = global_mean_val
        min_val = np.nanmin(mat)
        max_val = np.nanmax(mat)
        mat_pct = ((mat - min_val) / (max_val - min_val) * 100) if max_val > min_val else mat
        fig = go.Figure(data=[go.Surface(
            z=mat,
            x=np.arange(len(stat_fields)),
            y=np.arange(len(significant_talents)),
            surfacecolor=mat_pct,
            colorscale=custom_colorscale,
            colorbar=dict(title=f"% of Max {metric}", tickvals=[0, 25, 50, 75, 100]),
            cmin=0,
            cmax=100,
            showscale=True
        )])
        fig.update_layout(
            title=f"Talent vs Stat Synergy (Avg {metric} Surface, threshold ≥ {threshold}%)",
            scene=dict(
                xaxis=dict(title="Stat", tickvals=np.arange(len(stat_fields)), ticktext=stat_fields, color="#ffc8dd", backgroundcolor="rgba(30,30,47,0.85)"),
                yaxis=dict(title="Talent", tickvals=np.arange(len(significant_talents)), ticktext=significant_talents, color="#ffc8dd", backgroundcolor="rgba(30,30,47,0.85)"),
                zaxis=dict(title=f"Avg {metric}", color="#ffc8dd", backgroundcolor="rgba(30,30,47,0.85)")
            ),
            font=dict(color="#f2b5d4"),
            margin=dict(l=30, r=30, t=60, b=60),
            height=700,
            paper_bgcolor="#1e1e2f",
            plot_bgcolor="#1e1e2f"
        )
        return fig
    # Stat vs Stat (fully duplicated logic)
    elif heatmap_type == "stat_stat" and metric in ["dps", "priority_dps", "StdDev", "range_pct_dps"]:
        import plotly.graph_objects as go
        mat = np.zeros((len(stat_fields), len(stat_fields)), dtype=float)
        global_mean_val = df_heatmap[metric].mean()
        for i, s1 in enumerate(stat_fields):
            mask1 = df_heatmap[s1] != 0 if s1 in df_heatmap.columns else pd.Series([False]*len(df_heatmap))
            for j, s2 in enumerate(stat_fields):
                mask2 = df_heatmap[s2] != 0 if s2 in df_heatmap.columns else pd.Series([False]*len(df_heatmap))
                mask = mask1 & mask2 if not mask2.empty else mask1
                if mask.any():
                    mat[i, j] = df_heatmap.loc[mask, metric].mean()
                else:
                    mat[i, j] = global_mean_val
        min_val = np.nanmin(mat)
        max_val = np.nanmax(mat)
        mat_pct = ((mat - min_val) / (max_val - min_val) * 100) if max_val > min_val else mat
        fig = go.Figure(data=[go.Surface(
            z=mat,
            x=np.arange(len(stat_fields)),
            y=np.arange(len(stat_fields)),
            surfacecolor=mat_pct,
            colorscale=custom_colorscale,
            colorbar=dict(title=f"% of Max {metric}", tickvals=[0, 25, 50, 75, 100]),
            cmin=0,
            cmax=100,
            showscale=True
        )])
        fig.update_layout(
            title=f"Stat vs Stat Synergy (Avg {metric} Surface, threshold ≥ {threshold}%)",
            scene=dict(
                xaxis=dict(title="Stat", tickvals=np.arange(len(stat_fields)), ticktext=stat_fields, color="#ffc8dd", backgroundcolor="rgba(30,30,47,0.85)"),
                yaxis=dict(title="Stat", tickvals=np.arange(len(stat_fields)), ticktext=stat_fields, color="#ffc8dd", backgroundcolor="rgba(30,30,47,0.85)"),
                zaxis=dict(title=f"Avg {metric}", color="#ffc8dd", backgroundcolor="rgba(30,30,47,0.85)")
            ),
            font=dict(color="#f2b5d4"),
            margin=dict(l=30, r=30, t=60, b=60),
            height=700,
            paper_bgcolor="#1e1e2f",
            plot_bgcolor="#1e1e2f"
        )
        return fig
    elif heatmap_type == "talent_dps":
        mat = np.zeros((len(significant_talents), 1), dtype=float)
        for i, t in enumerate(significant_talents):
            mask = df_heatmap[t] != 0
            mat[i, 0] = df_heatmap.loc[mask, metric].mean() if mask.any() else np.nan
        min_val = np.nanmin(mat)
        max_val = np.nanmax(mat)
        mat_pct = ((mat - min_val) / (max_val - min_val) * 100) if max_val > min_val else mat
        fig = px.imshow(
            mat_pct,
            x=[metric],
            y=significant_talents,
            color_continuous_scale=[
                [0.0, '#1a1aff'],   # deep blue
                [0.2, '#6a4fb6'],  # purple
                [0.5, '#d16ba5'],  # pink
                [0.8, '#e75480'],  # magenta-pink
                [1.0, '#ff00cc']   # magenta
            ],
            labels=dict(x=metric, y="Talent", color=f"% of Max {metric}"),
            title=f"Talent vs {metric} (% of Max {metric}, StdDev ≥ {threshold})"
        )
        fig.update_layout(
            font=dict(color="#f2b5d4"),
            plot_bgcolor="#1e1e2f",
            paper_bgcolor="#1e1e2f",
            margin=dict(l=30, r=30, t=60, b=60),
            height=700,
            autosize=True
        )
        fig.update_coloraxes(cmin=0, cmax=100, colorbar_title="% of Max")
        return fig
    elif heatmap_type == "stat_dps":
        mat = np.zeros((len(stat_fields), 1), dtype=float)
        for i, s in enumerate(stat_fields):
            mat[i, 0] = df_heatmap[s].mean()
        min_val = np.nanmin(mat)
        max_val = np.nanmax(mat)
        mat_pct = ((mat - min_val) / (max_val - min_val) * 100) if max_val > min_val else mat
        fig = px.imshow(
            mat_pct,
            x=[metric],
            y=stat_fields,
            color_continuous_scale=[
                [0.0, '#1a1aff'],   # deep blue
                [0.2, '#6a4fb6'],  # purple
                [0.5, '#d16ba5'],  # pink
                [0.8, '#e75480'],  # magenta-pink
                [1.0, '#ff00cc']   # magenta
            ],
            labels=dict(x=metric, y="Stat", color=f"% of Max {metric}"),
            title=f"Stat vs {metric} (% of Max {metric})"
        )
        fig.update_layout(
            font=dict(color="#f2b5d4"),
            plot_bgcolor="#1e1e2f",
            paper_bgcolor="#1e1e2f",
            margin=dict(l=30, r=30, t=60, b=60),
            height=700,
            autosize=True
        )
        fig.update_coloraxes(cmin=0, cmax=100, colorbar_title="% of Max")
        return fig
    fig = px.imshow(np.zeros((2,2)), x=["A","B"], y=["A","B"], color_continuous_scale="Viridis",
                    labels=dict(x="", y="", color=""), title="No Data")
    fig.update_layout(
        font=dict(color="#f2b5d4"),
        plot_bgcolor="#1e1e2f",
        paper_bgcolor="#1e1e2f",
        margin=dict(l=30, r=30, t=60, b=60),
        height=700,
        autosize=True
    )
    return fig

@app.callback(
    Output("sample-percent", "value"),
    Input("race-filter", "value"),
    Input("class-filter", "value"),
    Input("spec-filter", "value"),
    Input("buildid-filter", "value"),
    Input("talent-filter", "value"),
    State("sample-percent", "value")
)
def auto_set_sample_percent(race, cls, spec, buildid, talent, current_value):
    df_copy = df.copy()
    # Apply same filtering logic used in build_chart
    if race:
        df_copy = df_copy[df_copy["race"].isin(race)]
    if cls:
        df_copy = df_copy[df_copy["class"].isin(cls)]
    if spec:
        df_copy = df_copy[df_copy["spec"].isin(spec)]
    if buildid:
        df_copy = df_copy[df_copy["builds"].isin(buildid)]
    if talent:
        if isinstance(talent, list) and talent:
            mask = (df_copy[talent] != 0).any(axis=1)
            df_copy = df_copy[mask]
        elif isinstance(talent, str):
            df_copy = df_copy[df_copy[talent] != 0]

    n_rows = len(df_copy)
    if n_rows < 2000:
        recommended = 100
    elif n_rows > 10000:
        recommended = 10
    else:
        recommended = 50

    # Avoid redundant updates
    if current_value == recommended:
        return dash.no_update
    return recommended

@app.callback(
    Output("copy-output", "children"),
    Output("talent-box", "value"),
    Input("scatter", "clickData")
)
def update_output(clickData):
    if not clickData:
        return "", ""
    talent_str = clickData["points"][0]["customdata"][0]
    return "📋 Talent string selected!", talent_str

# Point/outlier readout updater
@app.callback(
    Output("sample-readout", "children"),
    Input("sample-percent", "value"),
    Input("x-metric", "value"),
    Input("y-metric", "value"),
    Input("z-metric", "value"),
    Input("mode-selector", "value")
)
def update_point_readout(sample_percent, x_metric, y_metric, z_metric, mode):
    try:
        base_n = len(df)
        if sample_percent is None:
            sample_percent = 100
        sample_percent = max(0, min(100, sample_percent))
        sampled_n = max(1, int(base_n * (sample_percent / 100.0))) if base_n > 0 else 0
        return f"Total: {base_n} | Sampled: {sampled_n} ({sample_percent}%)"
    except Exception as e:
        return f"Readout error: {e}" 

@app.callback(
    Output("graph-title", "children"),
    Input("x-metric", "value"),
    Input("y-metric", "value"),
    Input("z-metric", "value"),
    Input("mode-selector", "value")
)
def update_title(x_metric, y_metric, z_metric, mode):
    aliases = {
        "dps": "DPS",
        "priority_dps": "Priority DPS",
        "StdDev": "DPS Std Dev",
        "range_pct_dps": "DPS Range %",
        "crit": "Crit Rating",
        "haste": "Haste Rating",
        "vers": "Versatility Rating",
        "mastery": "Mastery Rating",
        "targets": "# Targets"
    }
    x_label = aliases.get(x_metric, x_metric)
    y_label = aliases.get(y_metric, y_metric)
    if mode == "3d":
        z_label = aliases.get(z_metric, z_metric)
        return f"{x_label} vs {y_label} vs {z_label}"
    return f"{x_label} vs {y_label}"


@app.callback(
    Output("z-axis-block", "style"),
    Input("mode-selector", "value")
)
def toggle_z_visibility(mode):
    if mode == "3d":
        return {"display": "block", "marginRight": "30px"}
    else:
        return {"display": "none"}

# Add callback for best profiles table
@app.callback(
    Output('best-profiles-table', 'data'),
    [Input('update-rankings-btn', 'n_clicks')],
    [State('dps-weight', 'value'),
     State('priority-dps-weight', 'value')] +
    [State(f'target-{target}-weight', 'value') for target in sorted(df['targets'].unique())]
)
def update_best_profiles(n_clicks, dps_weight, priority_dps_weight, *target_weights):
    if n_clicks is None:
        return []
    
    try:
        df_profiles = df.copy()
        
        # Get unique target counts and create weight mapping
        unique_targets = sorted(df_profiles['targets'].unique())
        target_weight_map = {}
        
        for i, target_count in enumerate(unique_targets):
            weight = target_weights[i] if i < len(target_weights) and target_weights[i] is not None else 1.0
            target_weight_map[target_count] = weight
        
        # Group by profile (talent_str, trinkets, stats)
        profile_groups = df_profiles.groupby(['talent_str', 'trinket_1', 'trinket_2', 'crit', 'haste', 'vers', 'mastery'])
        
        profile_scores = []
        
        for profile_key, profile_data in profile_groups:
            total_weighted_score = 0
            total_weight = 0
            
            for _, row in profile_data.iterrows():
                target_weight = target_weight_map.get(row['targets'], 1.0)
                
                # Normalize DPS values within target group
                target_data = df_profiles[df_profiles['targets'] == row['targets']]
                max_dps = target_data['dps'].max()
                max_priority_dps = target_data['priority_dps'].max()
                
                if max_dps > 0 and max_priority_dps > 0:
                    normalized_dps = row['dps'] / max_dps
                    normalized_priority_dps = row['priority_dps'] / max_priority_dps
                    
                    # Calculate weighted score
                    score = (normalized_dps * (dps_weight or 0) + 
                            normalized_priority_dps * (priority_dps_weight or 0))
                    
                    total_weighted_score += score * target_weight
                    total_weight += target_weight
            
            if total_weight > 0:
                avg_weighted_score = total_weighted_score / total_weight
                
                profile_scores.append({
                    'talent_str': profile_key[0],
                    'trinket_1': profile_key[1],
                    'trinket_2': profile_key[2],
                    'crit': profile_key[3],
                    'haste': profile_key[4],
                    'vers': profile_key[5],
                    'mastery': profile_key[6],
                    'weighted_score': avg_weighted_score
                })
        
        # Sort by weighted score and get top 10
        profile_scores.sort(key=lambda x: x['weighted_score'], reverse=True)
        top_profiles = profile_scores[:10]
        
        # Add rank
        for i, profile in enumerate(top_profiles):
            profile['rank'] = i + 1
        
        return top_profiles
        
    except Exception as e:
        print(f"Error calculating best profiles: {e}")
        return []


if __name__ == "__main__":
    # Avoid Werkzeug reloader crash on UNC paths (e.g., \\SERVER\Share)
    # This preserves dashboard behavior and only disables the file reloader when necessary.
    import os
    from pathlib import Path
    try:
        script_dir = str(Path(__file__).resolve().parent)
        is_unc = script_dir.startswith("\\\\")
    except Exception:
        is_unc = False
    app.run(debug=True, use_reloader=not is_unc)