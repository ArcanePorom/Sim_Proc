import json
import os
from typing import Dict

# Optional merged profileset helper using generate_talents.py APIs
try:
    from generate_talents import TalentJSON, tokenize
except Exception:
    TalentJSON = None
    tokenize = None

def _talent_str(build: Dict[int, int]) -> str:
    """Format a build dict as SimC talent string id:points joined by '/'."""
    return '/'.join(f"{tid}:{pts}" for tid, pts in build.items())

def generate_merged_class_spec_profilesets(talents_json_path: str, class_name: str, spec_name: str, base_name: str) -> str:
    """
    Generate merged profileset lines (class_talents + spec_talents) without modifying generate_talents.py.
    Returns a single string with multiple copy= blocks.

    Example usage:
        text = generate_merged_class_spec_profilesets('talents.json', 'mage', 'arcane', 'arc_base')
    """
    if TalentJSON is None or tokenize is None:
        return ""
    tj = TalentJSON.from_file(talents_json_path)
    # Access specialization via tokenized attributes: tj.<class>.<spec>
    cls_attr = tokenize(class_name)
    spc_attr = tokenize(spec_name)
    try:
        spec = getattr(getattr(tj, cls_attr), spc_attr)
    except Exception:
        return ""
    lines = []
    idx = 1
    for class_build in spec.class_.generate_builds():
        for spec_build in spec.spec.generate_builds():
            name = f"{base_name}_{idx}"
            block = [f"copy={name},base",
                     f"class_talents={_talent_str(class_build)}",
                     f"spec_talents={_talent_str(spec_build)}"]
            lines.append("\n".join(block))
            idx += 1
    return "\n".join(lines) + ("\n" if lines else "")
import dash
from dash import html, dcc, Input, Output, State, ctx
import dash_bootstrap_components as dbc

# Load talent data (resolve relative to this script directory)
script_dir = os.path.dirname(os.path.abspath(__file__))
talents_path = os.path.join(script_dir, "talents.json")
default_class_name = None
default_spec_name = None
try:
    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(script_dir, "config.ini"))
    default_class_name = cfg.get("talents", "default_class", fallback=None)
    default_spec_name = cfg.get("talents", "default_spec", fallback=None)
except Exception:
    pass
with open(talents_path, encoding="utf-8") as f:
    raw = json.load(f)

class_map = {}
for entry in raw:
    cid = entry["classId"]
    cname = entry["className"]
    if cid not in class_map:
        class_map[cid] = cname

class_dropdown_options = [{"label": name, "value": cid} for cid, name in sorted(class_map.items())]
# Resolve defaults from config by names to IDs; fall back to Mage/Arcane if not found
def _resolve_ids(class_name, spec_name):
    if not class_name or not spec_name:
        return None, None
    class_id = next((cid for cid, cname in class_map.items() if str(cname).lower() == str(class_name).lower()), None)
    spec_id = None
    if class_id is not None:
        for t in raw:
            if t.get("classId") == class_id and str(t.get("specName", "")).lower() == str(spec_name).lower():
                spec_id = t.get("specId")
                break
    return class_id, spec_id

cfg_class_id, cfg_spec_id = _resolve_ids(default_class_name, default_spec_name)
DEFAULT_CLASS_ID = cfg_class_id if cfg_class_id is not None else 8
DEFAULT_SPEC_ID = cfg_spec_id if cfg_spec_id is not None else 62

def simplify(val, base=600): return int(val / base)
def clean(name):
    # Robust: handle ints when using spellId identifiers
    s = str(name) if name is not None else ""
    return s.replace("’", "")
def get_icon_url(icon_id):
    overrides = { 
        "spell_frost_ring_of_frost": "spell_frost_ring-of-frost",
        "spell_frost_ice_shards": "spell_frost_ice-shards",
        "spell_priest_void_blast": "spell_priest_void-blast",
        "warlock_summon__voidlord": "warlock_summon_-voidlord",
        "spell_priest_void_flay": "spell_priest_void-flay",
        "spell_priest_power_word": "spell_priest_power-word",
        "inv_10_specialreagentfoozles_tuskclaw_ice": "inv_10_specialreagentfoozles_tuskclaw-ice",
        "spell_firefrost-orb": "spell_firefrost_orb",
        "achievement_firelands_raid_ragnaros": "inv_misc_petragnaros",
        "warlock__bloodstone": "warlock_-bloodstone",
        "spell_frostfire_orb": "spell_frostfire-orb",
        "spell_firefrost_orb": "spell_firefrost-orb",
        "spell_frost_piercing_chill": "spell_frost_piercing-chill",
        }
    icon_id = overrides.get(icon_id, icon_id)
    return f"https://wow.zamimg.com/images/wow/icons/large/{icon_id}.jpg"

nodes = []

from collections import defaultdict

# Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE], suppress_callback_exceptions=True)
app.title = "Talent Picker"

# Layout
app.layout = html.Div([
    html.Div(html.H2("Talent Picker", className="text-white mt-4"), style={"textAlign": "center"}),

    html.Div([
        html.Div([
            html.Label("Class", style={"color": "white"}),
            dcc.Dropdown(
                id="class-selector",
                options=class_dropdown_options,
                value=DEFAULT_CLASS_ID,
                clearable=False
            )
        ], style={"width": "300px"}),
        
      # html.Div(id="class-grid", style={}),
      # html.Div(id="spec-grid", style={}),

        html.Div([
            html.Label("Specialization", style={"color": "white"}),
            dcc.Dropdown(
                id="spec-selector",
                options=[],  # Will be set by callback
                value=None,
                clearable=False
            )
        ], style={"width": "300px", "marginLeft": "20px"}),

    ], style={"display": "flex", "justifyContent": "center", "marginBottom": "30px"}),

    dbc.Container([
        dcc.Store(id="nodes-store", data={}),
        dcc.Store(id="included-store", data=[]),
        dcc.Store(id="blocked-store", data=[]),
        dcc.Store(id="included2-store", data={}),
        dcc.Store(id="grid-dimensions", data={}),

        dcc.Store(id="hero-dimensions", data={"rows": 1, "cols": 1}),
        html.Div([
            html.H4("Class Tree", className="text-info text-center"),
            html.Div(id="class-count", className="text-light", style={"textAlign": "center", "marginBottom": "6px"}),
            html.Div([
                html.Div(id="class-grid", style={
                    "display": "grid",
                    "gap": "3px", "padding": "0px"
                })
            ], style={"display": "flex", "justifyContent": "center"})
        ], style={"flex": "0 0 auto", "textAlign": "center"}),

        html.Div([
            html.H4("Hero Tree", className="text-info text-center"),
            html.Div(id="hero-count", className="text-light", style={"textAlign": "center", "marginBottom": "6px"}),
html.Div([
    html.Div([
        html.H5(id="hero-name-a", className="text-warning text-center"),
        html.Div(
            html.Div(id="hero-grid-a"),
            id="hero-wrapper-a",
            style={"position": "relative"}
        )
    ], style={"flex": "0 0 auto", "display": "flex", "flexDirection": "column"}),

    html.Div([
        html.H5(id="hero-name-b", className="text-warning text-center"),
        html.Div(
            html.Div(id="hero-grid-b"),
            id="hero-wrapper-b",
            style={"position": "relative"}
        )
    ], style={"flex": "0 0 auto", "display": "flex", "flexDirection": "column"}),
], style={
    "display": "flex",
    "gap": "2px",
    "alignItems": "stretch",
})

        ], style={"flex": "0 0 auto", "textAlign": "center"}),

        html.Div([
            html.H4("Spec Tree", className="text-info text-center"),
            html.Div(id="spec-count", className="text-light", style={"textAlign": "center", "marginBottom": "6px"}),
            html.Div([
                html.Div(id="spec-grid", style={
                    "display": "grid",
                    "gap": "3px", "padding": "0px"
                })
            ], style={"display": "flex", "justifyContent": "center"})
        ], style={"flex": "0 0 auto", "textAlign": "center"})
    ], style={
        "display": "flex",
        "justifyContent": "flex-start",
        "gap": "2px",
        "marginTop": "24px"
    }, className="px-0", fluid=True),

    html.Hr(),

    html.Div(id="talent-output", className="text-light", style={
        "whiteSpace": "pre-wrap",
        "fontFamily": "monospace",
        "fontSize": "14px",
        "maxHeight": "20vh",
        "overflow": "auto",
        "width": "100%",
        "boxSizing": "border-box",
        "marginLeft": "20px",
        "marginRight": "20px"
    }),
    html.Button("Generate Copy Profiles", id="generate-btn", n_clicks=0, style={"margin":"10px", "padding":"6px 12px", "fontSize":"14px"}),
    html.Div(id="generate-status", style={"color":"#9bd"}),
    html.Button("Generate Profilesets", id="generate-psets-btn", n_clicks=0, style={"margin":"10px", "padding":"6px 12px", "fontSize":"14px"}),
    html.Div(id="generate-psets-status", style={"color":"#9bd"}),
    html.Div(id="copy-status", style={"display": "none"})
])
html.Div(id="alignment-debug-overlay", style={
    "position": "absolute",
    "top": "0px",
    "left": "0px",
    "width": "100%",
    "height": "2px",
    "backgroundColor": "red",
    "zIndex": 9999
})

# Shared tile rendering
def render_tile(node, included, blocked, included2=None):
    row, col = node["row"] + 1, node["col"] + 1
    style = {
        "gridRow": row,
        # Allow an override to target a specific grid line range
        "gridColumn": node.get("grid_column_override", col),
        "border": "1px solid #444",
        "borderRadius": "6px",
        "padding": "4px",
        "cursor": "pointer",
        "textAlign": "center",
        "backgroundColor": "#2a2f36"
    }

    # Optional: control horizontal alignment inside the assigned grid area
    js = node.get("justify_self_override")
    if js:
        style["justifySelf"] = js

    if not node["is_choice"]:
        classes = ["talent-img"]
        img_style = {"width": "40px", "height": "40px"}
        # use spellId for membership
        if node.get("spell_id") in included:
            classes.append("included")
        elif node.get("spell_id") in blocked:
            classes.append("blocked")
        else:
            # Special first-click outline for maxRanks==2 tracked in included2
            mr = node.get("max_ranks", 1)
            sid = node.get("spell_id")
            if mr == 2 and included2 and included2.get(str(sid)) == 1:
                # Apply blue outline around the icon image (same placement as green/red)
                img_style["border"] = "2px solid #2e86ff"

        return html.Div(
            id={"type": "clickable", "node_id": node["id"], "name": node["name"]},
            children=html.Img(
                src=node["image"],
                className=" ".join(classes),
                style=img_style
            ),
            n_clicks=0,
            style=style,
            className="talent-box"
        )
    else:
        buttons = []
        for i, name in enumerate(node["all_names"]):
            icon = node["image"][i]
            classes = ["talent-img"]
            sid = (node.get("all_spell_ids") or [None]*len(node["all_names"]))[i]
            if sid in included:
                classes.append("included")
            elif sid in blocked:
                classes.append("blocked")

            buttons.append(html.Button(
                html.Img(src=icon, className=" ".join(classes), style={"width": "20px", "height": "40px"}),
                id={"type": "clickable", "node_id": node["id"], "name": name},
                n_clicks=0,
                style={"border": "none", "background": "none", "padding": "0"}
            ))

        return html.Div(children=buttons, style={**style, "display": "flex", "gap": "2px"}, className="talent-box")

# Callbacks
def update_node_state(node_id, selected_name, included, blocked, all_nodes, included2):
    included, blocked = set(included or []), set(blocked or [])
    included2 = dict(included2 or {})
    node = next((n for n in all_nodes if n["id"] == node_id), None)
    if not node:
        return list(included), list(blocked), included2

    if node["is_choice"]:
        if selected_name not in node["all_names"]:
            return list(included), list(blocked), included2

        # Map choice names to spellIds
        idx = node["all_names"].index(selected_name)
        selected_sid = (node.get("all_spell_ids") or [None]*len(node["all_names"]))[idx]
        # find the other option's spell id
        other_sid = None
        for i, nm in enumerate(node["all_names"]):
            if i != idx:
                other_sid = (node.get("all_spell_ids") or [None]*len(node["all_names"]))[i]
                break

        if selected_sid in included:
            included.remove(selected_sid)
            blocked.add(selected_sid)
        elif selected_sid in blocked:
            # toggle with other
            if other_sid in included:
                blocked.discard(selected_sid)
                included.discard(other_sid)
                blocked.add(other_sid)
                included.add(selected_sid)
            else:
                # flip selected only
                if selected_sid in blocked:
                    blocked.remove(selected_sid)
                else:
                    blocked.add(selected_sid)
        else:
            if other_sid in included:
                blocked.add(selected_sid)
            elif other_sid in blocked:
                blocked.add(selected_sid)
            else:
                included.add(selected_sid)
                if other_sid is not None:
                    blocked.add(other_sid)
    else:
        # Cycle through: included → blocked → neutral → included
        sid = node.get("spell_id")
        mr = node.get("max_ranks", 1)
        if mr == 2:
            key = str(sid)
            if sid in included:
                # advance to blocked
                included.discard(sid)
                included2.pop(key, None)
                blocked.add(sid)
            elif sid in blocked:
                # reset to neutral
                blocked.discard(sid)
                included2.pop(key, None)
            else:
                # neutral -> rank1 (blue outline) -> rank2 (green)
                if included2.get(key) == 1:
                    included2.pop(key, None)
                    included.add(sid)
                else:
                    included2[key] = 1
        else:
            if sid in included:
                included.remove(sid)
                blocked.add(sid)
            elif sid in blocked:
                blocked.remove(sid)
            else:
                if sid is not None:
                    included.add(sid)

    return list(included), list(blocked), included2

@app.callback(Output("class-grid", "children"),
              Input("included-store", "data"),
              Input("blocked-store", "data"),
              Input("included2-store", "data"),
              Input("nodes-store", "data"))
def render_class_grid(included, blocked, included2, nodes):
    included, blocked = set(included or []), set(blocked or [])
    class_nodes = nodes.get("class", [])

    top_row = min((n["row"] for n in class_nodes), default=0)
    left_col = min((n["col"] for n in class_nodes), default=0)

    normalized_nodes = [
        {
            **n,
            "row": n["row"] - top_row + 1,
            "col": n["col"] - left_col + 1
        }
        for n in class_nodes
    ]

    return [render_tile(n, included, blocked, included2) for n in normalized_nodes]

@app.callback(Output("spec-grid", "children"),
              Input("included-store", "data"),
              Input("blocked-store", "data"),
              Input("included2-store", "data"),
              Input("nodes-store", "data"))
def render_spec_grid(included, blocked, included2, nodes):
    included, blocked = set(included or []), set(blocked or [])
    spec_nodes = nodes.get("spec", [])

    top_row = min((n["row"] for n in spec_nodes), default=0)
    left_col = min((n["col"] for n in spec_nodes), default=0)

    normalized_nodes = [
        {
            **n,
            "row": n["row"] - top_row + 1,
            "col": n["col"] - left_col + 1
        }
        for n in spec_nodes
    ]

    return [render_tile(n, included, blocked, included2) for n in normalized_nodes]

@app.callback(Output("hero-grid", "children"),
              Input("included-store", "data"),
              Input("blocked-store", "data"),
              Input("included2-store", "data"),
              Input("nodes-store", "data"))
def render_hero_grid(included, blocked, included2, nodes):
    included, blocked = set(included or []), set(blocked or [])
    hero_nodes = nodes.get("hero", {})
    return [render_tile(n, included, blocked, included2) for n in hero_nodes]

@app.callback(
    Output("class-count", "children"),
    Output("spec-count", "children"),
    Output("hero-count", "children"),
    Input("included-store", "data"),
    Input("blocked-store", "data"),
    Input("included2-store", "data"),
    Input("nodes-store", "data"),
    State("class-selector", "value"),
    State("spec-selector", "value"),
    prevent_initial_call=False
)
def update_tree_counts(included, blocked, included2, nodes_store, class_id, spec_id):
    try:
        # Resolve specialization
        from generate_talents import TalentJSON, tokenize
        class_name = next((t["className"] for t in raw if t["classId"] == class_id), None)
        spec_name = next((t["specName"] for t in raw if t["specId"] == spec_id and t["classId"] == class_id), None)
        if not class_name or not spec_name:
            return "Class builds: 0", "Spec builds: 0", "Hero builds: 0"
        tj = TalentJSON.from_file(talents_path)
        spec_obj = getattr(getattr(tj, tokenize(class_name)), tokenize(spec_name))

        nodes_store = nodes_store or {}
        included = set(included or [])
        blocked = set(blocked or [])
        included2 = included2 or {}

        def build_requirements(group_nodes):
            choice_req = {}
            node_req = {}
            for n in group_nodes or []:
                if n.get("is_choice"):
                    for sid, eid in zip(n.get("all_spell_ids", []) or [], n.get("all_entry_ids", []) or []):
                        if sid in included or (str(sid) in included2 and included2.get(str(sid)) == 1):
                            choice_req[eid] = 1
                        elif sid in blocked:
                            choice_req[eid] = 0
                else:
                    sid = n.get("spell_id")
                    if sid in included:
                        node_req[n.get("id")] = n.get("max_ranks", 1)
                    elif str(sid) in included2 and included2.get(str(sid)) == 1:
                        node_req[n.get("id")] = 1
                    elif sid in blocked:
                        node_req[n.get("id")] = 0
            return choice_req, node_req

        # Class/spec requirements
        class_choice_req, class_node_req = build_requirements(nodes_store.get("class", []))
        spec_choice_req, spec_node_req = build_requirements(nodes_store.get("spec", []))

        class_cnt = spec_obj.class_.count_builds(choice_requirements=class_choice_req, node_requirements=class_node_req) if (class_choice_req or class_node_req) else 1
        spec_cnt = spec_obj.spec.count_builds(choice_requirements=spec_choice_req, node_requirements=spec_node_req) if (spec_choice_req or spec_node_req) else 1

        # Hero requirements per selected tree
        hero_groups = (nodes_store.get("hero", {}) or {})
        selected_hero_trees = []
        hero_reqs_by_tree = {}
        for tree_name, group_nodes in hero_groups.items():
            has_any = False
            for n in group_nodes or []:
                if n.get("is_choice"):
                    for sid in (n.get("all_spell_ids", []) or []):
                        if sid in included or sid in blocked or (str(sid) in included2 and included2.get(str(sid)) == 1):
                            has_any = True; break
                else:
                    sid = n.get("spell_id")
                    if sid in included or sid in blocked or (str(sid) in included2 and included2.get(str(sid)) == 1):
                        has_any = True
                if has_any:
                    break
            if has_any:
                selected_hero_trees.append(tree_name)
                hero_reqs_by_tree[tree_name] = build_requirements(group_nodes)

        if not selected_hero_trees:
            hero_cnt_total = 1
        else:
            hero_cnt_total = 0
            for tree_name in selected_hero_trees:
                h_choice_req, h_node_req = hero_reqs_by_tree[tree_name]
                # Block other tree nodes for counting this subtree
                h_choice_req = dict(h_choice_req)
                h_node_req = dict(h_node_req)
                for other_name, other_group in hero_groups.items():
                    if other_name == tree_name:
                        continue
                    for n in other_group:
                        if n.get("is_choice"):
                            for sid, eid in zip(n.get("all_spell_ids", []) or [], n.get("all_entry_ids", []) or []):
                                h_choice_req[eid] = 0
                        else:
                            h_node_req[n.get("id")] = 0
                hero_cnt_total += spec_obj.hero.count_builds(choice_requirements=h_choice_req, node_requirements=h_node_req)

        return f"Class builds: {class_cnt}", f"Spec builds: {spec_cnt}", f"Hero builds: {hero_cnt_total}"
    except Exception:
        return "Class builds: -", "Spec builds: -", "Hero builds: -"
@app.callback(
    Output("spec-selector", "options"),
    Output("spec-selector", "value"),
    Input("class-selector", "value")
)
def update_specs_for_class(class_id):
    specs = [
        {"label": t["specName"], "value": t["specId"]}
        for t in raw if t["classId"] == class_id
    ]
    # Prefer config default spec if it matches this class
    preferred_spec = DEFAULT_SPEC_ID if any(t.get("specId") == DEFAULT_SPEC_ID and t.get("classId") == class_id for t in raw) else None
    default_spec = preferred_spec or (specs[0]["value"] if specs else None)
    return specs, default_spec

# Reset selections when class or spec changes
@app.callback(
    Output("included-store", "data", allow_duplicate=True),
    Output("blocked-store", "data", allow_duplicate=True),
    Output("included2-store", "data", allow_duplicate=True),
    Input("class-selector", "value"),
    Input("spec-selector", "value"),
    prevent_initial_call='initial_duplicate'
)
def reset_on_class_spec_change(class_id, spec_id):
    return [], [], {}

@app.callback(
    Output("included-store", "data"),
    Output("blocked-store", "data"),
    Output("included2-store", "data"),
    Input({"type": "clickable", "node_id": dash.ALL, "name": dash.ALL}, "n_clicks"),
    Input("nodes-store", "data"),
    State("included-store", "data"),
    State("blocked-store", "data"),
    State("included2-store", "data"),
    prevent_initial_call=True
)
def handle_clicks(grid_clicks, nodes_store, included, blocked, included2):
    all_nodes = []

    # Flatten out all node groups
    for node_group in ["class", "spec"]:
        all_nodes.extend(nodes_store.get(node_group, []))
    for tree_nodes in nodes_store.get("hero", {}).values():
        all_nodes.extend(tree_nodes)

    try:
        for i, count in enumerate(grid_clicks):
            if count:
                clicked_id = ctx.inputs_list[0][i]["id"]
                if isinstance(clicked_id, dict) and clicked_id.get("type") == "clickable":
                    name = clicked_id["name"]
                    node_id = clicked_id["node_id"]
                    return update_node_state(node_id, name, included, blocked, all_nodes, included2)
    except Exception as e:
        print("Callback error:", str(e))
        raise dash.exceptions.PreventUpdate

    raise dash.exceptions.PreventUpdate

@app.callback(
    Output("nodes-store", "data"),
    Output("hero-dimensions", "data"),
    Output("grid-dimensions", "data"),
    Input("class-selector", "value"),
    Input("spec-selector", "value"),
    prevent_initial_call=False
)
def load_nodes(selected_class_id, selected_spec_id):
    tree = next((t for t in raw if t["classId"] == selected_class_id and t["specId"] == selected_spec_id), None)
    if not tree:
        empty_store = {"class": [], "spec": [], "hero": {}}
        empty_dims = {"class": {"rows": 1, "cols": 1}, "spec": {"rows": 1, "cols": 1}, "hero": {}}
        return empty_store, {}, empty_dims

    hero_ids = set(n["id"] for n in tree.get("heroNodes", []))
    class_ids = set(n["id"] for n in tree.get("classNodes", []))
    spec_ids  = set(n["id"] for n in tree.get("specNodes", []))
    hero_tree_map = {}
    for subtree in tree.get("subTreeNodes", []):
        for entry in subtree.get("entries", []):
            tree_name = entry.get("name", "UnnamedHeroTree")
            for nid in entry.get("nodes", []):
                hero_tree_map[nid] = tree_name
    nodes_raw = tree.get("classNodes", []) + tree.get("specNodes", []) + tree.get("heroNodes", [])
    parsed_store = {"class": [], "spec": [], "hero": {}}
    hero_dimensions = {}

    # Group hero nodes by tree name
    hero_groups = {}
    for node in tree.get("heroNodes", []):
        tree_name = node.get("heroTreeName") or "TreeDefault"
        hero_groups.setdefault(tree_name, []).append(node)
    hero_min_y = {}
    for node in tree.get("heroNodes", []):
        tid = node["id"]
        tree_name = hero_tree_map.get(tid, "TreeDefault")
        pos_y = node.get("posY", 0)
        hero_min_y[tree_name] = min(hero_min_y.get(tree_name, pos_y), pos_y)

    for node in nodes_raw:
        tid = node["id"]
        label = node["name"]
        row = simplify(node.get("posY"))
        col = simplify(node.get("posX"))
        entries = node.get("entries", [])
        # Skip placeholder/blank nodes:
        # - no entries
        # - or entries exist but contain only empty objects with no display/useful fields
        if not entries:
            continue
        if all(not (e.get("id") or e.get("icon") or e.get("name") or e.get("spellId") or e.get("maxRanks")) for e in entries):
            continue
        is_choice = node.get("type") == "choice"
        is_hero = tid in hero_ids
        is_class = tid in class_ids
        is_spec  = tid in spec_ids

        category = (
            "hero" if is_hero else
            "class" if is_class else
            "spec" if is_spec else
            "unknown"
        )

        if is_choice and len(entries) >= 2:
            # Build choice arrays robustly: some entries may lack 'icon' or 'name'
            all_names = []
            image = []
            all_spell_ids = []
            all_max_ranks = []
            all_entry_ids = []
            for e in entries[:2]:  # UI expects exactly two buttons; truncate extras
                nm = clean(e.get("name", label))
                ic = e.get("icon") or "inv_misc_questionmark"
                sid = e.get("spellId")
                mr = e.get("maxRanks", 1)
                eid = e.get("id")
                all_names.append(nm)
                image.append(get_icon_url(ic))
                # Fallback to entry id when spellId is missing
                all_spell_ids.append(sid if sid is not None else eid)
                all_max_ranks.append(mr)
                all_entry_ids.append(eid)
            name = all_names[0] if all_names else clean(label)
        else:
            # Non-choice or malformed choice: pick first valid entry with icon
            first_with_icon = next((e for e in entries if e.get("icon")), None)
            icon_id = (first_with_icon.get("icon") if first_with_icon else "inv_misc_questionmark")
            image = get_icon_url(icon_id)
            first_with_name = next((e for e in entries if e.get("name")), None)
            all_names = [clean(first_with_name.get("name"))] if first_with_name else []
            # Prefer an entry that has spellId; otherwise fall back to any entry with id
            sid_entry = next((e for e in entries if e.get("spellId") is not None), None)
            id_entry = next((e for e in entries if e.get("id") is not None), None)
            spell_id = sid_entry.get("spellId") if sid_entry else None
            entry_id = (sid_entry.get("id") if sid_entry and sid_entry.get("id") is not None else (id_entry.get("id") if id_entry else None))
            # max ranks: prefer entry value, else node value
            max_ranks = (
                sid_entry.get("maxRanks") if sid_entry and sid_entry.get("maxRanks") is not None
                else (id_entry.get("maxRanks") if id_entry and id_entry.get("maxRanks") is not None else node.get("maxRanks", 1))
            )
            name = clean(label)
            # Fallback: use entry_id as identifier when spellId is missing
            if spell_id is None and entry_id is not None:
                spell_id = entry_id
        
        parsed_node = {
            "id": tid,
            "name": name,
            "label": label,
            "image": image,
            "row": row,
            "col": col,
            "is_choice": is_choice,
            "all_names": all_names,
            "spell_id": spell_id if not is_choice else None,
            "entry_id": entry_id if not is_choice else None,
            "max_ranks": max_ranks if not is_choice else None,
            "all_spell_ids": all_spell_ids if is_choice else [],
            "all_max_ranks": all_max_ranks if is_choice else [],
            "all_entry_ids": all_entry_ids if is_choice else [],
            "is_hero": is_hero,
            "category": category,
            "tree_name": node.get("heroTreeName") if is_hero else ""
        }

        if category == "class":
            parsed_store["class"].append(parsed_node)
        elif category == "spec":
            parsed_store["spec"].append(parsed_node)
        elif category == "hero":
            tree_name = hero_tree_map.get(tid, "TreeDefault")
            parsed_node["tree_name"] = tree_name
            parsed_store["hero"].setdefault(tree_name, []).append(parsed_node)

    # Calculate normalized grid dimensions (remove empty leading rows/cols)
    # Hero: per-tree min/max normalization
    for tree_name, group in parsed_store["hero"].items():
        if group:
            min_r = min(n["row"] for n in group)
            max_r = max(n["row"] for n in group)
            min_c = min(n["col"] for n in group)
            max_c = max(n["col"] for n in group)
            rows = max(1, (max_r - min_r + 1))
            cols = max(1, (max_c - min_c + 1))
        else:
            rows = cols = 1
        hero_dimensions[tree_name] = {"rows": rows, "cols": cols}

    # Class/spec: normalize using per-category min/max
    if parsed_store["class"]:
        min_cr = min(n["row"] for n in parsed_store["class"])
        max_cr = max(n["row"] for n in parsed_store["class"])
        min_cc = min(n["col"] for n in parsed_store["class"])
        max_cc = max(n["col"] for n in parsed_store["class"])
        max_class_row = max(1, (max_cr - min_cr + 1))
        max_class_col = max(1, (max_cc - min_cc + 1))
    else:
        max_class_row = max_class_col = 1

    if parsed_store["spec"]:
        min_sr = min(n["row"] for n in parsed_store["spec"])
        max_sr = max(n["row"] for n in parsed_store["spec"])
        min_sc = min(n["col"] for n in parsed_store["spec"])
        max_sc = max(n["col"] for n in parsed_store["spec"])
        max_spec_row = max(1, (max_sr - min_sr + 1))
        max_spec_col = max(1, (max_sc - min_sc + 1))
    else:
        max_spec_row = max_spec_col = 1

    grid_dimensions = {
        "class": {"rows": max_class_row, "cols": max_class_col},
        "spec": {"rows": max_spec_row, "cols": max_spec_col},
        "hero": hero_dimensions
    }

    return parsed_store, hero_dimensions, grid_dimensions

@app.callback(
    Output("hero-name-a", "children"),
    Output("hero-name-b", "children"),
    Input("hero-dimensions", "data")
)
def update_hero_names(dim):
    tree_names = sorted(dim.keys())
    name_a = tree_names[0] if len(tree_names) > 0 else "Hero Tree A"
    name_b = tree_names[1] if len(tree_names) > 1 else "Hero Tree B"
    return name_a, name_b

@app.callback(
    Output("class-grid", "style"),
    Output("spec-grid", "style"),
    Input("grid-dimensions", "data")
)
def apply_class_spec_styles(dims):
    dims = dims or {}
    class_dims = dims.get("class", {"rows": 1, "cols": 1})
    spec_dims = dims.get("spec", {"rows": 1, "cols": 1})

    class_style = {
        "display": "grid",
        "gridTemplateRows": f"repeat({class_dims['rows']}, 1fr)",
        "gridTemplateColumns": f"repeat({class_dims['cols']}, 1fr)",
        "gap": "3px",
        "padding": "0px"
    }
    spec_style = {
        "display": "grid",
        "gridTemplateRows": f"repeat({spec_dims['rows']}, 1fr)",
        "gridTemplateColumns": f"repeat({spec_dims['cols']}, 1fr)",
        "gap": "3px",
        "padding": "0px"
    }
    return class_style, spec_style

@app.callback(Output("hero-grid-a", "children"),
              Input("included-store", "data"),
              Input("blocked-store", "data"),
              Input("included2-store", "data"),
              Input("nodes-store", "data"))
def render_hero_grid_a(included, blocked, included2, nodes):
    hero_nodes_dict = nodes.get("hero", {})
    tree_names = sorted(hero_nodes_dict.keys())
    tree_a_nodes = hero_nodes_dict.get(tree_names[0], []) if len(tree_names) >= 1 else []

    # Normalize both rows and columns
    top_row = min((n["row"] for n in tree_a_nodes), default=0)
    left_col = min((n["col"] for n in tree_a_nodes), default=0)

    normalized_nodes = [
        {
            **n,
            "row": n["row"] - top_row + 1,
            "col": n["col"] - left_col + 1
        }
        for n in tree_a_nodes
    ]

    # Place first/last row nodes centered above the space between the 2nd and 3rd primary columns
    if normalized_nodes:
        first_row = min(n["row"] for n in normalized_nodes)
        last_row = max(n["row"] for n in normalized_nodes)
        # Prefer a row with 4 distinct columns (the 4x3 interior rows)
        rows_map = {}
        for n in normalized_nodes:
            if first_row < n["row"] < last_row:
                rows_map.setdefault(n["row"], set()).add(n["col"])
        candidate_row = None
        candidate_cols = []
        for r, cols in sorted(rows_map.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            if len(cols) >= 4:
                candidate_row = r
                candidate_cols = sorted(cols)
                break
        if not candidate_cols:
            # Fallback to frequency-based detection across interior rows
            col_counts = {}
            for n in normalized_nodes:
                if first_row < n["row"] < last_row:
                    col_counts[n["col"]] = col_counts.get(n["col"], 0) + 1
            top4 = sorted(col_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]
            candidate_cols = sorted(c for c, _ in top4)
        if len(candidate_cols) >= 3:
            c2, c3 = candidate_cols[1], candidate_cols[2]
            for i, n in enumerate(normalized_nodes):
                if n["row"] == first_row or n["row"] == last_row:
                    # render_tile adds +1 to numeric columns; apply the same +1 to grid line indices
                    if c3 > c2 + 1:
                        # True empty track exists: target exactly that gap (shifted by +1)
                        gco = f"{c2 + 2} / {c3 + 1}"
                    else:
                        # Adjacent middle columns: span both (shifted by +1) and center within span
                        gco = f"{c2 + 1} / {c3 + 2}"
                    normalized_nodes[i] = {**n, "grid_column_override": gco, "justify_self_override": "center"}

    tiles = [render_tile(n, included, blocked, included2) for n in normalized_nodes]
    return tiles

@app.callback(Output("hero-grid-b", "children"),
              Input("included-store", "data"),
              Input("blocked-store", "data"),
              Input("included2-store", "data"),
              Input("nodes-store", "data"))
def render_hero_grid_b(included, blocked, included2, nodes):
    hero_nodes_dict = nodes.get("hero", {})
    tree_names = sorted(hero_nodes_dict.keys())
    tree_b_nodes = hero_nodes_dict.get(tree_names[1], []) if len(tree_names) > 1 else []

    # Normalize both rows and columns
    top_row = min((n["row"] for n in tree_b_nodes), default=0)
    left_col = min((n["col"] for n in tree_b_nodes), default=0)

    normalized_nodes = [
        {
            **n,
            "row": n["row"] - top_row + 1,
            "col": n["col"] - left_col + 1
        }
        for n in tree_b_nodes
    ]

    # Place first/last row nodes centered above the space between the 2nd and 3rd primary columns
    if normalized_nodes:
        first_row = min(n["row"] for n in normalized_nodes)
        last_row = max(n["row"] for n in normalized_nodes)
        rows_map = {}
        for n in normalized_nodes:
            if first_row < n["row"] < last_row:
                rows_map.setdefault(n["row"], set()).add(n["col"])
        candidate_row = None
        candidate_cols = []
        for r, cols in sorted(rows_map.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            if len(cols) >= 4:
                candidate_row = r
                candidate_cols = sorted(cols)
                break
        if not candidate_cols:
            col_counts = {}
            for n in normalized_nodes:
                if first_row < n["row"] < last_row:
                    col_counts[n["col"]] = col_counts.get(n["col"], 0) + 1
            top4 = sorted(col_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:4]
            candidate_cols = sorted(c for c, _ in top4)
        if len(candidate_cols) >= 3:
            c2, c3 = candidate_cols[1], candidate_cols[2]
            for i, n in enumerate(normalized_nodes):
                if n["row"] == first_row or n["row"] == last_row:
                    # Align grid line indices with render_tile's +1 for numeric columns
                    if c3 > c2 + 1:
                        gco = f"{c2 + 2} / {c3 + 1}"
                    else:
                        gco = f"{c2 + 1} / {c3 + 2}"
                    normalized_nodes[i] = {**n, "grid_column_override": gco, "justify_self_override": "center"}

    tiles = [render_tile(n, included, blocked, included2) for n in normalized_nodes]
    return tiles

@app.callback(
    Output("hero-grid-a", "style"),
    Output("hero-grid-b", "style"),
    Input("hero-dimensions", "data")
)
def update_hero_styles(dim):
    tree_names = list(dim.keys())
    dim_a = dim.get(tree_names[0], {"rows": 1, "cols": 1}) if len(tree_names) > 0 else {"rows": 1, "cols": 1}
    dim_b = dim.get(tree_names[1], {"rows": 1, "cols": 1}) if len(tree_names) > 1 else {"rows": 1, "cols": 1}

    # Sync the dimensions to the max of both
    max_rows = max(dim_a["rows"], dim_b["rows"])
    max_cols = max(dim_a["cols"], dim_b["cols"])

    style = {
        "display": "grid",
        "gridTemplateRows": f"repeat({max_rows}, 1fr)",
        "gridTemplateColumns": f"repeat({max_cols}, 1fr)",
        "gap": "3px",
        "padding": "0px",
        "minHeight": f"{max_rows * 60}px"
    }

    return style, style

@app.callback(
    Output("talent-output", "children"),
    Input("included-store", "data"),
    Input("blocked-store", "data"),
    Input("included2-store", "data"),
    State("nodes-store", "data"),
    prevent_initial_call=True
)
def render_output(included, blocked, included2, nodes_store):
    # Build lookups of spellId -> category, name, maxRanks, and entryId
    nodes_store = nodes_store or {}
    sid_to_cat = {}
    sid_to_name = {}
    sid_to_max = {}
    sid_to_entry = {}

    # Spec nodes
    for n in nodes_store.get("spec", []) or []:
        if n.get("is_choice"):
            for sid, nm, mr, eid in zip(n.get("all_spell_ids", []) or [], n.get("all_names", []) or [], n.get("all_max_ranks", []) or [], n.get("all_entry_ids", []) or []):
                if sid is not None:
                    sid_to_cat[sid] = "spec"
                    sid_to_name[sid] = clean(nm)
                    sid_to_max[sid] = mr if mr is not None else 1
                    sid_to_entry[sid] = eid
        else:
            sid = n.get("spell_id")
            if sid is not None:
                sid_to_cat[sid] = "spec"
                sid_to_name[sid] = clean(n.get("name", ""))
                sid_to_max[sid] = n.get("max_ranks", 1)
                sid_to_entry[sid] = n.get("entry_id")

    # Class nodes (process independently)
    for n in nodes_store.get("class", []) or []:
        if n.get("is_choice"):
            for sid, nm, mr, eid in zip(n.get("all_spell_ids", []) or [], n.get("all_names", []) or [], n.get("all_max_ranks", []) or [], n.get("all_entry_ids", []) or []):
                if sid is not None:
                    sid_to_cat[sid] = "class"
                    sid_to_name[sid] = clean(nm)
                    sid_to_max[sid] = mr if mr is not None else 1
                    sid_to_entry[sid] = eid
        else:
            sid = n.get("spell_id")
            if sid is not None:
                sid_to_cat[sid] = "class"
                sid_to_name[sid] = clean(n.get("name", ""))
                sid_to_max[sid] = n.get("max_ranks", 1)
                sid_to_entry[sid] = n.get("entry_id")

    # Hero nodes (process each group; previous indentation bug processed only last group)
    for group in (nodes_store.get("hero", {}) or {}).values():
        for n in group or []:
            if n.get("is_choice"):
                for sid, nm, mr, eid in zip(n.get("all_spell_ids", []) or [], n.get("all_names", []) or [], n.get("all_max_ranks", []) or [], n.get("all_entry_ids", []) or []):
                    if sid is not None:
                        sid_to_cat[sid] = "hero"
                        sid_to_name[sid] = clean(nm)
                        sid_to_max[sid] = mr if mr is not None else 1
                        sid_to_entry[sid] = eid
            else:
                sid = n.get("spell_id")
                if sid is not None:
                    sid_to_cat[sid] = "hero"
                    sid_to_name[sid] = clean(n.get("name", ""))
                    sid_to_max[sid] = n.get("max_ranks", 1)
                    sid_to_entry[sid] = n.get("entry_id")

    # Included IDs: union of fully included plus first-click rank1 from included2
    inc_primary = set(included or [])
    inc_blue = {int(k) for k, v in (included2 or {}).items() if v == 1}
    inc_ids = list(inc_primary | inc_blue)
    blk_ids = list(blocked or [])

    def fmt(entries):
        parts = []
        for sid in entries:
            mr = sid_to_max.get(sid, 1)
            eid = sid_to_entry.get(sid, sid)
            # If this is a rank-2 node and first-click state is set, use 1
            if mr == 2 and included2 and included2.get(str(sid)) == 1:
                parts.append(f"{eid}:1")
                continue
            parts.append(f"{eid}:{mr}")
        return "/".join(parts)

    def fmt0(entries):
        parts = []
        for sid in entries:
            eid = sid_to_entry.get(sid, sid)
            parts.append(f"{eid}:0")
        return "/".join(parts)

    spec_inc = sorted([sid for sid in inc_ids if sid_to_cat.get(sid) == "spec"])
    spec_blk = sorted([sid for sid in blk_ids if sid_to_cat.get(sid) == "spec"])
    class_inc = sorted([sid for sid in inc_ids if sid_to_cat.get(sid) == "class"])
    class_blk = sorted([sid for sid in blk_ids if sid_to_cat.get(sid) == "class"])
    hero_inc = sorted([sid for sid in inc_ids if sid_to_cat.get(sid) == "hero"])
    hero_blk = sorted([sid for sid in blk_ids if sid_to_cat.get(sid) == "hero"])

    lines = [
        ("class_inc", fmt(class_inc)),
        ("class_blk", fmt0(class_blk)),
        ("spec_inc", fmt(spec_inc)),
        ("spec_blk", fmt0(spec_blk)),
        ("hero_inc", fmt(hero_inc)),
        ("hero_blk", fmt0(hero_blk)),
    ]

    header = html.Div("Copy the below lines for talent profiler!", style={"marginBottom": "4px", "fontWeight": "600"})

    def row(line_id, text):
        return html.Div([
            html.Button("Copy", id={"type": "copy-btn", "line": line_id}, n_clicks=0,
                        style={"marginRight": "8px", "padding": "2px 6px", "fontSize": "12px", "cursor": "pointer"}),
            html.Div(text, id={"type": "copy-line", "line": line_id}, n_clicks=0,
                     **{"data-text": text},
                     style={"cursor": "pointer", "userSelect": "text"})
        ], style={"display": "flex", "alignItems": "center", "gap": "6px", "marginBottom": "2px"})

    children = [header] + [row(i, t) for i, t in lines]
    return children

@app.callback(
    Output("generate-status", "children"),
    Input("generate-btn", "n_clicks"),
    State("included-store", "data"),
    State("blocked-store", "data"),
    State("included2-store", "data"),
    State("nodes-store", "data"),
    State("class-selector", "value"),
    State("spec-selector", "value"),
    prevent_initial_call=True
)
def generate_profilesets(n_clicks, included, blocked, included2, nodes_store, class_id, spec_id):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    try:
        # print("[Generate] Button clicked; starting copy generation...")
        from generate_talents import tokenize
        class_name = next((t["className"] for t in raw if t["classId"] == class_id), None)
        spec_name = next((t["specName"] for t in raw if t["specId"] == spec_id and t["classId"] == class_id), None)
        if not class_name or not spec_name:
            print("[Generate] Invalid class/spec selection", class_id, spec_id)
            return "❌ Generation failed: invalid class/spec selection"
        base_name = f"{tokenize(class_name)}_{tokenize(spec_name)}"
        # print(f"[Generate] Using base_name={base_name}")
        # Inline merged generation to surface errors clearly
        try:
            from generate_talents import TalentJSON
            tj = TalentJSON.from_file(talents_path)
            cls_attr = tokenize(class_name)
            spc_attr = tokenize(spec_name)
            spec_obj = getattr(getattr(tj, cls_attr), spc_attr)

            # Build requirements from current UI selections; only include trees with selections
            nodes_store = nodes_store or {}
            included = set(included or [])
            blocked = set(blocked or [])
            included2 = included2 or {}

            def build_requirements(group_nodes):
                choice_req = {}
                node_req = {}
                for n in group_nodes or []:
                    if n.get("is_choice"):
                        for sid, eid in zip(n.get("all_spell_ids", []) or [], n.get("all_entry_ids", []) or []):
                            if sid in included:
                                choice_req[eid] = 1
                            elif str(sid) in included2 and included2.get(str(sid)) == 1:
                                choice_req[eid] = 1
                            elif sid in blocked:
                                choice_req[eid] = 0
                    else:
                        sid = n.get("spell_id")
                        if sid in included:
                            mr = n.get("max_ranks", 1)
                            node_req[n.get("id")] = mr
                        elif str(sid) in included2 and included2.get(str(sid)) == 1:
                            node_req[n.get("id")] = 1
                        elif sid in blocked:
                            node_req[n.get("id")] = 0
                return choice_req, node_req

            class_choice_req, class_node_req = build_requirements(nodes_store.get("class", []))
            spec_choice_req, spec_node_req = build_requirements(nodes_store.get("spec", []))
            # Hero tree selection logic: detect per-tree selections so only selected hero tree(s) generate.
            hero_groups = (nodes_store.get("hero", {}) or {})
            selected_hero_trees = []
            hero_reqs_by_tree = {}
            for tree_name, group_nodes in hero_groups.items():
                # Any node (choice option or single) that is included/blocked or rank1 outline marks this tree as selected
                has_any = False
                for n in group_nodes or []:
                    if n.get("is_choice"):
                        for sid in (n.get("all_spell_ids", []) or []):
                            if sid in included or sid in blocked or (str(sid) in included2 and included2.get(str(sid)) == 1):
                                has_any = True; break
                    else:
                        sid = n.get("spell_id")
                        if sid in included or sid in blocked or (str(sid) in included2 and included2.get(str(sid)) == 1):
                            has_any = True
                    if has_any:
                        break
                if has_any:
                    selected_hero_trees.append(tree_name)
                    hero_reqs_by_tree[tree_name] = build_requirements(group_nodes)

            include_class = bool(class_choice_req or class_node_req)
            include_spec = bool(spec_choice_req or spec_node_req)
            include_hero  = len(selected_hero_trees) > 0

            if not (include_class or include_spec or include_hero):
                print("[Generate] No selections; nothing to merge")
                return "⚠️ No selections made; nothing to generate."

            # Count builds with requirements and enforce cap
            class_cnt = spec_obj.class_.count_builds(choice_requirements=class_choice_req, node_requirements=class_node_req) if include_class else 1
            spec_cnt = spec_obj.spec.count_builds(choice_requirements=spec_choice_req, node_requirements=spec_node_req) if include_spec else 1
            if include_hero:
                # For each selected hero tree, force all nodes of the other tree(s) to zero
                # so generation is restricted to a single subtree per iteration.
                hero_cnts = {}
                for tree_name in selected_hero_trees:
                    h_choice_req, h_node_req = hero_reqs_by_tree[tree_name]
                    h_choice_req = dict(h_choice_req)
                    h_node_req = dict(h_node_req)
                    for other_name, other_group in hero_groups.items():
                        if other_name == tree_name:
                            continue
                        for n in other_group:
                            if n.get("is_choice"):
                                for sid, eid in zip(n.get("all_spell_ids", []) or [], n.get("all_entry_ids", []) or []):
                                    h_choice_req[eid] = 0
                            else:
                                h_node_req[n.get("id")] = 0
                    cnt = spec_obj.hero.count_builds(choice_requirements=h_choice_req, node_requirements=h_node_req)
                    hero_cnts[tree_name] = cnt
                hero_cnt_total = sum(hero_cnts.values()) if hero_cnts else 1
            else:
                hero_cnts = {}
                hero_cnt_total = 1
            total_cnt = class_cnt * spec_cnt * hero_cnt_total
            # print(f"[Generate] class_cnt={class_cnt} spec_cnt={spec_cnt} hero_cnt_total={hero_cnt_total} by_tree={hero_cnts} total={total_cnt}")
            MAX_TOTAL = 200000
            if total_cnt > MAX_TOTAL:
                return f"⚠️ Too many builds ({total_cnt}). Refine selections or lower cap ({MAX_TOTAL})."

            def talent_str(build):
                parts = [f"{tid}:{pts}" for tid, pts in sorted(build.items())]
                return '/'.join(parts)

            def zero_ids_for_other_hero_tree(other_group_nodes):
                zero_ids = []
                for n in other_group_nodes or []:
                    if n.get("is_choice"):
                        zero_ids.extend([eid for eid in (n.get("all_entry_ids", []) or []) if eid is not None])
                    else:
                        eid = n.get("entry_id")
                        if eid is not None:
                            zero_ids.append(eid)
                return sorted(set(zero_ids))

            def build_choice_pair_map(group_nodes):
                pair_map = {}
                for n in group_nodes or []:
                    if n.get("is_choice"):
                        ids = [eid for eid in (n.get("all_entry_ids", []) or []) if eid is not None]
                        if len(ids) == 2:
                            a, b = ids
                            pair_map[a] = b
                            pair_map[b] = a
                return pair_map

            class_choice_pairs = build_choice_pair_map(nodes_store.get("class", []))
            spec_choice_pairs  = build_choice_pair_map(nodes_store.get("spec", []))

            lines = []
            idx = 1
            # Materialize generators to avoid exhaustion when nested
            class_builds = list(spec_obj.class_.generate_builds(choice_requirements=class_choice_req, node_requirements=class_node_req)) if include_class else [{}]
            spec_builds  = list(spec_obj.spec.generate_builds(choice_requirements=spec_choice_req, node_requirements=spec_node_req)) if include_spec else [{}]
            # print(f"[Generate] materialized: class_builds={len(class_builds)} spec_builds={len(spec_builds)} selected_hero_trees={selected_hero_trees}")
            if include_hero:
                for tree_name in selected_hero_trees:
                    h_choice_req, h_node_req = hero_reqs_by_tree[tree_name]
                    # Block nodes of other trees for this iteration
                    h_choice_req = dict(h_choice_req)
                    h_node_req = dict(h_node_req)
                    for other_name, other_group in hero_groups.items():
                        if other_name == tree_name:
                            continue
                        for n in other_group:
                            if n.get("is_choice"):
                                for sid, eid in zip(n.get("all_spell_ids", []) or [], n.get("all_entry_ids", []) or []):
                                    h_choice_req[eid] = 0
                            else:
                                h_node_req[n.get("id")] = 0
                    hero_builds = list(spec_obj.hero.generate_builds(choice_requirements=h_choice_req, node_requirements=h_node_req))
                    # print(f"[Generate] hero tree '{tree_name}' builds={len(hero_builds)}")
                    for class_build in class_builds:
                        for spec_build in spec_builds:
                            for hero_build in hero_builds:
                                # Append hero tree token to copy name when hero is included
                                name = f"{base_name}_{tokenize(tree_name)}_{idx}"
                                block = [f"copy={name}"]
                                if include_class:
                                    ct_main = talent_str(class_build)
                                    ct_zero = []
                                    for tid, pts in class_build.items():
                                        if pts and pts > 0 and tid in class_choice_pairs:
                                            ct_zero.append(f"{class_choice_pairs[tid]}:0")
                                    ct_full = '/'.join([p for p in [ct_main] if p] + ct_zero)
                                    if ct_full:
                                        block.append(f"class_talents={ct_full}")
                                if include_spec:
                                    st_main = talent_str(spec_build)
                                    st_zero = []
                                    for tid, pts in spec_build.items():
                                        if pts and pts > 0 and tid in spec_choice_pairs:
                                            st_zero.append(f"{spec_choice_pairs[tid]}:0")
                                    st_full = '/'.join([p for p in [st_main] if p] + st_zero)
                                    if st_full:
                                        block.append(f"spec_talents={st_full}")
                                # Build hero talents string including zeroed nodes from the other hero tree(s)
                                # Zero sibling options in hero choice nodes
                                # Build hero pair map once per iteration tree_name
                                hero_choice_pairs = build_choice_pair_map(hero_groups.get(tree_name, []))
                                # Build deduped hero talents mapping: prefer non-zero ranks from main build
                                main_map = dict(hero_build)  # id -> rank
                                zero_ids_set = set()
                                # Zero sibling choices for selected choices in this hero tree
                                for tid, pts in hero_build.items():
                                    if pts and pts > 0 and tid in hero_choice_pairs:
                                        zero_ids_set.add(hero_choice_pairs[tid])
                                # Zero all entries from other hero tree(s)
                                for other_name, other_group in hero_groups.items():
                                    if other_name == tree_name:
                                        continue
                                    zeros = zero_ids_for_other_hero_tree(other_group)
                                    zero_ids_set.update(zeros)
                                # Merge: add zeros only for ids not present in main_map
                                merged_map = dict(main_map)
                                for eid in zero_ids_set:
                                    if eid not in merged_map:
                                        merged_map[eid] = 0
                                # Minimal enforcement: active tree opening = 1, other tree opening = 0
                                try:
                                    entries = spec_obj.subtree_node.get('entries', [])
                                    name_to_eid = {}
                                    for e in entries:
                                        stid = e.get('traitSubTreeId')
                                        candidates = [n for n in getattr(spec_obj.hero, 'nodes', {}).values() if getattr(n, 'sub_tree', None) == stid]
                                        if not candidates:
                                            continue
                                        opening_node = min(candidates, key=lambda n: getattr(n, 'req_points', 0))
                                        if getattr(opening_node, 'choices', []):
                                            opening_eid = opening_node.choices[0].id
                                            name_to_eid[tokenize(e.get('name', '') or '')] = opening_eid
                                    active_name = tokenize(tree_name)
                                    active_eid = name_to_eid.get(active_name)
                                    if active_eid is not None:
                                        merged_map[active_eid] = 1
                                    for nname, eid in name_to_eid.items():
                                        if nname != active_name and eid not in merged_map:
                                            merged_map[eid] = 0
                                except Exception:
                                    pass
                                if merged_map:
                                    ht_full = '/'.join(f"{tid}:{pts}" for tid, pts in sorted(merged_map.items()))
                                    block.append(f"hero_talents={ht_full}")
                                lines.append("\n".join(block))
                                idx += 1
            else:
                for class_build in class_builds:
                    for spec_build in spec_builds:
                        name = f"{base_name}_{idx}"
                        block = [f"copy={name}"]
                        if include_class:
                            ct = talent_str(class_build)
                            if ct:
                                block.append(f"class_talents={ct}")
                        if include_spec:
                            st = talent_str(spec_build)
                            if st:
                                block.append(f"spec_talents={st}")
                        lines.append("\n".join(block))
                        idx += 1
            merged_text = "\n".join(lines) + ("\n" if lines else "")
        except Exception as gen_err:
            import traceback; traceback.print_exc()
            return f"❌ Generation failed: {gen_err}"
        if not merged_text:
            # print("[Generate] Merged generation produced 0 lines")
            return "⚠️ No merged output produced (0 builds)."
        out_path = os.path.join(script_dir, 'sim_proc_output.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(merged_text)
        # Accurate count equals number of blocks we appended: idx started at 1
        written_total = idx - 1
        # print(f"[Generate] Wrote {written_total} profiles to {out_path}")
        return f"✅ Generated {written_total} copy profiles → {out_path}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Generation failed: {e}"

@app.callback(
    Output("generate-psets-status", "children"),
    Input("generate-psets-btn", "n_clicks"),
    State("included-store", "data"),
    State("blocked-store", "data"),
    State("included2-store", "data"),
    State("nodes-store", "data"),
    State("class-selector", "value"),
    State("spec-selector", "value"),
    prevent_initial_call=True
)
def generate_profilesets_psets(n_clicks, included, blocked, included2, nodes_store, class_id, spec_id):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    try:
        # print("[Generate-PSETS] Button clicked; starting profilesets generation...")
        from generate_talents import tokenize
        class_name = next((t["className"] for t in raw if t["classId"] == class_id), None)
        spec_name = next((t["specName"] for t in raw if t["specId"] == spec_id and t["classId"] == class_id), None)
        if not class_name or not spec_name:
            print("[Generate-PSETS] Invalid class/spec selection", class_id, spec_id)
            return "❌ Generation failed: invalid class/spec selection"
        base_name = f"{tokenize(class_name)}_{tokenize(spec_name)}"
        # print(f"[Generate-PSETS] Using base_name={base_name}")
        try:
            from generate_talents import TalentJSON
            tj = TalentJSON.from_file(talents_path)
            cls_attr = tokenize(class_name)
            spc_attr = tokenize(spec_name)
            spec_obj = getattr(getattr(tj, cls_attr), spc_attr)

            nodes_store = nodes_store or {}
            included = set(included or [])
            blocked = set(blocked or [])
            included2 = included2 or {}

            def build_requirements(group_nodes):
                choice_req = {}
                node_req = {}
                for n in group_nodes or []:
                    if n.get("is_choice"):
                        for sid, eid in zip(n.get("all_spell_ids", []) or [], n.get("all_entry_ids", []) or []):
                            if sid in included:
                                choice_req[eid] = 1
                            elif str(sid) in included2 and included2.get(str(sid)) == 1:
                                choice_req[eid] = 1
                            elif sid in blocked:
                                choice_req[eid] = 0
                    else:
                        sid = n.get("spell_id")
                        if sid in included:
                            mr = n.get("max_ranks", 1)
                            node_req[n.get("id")] = mr
                        elif str(sid) in included2 and included2.get(str(sid)) == 1:
                            node_req[n.get("id")] = 1
                        elif sid in blocked:
                            node_req[n.get("id")] = 0
                return choice_req, node_req
            # Sibling choice maps for class/spec to zero the other side
            def build_choice_pair_map(group_nodes):
                pair_map = {}
                for n in group_nodes or []:
                    if n.get("is_choice"):
                        ids = [eid for eid in (n.get("all_entry_ids", []) or []) if eid is not None]
                        if len(ids) == 2:
                            a, b = ids
                            pair_map[a] = b
                            pair_map[b] = a
                return pair_map

            class_choice_pairs = build_choice_pair_map(nodes_store.get("class", []))
            spec_choice_pairs  = build_choice_pair_map(nodes_store.get("spec", []))

            class_choice_req, class_node_req = build_requirements(nodes_store.get("class", []))
            spec_choice_req, spec_node_req = build_requirements(nodes_store.get("spec", []))
            hero_groups = (nodes_store.get("hero", {}) or {})
            selected_hero_trees = []
            hero_reqs_by_tree = {}
            for tree_name, group_nodes in hero_groups.items():
                has_any = False
                for n in group_nodes or []:
                    if n.get("is_choice"):
                        for sid in (n.get("all_spell_ids", []) or []):
                            if sid in included or sid in blocked or (str(sid) in included2 and included2.get(str(sid)) == 1):
                                has_any = True; break
                    else:
                        sid = n.get("spell_id")
                        if sid in included or sid in blocked or (str(sid) in included2 and included2.get(str(sid)) == 1):
                            has_any = True
                    if has_any:
                        break
                if has_any:
                    selected_hero_trees.append(tree_name)
                    hero_reqs_by_tree[tree_name] = build_requirements(group_nodes)

            include_class = bool(class_choice_req or class_node_req)
            include_spec = bool(spec_choice_req or spec_node_req)
            include_hero  = len(selected_hero_trees) > 0

            if not (include_class or include_spec or include_hero):
                print("[Generate-PSETS] No selections; nothing to generate")
                return "⚠️ No selections made; nothing to generate."

            # Count
            class_cnt = spec_obj.class_.count_builds(choice_requirements=class_choice_req, node_requirements=class_node_req) if include_class else 1
            spec_cnt = spec_obj.spec.count_builds(choice_requirements=spec_choice_req, node_requirements=spec_node_req) if include_spec else 1
            if include_hero:
                hero_cnts = {}
                for tree_name in selected_hero_trees:
                    h_choice_req, h_node_req = hero_reqs_by_tree[tree_name]
                    h_choice_req = dict(h_choice_req)
                    h_node_req = dict(h_node_req)
                    for other_name, other_group in hero_groups.items():
                        if other_name == tree_name:
                            continue
                        for n in other_group:
                            if n.get("is_choice"):
                                for sid, eid in zip(n.get("all_spell_ids", []) or [], n.get("all_entry_ids", []) or []):
                                    h_choice_req[eid] = 0
                            else:
                                h_node_req[n.get("id")] = 0
                    cnt = spec_obj.hero.count_builds(choice_requirements=h_choice_req, node_requirements=h_node_req)
                    hero_cnts[tree_name] = cnt
                hero_cnt_total = sum(hero_cnts.values()) if hero_cnts else 1
            else:
                hero_cnts = {}
                hero_cnt_total = 1
            total_cnt = class_cnt * spec_cnt * hero_cnt_total
            # print(f"[Generate-PSETS] class_cnt={class_cnt} spec_cnt={spec_cnt} hero_cnt_total={hero_cnt_total} by_tree={hero_cnts} total={total_cnt}")
            MAX_TOTAL = 200000
            if total_cnt > MAX_TOTAL:
                return f"⚠️ Too many builds ({total_cnt}). Refine selections or lower cap ({MAX_TOTAL})."

            def talent_str(build):
                parts = [f"{tid}:{pts}" for tid, pts in sorted(build.items())]
                return '/'.join(parts)

            def zero_ids_for_other_hero_tree(other_group_nodes):
                zero_ids = []
                for n in other_group_nodes or []:
                    if n.get("is_choice"):
                        zero_ids.extend([eid for eid in (n.get("all_entry_ids", []) or []) if eid is not None])
                    else:
                        eid = n.get("entry_id")
                        if eid is not None:
                            zero_ids.append(eid)
                return sorted(set(zero_ids))

            lines = []
            idx = 1
            class_builds = list(spec_obj.class_.generate_builds(choice_requirements=class_choice_req, node_requirements=class_node_req)) if include_class else [{}]
            spec_builds  = list(spec_obj.spec.generate_builds(choice_requirements=spec_choice_req, node_requirements=spec_node_req)) if include_spec else [{}]

            if include_hero:
                # print(f"[Generate-PSETS] Selected hero trees: {selected_hero_trees}")
                for tree_name in selected_hero_trees:
                    h_choice_req, h_node_req = hero_reqs_by_tree[tree_name]
                    h_choice_req = dict(h_choice_req)
                    h_node_req = dict(h_node_req)
                    for other_name, other_group in hero_groups.items():
                        if other_name == tree_name:
                            continue
                        for n in other_group:
                            if n.get("is_choice"):
                                for sid, eid in zip(n.get("all_spell_ids", []) or [], n.get("all_entry_ids", []) or []):
                                    h_choice_req[eid] = 0
                            else:
                                h_node_req[n.get("id")] = 0
                    hero_builds = list(spec_obj.hero.generate_builds(choice_requirements=h_choice_req, node_requirements=h_node_req))
                    # print(f"[Generate-PSETS] hero tree '{tree_name}' builds={len(hero_builds)}")
                    for class_build in class_builds:
                        for spec_build in spec_builds:
                            for hero_build in hero_builds:
                                name = f"{base_name}_{tokenize(tree_name)}_{idx}"
                                if include_class:
                                    ct_main = talent_str(class_build)
                                    ct_zero = []
                                    # Zero sibling choices for class
                                    for tid, pts in class_build.items():
                                        if pts and pts > 0 and tid in class_choice_pairs:
                                            ct_zero.append(f"{class_choice_pairs[tid]}:0")
                                    ct_full = '/'.join([p for p in [ct_main] if p] + ct_zero)
                                    if ct_full:
                                        lines.append(f"profileset.\"{name}\"+=class_talents={ct_full}")
                                if include_spec:
                                    st_main = talent_str(spec_build)
                                    st_zero = []
                                    # Zero sibling choices for spec
                                    for tid, pts in spec_build.items():
                                        if pts and pts > 0 and tid in spec_choice_pairs:
                                            st_zero.append(f"{spec_choice_pairs[tid]}:0")
                                    st_full = '/'.join([p for p in [st_main] if p] + st_zero)
                                    if st_full:
                                        lines.append(f"profileset.\"{name}\"+=spec_talents={st_full}")
                                # Hero talents including zeroed sibling choices and other hero tree nodes
                                hero_choice_pairs = {}
                                for n in hero_groups.get(tree_name, []) or []:
                                    if n.get("is_choice"):
                                        ids = [eid for eid in (n.get("all_entry_ids", []) or []) if eid is not None]
                                        if len(ids) == 2:
                                            a, b = ids
                                            hero_choice_pairs[a] = b
                                            hero_choice_pairs[b] = a
                                # Dedup hero talents: prefer non-zero ranks from main build
                                main_map = dict(hero_build)
                                zero_ids_set = set()
                                for tid, pts in hero_build.items():
                                    if pts and pts > 0 and tid in hero_choice_pairs:
                                        zero_ids_set.add(hero_choice_pairs[tid])
                                for other_name, other_group in hero_groups.items():
                                    if other_name == tree_name:
                                        continue
                                    zeros = zero_ids_for_other_hero_tree(other_group)
                                    zero_ids_set.update(zeros)
                                merged_map = dict(main_map)
                                for eid in zero_ids_set:
                                    if eid not in merged_map:
                                        merged_map[eid] = 0
                                # Minimal enforcement: active tree opening = 1, other tree opening = 0
                                try:
                                    entries = spec_obj.subtree_node.get('entries', [])
                                    name_to_eid = {}
                                    for e in entries:
                                        stid = e.get('traitSubTreeId')
                                        candidates = [n for n in getattr(spec_obj.hero, 'nodes', {}).values() if getattr(n, 'sub_tree', None) == stid]
                                        if not candidates:
                                            continue
                                        opening_node = min(candidates, key=lambda n: getattr(n, 'req_points', 0))
                                        if getattr(opening_node, 'choices', []):
                                            opening_eid = opening_node.choices[0].id
                                            name_to_eid[tokenize(e.get('name', '') or '')] = opening_eid
                                    active_name = tokenize(tree_name)
                                    active_eid = name_to_eid.get(active_name)
                                    if active_eid is not None:
                                        merged_map[active_eid] = 1
                                    for nname, eid in name_to_eid.items():
                                        if nname != active_name and eid not in merged_map:
                                            merged_map[eid] = 0
                                except Exception:
                                    pass
                                if merged_map:
                                    ht_full = '/'.join(f"{tid}:{pts}" for tid, pts in sorted(merged_map.items()))
                                    lines.append(f"profileset.\"{name}\"+=hero_talents={ht_full}")
                                idx += 1
            else:
                # print("[Generate-PSETS] No hero selections; generating without hero tree")
                for class_build in class_builds:
                    for spec_build in spec_builds:
                        name = f"{base_name}_{idx}"
                        if include_class:   
                            ct = talent_str(class_build)
                            if ct:
                                lines.append(f"profileset.\"{name}\"+=class_talents={ct}")
                        if include_spec:
                            st = talent_str(spec_build)
                            if st:
                                lines.append(f"profileset.\"{name}\"+=spec_talents={st}")
                        idx += 1

            text = "\n".join(lines) + ("\n" if lines else "")
        except Exception as gen_err:
            import traceback; traceback.print_exc()
            return f"❌ Generation failed: {gen_err}"

        if not text:
            # print("[Generate-PSETS] Generation produced 0 lines")
            return "⚠️ No profilesets produced (0 builds)."

        out_path = os.path.join(script_dir, 'sim_proc_output.txt')
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)
        written_total = idx - 1
        # print(f"[Generate-PSETS] Wrote {written_total} profilesets to {out_path}")
        return f"✅ Generated {written_total} profilesets → {out_path}"
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Generation failed: {e}"

if __name__ == "__main__":
    print("Starting Dash on http://127.0.0.1:8050")
    app.run(host="127.0.0.1", port=8050, debug=True, use_reloader=False)