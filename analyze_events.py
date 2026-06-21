import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import os

# =====================================================
# CONFIG
# =====================================================

CSV_FILE = "player_positions.csv"
OUTPUT_DIR = "graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# EVENTS
# =====================================================

GAME_STARTS = [
    "00:04:40",
    "00:13:17",
    "00:19:31",
    "00:30:20",
    "00:41:10",
    "00:51:57",
    "00:58:05"
]

KICKOFFS = [
    "00:11:29",
    "00:17:26",
    "00:24:44",
    "00:36:52",
    "00:46:26",
    "00:54:48",
    "00:58:44"
]

END_TIME = "00:59:17"

# =====================================================
# HELPERS
# =====================================================

def hms_to_seconds(txt):
    h, m, s = map(int, txt.split(":"))
    return h * 3600 + m * 60 + s

game_start_sec = [hms_to_seconds(t) for t in GAME_STARTS]
kickoff_sec = [hms_to_seconds(t) for t in KICKOFFS]
end_sec = hms_to_seconds(END_TIME)

# =====================================================
# LOAD DATA
# =====================================================

print("Loading CSV...")

df = pd.read_csv(CSV_FILE)

df["seconds"] = df["seconds"].round(3)

# determine teams present
teams = sorted(df["team"].dropna().unique())

print("Teams found:", teams)

# =====================================================
# BUILD TIMELINE METRICS
# =====================================================

records = []

for sec, g in df.groupby("seconds"):

    row = {
        "seconds": sec,
        "player_count": len(g)
    }

    # -----------------------------
    # team counts
    # -----------------------------

    for team in teams:
        row[f"count_{team}"] = (g["team"] == team).sum()

    # -----------------------------
    # centroid
    # -----------------------------

    row["centroid_x"] = g["x"].mean()
    row["centroid_y"] = g["y"].mean()

    # -----------------------------
    # spread
    # -----------------------------

    row["spread_x"] = g["x"].std()
    row["spread_y"] = g["y"].std()

    # combined spread

    dx = g["x"] - row["centroid_x"]
    dy = g["y"] - row["centroid_y"]

    row["spread_total"] = np.sqrt(
        (dx**2 + dy**2).mean()
    )

    # -----------------------------
    # convex hull area
    # -----------------------------

    if len(g) >= 3:

        pts = g[["x", "y"]].values

        try:
            hull = ConvexHull(pts)
            row["hull_area"] = hull.volume
        except:
            row["hull_area"] = np.nan

    else:
        row["hull_area"] = np.nan

    records.append(row)

timeline = pd.DataFrame(records)
timeline = timeline.sort_values("seconds")

# =====================================================
# MOVEMENT
# =====================================================

timeline["movement"] = np.sqrt(
    np.diff(
        timeline["centroid_x"],
        prepend=timeline["centroid_x"].iloc[0]
    )**2
    +
    np.diff(
        timeline["centroid_y"],
        prepend=timeline["centroid_y"].iloc[0]
    )**2
)

# =====================================================
# PLOT FUNCTION
# =====================================================
def add_event_markers():

    first_gs = True
    first_ko = True

    for t in game_start_sec:

        plt.axvline(
            t,
            color="green",
            linestyle="--",
            linewidth=1,
            label="Game Start" if first_gs else None
        )

        first_gs = False

    for t in kickoff_sec:

        plt.axvline(
            t,
            color="red",
            linestyle=":",
            linewidth=1,
            label="Kickoff" if first_ko else None
        )

        first_ko = False

    plt.axvline(
        end_sec,
        color="black",
        linewidth=2,
        label="End"
    )
    

def save_plot(y, title, filename):

    plt.figure(figsize=(16, 6))

    plt.plot(
        timeline["seconds"],
        timeline[y]
    )

    add_event_markers()

    plt.title(title)
    plt.xlabel("Seconds")
    plt.ylabel(y)

    plt.tight_layout()

    outfile = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(outfile)
    plt.close()

    print("Saved:", outfile)

# =====================================================
# GRAPHS
# =====================================================

print("Generating graphs...")

save_plot(
    "player_count",
    "Total Players",
    "01_player_count.png"
)

save_plot(
    "centroid_x",
    "Field Centroid X",
    "02_centroid_x.png"
)

save_plot(
    "centroid_y",
    "Field Centroid Y",
    "03_centroid_y.png"
)

save_plot(
    "spread_total",
    "Player Spread",
    "04_spread_total.png"
)

save_plot(
    "hull_area",
    "Convex Hull Area",
    "05_hull_area.png"
)

save_plot(
    "movement",
    "Centroid Movement",
    "06_movement.png"
)

# =====================================================
# TEAM COUNTS
# =====================================================

plt.figure(figsize=(16, 6))

for team in teams:

    col = f"count_{team}"

    if col in timeline.columns:

        plt.plot(
            timeline["seconds"],
            timeline[col],
            label=team
        )

add_event_markers()

plt.title("Team Counts")
plt.xlabel("Seconds")
plt.ylabel("Players")
plt.legend()

plt.tight_layout()

outfile = os.path.join(
    OUTPUT_DIR,
    "07_team_counts.png"
)

plt.savefig(outfile)
plt.close()

print("Saved:", outfile)

# =====================================================
# TEAM SPREADS
# =====================================================

team_spread = []

for sec, g in df.groupby("seconds"):

    row = {"seconds": sec}

    for team in teams:

        gt = g[g["team"] == team]

        if len(gt) > 1:

            cx = gt["x"].mean()
            cy = gt["y"].mean()

            d = np.sqrt(
                (gt["x"] - cx)**2 +
                (gt["y"] - cy)**2
            )

            row[f"spread_{team}"] = d.mean()

        else:

            row[f"spread_{team}"] = np.nan

    team_spread.append(row)

team_spread = pd.DataFrame(team_spread)



plt.figure(figsize=(16, 6))

for team in teams:

    col = f"spread_{team}"

    if col in team_spread.columns:

        plt.plot(
            team_spread["seconds"],
            team_spread[col],
            label=team
        )

add_event_markers()

plt.title("Team Spread")
plt.xlabel("Seconds")
plt.ylabel("Spread")
plt.legend()

plt.tight_layout()

outfile = os.path.join(
    OUTPUT_DIR,
    "08_team_spread.png"
)

plt.savefig(outfile)
plt.close()

print("Saved:", outfile)

# =====================================================
# TEAM Y SPREAD
# =====================================================

team_yspread = []

for sec, g in df.groupby("seconds"):

    row = {"seconds": sec}

    for team in teams:

        gt = g[g["team"] == team]

        if len(gt) > 1:
            row[f"yspread_{team}"] = gt["y"].std()
        else:
            row[f"yspread_{team}"] = np.nan

    team_yspread.append(row)

team_yspread = pd.DataFrame(team_yspread)

plt.figure(figsize=(16, 6))

for team in teams:

    col = f"yspread_{team}"

    if col in team_yspread.columns:

        plt.plot(
            team_yspread["seconds"],
            team_yspread[col],
            label=team
        )

add_event_markers()

plt.title("Team Y Spread")
plt.xlabel("Seconds")
plt.ylabel("Y Standard Deviation")
plt.legend()

plt.tight_layout()

outfile = os.path.join(
    OUTPUT_DIR,
    "09_team_yspread.png"
)

plt.savefig(outfile)
plt.close()

print("Saved:", outfile)

# =====================================================
# TEAM X SPREAD
# =====================================================

team_xspread = []

for sec, g in df.groupby("seconds"):

    row = {"seconds": sec}

    for team in teams:

        gt = g[g["team"] == team]

        if len(gt) > 1:
            row[f"xspread_{team}"] = gt["x"].std()
        else:
            row[f"xspread_{team}"] = np.nan

    team_xspread.append(row)

team_xspread = pd.DataFrame(team_xspread)

plt.figure(figsize=(16, 6))

for team in teams:

    col = f"xspread_{team}"

    if col in team_xspread.columns:

        plt.plot(
            team_xspread["seconds"],
            team_xspread[col],
            label=team
        )

add_event_markers()

plt.title("Team X Spread")
plt.xlabel("Seconds")
plt.ylabel("X Standard Deviation")
plt.legend()

plt.tight_layout()

outfile = os.path.join(
    OUTPUT_DIR,
    "10_team_xspread.png"
)

plt.savefig(outfile)
plt.close()

print("Saved:", outfile)

# =====================================================
# TEAM CENTROID Y DISTANCE
# =====================================================

if len(teams) >= 2:

    team_a = teams[0]
    team_b = teams[1]

    centroid_rows = []

    for sec, g in df.groupby("seconds"):

        ga = g[g["team"] == team_a]
        gb = g[g["team"] == team_b]

        if len(ga) > 0 and len(gb) > 0:

            centroid_rows.append({
                "seconds": sec,
                "distance":
                    abs(
                        ga["y"].mean()
                        -
                        gb["y"].mean()
                    )
            })

    centroid_df = pd.DataFrame(
        centroid_rows
    )

    plt.figure(figsize=(16, 6))

    plt.plot(
        centroid_df["seconds"],
        centroid_df["distance"]
    )

    add_event_markers()

    plt.title(
        f"Centroid Y Distance ({team_a} vs {team_b})"
    )

    plt.xlabel("Seconds")
    plt.ylabel("Distance")

    plt.tight_layout()

    outfile = os.path.join(
        OUTPUT_DIR,
        "11_team_centroid_y_distance.png"
    )

    plt.savefig(outfile)
    plt.close()

    print("Saved:", outfile)
    
    # =====================================================
# TEAM CENTROID X DISTANCE
# =====================================================

if len(teams) >= 2:

    team_a = teams[0]
    team_b = teams[1]

    centroid_rows = []

    for sec, g in df.groupby("seconds"):

        ga = g[g["team"] == team_a]
        gb = g[g["team"] == team_b]

        if len(ga) > 0 and len(gb) > 0:

            centroid_rows.append({
                "seconds": sec,
                "distance":
                    abs(
                        ga["x"].mean()
                        -
                        gb["x"].mean()
                    )
            })

    centroid_df = pd.DataFrame(
        centroid_rows
    )

    plt.figure(figsize=(16, 6))

    plt.plot(
        centroid_df["seconds"],
        centroid_df["distance"]
    )

    add_event_markers()

    plt.title(
        f"Centroid X Distance ({team_a} vs {team_b})"
    )

    plt.xlabel("Seconds")
    plt.ylabel("Distance")

    plt.tight_layout()

    outfile = os.path.join(
        OUTPUT_DIR,
        "12_team_centroid_x_distance.png"
    )

    plt.savefig(outfile)
    plt.close()

    print("Saved:", outfile)
    


print()
print("Done.")
print(f"Graphs written to: {OUTPUT_DIR}")