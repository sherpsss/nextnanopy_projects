"""Cleaving plan for the piece currently on the bench.

Starting material: 2" wafer, already cleaved along a line 5 mm off the centre
line; the larger of the two pieces is the one being diced.

Edit the numbers below and re-run.  For a prompt-driven session instead, run
``python wafer_cleave.py`` with no arguments.
"""

import matplotlib.pyplot as plt

from wafer_cleave import Wafer, Cut, DieType, best_layout, report, plot_layout

# --- starting material ------------------------------------------------------
wafer = Wafer(
    diameter=50.8,        # 2 inch
    flat_length=17.0,      # e.g. 16.0 for a 2" GaAs primary flat; 0 = ignore
    flat_side="bottom",
    edge_exclusion=1.0,   # mm of rim to stay away from
)

cuts = [
    # the cleave already made: 5 mm off centre, keeping the larger piece
    Cut(axis="x", position=5.0, keep="below"),
]

# --- what to dice it into (priority order: first type gets the good edge) ----
# strip_axis pins the axis a type's strips march along: "x" gives vertical
# columns cleaved off the existing x = +5 edge, which are then diced crosswise.
# Leave it None to let the planner pick whichever orientation yields more.
dies = [
    DieType("big",   width=14.0, height=9.5, max_count=4, strip_axis="y"),
    DieType("small", width=13.0, height=13.0, max_count=2,strip_axis="y"),  # None =
]

# --- plan and draw ----------------------------------------------------------
# align="grid" puts every die of a type on one shared lattice so the cross cuts
# line up strip to strip; align="center" centres each strip independently,
# which fits a die or two more but staggers the rows.
layout = best_layout(wafer, cuts, dies, axis="both", kerf=0.0, align="grid")

print(report(layout))
plot_layout(layout, out="cleaving_plan.png")
plt.show()
