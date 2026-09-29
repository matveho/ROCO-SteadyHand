# Offline board annotation tool

`tools/board_annotation_tool.py` is a supervised desktop tool for measuring
the initial and final locations in board photographs. It is designed for an
offline laptop session; it does not connect to the robot or alter live
calibration.

Install the two UI dependencies on the computer that will run it:

```bash
python3 -m pip install Pillow
# Linux distributions may provide Tk separately, for example:
# sudo apt install python3-tk
```

Start it from the repository checkout:

```bash
cd repository-audit
python3 tools/board_annotation_tool.py \
  --initial /path/to/initial.jpg \
  --final /path/to/final.jpg \
  --output /path/to/board_annotations.json
```

The same paths can be selected with **Open initial** and **Open final** after
launching with no arguments.

For each image, use **Set 4 board corners** and click top-left, top-right,
bottom-right, and bottom-left. **Rectify board** computes the exact four-point
planar homography and creates a front-facing board view. Enter the known board
width and height in metres before rectifying. If a measured rectangle is
visible, use **Measured rectangle** in either source view (after rectification)
or the rectified view and enter its measured width, height, and units. A ruler
can be marked with **Set ruler scale** by clicking its two endpoints in either
view. The measured rectangle provides independent X/Y
scale; otherwise the ruler provides a common scale. The calibration values and
both homography matrices are retained in the audit JSON.

Use **circle** by clicking the center and one edge point. Use **pen** by
clicking vertices and then **Close polygon** (or clicking the first vertex).
Open polylines are never exported as annotations. Polygon centers use the
area-weighted centroid; circles retain the clicked center and radius. Part
names and center coordinates can be toggled on the image, while the annotation
list on the right keeps labels out of the board surface.
Use **Move image** for left-button panning, or middle-button drag as a
shortcut. **Zoom +**, **Zoom −**, **Fit view**, and the mouse wheel change only
the viewport; stored board corners and shape coordinates stay in image pixels.
`Ctrl+Z` undoes the last point, annotation, zoom, pan, crop, or rotation.

The export writes two JSON files:

* `board_annotations.json` contains source-image hashes, board corner pixels,
  homographies, ruler/rectangle measurements, every closed shape, source and
  rectified geometry, center coordinates, and export warnings.
* `board_annotations.task_coordinates.json` contains the robot-facing
  board-local task points. `pick` comes from the initial image and `place`
  comes from the final image. Coordinates are metres from the rectified board
  top-left, with `source_board_center_xy_m` declaring the center that the
  existing `steadyhand` live board-axis mapping subtracts. The third coordinate
  is `0.0` by design: live board-plane calibration owns robot Z.

The export also writes processed and rectified PNG companions for each state
next to the audit JSON. Their paths and hashes are recorded in the project so
the click coordinates can be reviewed without relying on a mutable camera
source file.

The task file declares `source_pose_frame` as `board_local_annotation`, so it
must be reviewed by the robot-code owner before replacing any existing
organizer or runtime target file. Missing or duplicate part annotations are
listed in its `warnings` array rather than being silently discarded.

Crop and rotate are available in source view. Either operation resets that
image's board calibration and annotations, which prevents stale pixel geometry
from being mixed with a newly transformed photograph.
