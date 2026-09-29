"""Reusable head-camera scene perception for the physical Vega task board.

This is the live equivalent of tools/identify_board_snapshot.py. It deliberately
uses exactly the same board-corner and dark-part detector so offline validation
and robot-time behavior do not diverge.

Outputs are coarse/global: board corners in image/base coordinates and part
centers in board/base coordinates. Fine grasp alignment remains wrist-camera work.
"""

from __future__ import annotations

from .board import (
    board_frame_from_corners,
    detect_white_board_corners,
    head_left_optical_transform,
    pixels_to_horizontal_plane,
)
from .task_parts import (
    BOARD_SIZE_M,
    detect_dark_part_boxes,
    label_final_layout,
    project_part_boxes_to_image,
    rectify_board,
)


def detect_head_task_scene(
    rgb,
    camera_info,
    head_q_rad,
    *,
    plane_z_m,
    lift_m=0.0,
    torso_flip_rad=0.22689280275926285,
    layout="unlabeled",
    board_min_value=150,
    board_max_chroma=65,
    part_dark_threshold=140,
):
    """Detect board + parts and return a JSON-serializable scene record.

    layout="unlabeled" returns generic spatial detections.
    layout="final" attaches operator-confirmed assembled-layout names.
    Board-relative coordinates use the known physical 0.386 m square board,
    with origin at board center, +X TL->TR and +Y TL->BL.
    """
    import numpy as np
    from steadyhand.cameras.vega import intrinsics_from_camera_info

    fx, fy, cx, cy = intrinsics_from_camera_info(camera_info)
    corners_px = detect_white_board_corners(
        rgb,
        min_value=int(board_min_value),
        max_chroma=int(board_max_chroma),
    )

    rectified, H_image_to_board = rectify_board(rgb, corners_px)
    parts = detect_dark_part_boxes(
        rectified,
        dark_threshold=int(part_dark_threshold),
    )
    parts = project_part_boxes_to_image(parts, H_image_to_board)
    if layout == "final":
        parts = label_final_layout(parts)
    elif layout != "unlabeled":
        raise ValueError("layout must be unlabeled or final")

    T_base_camera = head_left_optical_transform(
        head_q_rad,
        lift_m=float(lift_m),
        torso_flip_rad=float(torso_flip_rad),
    )
    corners_base = pixels_to_horizontal_plane(
        corners_px,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        T_base_camera=T_base_camera,
        plane_z_m=float(plane_z_m),
    )
    T_base_board, observed_width_m, observed_height_m = board_frame_from_corners(
        corners_base
    )
    center_base = T_base_board[:3, 3]

    out_parts = []
    for part in parts:
        fx_board, fy_board = (float(v) for v in part["center_board_fraction"])
        board_local = np.asarray(
            [
                (fx_board - 0.5) * BOARD_SIZE_M,
                (fy_board - 0.5) * BOARD_SIZE_M,
                0.0,
                1.0,
            ],
            dtype=float,
        )
        base = T_base_board @ board_local
        item = dict(part)
        item["center_board_m"] = [
            float(board_local[0]),
            float(board_local[1]),
            0.0,
        ]
        item["center_base_m_coarse"] = [
            float(base[0]),
            float(base[1]),
            float(plane_z_m),
        ]
        out_parts.append(item)

    labels = ("tl", "tr", "br", "bl")
    return {
        "board": {
            "known_size_m": [BOARD_SIZE_M, BOARD_SIZE_M],
            "corners_px": {
                label: [int(x), int(y)]
                for label, (x, y) in zip(labels, corners_px)
            },
            "corners_base_m_coarse": {
                label: [float(v) for v in point]
                for label, point in zip(labels, corners_base)
            },
            "center_base_m_coarse": [float(v) for v in center_base],
            "observed_size_m_from_coarse_extrinsic": [
                float(observed_width_m),
                float(observed_height_m),
            ],
            "T_base_board_center": np.asarray(T_base_board, dtype=float).tolist(),
        },
        "parts": out_parts,
        "camera": {
            "intrinsics": {
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
            },
            "head_q_rad": [float(v) for v in head_q_rad],
            "plane_z_m": float(plane_z_m),
        },
    }


def render_scene_overlay(rgb, scene):
    """Return RGB image with green board edges and red part boxes/labels."""
    import cv2
    import numpy as np

    image = np.asarray(rgb).copy()
    corners_map = scene["board"]["corners_px"]
    corners = np.asarray(
        [corners_map[k] for k in ("tl", "tr", "br", "bl")],
        dtype=np.int32,
    )
    bgr = cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2BGR)
    cv2.polylines(bgr, [corners], True, (0, 255, 0), 3)

    for part in scene["parts"]:
        quad = np.rint(part["quad_image_px"]).astype(np.int32)
        cv2.polylines(bgr, [quad], True, (0, 0, 255), 2)
        x, y = (int(v) for v in quad[0])
        label = part.get("name") or str(part["index"])
        cv2.putText(
            bgr,
            label,
            (x, max(18, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
