#!/usr/bin/env python3
"""Rerun visualization wrapper for easier 3D visualization.

This module provides a high-level wrapper around the Rerun SDK to simplify
common visualization tasks like logging point clouds, coordinate frames,
camera poses, and paths.

Example:
    from src.utils.rerun_visualizer import RerunVisualizer
    
    vis = RerunVisualizer("MyVisualization", spawn=True, fps=5.0)
    vis.log_point_cloud("cloud/points", points, colors)
    vis.log_coordinate_frame("camera", pose_matrix)
    vis.set_frame(0)
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


class RerunVisualizer:
    """High-level wrapper for Rerun visualization.
    
    This class provides convenient methods for common 3D visualization tasks
    including point clouds, coordinate frames, camera poses, paths, and lines.
    """
    
    # Default colors (RGBA, 0-255)
    DEFAULT_AXIS_COLORS = np.array(
        [
            [255, 0, 0, 255],  # Red (X)
            [0, 255, 0, 255],  # Green (Y)
            [0, 0, 255, 255],  # Blue (Z)
        ],
        dtype=np.uint8,
    )
    
    DEFAULT_CAMERA_PATH_COLOR = np.array([[255, 200, 0, 255]], dtype=np.uint8)  # Orange
    DEFAULT_LINE_COLOR = np.array([[255, 0, 255, 255]], dtype=np.uint8)  # Magenta
    
    def __init__(
        self,
        name: str,
        spawn: bool = True,
        fps: float | None = None,
        view_coordinates: str = "RDF",
    ):
        """Initialize Rerun visualizer.
        
        Args:
            name: Name of the visualization session
            spawn: Whether to spawn a separate viewer window
            fps: Playback speed (frames per second). If None, timeline speed is not set.
            view_coordinates: View coordinate system (default: "RDF" for Right-Down-Forward)
        """
        self._import_rerun()
        self.rr.init(name, spawn=spawn)
        
        # Set view coordinates
        try:
            if view_coordinates == "RDF":
                self.rr.log("world", self.rr.ViewCoordinates.RDF)
            else:
                # Allow other coordinate systems if needed
                self.rr.log("world", getattr(self.rr.ViewCoordinates, view_coordinates))
        except Exception:
            pass
        
        # Set playback speed if FPS is provided
        if fps is not None and fps > 1e-6:
            try:
                seconds_per_timeline = 1.0 / fps
                self.rr.set_time_seconds_per_timeline("frame", seconds_per_timeline)
            except AttributeError:
                # API not available in this version
                pass
            except Exception:
                pass
        
        self._name = name
    
    def _import_rerun(self):
        """Import rerun module and store it."""
        try:
            import rerun as rr  # type: ignore
        except Exception as exc:
            raise RuntimeError("Rerun package is required. Install with `pip install rerun-sdk`.") from exc
        self.rr = rr
    
    def set_frame(self, frame_id: int | str, timeline: str = "frame"):
        """Set the current timeline frame.
        
        Args:
            frame_id: Frame identifier (can be int or string)
            timeline: Timeline name (default: "frame")
        """
        try:
            time_idx = int(frame_id)
        except ValueError:
            time_idx = frame_id
        self.rr.set_time(timeline, sequence=time_idx)
    
    def log_transform(self, entity: str, pose: np.ndarray):
        """Log a 3D transform (pose).
        
        Args:
            entity: Entity path (e.g., "camera/pose")
            pose: 4x4 transformation matrix
        """
        self.rr.log(
            entity,
            self.rr.Transform3D(
                translation=pose[:3, 3].astype(np.float32),
                mat3x3=pose[:3, :3].astype(np.float32),
            ),
        )
    
    def log_coordinate_frame(
        self,
        entity: str,
        pose: np.ndarray,
        axis_len: float = 0.05,
        colors: np.ndarray | None = None,
    ):
        """Log a coordinate frame with axes.
        
        Args:
            entity: Entity path (e.g., "camera/frame")
            pose: 4x4 transformation matrix
            axis_len: Length of axis arrows in meters
            colors: Optional 3x4 array of RGBA colors for X, Y, Z axes (0-255).
                   If None, uses default red/green/blue colors.
        """
        # Log the transform
        self.log_transform(entity, pose)
        
        # Set default colors if not provided
        if colors is None:
            colors = self.DEFAULT_AXIS_COLORS
        
        # Log axes
        origins = np.zeros((3, 3), dtype=np.float32)
        vectors = (np.eye(3, dtype=np.float32) * axis_len).astype(np.float32)
        self.rr.log(
            f"{entity}/axes",
            self.rr.Arrows3D(
                origins=origins,
                vectors=vectors,
                colors=colors,
                radii=np.full(3, axis_len * 0.05, dtype=np.float32),
            ),
        )
    
    def log_point_cloud(
        self,
        entity: str,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        radius: float = 0.002,
        clear: bool = False,
    ):
        """Log a point cloud.
        
        Args:
            entity: Entity path (e.g., "cloud/points")
            points: Point positions (N, 3) array
            colors: Optional point colors (N, 3) or (N, 4) array in 0-255 range.
                   If None, points will be white.
            radius: Point radius in meters
            clear: Whether to clear the entity before logging (useful for frame-by-frame updates)
        """
        if clear:
            self.rr.log(entity, self.rr.Clear(recursive=False))
        
        if colors is None:
            # Default to white if no colors provided
            colors = np.full((points.shape[0], 3), 255, dtype=np.uint8)
        elif colors.dtype != np.uint8:
            # Ensure colors are in 0-255 range
            if colors.max() <= 1.0:
                colors = (colors * 255.0).astype(np.uint8)
            else:
                colors = colors.astype(np.uint8)
        
        # Ensure colors have correct shape
        if colors.shape[1] == 3:
            # Add alpha channel if missing
            colors_with_alpha = np.zeros((colors.shape[0], 4), dtype=np.uint8)
            colors_with_alpha[:, :3] = colors
            colors_with_alpha[:, 3] = 255
            colors = colors_with_alpha
        
        self.rr.log(
            entity,
            self.rr.Points3D(
                positions=points.astype(np.float32),
                colors=colors,
                radii=radius,
            ),
        )
    
    def log_camera_pose(
        self,
        entity: str,
        pose: np.ndarray,
        axis_len: float = 0.05,
        colors: np.ndarray | None = None,
    ):
        """Log a camera pose with coordinate frame axes.
        
        Args:
            entity: Entity path (e.g., "camera/pose")
            pose: 4x4 transformation matrix
            axis_len: Length of axis arrows in meters
            colors: Optional 3x4 array of RGBA colors for axes
        """
        self.log_transform(entity, pose)
        
        if colors is None:
            colors = self.DEFAULT_AXIS_COLORS
        
        self.rr.log(
            f"{entity}/axes",
            self.rr.Arrows3D(
                origins=np.zeros((3, 3), dtype=np.float32),
                vectors=np.eye(3, dtype=np.float32) * axis_len,
                colors=colors,
            ),
        )
    
    def log_camera_path(
        self,
        entity: str,
        positions: List[np.ndarray] | np.ndarray,
        radius: float = 0.002,
        color: np.ndarray | None = None,
    ):
        """Log a camera path as a line strip.
        
        Args:
            entity: Entity path (e.g., "camera/path")
            positions: List of 3D positions or array of shape (N, 3)
            radius: Line radius in meters
            color: Optional RGBA color (4,) array in 0-255 range
        """
        if isinstance(positions, list):
            positions_array = np.asarray(positions, dtype=np.float32)
        else:
            positions_array = positions.astype(np.float32)
        
        if color is None:
            color = self.DEFAULT_CAMERA_PATH_COLOR
        
        self.rr.log(
            entity,
            self.rr.LineStrips3D(
                [positions_array],
                radii=radius,
                colors=color,
            ),
        )
    
    def log_line(
        self,
        entity: str,
        start: np.ndarray,
        end: np.ndarray,
        radius: float = 0.002,
        color: np.ndarray | None = None,
    ):
        """Log a line segment between two points.
        
        Args:
            entity: Entity path (e.g., "camera_to_aruco")
            start: Start point (3,) array
            end: End point (3,) array
            radius: Line radius in meters
            color: Optional RGBA color (4,) array in 0-255 range
        """
        if color is None:
            color = self.DEFAULT_LINE_COLOR
        
        self.rr.log(
            entity,
            self.rr.LineStrips3D(
                [np.array([start, end], dtype=np.float32)],
                radii=radius,
                colors=color,
            ),
        )
    
    def log_lines(
        self,
        entity: str,
        lines: List[Tuple[np.ndarray, np.ndarray]] | np.ndarray,
        radius: float = 0.002,
        color: np.ndarray | None = None,
    ):
        """Log multiple line segments.
        
        Args:
            entity: Entity path (e.g., "lines")
            lines: List of (start, end) tuples or array of shape (N, 2, 3)
            radius: Line radius in meters
            color: Optional RGBA color (4,) array in 0-255 range
        """
        if isinstance(lines, list):
            # Convert list of tuples to array
            line_array = np.array([np.array([start, end], dtype=np.float32) for start, end in lines])
        else:
            line_array = lines.astype(np.float32)
        
        if color is None:
            color = self.DEFAULT_LINE_COLOR
        
        # Expand color to match number of lines if needed
        if color.shape[0] == 1 and line_array.shape[0] > 1:
            color = np.repeat(color, line_array.shape[0], axis=0)
        
        self.rr.log(
            entity,
            self.rr.LineStrips3D(
                line_array,
                radii=radius,
                colors=color,
            ),
        )
    
    def clear(self, entity: str, recursive: bool = False):
        """Clear an entity.
        
        Args:
            entity: Entity path to clear
            recursive: Whether to clear child entities recursively
        """
        self.rr.log(entity, self.rr.Clear(recursive=recursive))
    
    def log_image(self, entity: str, image: np.ndarray):
        """Log a 2D image.
        
        Args:
            entity: Entity path (e.g., "camera/image")
            image: Image array (H, W, 3) or (H, W) in uint8 format
        """
        self.rr.log(entity, self.rr.Image(image))
    
    def log_text(self, entity: str, text: str):
        """Log text annotation.
        
        Args:
            entity: Entity path (e.g., "info/text")
            text: Text string to display
        """
        self.rr.log(entity, self.rr.TextLog(text))

