"""Core data structures for the package router."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Dict, Optional, Set, FrozenSet
import math


class Direction(Enum):
    """Segment direction categories for crosstalk grouping."""
    HORIZONTAL = auto()
    VERTICAL = auto()
    DIAG_POS = auto()  # NE-SW diagonal: slope = +1
    DIAG_NEG = auto()  # NW-SE diagonal: slope = -1


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def scaled(self, s: float) -> "Point":
        return Point(self.x * s, self.y * s)


@dataclass
class Segment:
    """A wire segment on a single layer."""
    start: Point
    end: Point
    layer: int

    @property
    def length(self) -> float:
        return self.start.distance_to(self.end)

    @property
    def direction(self) -> Direction:
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        if abs(dy) < 1e-9:
            return Direction.HORIZONTAL
        if abs(dx) < 1e-9:
            return Direction.VERTICAL
        if (dx > 0 and dy > 0) or (dx < 0 and dy < 0):
            return Direction.DIAG_POS
        return Direction.DIAG_NEG

    @property
    def min_x(self) -> float:
        return min(self.start.x, self.end.x)

    @property
    def max_x(self) -> float:
        return max(self.start.x, self.end.x)

    @property
    def min_y(self) -> float:
        return min(self.start.y, self.end.y)

    @property
    def max_y(self) -> float:
        return max(self.start.y, self.end.y)


@dataclass
class Via:
    """A via connecting two layers at a point."""
    pos: Point
    from_layer: int
    to_layer: int


@dataclass(frozen=True)
class ViaObstacle:
    """A physical via or pad-via occupying space on a set of layers."""
    center: Point
    width: float
    layers: FrozenSet[int]


@dataclass
class RoutePattern:
    """One candidate route for a net."""
    net_idx: int
    pattern_idx: int
    segments: List[Segment]
    vias: List[Via]
    wirelength: float = 0.0
    _layers: Optional[frozenset] = field(default=None, repr=False, compare=False)
    _bbox: Optional[Tuple[float, float, float, float]] = field(
        default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.wirelength == 0.0 and self.segments:
            self.wirelength = sum(s.length for s in self.segments)
        self._cache()

    def _cache(self):
        """Pre-cache derived properties for fast access."""
        self._layers = frozenset(s.layer for s in self.segments)
        if self.segments:
            xs, ys = [], []
            for s in self.segments:
                xs.extend([s.start.x, s.end.x])
                ys.extend([s.start.y, s.end.y])
            self._bbox = (min(xs), min(ys), max(xs), max(ys))
        else:
            self._bbox = (0.0, 0.0, 0.0, 0.0)

    @property
    def num_vias(self) -> int:
        return len(self.vias)

    @property
    def layers_used(self) -> frozenset:
        if self._layers is None:
            self._cache()
        return self._layers

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """(min_x, min_y, max_x, max_y)"""
        if self._bbox is None:
            self._cache()
        return self._bbox


@dataclass
class Pad:
    """An IO pad on a die."""
    name: str
    die: str
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)

    @property
    def center(self) -> Point:
        return Point(
            (self.bbox[0] + self.bbox[2]) / 2.0,
            (self.bbox[1] + self.bbox[3]) / 2.0,
        )

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class Net:
    """A 2-pin chip-to-chip net."""
    idx: int
    name: str
    pad1_name: str
    pad2_name: str
    pad1: Pad
    pad2: Pad

    @property
    def pt1(self) -> Point:
        return self.pad1.center

    @property
    def pt2(self) -> Point:
        return self.pad2.center

    @property
    def hpwl(self) -> float:
        """Half-perimeter wirelength (Manhattan distance)."""
        return abs(self.pt1.x - self.pt2.x) + abs(self.pt1.y - self.pt2.y)


@dataclass
class DesignParams:
    """Global design parameters from .gp file."""
    pkg_width: float
    pkg_height: float
    num_layers: int
    wire_width: float
    via_width: float
    pitch: float
    spacing: float  # minimum edge-to-edge spacing between wires

    @property
    def min_center_distance(self) -> float:
        """Minimum center-to-center distance between two wires."""
        return self.wire_width + self.spacing


@dataclass
class Package:
    """Complete package description."""
    params: DesignParams
    dies: Dict[str, Dict[str, Pad]]   # die_name -> {pad_name: Pad}
    nets: List[Net]
    bumps: List[Tuple[float, float, float, float]]
    all_pads: Dict[str, Pad] = field(default_factory=dict)

    def __post_init__(self):
        if not self.all_pads:
            for die_name, pads in self.dies.items():
                self.all_pads.update(pads)


@dataclass
class RoutingSolution:
    """Complete routing solution for all nets."""
    package: Package
    selected: List[int]               # selected[i] = index of chosen pattern for net i
    candidates: List[List[RoutePattern]]  # candidates[i] = list of patterns for net i
    total_wirelength: float = 0.0
    total_crosstalk: float = 0.0
    max_crosstalk: float = 0.0
    num_vias: int = 0
    num_crossings: int = 0
    num_drc_violations: int = 0

    def get_route(self, net_idx: int) -> RoutePattern:
        return self.candidates[net_idx][self.selected[net_idx]]

    @property
    def all_routes(self) -> List[RoutePattern]:
        return [self.get_route(i) for i in range(len(self.selected))]
