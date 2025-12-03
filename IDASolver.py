import Facelet_to_Cube
from Cube import Cube
import os
import pickle
import json

PDB_DIR = "pdb_cache"
os.makedirs(PDB_DIR, exist_ok=True)

# Create heuristic tables for corner orientations
# encode: [ori0, ori1, ..., ori6] -> int in [0, 3**7)
def encode_corner_ori(corners):
    code = 0
    for i in range(7):
        code = 3 * code + corners[i][1]  # use ori (ignore pid)
    return code

from collections import deque
import math

def build_corner_ori_pdb():
    # 3^7 states
    N = 3**7
    pdb = [math.inf] * N

    # solved cube
    solved = Cube(
        corners=[(i, 0) for i in range(8)],
        edges=[(i, 0) for i in range(12)],
    )

    start_code = encode_corner_ori(solved.corners)
    pdb[start_code] = 0

    q = deque()
    q.append(solved)

    MOVES = ["U","U'","D","D'","F","F'","B","B'","R","R'","L","L'"]

    while q:
        cube = q.popleft()
        depth = pdb[encode_corner_ori(cube.corners)]

        for m in MOVES:
            nxt = cube.clone()
            nxt.apply_move(m)

            code = encode_corner_ori(nxt.corners)
            if pdb[code] == math.inf:
                pdb[code] = depth + 1
                q.append(nxt)

    return pdb
corner_ori_pdb = 0

if os.path.exists(os.path.join(PDB_DIR, "corner_ori.pkl")):
    with open(os.path.join(PDB_DIR, "corner_ori.pkl"), "rb") as f:
        corner_ori_pdb = pickle.load(f)
else:
    corner_ori_pdb = build_corner_ori_pdb()
    with open(os.path.join(PDB_DIR, "corner_ori.pkl"), "wb") as f:
        pickle.dump(corner_ori_pdb, f, protocol=pickle.HIGHEST_PROTOCOL)

def h_corner_ori(cube):
    code = encode_corner_ori(cube.corners)
    return corner_ori_pdb[code]

# edges[0].ori is the most significant bit,
# edges[10].ori is the least significant bit.
def encode_edge_ori(edges):
    code = 0
    for i in range(11):              # 0..10
        _, ori = edges[i]
        code = (code << 1) | (ori & 1)
    return code  # in [0, 2**11)
from collections import deque
import math

MOVES = ["U", "U'", "D", "D'",
         "F", "F'", "B", "B'",
         "R", "R'", "L", "L'"]

def encode_edge_ori(edges):
    code = 0
    for i in range(11):
        _, ori = edges[i]
        code = (code << 1) | (ori & 1)
    return code


def build_edge_ori_pdb():
    NUM_STATES = 2**11   # 2048 possible orientation patterns
    INF = math.inf
    pdb = [INF] * NUM_STATES

    # solved cube: pid == position, ori == 0
    solved = Cube(
        corners=[(i, 0) for i in range(8)],
        edges=[(i, 0) for i in range(12)],
    )

    start_code = encode_edge_ori(solved.edges)
    pdb[start_code] = 0

    q = deque()
    q.append(solved)

    while q:
        cube = q.popleft()
        cur_code = encode_edge_ori(cube.edges)
        cur_dist = pdb[cur_code]

        for m in MOVES:
            nxt = cube.clone()
            nxt.apply_move(m)

            code = encode_edge_ori(nxt.edges)
            if pdb[code] is INF:
                pdb[code] = cur_dist + 1
                q.append(nxt)

    return pdb

edge_ori_pdb = 0

if os.path.exists(os.path.join(PDB_DIR, "edge_ori.pkl")):
    with open(os.path.join(PDB_DIR, "edge_ori.pkl"), "rb") as f:
        edge_ori_pdb = pickle.load(f)
else:
    edge_ori_pdb = build_edge_ori_pdb()
    with open(os.path.join(PDB_DIR, "edge_ori.pkl"), "wb") as f:
        pickle.dump(edge_ori_pdb, f, protocol=pickle.HIGHEST_PROTOCOL)



def h_edge_ori(cube):
    code = encode_edge_ori(cube.edges)
    return edge_ori_pdb[code]

from math import comb  # Python 3.8+

COMB = [[0]*5 for _ in range(13)]
for n in range(13):
    for k in range(5):
        if k <= n:
            COMB[n][k] = comb(n, k)

SLICE_EDGE_IDS = {8, 9, 10, 11}

def encode_udslice(edges):
    """
    edges: list of (pid, ori), len 12
    Returns: int in [0, 495) describing which positions contain the 4 UD-slice edges.
    """
    positions = [pos for pos, (pid, _) in enumerate(edges) if pid in SLICE_EDGE_IDS]
    positions.sort()
    assert len(positions) == 4, "Cube must have exactly 4 slice edges"

    p0, p1, p2, p3 = positions
    idx = (COMB[p0][1] +
           COMB[p1][2] +
           COMB[p2][3] +
           COMB[p3][4])
    return idx  # 0..494

EO_STATES = 2**11
UDSLICE_STATES = 495

def encode_phase1_coord(edges):
    eo = encode_edge_ori(edges)         # 0..2047
    us = encode_udslice(edges)          # 0..494
    return eo * UDSLICE_STATES + us     # 0..(EO_STATES*UDSLICE_STATES - 1)

from collections import deque
import math

MOVES = ["U", "U'", "D", "D'",
         "F", "F'", "B", "B'",
         "R", "R'", "L", "L'"]

EO_STATES = 2**11
UDSLICE_STATES = 495
PHASE1_TABLE_SIZE = EO_STATES * UDSLICE_STATES

def encode_edge_ori(edges):
    code = 0
    for i in range(11):
        _, ori = edges[i]
        code = (code << 1) | (ori & 1)
    return code

def encode_phase1_coord(edges):
    eo = encode_edge_ori(edges)
    us = encode_udslice(edges)
    return eo * UDSLICE_STATES + us


def build_phase1_eo_udslice_table():
    INF = 255  # enough for distances under 255 moves; we only need < 20
    table = bytearray([INF] * PHASE1_TABLE_SIZE)

    # solved cube
    solved = Cube(
        corners=[(i, 0) for i in range(8)],
        edges=[(i, 0) for i in range(12)],
    )

    start_idx = encode_phase1_coord(solved.edges)
    table[start_idx] = 0

    q = deque()
    q.append(solved)

    while q:
        cube = q.popleft()
        cur_idx = encode_phase1_coord(cube.edges)
        cur_dist = table[cur_idx]

        for m in MOVES:
            nxt = cube.clone()
            nxt.apply_move(m)

            idx = encode_phase1_coord(nxt.edges)
            if table[idx] == INF:
                table[idx] = cur_dist + 1
                q.append(nxt)

    return table

table = 0

if os.path.exists(os.path.join(PDB_DIR, "udslice.pkl")):
    with open(os.path.join(PDB_DIR, "udslice.pkl"), "rb") as f:
        table = pickle.load(f)
else:
    table = build_phase1_eo_udslice_table()
    with open(os.path.join(PDB_DIR, "udslice.pkl"), "wb") as f:
        pickle.dump(table, f, protocol=pickle.HIGHEST_PROTOCOL)

def h_phase1_eo_udslice(cube):
    idx = encode_phase1_coord(cube.edges)
    return table[idx]

def heuristic(cube) -> int:
    h1 = h_corner_ori(cube)
    h3 = h_edge_ori(cube)
    h_eo_us = h_phase1_eo_udslice(cube)

    return max(h1,h_eo_us, h3)

MOVES = ["U", "U'",
         "D", "D'",
         "F", "F'",
         "B", "B'",
         "R", "R'",
         "L", "L'"]

def move_face(m: str) -> str:
    # 'U' for 'U' or "U'"
    return m[0]

INVERSE_MOVE = {
    "U": "U'", "U'": "U",
    "D": "D'", "D'": "D",
    "F": "F'", "F'": "F",
    "B": "B'", "B'": "B",
    "R": "R'", "R'": "R",
    "L": "L'", "L'": "L",
}

import math

class IDASolver:
    def __init__(self):
        self.MOVES = MOVES
        self.heuristic = heuristic
        self.path = []  # list of moves leading to current node

    def solve(self, max_depth=25):
        """
        Return a list of moves solving the cube, or None
        if no solution found up to max_depth.
        """
        bound = self.heuristic(self.start)

        while bound <= max_depth:
            # print(f"IDA* bound = {bound}")
            t = self._search(self.start, g=0, bound=bound, last_move=None)
            if isinstance(t, list):
                # Found a solution path
                return t
            if t == math.inf:
                # No solution within this or any larger bound (in theory)
                return None
            bound = t  # increase to smallest f that exceeded previous bound

        return None

    def _search(self, cube, g, bound, last_move):
        """
        Depth-first search with pruning.

        Returns:
          - list (solution path) if found
          - float (min f that exceeded bound) otherwise
        """
        h = self.heuristic(cube)
        f = g + h
        if f > bound:
            return f
        if cube.is_solved():
            return self.path.copy()

        min_excess = math.inf

        for m in self.MOVES:
            # --- pruning ---
            if last_move is not None:
                # don't immediately undo the previous move
                if INVERSE_MOVE[last_move] == m:
                    continue
                # optional: disallow same face twice in a row (U U, R R, etc.)
                #if move_face(m) == move_face(last_move):
                    # allow sequences like "R R'"? here we already skipped inverse above
                    # so this mainly avoids "R R"
                    #continue

            new_cube = cube.clone()
            new_cube.apply_move(m)

            self.path.append(m)
            t = self._search(new_cube, g + 1, bound, last_move=m)
            self.path.pop()

            if isinstance(t, list):
                return t  # solution
            if t < min_excess:
                min_excess = t

        return min_excess

    def RubikAStar(self):
        formatted_obj = {}
        with open('AStar_in.json', 'r') as file:
            input_obj = json.load(file)
        input_obj = input_obj["cube"]
        for i in ("up", "down", "left", "right", "front", "back"):
            face = input_obj[i]
            formatted_obj[i] = face
        cube = Facelet_to_Cube.facelet_to_cube(formatted_obj)
        self.start = cube
        solution = self.solve(max_depth=21)
        solution = " ".join(solution)

        with open('AStar_out.txt', 'w') as file:
            file.write(solution)
