class Cube:
    def __init__(self, corners, edges):
        # corners = [(piece_id, ori), ... length 8]
        # edges   = [(piece_id, ori), ... length 12]
        self.corners = corners[:]   # shallow copy of list is OK here
        self.edges   = edges[:]

    def log_cube(self):
      print("corners", self.corners)
      print("edges", self.edges)

    def clone(self):
        new_corners = [(pid, ori) for (pid, ori) in self.corners]
        new_edges   = [(pid, ori) for (pid, ori) in self.edges]
        return Cube(new_corners, new_edges)

    def apply_move(self, move):
        if move == "U":
            self._move_U()
        elif move == "U'":
            self._move_U(); self._move_U(); self._move_U()
        elif move == "U2":
            self._move_U(); self._move_U()
        elif move == "D":
            self._move_D()
        elif move == "D'":
            self._move_D(); self._move_D(); self._move_D()
        elif move == "D2":
            self._move_D(); self._move_D()
        elif move == "F":
            self._move_F()
        elif move == "F'":
            self._move_F(); self._move_F(); self._move_F()
        elif move == "F2":
            self._move_F(); self._move_F()
        elif move == "B":
            self._move_B()
        elif move == "B'":
            self._move_B(); self._move_B(); self._move_B()
        elif move == "B2":
            self._move_B(); self._move_B()
        elif move == "L":
            self._move_L()
        elif move == "L'":
            self._move_L(); self._move_L(); self._move_L()
        elif move == "L2":
            self._move_L(); self._move_L()
        elif move == "R":
            self._move_R()
        elif move == "R'":
            self._move_R(); self._move_R(); self._move_R()
        elif move == "R2":
            self._move_R(); self._move_R()
        else:
            raise NotImplementedError(f"Move {move} not implemented yet.")

    def _move_U(self):
        """Clockwise U turn of the Up face."""
        # ---- Corners cycle ----
        # positions: 0 → 1 → 3 → 2 → 0
        c = self.corners
        c[0], c[1], c[3], c[2] = c[2], c[0], c[1], c[3]

        # ---- Edges cycle ----
        # positions: 0 → 1 → 2 → 3 → 0
        e = self.edges
        e[0], e[1], e[2], e[3] = e[3], e[0], e[1], e[2]

    def _move_D(self):
      c = self.corners
      e = self.edges
      # corners: 4 -> 5 -> 7 -> 6 -> 4
      c[4], c[5], c[7], c[6] = c[6], c[4], c[5], c[7]

      # edges: 4 -> 5 -> 6 -> 7 -> 4
      e[4], e[5], e[6], e[7] = e[7], e[4], e[5], e[6]

    def _move_F(self):
      c = self.corners
      e = self.edges
      # Permute
      # corners: 2 -> 3 -> 5 -> 4 -> 2
      c[2], c[3], c[5], c[4] = c[4], c[2], c[3], c[5]

      # edges: 2 -> 9 -> 4 -> 8 -> 2
      e[2], e[9], e[4], e[8] = e[8], e[2], e[9], e[4]

      # Orient
      # corners: 
      for i in (2, 3, 4, 5):
        pid, ori = c[i]
        if i in (3, 4):   
          ori = (ori + 1) % 3
        else:             
          ori = (ori + 2) % 3  # -1 mod 3
        c[i] = (pid, ori)
      
      #edges:
      for i in (2, 4, 8, 9):
        pid, ori = e[i]
        e[i] = (pid, ori ^ 1)
    
    def _move_B(self):
      c = self.corners
      e = self.edges
      # Permute
      # corners: 0 -> 6 -> 7 -> 1 -> 0
      c[0], c[6], c[7], c[1] = c[1], c[0], c[6], c[7]

      # edges: 0 -> 10 -> 6 -> 11
      e[0], e[10], e[6], e[11] = e[11], e[0], e[10], e[6]

      # Orient
      # corners:
      for i in (0, 1, 6, 7):
        pid, ori = c[i]
        if i in (0, 7):
          ori = (ori + 1) % 3
        else:
          ori = (ori + 2) % 3
        c[i] = (pid, ori)
      # edges:
      for i in (0, 6, 10, 11):
        pid, ori = e[i]
        e[i] = (pid, ori ^ 1)

    def _move_L(self):
      c = self.corners
      e = self.edges
      # Permute
      # corners: 0 -> 2 -> 4 -> 6 -> 0
      c[0], c[2], c[4], c[6] = c[6], c[0], c[2], c[4]

      # edges: 3 -> 8 -> 7 -> 10 -> 3
      e[3], e[8], e[7], e[10] = e[10], e[3], e[8], e[7]

      # Orient
      # corners:
      for i in (0, 2, 4, 6):
        pid, ori = c[i]
        if i in (2, 6):
          ori = (ori + 1) % 3
        else:
          ori = (ori + 2) % 3
        c[i] = (pid, ori)

    def _move_R(self):
      c = self.corners
      e = self.edges
      # Permute
      # corners: 1 -> 7 -> 5 -> 3 -> 1
      c[1], c[7], c[5], c[3] = c[3], c[1], c[7], c[5]

      # edges: 1 -> 11 -> 5 -> 9 -> 1
      e[1], e[11], e[5], e[9] = e[9], e[1], e[11], e[5]

      # Orient
      # corners:
      for i in (1, 3, 5, 7):
        pid, ori = c[i]
        if i in (1, 5):
          ori = (ori + 1) % 3
        else:
          ori = (ori + 2) % 3
        c[i] = (pid, ori)

    def is_solved(self):
        """
        A cube is solved if every position holds its own piece_id and oriented.
        """
        for pos, (pid, ori) in enumerate(self.corners):
            if pid != pos or ori != 0:
                return False
        for pos, (pid, ori) in enumerate(self.edges):
            if pid != pos or ori != 0:
                return False
        return True
