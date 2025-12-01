import Facelet_to_Cube
from IDASolver import IDASolver
from Cube import Cube

class CubeSolver:

    def RubikAStar(self, input_obj):
        cube = Facelet_to_Cube.facelet_to_cube(input_obj)
        solver = IDASolver(cube)
        return solver.solve(max_depth=21)

