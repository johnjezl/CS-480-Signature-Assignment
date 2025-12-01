import Facelet_to_Cube
from IDASolver import IDASolver
from Cube import Cube

class CubeSolver:

    def RubikAStar(self, input_obj):
        formatted_obj = {}
        for i in ("up", "down", "left", "right", "front", "back"):
            face = input_obj[i]
            flat = []
            for j in range(3):
                flat.append(face[j][0])
                flat.append(face[j][1])
                flat.append(face[j][2])
            formatted_obj[i] = flat
        cube = Facelet_to_Cube.facelet_to_cube(formatted_obj)
        solver = IDASolver(cube)
        return solver.solve(max_depth=21)

