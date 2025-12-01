from Cube import Cube

corner_notation = {
    "YGO": 0,
    "YGR": 1,
    "YBO": 2,
    "YBR": 3,
    "WBO": 4,
    "WBR": 5,
    "WGO": 6,
    "WGR": 7
}

edge_notation = {
    "YG": 0,
    "YR": 1,
    "YB": 2,
    "YO": 3,
    "WB": 4,
    "WR": 5,
    "WG": 6,
    "WO": 7,
    "BO": 8,
    "BR": 9,
    "GO": 10,
    "GR": 11
}

def corner_orientation(idx,x,y,z):
  if y == "W" or y == "Y":
    return 0
  if x == "W" or x == "Y":
    if idx == 1 or idx == 2 or idx == 5 or idx == 6:
      return 2
    return 1
  if z == "W" or z == "Y":
    if idx == 1 or idx == 2 or idx == 5 or idx == 6:
      return 1
    return 2

def edge_orientation(idx,x,y):
  if x == "W" or x == "Y" or y == "W" or y == "Y":
    if x == "W" or x == "Y":
      return 0
    return 1
  if x == "B" or x == "G" or y == "B" or y == "G":
    if x == "B" or x == "G":
      return 0
    return 1

def format_piece(piece):
  return (piece["pid"], piece["ori"])

def facelet_to_piece(facelets):
  # Do corners first
  # An entry for each position in the cube describing the colors and orientation
  corners = [{"colors": [facelets["up"][0], facelets["back"][2], facelets["left"][0]], "ori": "0"},
            {"colors": [facelets["up"][2], facelets["back"][0], facelets["right"][2]], "ori": "0"},
            {"colors": [facelets["up"][6], facelets["front"][0], facelets["left"][2]], "ori": "0"},
            {"colors": [facelets["up"][8], facelets["front"][2], facelets["right"][0]], "ori": "0"},
            {"colors": [facelets["down"][0], facelets["front"][6], facelets["left"][8]], "ori": "0"},
            {"colors": [facelets["down"][2], facelets["front"][8], facelets["right"][6]], "ori": "0"},
            {"colors": [facelets["down"][6], facelets["back"][8], facelets["left"][6]], "ori": "0"},
            {"colors": [facelets["down"][8], facelets["back"][6], facelets["right"][8]], "ori": "0"}]
  for idx,piece in enumerate(corners):
    letters = ["A","A","A"]
    for color in piece["colors"]:
      if color == "W" or color == "Y":
        letters[0] = color
      if color == "B" or color == "G":
        letters[1] = color
      if color == "O" or color == "R":
        letters[2] = color
    piece["pid"] = corner_notation[letters[0]+letters[1]+letters[2]]
    piece["ori"] = corner_orientation(idx,piece["colors"][2], piece["colors"][0], piece["colors"][1])
  edges = [{"colors": [facelets["up"][1], facelets["back"][1]], "ori": "0"},
           {"colors": [facelets["up"][5], facelets["right"][1]], "ori": "0"},
           {"colors": [facelets["up"][7], facelets["front"][1]], "ori": "0"},
           {"colors": [facelets["up"][3], facelets["left"][1]], "ori": "0"},
           {"colors": [facelets["down"][1], facelets["front"][7]], "ori": "0"},
           {"colors": [facelets["down"][5], facelets["right"][7]], "ori": "0"},
           {"colors": [facelets["down"][7], facelets["back"][7]], "ori": "0"},
           {"colors": [facelets["down"][3], facelets["left"][7]], "ori": "0"},
           {"colors": [facelets["front"][3], facelets["left"][5]], "ori": "0"},
           {"colors": [facelets["front"][5], facelets["right"][3]], "ori": "0"},
           {"colors": [facelets["back"][5], facelets["left"][3]], "ori": "0"},
           {"colors": [facelets["back"][3], facelets["right"][5]], "ori": "0"}
           ]
  for idx, piece in enumerate(edges):
    letters = [piece["colors"][0], piece["colors"][1]]
    if letters[0] == "W" or letters[0] == "Y":
      letters = letters
    elif letters[1] == "W" or letters[1] == "Y":
      letters = [letters[1], letters[0]]
    elif letters[1] == "B" or letters[1] == "G":
      letters = [letters[1], letters[0]]
    piece["pid"] = edge_notation[letters[0]+letters[1]]
    piece["ori"] = edge_orientation(idx,piece["colors"][0],piece["colors"][1])
  return corners, edges

def facelet_to_cube(cube_obj):
    corners, edges = facelet_to_piece(cube_obj)
    final_corners = []
    for corner in corners:
      final_corners.append(format_piece(corner))
    final_edges = []
    for edge in edges:
      final_edges.append(format_piece(edge))

    new_cube = Cube(final_corners,final_edges)
    return new_cube
