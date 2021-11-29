# N x NAtoms x 3
def put_at_center(coordinates):
    coordinates -= coordinates.mean(axis=1).reshape([-1, 1, 3])
    return coordinates
