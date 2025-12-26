import csv
import re
from sympy import Point, Polygon
from sympy.geometry.polygon import Triangle
from sympy.utilities.iterables import rotate_left


def verify_triangles(input: str) -> str:
    lines = input.strip().split('\n')
    if len(lines) != 2:
        return "Wrong answer"

    l1, l2 = [l.strip() for l in lines]

    if l1.startswith('2') and l2.startswith('1'):
        l1, l2 = l2, l1

    if not l1.startswith('1') or not l2.startswith('2'):
        return "Wrong answer"

    # parse line 1 of the input into "1; A; B", where A is an array of 5 2-vectors and B is an array of 3 2-vectors
    try:
        _, A1, B1 = [l for l in l1.split(';') if l != ""]
        A1_vec = get_points(A1)
        B1_vec = get_points(B1)
    except Exception as e:
        print(e)
        return "Wrong format of an example (n=1)"

    a1_poly = Polygon(*A1_vec)
    b1_poly = Polygon(*B1_vec)
    # Polygon constructor may return Triangle, Segment, or Point for <= 3 points
    # Only Polygon and Triangle have is_convex() method
    # if not all(isinstance(poly, (Polygon, Triangle)) for poly in [a1_poly, b1_poly]):
    #    return "Point do not define a polygon (n=1)"
    # if not all(poly.is_convex() for poly in [a1_poly, b1_poly]): # type: ignore
    #    return "Point do not define a convex polygon (n=1)"

    try:
        for vec in [A1_vec, B1_vec]:
            if len(concavities(vec)) > 0:
                return "Polygon not convex (n=1)"
    except Exception as e:
        return str(e) + " (n=1)"

    # reorder A cyclically so that { A[0], A[1], A[2] } = { B[0], B[1], B[2] } as sets
    try:
        reorder_points(A1_vec, ([0, 1, 2], B1_vec))
    except Exception:
        return "Wrong position of the trangle (n=1)"

    if not orient(A1_vec[0], A1_vec[1], A1_vec[2], A1_vec[3]) < 0:
        return "Wrong shape of the pentagon (n=1)"
    if not orient(A1_vec[4], A1_vec[0], A1_vec[1], A1_vec[2]) < 0:
        return "Wrong shape of the pentagon (n=1)"

    #   parse line 2 of the input into "2; A; B, C", where A is an array of 5 2-vectors, and B and C are arrays of 3 2-vectors
    try:
        _, A2, B2, C2 = [l for l in l2.split(';') if l != ""]
        A2_vec = get_points(A2)
        B2_vec = get_points(B2)
        C2_vec = get_points(C2)
    except Exception as e:
        print(e)
        return "Wrong format of an example (n=2)"

    # # Check that all polygons are valid (not Segments or Points) and convex
    # if not all(poly.is_convex() for vec in [A2_vec, B2_vec, C2_vec] if isinstance(poly:=Polygon(*vec), (Polygon, Triangle))):
    #     return "Point do not define a convex polygon (n=2)"

    try:
        for vec in [A2_vec, B2_vec, C2_vec]:
            if len(concavities(vec)) > 0:
                return "Polygon not convex (n=2)"
    except Exception as e:
        return str(e) + " (n=2)"

    # reorder A cyclically so that { A[0], A[1], A[3] } = { B[0], B[1], B[2] } and { A[0], A[2], A[4] } = { C[0], C[1], C[2] } as sets
    try:
        # TODO: Its also fine if exchanging B and C passes this test! --> adapt the reorder code to implement this.
        # Done with the second call of reorder_points(): the resulting reordering of A2_vec is still ok for what follows
        reorder_points(A2_vec, ([0, 1, 3], B2_vec), ([0, 2, 4], C2_vec))
    except Exception:
        try:
            reorder_points(A2_vec, ([0, 1, 3], C2_vec), ([0, 2, 4], B2_vec))
        except Exception:
            return "Wrong position of a triangle (n=2)"

    if not orient(A2_vec[4], A2_vec[0], A2_vec[1], A2_vec[3]) < 0:
        return "Wrong shape of the pentagon (n=2)"
    if not orient(A2_vec[4], A2_vec[0], A2_vec[1], A2_vec[2]) > 0:
        return "Wrong shape of the pentagon (n=2)"
    if not orient(A2_vec[2], A2_vec[4], A2_vec[0], A2_vec[1]) < 0:
        return "Wrong shape of the pentagon (n=2)"
    if not orient(A2_vec[3], A2_vec[4], A2_vec[0], A2_vec[1]) > 0:
        return "Wrong shape of the pentagon (n=2)"
    if not orient(A2_vec[1], A2_vec[2], A2_vec[3], A2_vec[0]) > 0:
        return "Wrong shape of the pentagon (n=2)"
    if not orient(A2_vec[0], A2_vec[2], A2_vec[3], A2_vec[4]) > 0:
        return "Wrong shape of the pentagon (n=2)"

    return "True"


def get_points(s: str) -> list[Point]:
    # Match patterns of the form (float,float)
    pattern = r'\(([-\d.]+),([-\d.]+)\)'
    matches = re.findall(pattern, s)
    return [
        Point(float(x), float(y))
        for x, y in matches
    ]


def orient(P: Point, Q: Point, R: Point, S: Point) -> int:
    PQ = P - Q
    RQ = R - Q
    SR = S - R

    det1 = det(PQ, RQ)
    det2 = det(PQ, SR)
    return sign(det1 * det2)


def det(v1: Point, v2: Point) -> float:
    return float(v1[0] * v2[1] - v1[1] * v2[0])


def sign(x: float) -> int:
    return 1 if x > 0 else -1 if x < 0 else 0


def reorder_points(A_vec: list[Point], *constraints: tuple[list[int], list[Point]]) -> None:
    """
    Reorder A_vec cyclically to satisfy one or more constraints.

    Each constraint is a tuple of (indices, target_points) where:
    - indices: list of indices into A_vec that should match target_points as a set
    - target_points: list of points that the selected indices should match

    Examples:
    - reorder_points(A_vec, ([0, 1, 2], B_vec)) matches {A[0], A[1], A[2]} = {B[0], B[1], B[2]}
    - reorder_points(A_vec, ([0, 1, 3], B_vec), ([0, 2, 4], C_vec)) matches both constraints
    """
    for indices, target_points in constraints:
        if len(set(target_points)) != len(target_points):
            raise ValueError("Target points must be unique")
        if len(indices) != len(target_points):
            raise ValueError("Number of indices must match number of target points")
        if any(idx >= len(A_vec) for idx in indices):
            raise ValueError("Index out of range for A_vec")

    for start_idx in range(len(A_vec)):
        rotated = rotate_left(A_vec, start_idx)

        all_match = True
        for indices, target_points in constraints:
            selected_points = [rotated[idx] for idx in indices]
            if not compare_points(selected_points, target_points):
                all_match = False
                break

        if all_match:
            A_vec[:] = rotated
            return

    raise ValueError("Could not reorder points to satisfy constraints")


def reorder_points_dual(A_vec: list[Point], B_vec: list[Point], C_vec: list[Point]) -> None:
    """
    Reorder A cyclically so that:
    - {A[0], A[1], A[3]} = {B[0], B[1], B[2]} as sets
    - {A[0], A[2], A[4]} = {C[0], C[1], C[2]} as sets
    """
    if len(B_vec) != 3 or len(C_vec) != 3:
        raise ValueError("B and C must each have 3 unique points")
    if len(A_vec) != 5:
        raise ValueError("A must have 5 points")

    reorder_points(A_vec, ([0, 1, 3], B_vec), ([0, 2, 4], C_vec))


def compare_points(vec1: list[Point], vec2: list[Point]) -> bool:
    """Compare two lists of points as sets. Points are compared by their coordinates."""
    return set(vec1) == set(vec2)


def segments_intersect(A: Point, B: Point, C: Point, D: Point) -> bool:
    """check that AB intersects CD (possibly by the end points)"""
    return (det(A - C, A - D) * det(B - C, B - D) <= 0) and (det(C - A, C - B) * det(D - A, D - B) <= 0)


def concavities(p: list[Point]) -> list[int]:
    """returns the list of indices of concave vertices of a polygon (empty for a convex polygon), raises errors if not polygon"""
    # more than 2 vertices?
    if len(p) < 3:
        raise ValueError("Less than 3 vertices in polygon " + str(p))
    # Repeated vertex?
    for i in range(len(p)):
        if p[i] == p[(i + 1) % len(p)]:
            raise ValueError("Repeated vertex " + str(i) + " in polygon " + str(p))
    # An angle is a multiple of \pi?
    for i in range(len(p)):
        if det(p[i] - p[(i + 1) % len(p)], p[i] - p[(i - 1) % len(p)]) == 0:
            raise ValueError("Degenerate vertex " + str(i) + " in polygon " + str(p))
    # the first-to-last edge intersects a non-consecutive edge?
    for i in range(1, len(p) - 2):
        if segments_intersect(p[i], p[i + 1], p[0], p[len(p) - 1]):
            raise ValueError("Intersecting edges -1 and " + str(i) + " in polygon " + str(p))
    # two other non-consecutive edges intersect?
    for i in range(len(p) - 3):
        for j in range(i + 2, len(p) - 1):
            if segments_intersect(p[i], p[i + 1], p[j], p[j + 1]):
                raise ValueError("Intersecting edges " + str(i) + " and " + str(j) + " in polygon " + str(p))
    # clockwise or counterclockwise?
    ar = 0
    for i in range(len(p)):
        ar += det(p[i], p[(i + 1) % len(p)])
    # list concave angles
    conc = []
    for i in range(len(p)):
        if sign(det(p[(i - 1) % len(p)] - p[i], p[(i + 1) % len(p)] - p[i])) == sign(ar):
            conc.append(i)
    return conc


if __name__ == "__main__":

    #print(get_points( " (0,5), (0,0), (10,0)"))
    #quit()

    results = []
    reader = csv.reader(open("input.csv"))  # , quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:  # each row is a list
        results.append(row)

    for row in results:
        row.append(verify_triangles(row[0]))

    writer = csv.writer(open('output.csv', 'w'), delimiter=',', lineterminator='\n')
    for row in results: writer.writerow([row])

    quit()

    input_text = """1; (0,0), (6,0), (5,2), (2,5), (0,6); (0,0), (6,0), (0,6);
    2; (0,0), (6,0), (7,5), (5,7), (6,0); (0,0), (6,0), (5,7); (0,0), (7,5), (6,0);  
    """

    verified = verify_triangles(input_text)
    print(verified)

    quit()

    try:
        print(concavities([Point(-1, 7), Point(1, 7), Point(6, 6), Point(0, 9), Point(-6, 6)]))
    except ValueError as e:
        print(e)