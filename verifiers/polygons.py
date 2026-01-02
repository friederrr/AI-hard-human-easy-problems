import re
from sympy import Point
from dataclasses import dataclass
from typing import Callable, Iterable
from sympy.utilities.iterables import rotate_left


def verify_polygon_problem(input: str, problem_version: int) -> tuple[bool, str]:
    """

    Verifies an answer for a problem in the class of polygon problems.
    Expected input format:
    1. (0,0), (1,0), (0,1); (1,2); (2,1)
    2. (0,0), (1,0), (0,1); (1,2); (2,1); (0,2)
    ...

    Returns:
        tuple[bool, str]: bool is True if the answer is valid, False otherwise.
        str is an error message if the answer is invalid, empty string if the answer is valid.
    """
    
    if problem_version == 1:
            return verify_p1(input)

    if problem_version not in [2,3,4]:
        return False, f"Unknown problem version: {problem_version}"

    lines = input.strip().split('\n')   
    accepted_n = [1,2,3]
    
    for line in lines:
        n = int(line.split(' ')[0].strip())
        if n not in accepted_n:
            return False, f"Wrong line enumeration: {n} not in {accepted_n}"
        
        P, *T  = [get_points(p.strip()) for p in line.split(';') if p != ""]

        if len(T) != n * 3:
            return False, f"Wrong number of triangles for line {n}"

        try:
            if not all(area(*t) > 0 for t in T):
                return False, f"Degenerate triangle in line {n}"
        except (ValueError, TypeError):
            return False, f"Invalid triangle definition in line {n}"

        try:
            C = concavities(P)
        except (ValueError, TypeError):
            return False, f"Count not check pentagon P for concavities in line {n}"

        inputs = (P,T,C)
        match problem_version:
            case 2:
                return check_pi(*inputs, [2], correct_answer_p2)
            case 3:
                return check_pi(*inputs, [0, 4], correct_answer_p3)
            case 4:
                return check_pi(*inputs, [1, 3], correct_answer_p4)

    # At this point, the answer is correct.

    if len(lines) > 3:
        return True, "Correct answer, and OMG it found a correct example with more than 3 maximal triangles…"
    
    return True, ''

def verify_p1(input: str) -> tuple[bool, str]:
    """ Verifies an answer for the triangles problem.
    Returns:
        tuple[bool, str]: bool is True if the answer is valid, False otherwise.
        str is an error message if the answer is invalid, empty string if the answer is valid.
    """
    lines = input.strip().split('\n')
    if len(lines) != 2:
        return False, "'n' != 2, too many or too few solutions"

    l1, l2 = [l.strip() for l in lines]

    if l1.startswith('2') and l2.startswith('1'):
        l1, l2 = l2, l1

    if not l1.startswith('1') or not l2.startswith('2'):
        return False, "Wrong line enumeration"

    # parse line 1 of the input into "1; A; B", where A is an array of 5 2-vectors and B is an array of 3 2-vectors
    try:
        _, A1, B1 = [l for l in l1.split(';') if l != ""]
        A1_vec = get_points(A1)
        B1_vec = get_points(B1)
    except Exception as e:
        print(e)
        return False, "Line 1 can't be converted to points"

    try:
        if not all(is_convex(vec) for vec in [A1_vec, B1_vec]):
            return False, "Line 1 contains non-convex polygons"
    except ValueError as e:
        return False, f"Line 1: {str(e)}"

    # reorder A cyclically so that { A[0], A[1], A[2] } = { B[0], B[1], B[2] } as sets
    try:
        if not reorder_points(A1_vec, ([0, 1, 2], B1_vec)):
            return False, "Line 1 can't be reordered to satisfy the constraint"
    except Exception:
        return False, "Line 1 can't be reordered to satisfy the constraint"

    if not orient(A1_vec[0], A1_vec[1], A1_vec[2], A1_vec[3]) < 0:
        return False, "Line 1 doesn't satisfy the orientation constraint"
    if not orient(A1_vec[4], A1_vec[0], A1_vec[1], A1_vec[2]) < 0:
        return False, "Line 1 doesn't satisfy the orientation constraint"

    #   parse line 2 of the input into "2; A; B, C", where A is an array of 5 2-vectors, and B and C are arrays of 3 2-vectors
    try:
        _, A2, B2, C2 = [l for l in l2.split(';') if l != ""]
        A2_vec = get_points(A2)
        B2_vec = get_points(B2)
        C2_vec = get_points(C2)
    except Exception as e:
        print(e)
        return False, "Line 2 can't be converted to points"

    # # Check that all polygons are valid (not Segments or Points) and convex
    try:
        if not all(is_convex(vec) for vec in [A2_vec, B2_vec, C2_vec]):
            return False, "Line 2 contains non-convex polygons"
    except ValueError as e:
        return False, f"Line 2: {str(e)}"

    # reorder A cyclically so that { A[0], A[1], A[3] } = { B[0], B[1], B[2] } and { A[0], A[2], A[4] } = { C[0], C[1], C[2] } as sets
    try:
        # Try both possible orderings of B and C to see if one works
        if not reorder_points(A2_vec, ([0, 1, 3], B2_vec), ([0, 2, 4], C2_vec)) and not reorder_points(A2_vec, ([0, 1, 3], C2_vec), ([0, 2, 4], B2_vec)):
            return False, "Line 2 can't be reordered to satisfy the constraint"
    except Exception:
        return False, "Line 2 can't be reordered to satisfy the constraint"

    if not orient(A2_vec[4], A2_vec[0], A2_vec[1], A2_vec[3]) < 0:
        return False, "Line 2 doesn't satisfy the orientation constraint"
    if not orient(A2_vec[4], A2_vec[0], A2_vec[1], A2_vec[2]) > 0:
        return False, "Line 2 doesn't satisfy the orientation constraint"
    if not orient(A2_vec[2], A2_vec[4], A2_vec[0], A2_vec[1]) < 0:
        return False, "Line 2 doesn't satisfy the orientation constraint"
    if not orient(A2_vec[3], A2_vec[4], A2_vec[0], A2_vec[1]) > 0:
        return False, "Line 2 doesn't satisfy the orientation constraint"
    if not orient(A2_vec[1], A2_vec[2], A2_vec[3], A2_vec[0]) > 0:
        return False, "Line 2 doesn't satisfy the orientation constraint"
    if not orient(A2_vec[0], A2_vec[2], A2_vec[3], A2_vec[4]) > 0:
        return False, "Line 2 doesn't satisfy the orientation constraint"

    return True, ""

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

def reorder_points(A_vec: list[Point], *constraints: tuple[list[int], list[Point]]) -> bool:
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
            return True
    
    return False

def reorder_points_dual(A_vec: list[Point], B_vec: list[Point], C_vec: list[Point]) -> bool:
    """
    Reorder A cyclically so that:
    - {A[0], A[1], A[3]} = {B[0], B[1], B[2]} as sets
    - {A[0], A[2], A[4]} = {C[0], C[1], C[2]} as sets
    """
    if len(B_vec) != 3 or len(C_vec) != 3:
        raise ValueError("B and C must each have 3 unique points")
    if len(A_vec) != 5:
        raise ValueError("A must have 5 points")
    
    return reorder_points(A_vec, ([0, 1, 3], B_vec), ([0, 2, 4], C_vec))

def compare_points(vec1: list[Point], vec2: list[Point]) -> bool:
    """Compare two lists of points as sets. Points are compared by their coordinates."""
    return set(vec1) == set(vec2)

def is_convex(p: list[Point]) -> bool:
    """returns True if the polygon is convex, False otherwise"""
    return len(concavities(p)) == 0

def concavities(p: list[Point]) -> list[int]:
    """returns the list of indices of concave vertices of a polygon (empty for a convex polygon), raises errors if not polygon"""
    if len(p) < 3:
        raise ValueError(f"Less than 3 vertices in polygon {p}")
    
    # check repeated vertices, degenerate vertices, calculate area, and detect concavities
    ar = 0.0
    conc = []
    n = len(p)
    for i in range(n):
        next_point = p[(i + 1) % n]
        prev_point = p[(i - 1) % n]
        current_point = p[i]
        # Calculate signed area
        ar += det(current_point, next_point)

        # Check for repeated vertex
        if p[i] == next_point:
            raise ValueError(f"Repeated vertex {i} in polygon {p}")

        # Check for degenerate vertex
        if det(current_point - next_point, current_point - prev_point) == 0:
            raise ValueError(f"Degenerate vertex {i} in polygon {p}")

        # the first-to-last edge intersects a non-consecutive edge?
        if (i > 1 and i < n - 2) and segments_intersect(current_point, next_point, p[0], p[n - 1]):
            raise ValueError(f"Intersecting edges {i} and {n - 1} in polygon {p}")

        # two other non-consecutive edges intersect?
        if i >= n - 3:
            continue
        for j in range(i + 2, n - 1):
            if segments_intersect(current_point, next_point, p[j], p[j + 1]):
                raise ValueError(f"Intersecting edges {i} and {j} in polygon {p}")
    
    # Now check concavities using the calculated area
    ar_sign = sign(ar)
    conc = [
        i for i in range(n) if sign(det(p[(i - 1) % n] - p[i], p[(i + 1) % n] - p[i])) == ar_sign
    ]
    return conc

def segments_intersect(A: Point, B: Point, C: Point, D: Point) -> bool:
    """check that AB intersects CD (possibly by the end points)"""
    return (det(A - C, A - D) * det(B - C, B - D) <= 0) and (det(C - A, C - B) * det(D - A, D - B) <= 0)

def area(P: Point, Q: Point, R: Point) -> float:
    return abs(det(Q-P, R-P)) / 2

@dataclass
class IntersectionResult:
    p: Point
    first: float
    second: float

def intersection(A: Point, B: Point, C: Point, D: Point) -> IntersectionResult:

    ba = B - A
    dc = D - C
    ca = C - A

    # Apply Cramer's rule to solve for t and s
    q = det(ba, -dc)

    if q == 0: # Lines are parallel
        raise ValueError("Lines are parallel")
    
    t = det(ca, -dc) / q
    s = det(ba, ca) / q

    return IntersectionResult(A + ba * t, t, s)

def maxima(A: list[float], threshold: float = 0.999) -> list[int]:
    m = max(A)
    return [i for i, a in enumerate(A) if a > threshold * m]

def check_pi(P: list[Point], T: list[list[Point]], C: list[int], expected_vertices: Iterable[int], check_fn: Callable[[list[Point]], list[list[Point]]], k: int = 5) -> tuple[bool, str]:

    valid_k = check_concave_vertices(C, range(0, k), expected_vertices)
    if not valid_k:
        return False, f"Pentagon {P} has concave vertices {C}"
    P_rot = rotate_left(P, valid_k)
    correct_triangles = _get_max_area_triangles(check_fn(P_rot))
    corrected_triangles_set = set(frozenset(t) for t in correct_triangles)
    predicted_triangles_set = set(frozenset(t) for t in T)

    if corrected_triangles_set != predicted_triangles_set:
        return False, f"Wrong triangles: {predicted_triangles_set} != {corrected_triangles_set}"

    return True, ""

def check_concave_vertices(concavities: list[int], ks: Iterable[int], expected_vertices: Iterable[int]) -> int | None:
    ks = list(ks)
    expected_vertices = set(expected_vertices)
    valid_k = None
    for k in ks:
        C_k = set(c + k % 5 for c in concavities)
        if C_k == expected_vertices:
            valid_k = k
            break

    return valid_k


def correct_answer_p2(P: list[Point]) -> list[list[Point]]:
    Q = list(reversed(P))
    T = []

    p = intersection(P[0], P[2], P[4], P[1]).first
    q = intersection(Q[0], Q[2], Q[4], Q[1]).first
    
    if abs(det(P[1] - P[0], P[4] - P[3])) < 0.001 and (p < 1 or q < 1):
        # one of these quadrangles is strictly concave ⇔ infinitely many maxinmal triangles
        raise ValueError("Infinitely many maximal triangles")

    T.extend(_get_triangles_p2(P, p))
    T.extend(_get_triangles_p2(Q, q))

    return T

def _get_triangles_p2(P: list[Point], p: float) -> list[list[Point]]:
    T = []
    if p < 0.999:
        T.append([P[0], P[1], P[4]])
        T.append([P[0], P[1], intersection(P[1], P[2], P[3], P[4]).p])
    elif p > 1.001:
        T.append([P[0], P[1], intersection(P[1], P[2], P[4], P[0]).p])
        T.append([P[4], P[0], intersection(P[0], P[1], P[4], P[2]).p])
    else:
        T.append([P[0], P[1], P[4]])
    return T


def _get_max_area_triangles(T: list[list[Point]]) -> list[list[Point]]:
    areas = [area(*t) for t in T]
    max_area = max(areas)
    return [t for t, a in zip(T, areas) if a == max_area]

def correct_answer_p3(P: list[Point]) -> list[list[Point]]:
    T = []

    T.append([P[2], intersection(P[0], P[4], P[2], P[3]).p, intersection(P[0], P[4], P[2], P[1]).p])
    T.append([P[2], P[3], intersection(P[3], P[4], P[1], P[2]).p])

    P_rev = list(reversed(P))
    T.append([P_rev[2], P_rev[3], intersection(P_rev[3], P_rev[4], P_rev[1], P_rev[2]).p])

    return T

def correct_answer_p4(P: list[Point]) -> list[list[Point]]:
    P_rev = list(reversed(P))
    T = []

    T.append([P[2], intersection(P[0], P[4], P[2], P[3]).p, intersection(P[0], P[4], P[2], P[1]).p])
    
    isec = intersection(P[0], P[3], P[4], P[1])
    if isec.first < 1 and isec.second < 1:
        T.append([P[4], P[0], intersection(P[0], P[1], P[4], P[3]).p])
    
    if isec.first < 1:
        T.append([P[4], intersection(P[4], P[0], P[1], P[2]).p, intersection(P[4], P[3], P[1], P[2]).p])
        T.append([P_rev[4], intersection(P_rev[4], P_rev[0], P_rev[1], P_rev[2]).p, intersection(P_rev[4], P_rev[3], P_rev[1], P_rev[2]).p])
        
    return T