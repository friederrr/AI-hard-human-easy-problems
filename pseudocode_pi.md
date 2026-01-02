def area (P, Q, R : 2-vectors):

  return abs(det(Q-P,R-P))


def det ((a,b),(c,d)):

  return ad-bc


def inters(a,b,c,d : 2-vectors):

# returns the intersection point X of the lines ab and cd and two numbers t and s:

# t is the barycentric coordinate of X on the first line (0 if X=a and 1if X=b), s on the second line


  B=b-a 

  D=d-c

  R=c-a



  # Cramer’s rule for a+Bt=c+Ds

  Q=det(B,-D)

  if Q=0

    return {} #lines are parallel


  t=det(R,-D)/Q

  s=det(B,R)/Q

  return {‘p’ : a+B*t, ‘1st’ : t, ‘2nd’ : s}


def maxima (array of numbers A):

  m = maximum(A)

  return the array of all indices i such that A[i] > 0.999 * m



def correct_answer_P2111(P : array):

# takes a pentagon with only P[2] vertex concave, gives the correct list of its maximal triangles

# ! v2 : approximate-proof !

  Q = reversed(P) # the concave version of Q has the same index 2

  T = [] # the list of candidates for maximal triangles


  p = inters(P[0],P[2],P[4],P[1])[‘1st’] # cannot be parallel if the input is correct

  # p<1 iff the vertices of P with omitted P[3] are in strictly convex position

  q = inters(Q[0],Q[2],Q[4],Q[1])[‘1st’] # cannot be parallel if the input is correct

 

  if abs(det(P[1]-P[0],P[4]-P[3])) < 0.001 AND (p<1 OR q<1):

    # one of these quadrangles is strictly concave ⇔ infinitely many maxinmal triangles

    escalate exception “Wrong answer: infinitely many maximal triangles”

  

  if p<0.999:

    T.append((P[0],P[1],P[4])) 

    T.append((P[0],P[1],inters(P[1],P[2],P[3],P[4])[‘p’]))

  elif p>1.001:

    T.append((P[0],P[1],inters(P[1],P[2],P[4],P[0])[‘p’]))    

    T.append((P[4],P[0],inters(P[0],P[1],P[4],P[2])[‘p’]))

  otherwise 

    T.append((P[0],P[1],P[4]))    


  P = Q # repeating the same five appends with P reversed

  if q<0.999:

    T.append((P[0],P[1],P[4])) 

    T.append((P[0],P[1],inters(P[1],P[2],P[3],P[4])[‘p’]))

  elif q>1.001:

    T.append((P[0],P[1],inters(P[1],P[2],P[4],P[0])[‘p’]))    

    T.append((P[4],P[0],inters(P[0],P[1],P[4],P[2])[‘p’]))

  otherwise 

    T.append((P[0],P[1],P[4]))    


  return the array of all T[i] for i in maxima(area(T))


def approx_equal(A,B):

# recursive approximate equality of two “sets of sets of … of reals”

# we assume that the sets A and B are represented by lists with no repetitions 


  if A is real AND B is real:

    return abs(A-B)<0.000.1

  if NOT (A is list AND B is list):

    return FALSE


  if length(A) != length(B):

    return FALSE

  n = length(A)



  for AA in list(itertools.permutations(A)):

    all_equal=TRUE

    for i from 1 to n:

      all_equal *= approx_equal(AA[i], B[i])

    if all_equal:

      return TRUE


  return FALSE


def correct_answer_P4111(P : array): # ! v2: some redundant lines removed !

# takes a pentagon with P[1] and P[3] vertices concave, gives the correct list of its maximal triangles


  T = [(P[2], inters(P[0],P[4],P[2],P[3])[‘p’], inters(P[0],P[4],P[2],P[1])[‘p’])]

  (p, 1st, 2nd) = inters(P[0],P[3],P[4],P[1]) # cannot be parallel


  if 1st < 1 AND 2nd < 1 :

    T.append((P[4], P[0], inters(P[0],P[1],P[4],P[3])[‘p’]))


  if 1st < 1 

    T.append((P[4], inters(P[4],P[0],P[1],P[2])[‘p’], inters(P[4],P[3],P[1],P[2])[‘p’]))


  P.reverse() # repeating the same append with P reversed

  if 1st < 1 

    T.append((P[4], inters(P[4],P[0],P[1],P[2])[‘p’], inters(P[4],P[3],P[1],P[2])[‘p’]))


  return the array of all T[i] for i in maxima(area(T)) 

def correct_answer_P3111(P : array):

# takes a pentagon with P[0] and P[4] vertices concave, gives the correct list of its maximal triangles


  T = [(P[2], inters(P[0],P[4],P[2],P[3])[‘p’], inters(P[0],P[4],P[2],P[1])[‘p’])]

  T.append((P[2], P[3], inters(P[3],P[4],P[1],P[2])[‘p’]))


  P.reverse() # repeating the same append with P reversed

  T.append((P[2], P[3], inters(P[3],P[4],P[1],P[2])[‘p’])) 


  return the array of all T[i] for i in maxima(area(T))



def verify_Pi111(answer : string, version : integer) # version=1..4 is the version of a problem


  if i=1:

    return verify_triangles (answer) # the function that we already have for the initial problem


  if two lines in answer start with the same number:

    escalate exception “wrong format of the answer”


  for n=1,2,3:

   if no line in answer starts with n

     escalate exception “wrong answer: no example with ” + n + “ triangles”

 

  for each line L in answer

    try parsing L into 

      n = opening number

      P = array of 5 2-vectors # representing the pentagon

      T = n x 3 array of 2-vectors # representing the triangles

    otherwise escalate exception “Wrong answer: wrong number of vertices or triangles in line ” + n


    for t in T:

      if area(t)=0:

        return “Wrong answer: degenerate triangle in line ” + n


    try C = concavities(P): # already written, returns the indices of the concave vertices as an array

    except Exception as e:

      return e


# below we compare two arrays of arrays as sets of sets. I give an example of this:

# {{1,2}, {3,4}} = {{1,2}, {1,2}, {3,4}} = {{1,1,2}, {3,4}} = {{3,4}, {1,2}} = {{2,1}, {3,4}} != {{2,3}, {4,1}}


    match i:

      case 2:

        unless there exists k=0..4 such that (componentwise C+k mod 5) as a set = [2] as a set

          return “Wrong answer: the “ + n + “‘th pentagon has concave vertices ”+ C

        cyclically permute P by k # so that the new P[2] is the old P[C[0]]

        if T as a set of sets != correct_answer_P2111(P) as a set of sets :

          return “Wrong answer: the wrong “ + n + “ triangle(s)”


      case 3:

        unless there exists k=0..4 such that (componentwise C+k mod 5) as a set = [0, 4] as a set

          return “Wrong answer: the “ + n + “‘th pentagon has concave vertices ”+ C

        cyclically permute P by k # so that the new P[0,4] is the old P[C[0,1]]

        if T as a set of sets != correct_answer_P3111(P) as a set of sets :

          return “Wrong answer: the wrong “ + n + “ triangle(s)”


      case 4:

        unless there exists k=0..4 such that (componentwise C+k mod 5) as a set = [1, 3] as a set

          return “Wrong answer: the “ + n + “‘th pentagon has concave vertices ”+ C

        cyclically permute P by k # so that the new P[1,3] is the old P[C[0,1]]

        if T as a set of sets != correct_answer_P4111(P) as a set of sets :

          return “Wrong answer: the wrong “ + n + “ triangle(s)”


  # if we are here, the answer is correct


  if (number of lines in the answer) > 3:

   “Correct answer, and OMG it found a correct example with more than 3 maximal triangles…“


  return “Correct answer”        


