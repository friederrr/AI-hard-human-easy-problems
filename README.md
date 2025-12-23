# AI-hard-human-easy-problems

## Answer Parsers

Our answer parsers are found in `answer_parsers.py`, with the tests being available in `test_answer_parsers.py`.
Run the tests with `uv run pytest test_answer_parsers.py`.

Currently, we implement the following answer parsing methods:

* `verify`: Takes a single string as input with two lines in the following format:
```text
1, (the coordinate representation of the 5 vertices of a pentagon with one maximal triangle), (the coordinate representation of the vertices of its maximal triangle);
2, (the coordinate representation of the 5 vertices of a pentagon with two maximal triangles), (the coordinate representation of the vertices of its 1st maximal triangle), (the coordinate representation of the vertices of its 2nd maximal triangle);

For example:
1; (0,0), (6,0), (5,2), (2,5), (0,6); (0,0), (6,0), (0,6);
2; (0,0), (0,6). (5,7), (7,5), (6,0); (0,0), (6,0), (5,7); (0,0), (7,5), (0,6); 
```
