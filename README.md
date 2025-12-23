# AI-hard-human-easy-problems

## Installation 
We use Python, so please install that to your system.

We use Astral's `uv` package manager for this project.
Please proceed with the installation [as described in their documentation](https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_2).

Once you have installed uv, simply use `uv sync` to install all required dependencies.
To run **any Python script**, make sure to use **uv run <your_script.py>** instead of just using **python**.
(Alternatively, you can use **uv sync** and then **source .venv/bin/activate** if you prefer using **python** directly to activate the managed virtual environment).

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
