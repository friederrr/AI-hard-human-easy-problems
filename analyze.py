from pathlib import Path
import polars as pl
import typer

from answer_parsers import extract_boxed, verify_triangles

app = typer.Typer()

@app.command()
def triangles(path: Path):
    typer.echo(f"Analyzing triangles at {path}")
    df = pl.read_csv(path)
    df = (
        df.with_columns(
            boxed_answer=pl.col("outer_gen_text").map_elements(extract_boxed, return_dtype=pl.Utf8),
        ).with_columns(
            result=pl.col("boxed_answer").map_elements(
                lambda x: {"valid": (r := verify_triangles(x))[0], "error": r[1]},
                return_dtype=pl.Struct([pl.Field("valid", pl.Boolean), pl.Field("error", pl.Utf8)])
            )
        ).with_columns(
            valid=pl.col("result").struct.field("valid"),
            error=pl.col("result").struct.field("error")
        ).drop("result")
    )

    print(df.select("boxed_answer", "valid", "error"))
    score = df['valid'].sum() / len(df)
    print(f"Score: {score}")
    

if __name__ == "__main__":
    app()