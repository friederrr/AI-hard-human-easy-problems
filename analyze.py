from pathlib import Path
import polars as pl
import typer

from answer_parsers import extract_last_enclosed_answer, verify_p1

app = typer.Typer()

@app.command()
def triangles(path: Path):
    typer.echo(f"Analyzing triangles at {path}\n")
    df = pl.read_csv(path)
    model_name = df["model_name"].unique().item().split("/")[-1]
    
    df = (
        df.with_columns(
            answer=pl.col("outer_gen_text").map_elements(extract_last_enclosed_answer, return_dtype=pl.Utf8),
        ).with_columns(
            result=pl.col("answer").map_elements(
                lambda x: {"valid": (r := verify_p1(x))[0], "error": r[1]},
                return_dtype=pl.Struct([pl.Field("valid", pl.Boolean), pl.Field("error", pl.Utf8)])
            )
        ).with_columns(
            valid=pl.col("result").struct.field("valid"),
            error=pl.col("result").struct.field("error")
        ).drop("result")
    )

    error_frequency = df["error"].value_counts().sort("count", descending=True).to_dicts()
    error_section = "Error Frequency".center(80,'-') + "\n" + "\n".join(f"{e['error']}: {e['count']}" for e in error_frequency)
    typer.echo(f"|| {model_name} ||".center(80,'='))
    typer.echo(error_section)

    score = df['valid'].sum() / len(df)
    typer.echo("Score".center(80,'-') + f"\n{score:.2f}")

@app.command()
def to_txt(path: Path, outfile: Path, n_tail: int | None = None):
    typer.echo(f"Reading model run data from {path}")
    df = pl.read_csv(path)
    line_length = 80
    delim = "-"*line_length 

    df = (
        df.select("prompt", "outer_gen_text")
        .with_columns(
            gen_last_section=pl.col("outer_gen_text").str.slice(-n_tail) if n_tail else pl.col("outer_gen_text")
        )
        .group_by("prompt")
        .agg("gen_last_section")
    )
    
    texts = []
    for row in df.iter_rows(named=True):
        prompt = delim + f"\n{row["prompt"]}\n" + delim + "\n\n"
        solution_str = '\n'.join(
            f"\n SOLUTION {idx}" + "/"*line_length + "\n\n .... "+ gen_text_ending for idx, gen_text_ending in enumerate(row["gen_last_section"])
        )
        texts.append(prompt + solution_str)


    outfile.write_text('\n'.join(texts), encoding="utf-8")
    typer.echo(f"Exported to {outfile}")

if __name__ == "__main__":
    app()