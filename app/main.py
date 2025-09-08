import argparse, json, sys
from pathlib import Path
from src.tlco_analysis import compute_tlco

def parse_args():
    p = argparse.ArgumentParser(description="Compute TLCO from cars.json and write results.")
    p.add_argument("--cars", type=str, default="/work/cars.json", help="Path to cars.json")
    p.add_argument("--outdir", type=str, default="/work/data", help="Directory to write results")
    return p.parse_args()

def main():
    args = parse_args()
    cars_path = Path(args.cars)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        if not cars_path.exists():
            raise FileNotFoundError(f"cars file not found: {cars_path}")

        data = json.loads(cars_path.read_text())
        log_path = str(outdir / "tlco.log")
        df = compute_tlco(data, log_path)

        # Primary tabular output: CSV
        csv_path = outdir / "results.csv"
        df.to_csv(csv_path, index=False)

        # Mirror JSON for programmatic use
        json_path = outdir / "results.json"
        json_path.write_text(df.to_json(orient="records", indent=2))

        print(f"Wrote: {csv_path}")
        print(f"Wrote: {json_path}")
        print(f"Logs:  {log_path}")
    except Exception as e:
        # Graceful error
        err_path = outdir / "error.txt"
        err_msg = f"ERROR: {e}\n"
        err_path.write_text(err_msg)
        print(err_msg, file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
