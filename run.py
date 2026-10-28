from backdoor_project.cli import main

if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as error:
        import sys
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
