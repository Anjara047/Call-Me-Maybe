from pathlib import Path
import argparse

#base_dir = Path(__file__).resolve().parent.parent

def parser_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description = "Translate the prompt from natural language into a function call..."
    )
    
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B"
    )
    return parser.parse_args()