from src.parser import parser_config
from src.file_loader import load_function_definition
from src.file_loader import load_prompt


def main() -> None:
    print("🚀 Starting...")
    args = parser_config()

    # print(args.input)
    # print(args.output)
    # print(args.functions_definition)
    # print(args.model)
    
    function = load_function_definition(args.functions_definition)
    if function is None:
        print("Please fix the format in functions definition due to its wrong format")
        return
    for func in function:
        print(f"{func}\n\n")

    prompt = load_prompt(args.input)
    if prompt is None:
        print("Please fix the format in functions definition due to its wrong format")
        return
    for porn in prompt:
        print(f"{porn}\n\n")

if __name__ == "__main__":
    main()