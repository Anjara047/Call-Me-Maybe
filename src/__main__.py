from src.parser import parser_config
from src.file_loader import load_function_definition
from src.file_loader import load_prompt
from src.constrained_decoding import build_sytem_prompt
from src.constrained_decoding import load_vocabulary
from src.constrained_decoding import build_json_valid_id
import sys
try:
    from llm_sdk import Small_LLM_Model
except (ImportError, ModuleNotFoundError, KeyboardInterrupt):
    print("Program stopped by the user")
    sys.exit()


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
    #for func in function:
    #    print(f"{func}\n\n")

    prompt = load_prompt(args.input)
    if prompt is None:
        print("Please fix the format in functions definition due to its wrong format")
        return
   # for porn in prompt:
    #    print(f"{porn}\n\n")

    print("Building system prompt ...")
    system = build_sytem_prompt(function)
    #print(system)

    print(f"loading model: {args.model}")
    try:
        model = Small_LLM_Model(model_name=args.model)
        #print(model)
    except OSError:
        print(f"Model: {args.model} not found or failed to download")

    vocab = load_vocabulary(model)
    valid_id = build_json_valid_id(vocab)

    print("Precessing prompt ...")
    for promp in prompt:
        prompt = promp.prompt
        #print(f"Processing prompt :{prompt}")
        all_prompt = f"User prompt: {prompt}\n{system}\n"
        #print(all_prompt)
        input_ids = model.encode(all_prompt)
        print(input_ids)


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        print("Insufficient Memory, There is no enough memory left on device")
        sys.exit()
    except (KeyboardInterrupt, EOFError):
        print("Program stopped by the user")
    #except Exception as error:
    #    print(f"Unexpected error: {error}")
    #    sys.exit(1)