from src.parser import parser_config
from src.file_loader import load_function_definition
from src.file_loader import load_prompt
from src.constrained_decoding import build_system_prompt
from src.constrained_decoding import load_vocabulary
from src.constrained_decoding import build_json_valid_id
from src.constrained_decoding import get_best_valid_token
from src.constrained_decoding import extract_only_expected
import sys
import json
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
        sys.exit()
    #for func in function:
    #    print(f"{func}\n\n")

    prompt = load_prompt(args.input)
    if prompt is None:
        print("Please fix the format in function calling due to its wrong format")
        sys.exit()
   # for porn in prompt:
    #    print(f"{porn}\n\n")

    print("Building system prompt ...")
    system = build_system_prompt(function)
    #print(system)

    print(f"loading model: {args.model}")
    try:
        model = Small_LLM_Model(model_name=args.model)
        #print(model)
    except OSError:
        print(f"Model: {args.model} not found or failed to download")
        sys.exit()

    vocab = load_vocabulary(model)
    #print("Vocab: ", len(vocab))

    #for token, token_id in vocab.items():
    #    if token_id == 15:
    #        print("Vocabulary token:", repr(token))
    #        break

    valid_id = build_json_valid_id(vocab)
    #print("Valid tokens: ", len(valid_id))

    print("Precessing prompt ...")
    for promp in prompt:
        user_prompt = promp.prompt
        #print(f"Processing prompt :{prompt}")
        all_prompt = f"{system}\nUser prompt: {user_prompt}\nAssistant: "
        #print(all_prompt)
        input_ids = model.encode(all_prompt)
        generated_ids = input_ids[0].tolist()

        all_generated_id = []
        excepted_json = None
        all_generated_id.extend(model.encode('{"name" : "')[0].tolist())
        while not excepted_json:
            logits = model.get_logits_from_input_ids(generated_ids + all_generated_id)
            #print(max(logits))
            next_id = get_best_valid_token(logits, valid_id)
            all_generated_id.append(next_id)
            generated_text = model.decode(all_generated_id)
            #print("Token:", repr(model.decode([next_id])))
            print(generated_text)
            if generated_text.rstrip().endswith("}}"):
                break
            excepted_json = extract_only_expected(generated_text)
            if excepted_json:
                try:
                    parsed = json.loads(excepted_json)
                    break
                except Exception:
                    pass
            #print(excepted_output)
    return excepted_json
#        all_generated_id = []
#        generated_text = ""
#        for _ in range(50):
#            logits = model.get_logits_from_input_ids(input_token_ids + all_generated_id)
#            valid_id = build_json_valid_id(vocab, generated_text)
#            if not valid_id:
#                break
#            next_id = get_best_valid_token(logits, valid_id)
#            all_generated_id.append(next_id)
#            generated_text = model.decode(all_generated_id)
#            print(generated_text)
#            if generated_text.count('{') == generated_text.count('}') and generated_text.rstrip().endswith('}'):
#                break


if __name__ == "__main__":
    try:
        main()
    except MemoryError:
        print("Insufficient Memory, There is no enough memory left on device")
        sys.exit()
    except (KeyboardInterrupt, EOFError):
        print("Program stopped by the user")
        sys.exit()
    #except Exception as error:
    #    print(f"Unexpected error: {error}")
    #    sys.exit(1)