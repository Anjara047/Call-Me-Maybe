"""Run the main workflow of the function-calling project."""
import json
import sys
import time

from src.models.valid_parameters import casting_parameters
from src.parser import parser_config
from src.file_loader import load_function_definition
from src.file_loader import load_prompt
from src.file_loader import save_results
from src.constrained_decoding import build_system_prompt
from src.function_calling import initialize_model
from src.function_calling import generate_response


def main() -> None:
    """
    Run the function-calling pipeline from input to output.

    The pipeline loads function definitions and prompts,
    initializes the language model, generates constrained JSON,
    validates parameters, and saves the results.

    Returns:
        None.
    """
#    heading = """
#   \t\t\t\t ██╗    ██╗███████╗██╗      ██████╗   ██████╗  ███╗   ███╗███████╗
#   \t\t\t\t ██║    ██║██╔════╝██║     ██╔════╝  ██╔═══██╗ ████╗ ████║██╔════╝
#   \t\t\t\t ██║ █╗ ██║█████╗  ██║     ██║       ██║   ██║ ██╔████╔██║█████╗
#   \t\t\t\t ██║███╗██║██╔══╝  ██║     ██║       ██║   ██║ ██║╚██╔╝██║██╔══╝
#   \t\t\t\t ╚███╔███╔╝███████╗███████╗╚██████╗  ╚██████╔╝ ██║ ╚═╝ ██║███████╗
#   \t\t\t\t  ╚══╝╚══╝ ╚══════╝╚══════╝ ╚═════╝   ╚═════╝  ╚═╝     ╚═╝╚══════╝
#
#   \t\t\t\t         ████████╗ ██████╗     ███╗   ███╗██╗   ██╗
#   \t\t\t\t         ╚══██╔══╝██╔═══██╗    ████╗ ████║╚██╗ ██╔╝
#   \t\t\t\t            ██║   ██║   ██║    ██╔████╔██║ ╚████╔╝
#   \t\t\t\t            ██║   ██║   ██║    ██║╚██╔╝██║  ╚██╔╝
#   \t\t\t\t            ██║   ╚██████╔╝    ██║ ╚═╝ ██║   ██║
#   \t\t\t\t            ╚═╝    ╚═════╝     ╚═╝     ╚═╝   ╚═╝
#   \t\t\t\t     ██████╗  █████╗ ██╗     ██╗         ███╗   ███╗███████╗
#   \t\t\t\t     ██╔════╝██╔══██╗██║     ██║         ████╗ ████║██╔════╝
#   \t\t\t\t     ██║     ███████║██║     ██║         ██╔████╔██║█████╗
#   \t\t\t\t     ██║     ██╔══██║██║     ██║         ██║╚██╔╝██║██╔══╝
#   \t\t\t\t     ╚██████╗██║  ██║███████╗███████╗    ██║ ╚═╝ ██║███████╗
#   \t\t\t\t      ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝    ╚═╝     ╚═╝╚══════╝
#
#   \t\t\t\t          ███╗   ███╗ █████╗ ██╗   ██╗██████╗ ███████╗
#   \t\t\t\t          ████╗ ████║██╔══██╗╚██╗ ██╔╝██╔══██╗██╔════╝
#   \t\t\t\t          ██╔████╔██║███████║ ╚████╔╝ ██████╔╝█████╗
#   \t\t\t\t          ██║╚██╔╝██║██╔══██║  ╚██╔╝  ██╔══██╗██╔══╝
#   \t\t\t\t          ██║ ╚═╝ ██║██║  ██║   ██║   ██████╔╝███████╗
#   \t\t\t\t          ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═════╝ ╚══════╝
#    """
    #print(heading)
    print("🚀 Starting...")
    args = parser_config()

    function = load_function_definition(args.functions_definition)
    if function is None:
        print("💡 Please fix the format in functions", end="")
        print(" definition due to its wrong format")
        print("🔏 Or Give the permission to this file")
        print("So up to now, the program was not launched yet")
        sys.exit()
    function_name = [fn.name for fn in function]

    prompt = load_prompt(args.input)
    if prompt is None:
        print("💡 Please fix the format in function", end="")
        print(" calling due to its wrong format")
        print("🔏Or Give the permission to this file", end="")
        print(" if denied")
        print("So up to now, the program was not launched yet")
        sys.exit()

    #print("📂 Building system prompt ...")
    system = build_system_prompt(function)
#
    #print(f"🔥 Loading model: {args.model}")
    #try:
    #    model = Small_LLM_Model(model_name=args.model)
    #except OSError:
    #    print(f"Model: {args.model} not found or failed to download")
    #    print("This is the most probably due to unsufficient Memory")
    #    sys.exit()
#
    #vocab = load_vocabulary(model)
    #valid_id = build_json_valid_id(vocab)

    model, valid_id = initialize_model(args.model)

    final_result = []
    all_result = []
    start_time = time.time()

    for promp in prompt:
        user_prompt = promp.prompt
        #all_prompt = f"{system}\nUser prompt: {user_prompt}\nAssistant: "
        #input_ids = model.encode(all_prompt)
        #generated_ids = input_ids[0].tolist()
#
        #all_generated_id = []
        #excepted_json = None
        #parsed = None
        #all_generated_id.extend(model.encode('{"name" : "')[0].tolist())
        #stop_event = threading.Event()
        #animation_thread = threading.Thread(
        #    target=loading_animation,
        #    args=(stop_event,)
        #)
        #animation_thread.start()
        #try:
        #    while not excepted_json:
        #        logits = model.get_logits_from_input_ids(
        #            generated_ids + all_generated_id)
        #        next_id = get_best_valid_token(logits, valid_id)
        #        if len(all_generated_id) > 200:
        #            break
        #        all_generated_id.append(next_id)
        #        generated_text = model.decode(all_generated_id)
        #        excepted_json = extract_only_expected(generated_text)
        #        if excepted_json:
        #            try:
        #                parsed = json.loads(excepted_json)
        #                break
        #            except Exception:
        #                pass
        #    if parsed is None:
        #        print("\n⚠️ Could not generate valid", end="")
        #        print(" JSON within the token limit")
        #finally:
        #    stop_event.set()
        #animation_thread.join()
        parsed = generate_response(
            model,
            valid_id,
            system,
            user_prompt
        )
        #print("\n🔍 Generated text:")
        #print(model.decode(all_generated_id))
        print(f"➡️ User prompt: {user_prompt}")
        print("✅ Done: Yes, prompt generated")
        print("👇Here is the result:")
        if parsed is None or parsed.get("name") not in function_name:
            parsed = {"name": "None", "args": {}}
        else:
            for fn in function:
                if fn.name == parsed["name"]:
                    casted_args = casting_parameters(
                        parsed.get("args", {}),
                        fn.parameters
                    )
                    if casted_args is None:
                        parsed = {"name": "None", "args": {}}
                    else:
                        parsed["args"] = casted_args
                    break

        all_result.append({
            "prompt": user_prompt,
            "name": parsed.get("name", "None"),
            "parameters": parsed.get("args", {})
        })

        if parsed.get("name", "None") == "None":
            print("\n\t➠ Unfortunately,", end="")
            print(" There is no function provided for this prompt\n")
            result = all_result[-1]
            print(json.dumps(result, indent=4))
        else:
            result = all_result[-1]
            print(json.dumps(result, indent=4))
    total_time = (time.time() - start_time) / 60
#    os.makedirs(os.path.dirname(args.output), exist_ok=True)
#    try:
#        with open(args.output, 'w') as file:
#            json.dump(all_result, file, ensure_ascii=False, indent=2)
#        print(f"\n╰┈➤ˎˊ˗ Result saved to : {args.output}")
#        print(f"🕐 It takes {total_time:.2f}", end="")
#        print(" minutes to generate the total of your prompt")
#        each_prompt = total_time / len(prompt) * 60
#        print(f"📈 Time average for each prompt takes {each_prompt:.2f} second")
#    except (PermissionError):
#        print("⚠️You denied the permission from the file to save the result")
#        print("So the result is not saved anywhere")
    save_results(args.output, all_result)


if __name__ == "__main__":
    try:
        main()
    except (MemoryError, OSError):
        print("⚠️Insufficient Memory,", end="")
        print(" There is no enough memory left on device")
        sys.exit()
    except (KeyboardInterrupt, EOFError):
        print("\n💡If you want to see the result,", end="")
        print(" do not interrupt the program\n")
        print("🚫 Since you stopped the program, there is nothing to do with")
        print("That means the result", end="")
        print(" was not generateed so it was not saved anywhere")
        sys.exit(0)
    except Exception as error:
        print(f"Unexpected error: {error}")
        sys.exit(0)
    finally:
        print("\n\t\t", "=" * 8)
        print("\n\nThe program touches its end\n".upper())
        print("Try another prompt to see more test")
        print("\n\tOr\n")
        print("'make clean' to remove all the temporary file")
        print("'make fclean' to remove all the newly", end="")
        print(" created cache and some dependencies in goinfre\n")
