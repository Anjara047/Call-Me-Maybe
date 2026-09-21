"""Run the main workflow of the function-calling project."""
import json
import sys
import time
from typing import Any

from src.models.valid_parameters import casting_parameters
from src.parser import parser_config
from src.file_loader import load_function_definition
from src.file_loader import load_prompt
from src.file_loader import save_results
from src.constrained_decoding import build_system_prompt
from src.function_calling import initialize_model
from src.function_calling import generate_response
from src.animation import heading


def main() -> None:
    """
    Run the function-calling pipeline from input to output.

    The pipeline loads function definitions and prompts,
    initializes the language model, generates constrained JSON,
    validates parameters, and saves the results.

    Returns:
        None.
    """
    heading()
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

    print("📂 Building system prompt ...")
    system = build_system_prompt(function)
    model, valid_id = initialize_model(args.model)

    all_result = []
    dup_prompt: dict[str, dict[str, Any] | None] = {}
    start_time = time.perf_counter()
    for promp in prompt:
        user_prompt = promp.prompt
        if user_prompt in dup_prompt:
            parsed = dup_prompt[user_prompt]
            #print("♻️ Reusing cached response")
        else:
            parsed = generate_response(
                model,
                valid_id,
                system,
                user_prompt
            )
            dup_prompt[user_prompt] = parsed
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
    total_time = (time.perf_counter() - start_time) / 60
    print(f"🕐 Total time: {total_time:.2f} minutes")
    if len(prompt) > 0:
        each_prompt = total_time / len(prompt) * 60
        print(
            f"📈 Average time per prompt: "
            f"{each_prompt:.2f} seconds"
        )

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
