from src.animation import loading_animation
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
import time
import os
import threading
try:
    from llm_sdk import Small_LLM_Model
except (ImportError, ModuleNotFoundError, KeyboardInterrupt):
    print("🚫 Program stopped by the user, llm could be missing")
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
        print("💡 Please fix the format in functions definition due to its wrong format")
        print("🔏Or Give the permission to this file and if denied the problem will be solved")
        print("So up to now, the program was not launched yet")
        sys.exit()
    function_name =[fn.name for fn in function] 

    prompt = load_prompt(args.input)
    if prompt is None:
        print("💡 Please fix the format in function calling due to its wrong format")
        print("🔏Or Give the permission to this file  if denied and the problem will be solved")
        print("So up to now, the program was not launched yet")
        sys.exit()

    print("📂 Building system prompt ...")
    system = build_system_prompt(function)

    print(f"🔥 Loading model: {args.model}")
    try:
        model = Small_LLM_Model(model_name=args.model)
    except OSError:
        print(f"Model: {args.model} not found or failed to download")
        print("This is the most probably due to unsufficient Memory")
        sys.exit()

    vocab = load_vocabulary(model)
    valid_id = build_json_valid_id(vocab)

    all_result = []
    start_time = time.time()

    #print("Precessing prompt ...")
    for promp in prompt:
        user_prompt = promp.prompt
        #print(f"User prompt: {user_prompt}")
        all_prompt = f"{system}\nUser prompt: {user_prompt}\nAssistant: "
        #print("\n\n")
        #print(all_prompt)
        #print("\n\n")
        input_ids = model.encode(all_prompt)
        generated_ids = input_ids[0].tolist()

        all_generated_id = []
        excepted_json = None
        parsed = None
        all_generated_id.extend(model.encode('{"name" : "')[0].tolist())
        stop_event = threading.Event()
        animation_thread = threading.Thread(
            target = loading_animation,
            args=(stop_event,)
        )
        animation_thread.start()
        try:
            while not excepted_json:
                logits = model.get_logits_from_input_ids(generated_ids + all_generated_id)
                next_id = get_best_valid_token(logits, valid_id)
                #if len(all_generated_id) > 50:
                #    break
                all_generated_id.append(next_id)
                generated_text = model.decode(all_generated_id)
                #print(generated_text)
                excepted_json = extract_only_expected(generated_text)
                if excepted_json:
                    try:
                        parsed = json.loads(excepted_json)
                        break
                    except Exception:
                        pass
        finally:
            stop_event.set()
        animation_thread.join()
        print(f"➡️ User prompt: {user_prompt}")
        print("✅ done: Yes, prompt generated")
        print("👇Here is the result:")
        if parsed.get("name") not in function_name:
            parsed = {"name": "none", "args": {}}

        all_result.append({
            "prompt":user_prompt,
            "name": parsed.get("name", "none"),
            "parameters": parsed.get("args", {})
        })

        if parsed.get("name", "none")  == "none":
            print(f"\n\t➠ Unfortunately, There is no function provided for this prompt")
            result = all_result[-1]
            print(json.dumps(result, indent=4))
        else:
            #result = f"\n\t➠ {parsed['name']}({parsed['args']})\n"
            result = all_result[-1]
            print(json.dumps(result, indent=4))
    total_time = time.time() - start_time
    #print(all_result)
    #all_parsed_result = [result for result in all_result if result['name'] != "none"]
    #not_parsed_result = [result for result in all_result if result['name'] == "none"]
    #print(all_parsed_result)
    os.makedirs(os.path.dirname(args.output), exist_ok = True)
    try:
        with open(args.output, 'w') as file:

            #json.dump(not_parsed_result, file, ensure_ascii=False, indent =2)
            json.dump(all_result, file, ensure_ascii=False, indent =2)
        print(f"\n╰┈➤ˎˊ˗ Result saved to : {args.output}")
        print(f"🕐 It takes {total_time:.2f} second to generate the total of your prompt")
        print(f"📈 Time average for each prompt takes {total_time/len(prompt):.2f} second")
        print(f"📈 Success rate: {len(all_parsed_result)}/{len(prompt)} ({len(all_parsed_result)/len(prompt)*100:.2f}%)")
    except (PermissionError):
        print("⚠️You denied the permission from the file to save the result")
        print("So the result is not saved anywhere")


if __name__ == "__main__":
    print("\t\t\t\tWELCOME TO MY CALL ME MAYBE\n")
    try:
        main()
    except (MemoryError, OSError):
        print("⚠️Insufficient Memory, There is no enough memory left on device")
        sys.exit()
    except (KeyboardInterrupt, EOFError):
        print("\n💡If you want to see the result, do not interrupt the program\n")
        print("🚫 Since you stopped the program, there is nothing to do with")
        print("That means the result was not generateed so it was not saved anywhere")
        sys.exit(0)
    except Exception as error:
        print(f"Unexpected error: {error}")
        sys.exit(0)
    finally:
        print("\n\t\t\t\t\t\t\t\t", "="*4)
        print("\n\n\tThe program touches it end\n".upper())
        print("\tTry another prompt to see more test")
        print("\n\tOr\n")
        print("\t'make clean' to remove all the temporary file")
        print("\t'make fclean' to remove all the newly created cache and some dependencies in goinfre\n")