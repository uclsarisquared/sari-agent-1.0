def determine_mode(timestep, current_state_log, plan, previous_mode):
    from google import genai
    from google.genai import types
    import dotenv
    import os

    dotenv.load_dotenv('api.env')
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=GEMINI_API_KEY)
    model_name = "gemini-2.5-pro-preview-05-06"

    prompt = (
        "You are the brain of an AI agent in a virtual grocery store environment. "
        "You will determine whether the agent is in a perception, navigation, or "
        "manipulation mode. You will be given the Time Step, Current State Log, "
        "Plan, and Previous Mode. The agent can only be in one mode at a time. "
        "Your task is to decide the mode based on the provided information. "
        "Return a JSON object with the mode as a string. "
        "The modes are: perception, navigation, manipulation. "
        "Example: {\"mode\": \"perception\"}\n\n"
        f"Time Step:\n{timestep}\n\n"
        f"Current State Log:\n{current_state_log}\n\n"
        f"Plan:\n{plan}\n\n"
        f"Previous Mode:\n{previous_mode}\n\n"
    )

    response = client.models.generate_content(
        model=model_name,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=0.5
        )
    )
    return response.text