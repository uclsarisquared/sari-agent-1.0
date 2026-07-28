import os
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
from operator import itemgetter

# Repo-root config.env, resolved from __file__ so it loads regardless of CWD.
# (load_dotenv was imported but never called here; GEMINI_API_KEY below relied on ambient env.)
load_dotenv(Path(__file__).resolve().parent / 'config.env')


# Define the function declaration for the model
PERCEPTION_FUNCTIONS = [
    {
        "name": "center_object_on_screen",
        "description": "Centers the agent's camera on the specified object using visual feedback from an object detector.",
        "parameters": {
            "type": "object",
            "properties": {
                "target_name": {
                    "type": "string",
                    "description": "The name or description of the object the agent should center in view."
                }
            },
            "required": ["target_name"]
        }
    },
    {
        "name": "stop",
        "description": "Stop the agent's execution when goals are met.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

MANIPULATION_FUNCTIONS = [
    {
        "name": "grab_and_read_item",
        "description": "Extends, grasps, and inspects an object directly in front of the agent using the specified hand. Returns OCR-extracted details.",
        "parameters": {
            "type": "object",
            "properties": {
                "hand": {
                    "type": "string",
                    "enum": ["left"],
                    "description": "The hand to use for grasping the object."
                }
            },
            "required": ["hand"]
        }
    },
    {
        "name": "extend_left_hand_forward",
        "description": "Extends the agent's left hand forward by 0.025 units.",
        "parameters": {
            "type": "object",
            "properties": {
                "units": {
                    "type": "integer",
                    "description": "Number of times to move left hand by 0.025 units forward."
                }
            },
            "required": ["units"]
        }
    },
    {
        "name": "extend_right_hand_forward",
        "description": "Extends the agent's right hand forward by 0.025 units.",
        "parameters": {
            "type": "object",
            "properties": {
                "units": {
                    "type": "integer",
                    "description": "Number of times to move right hand by 0.025 units forward."
                }
            },
            "required": ["units"]
        }
    },
    {
        "name": "pull_left_hand_backward",
        "description": "Pulls the agent's left hand backward by 0.025 units.",
        "parameters": {
            "type": "object",
            "properties": {
                "units": {
                    "type": "integer",
                    "description": "Number of times to move left hand by 0.025 units backward."
                }
            },
            "required": ["units"]
        }
    },
    {
        "name": "pull_right_hand_backward",
        "description": "Pulls the agent's right hand backward by 0.025 units.",
        "parameters": {
            "type": "object",
            "properties": {
                "units": {
                    "type": "integer",
                    "description": "Number of times to move right hand by 0.025 units backward."
                }
            },
            "required": ["units"]
        }
    },
    {
        "name": "raise_left_hand",
        "description": "Raises the agent's left hand by 0.025 units.",
        "parameters": {
            "type": "object",
            "properties": {
                "units": {
                    "type": "integer",
                    "description": "Number of times to raise left hand by 0.025 units."
                }
            },
            "required": ["units"]
        }
    },
    {
        "name": "raise_right_hand",
        "description": "Raises the agent's right hand by 0.025 units.",
        "parameters": {
            "type": "object",
            "properties": {
                "units": {
                    "type": "integer",
                    "description": "Number of times to raise right hand by 0.025 units."
                }
            },
            "required": ["units"]
        }
    },
    {
        "name": "lower_left_hand",
        "description": "Lowers the agent's left hand by 0.025 units.",
        "parameters": {
            "type": "object",
            "properties": {
                "units": {
                    "type": "integer",
                    "description": "Number of times to lower left hand by 0.025 units."
                }
            },
            "required": ["units"]
        }
    },
    {
        "name": "lower_right_hand",
        "description": "Lowers the agent's right hand by 0.025 units.",
        "parameters": {
            "type": "object",
            "properties": {
                "units": {
                    "type": "integer",
                    "description": "Number of times to lower right hand by 0.025 units."
                }
            },
            "required": ["units"]
        }
    },
    {
        "name": "toggle_left_grip",
        "description": "Toggles the grip of the left hand.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "toggle_right_grip",
        "description": "Toggles the grip of the right hand.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "rotate_and_read",
        "description": "Inspects an object directly in front of the agent using the specified hand by rotating an already-grabbed object in clockwise direction. Returns OCR-extracted details.",
        "parameters": {
            "type": "object",
            "properties": {
                "hand": {
                    "type": "string",
                    "enum": ["left", "right"],
                    "description": "The hand to use for grasping the object."
                }
            },
            "required": ["hand"]
        }
    },
    {
        "name": "stop",
        "description": "Stop the agent's execution when goals are met.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


NAVIGATION_FUNCTIONS = navigation_functions = [
    {
        "name": "move_forward",
        "description": "Moves the agent forward by (units × 0.1) units.",
        "parameters": {
            "type": "object",
            "properties": {
                "units": {
                    "type": "integer",
                    "description": "Number of times to move 0.1 units forward."
                }
            },
            "required": ["units"]
        }
    },
    {
        "name": "move_backward",
        "description": "Moves the agent backward by (units × 0.1) units.",
        "parameters": {
            "type": "object",
            "properties": {
                "units": {
                    "type": "integer",
                    "description": "Number of times to move 0.1 units backward."
                }
            },
            "required": ["units"]
        }
    },
    {
        "name": "move_left",
        "description": "Moves the agent to the left by (units × 0.1) units.",
        "parameters": {
            "type": "object",
            "properties": {
                "units": {
                    "type": "integer",
                    "description": "Number of times to move 0.1 units to the left."
                }
            },
            "required": ["units"]
        }
    },
    {
        "name": "move_right",
        "description": "Moves the agent to the right by (units × 0.1) units.",
        "parameters": {
            "type": "object",
            "properties": {
                "units": {
                    "type": "integer",
                    "description": "Number of times to move 0.1 units to the right."
                }
            },
            "required": ["units"]
        }
    },
    {
        "name": "pan_left",
        "description": "Pans the agent's camera to the left by (units × 2.5) degrees.",
        "parameters": {
            "type": "object",
            "properties": {
                "units": {
                    "type": "integer",
                    "description": "Number of times to pan 2.5 degrees left."
                }
            },
            "required": ["units"]
        }
    },
    {
        "name": "pan_right",
        "description": "Pans the agent's camera to the right by (units × 2.5) degrees.",
        "parameters": {
            "type": "object",
            "properties": {
                "units": {
                    "type": "integer",
                    "description": "Number of times to pan 2.5 degrees right."
                }
            },
            "required": ["units"]
        }
    },
    {
        "name": "pan_up",
        "description": "Pans the agent's camera upward by (units × 2.5) degrees.",
        "parameters": {
            "type": "object",
            "properties": {
                "units": {
                    "type": "integer",
                    "description": "Number of times to pan 2.5 degrees upward."
                }
            },
            "required": ["units"]
        }
    },
    {
        "name": "pan_down",
        "description": "Pans the agent's camera downward by (units × 2.5) degrees.",
        "parameters": {
            "type": "object",
            "properties": {
                "units": {
                    "type": "integer",
                    "description": "Number of times to pan 2.5 degrees downward."
                }
            },
            "required": ["units"]
        }
    },
    {
        "name": "stop",
        "description": "Stop the agent's execution when goals are met.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


def call_agent_action(mode, payload):
    """
    Call the agent action based on the mode.
    :param mode: The mode to call (perception, manipulation, navigation).
    """
    timestep, current_state_log, plan, mode = itemgetter('timestep', 'current_state_log', 'plan', 'mode')(payload)

    if mode == "perception":
        tools = PERCEPTION_FUNCTIONS
    elif mode == "manipulation":
        tools = MANIPULATION_FUNCTIONS
    elif mode == "navigation":
        tools = NAVIGATION_FUNCTIONS
    else:
        raise ValueError("Invalid mode specified.")
    
    model_name = "gemini-2.0-flash"

    # Configure the client and tools
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    tools = types.Tool(function_declarations=PERCEPTION_FUNCTIONS)
    config = types.GenerateContentConfig(tools=[tools])

    # Send request with function declarations
    response = client.models.generate_content(
        model=model_name,
        contents=(
            "You are an AI agent in a virtual grocery store environment that "
            "executes one action at a time based on a provided plan. You will "
            "given the current plan, previous actions, and the current state of the "
            "environment. Your task is to decide the next best action to take from "
            "the available tools. You can use the tools to navigate, manipulate objects, "
            "or perceive the environment.\n\n"
            f"Time Step:\n{timestep}\n\n"
            f"Current State Log:\n{current_state_log}\n\n"
            f"Plan:\n{plan}\n\n"
            f"Mode:\n{mode}\n\n"
            "What is the next best action?"),
        config=config,
    )
    print("CALL AGENT ACTION RESPONSE:", response)

    # Check for a function call
    if response.candidates[0].content.parts[0].function_call:
        function_call = response.candidates[0].content.parts[0].function_call
        print(f"Function to call: {function_call.name}")
        print(f"Arguments: {function_call.args}")
        #  In a real app, you would call your function here:
        #  result = schedule_meeting(**function_call.args)
        return function_call
    else:
        print("No function call found in the response.")
        print(response.text)
