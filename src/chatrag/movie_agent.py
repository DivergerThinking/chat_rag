from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai import OpenAI
from chatrag.tools import movie_tool_oai_format
import json
import logging


def movie_agent(
    history: list[dict[str, str] | ChatCompletionMessageParam | ChatCompletionMessage],
    oai_client: OpenAI,
    tool_name_dict: dict,
    model_name: str = "gpt-3.5-turbo-0125",
) -> str:
    response = oai_client.chat.completions.create(
        model=model_name,
        messages=history,  # type: ignore
        tools=movie_tool_oai_format,  # type: ignore
        tool_choice="auto",
    )
    choice = response.choices[0]
    if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
        tool_call = choice.message.tool_calls[0]
        function_name = tool_call.function.name
        arguments_json = tool_call.function.arguments
        # Convert the JSON string of arguments to a dictionary
        arguments_dict = json.loads(arguments_json)
        function = tool_name_dict[function_name]
        context = function(**arguments_dict)
        history = history + [choice.message] + [{"role": "tool", "tool_call_id": tool_call.id, "content": context}]
        logging.info(f"Tool {function_name} called with arguments:\n\n{arguments_dict}.")
        return movie_agent(history, oai_client, tool_name_dict)
    elif response.choices[0].message.content:
        return response.choices[0].message.content
    else:
        raise ValueError("Hay algo mal en el objeto response:\n", response)
