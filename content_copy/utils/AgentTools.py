from langchain_mistralai import ChatMistralAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

llm = ChatMistralAI()
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, max_results=5)

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import ValidationError

from pydantic import BaseModel, Field


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the question. Provide an answer, reflection, and then follow up with search queries to improve the answer."""

    answer: str = Field(description="~250 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    search_queries: list[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer."
    )


class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    def respond(self, state: dict):
        response = []
        for attempt in range(3):
            response = self.runnable.invoke(
                {"messages": state["messages"]}, {"tags": [f"attempt:{attempt}"]}
            )
            try:
                self.validator.invoke(response)
                return {"messages": response}
            except ValidationError as e:
                state = state + [
                    response,
                    ToolMessage(
                        content=f"{repr(e)}\n\nPay close attention to the function schema.\n\n"
                        + self.validator.schema_json()
                        + " Respond by fixing all validation errors.",
                        tool_call_id=response.tool_calls[0]["id"],
                    ),
                ]
        return {"messages": response}


# def improvising_tool(self, speech_text):
#     improviser_prompt = f"""
#         You are a skilled orator and storytelling expert. Your task is to take the provided speech and make it more engaging, dynamic, and memorable.
#         Follow these guidelines:

#         - Use vivid and descriptive language to improve storytelling.
#         - Add rhetorical devices (e.g., metaphors, alliteration) where appropriate.
#         - Adjust the tone to be more engaging and inspiring.
#         - Ensure the structure remains logical, with a clear introduction, body, and conclusion.
#         - Retain the original meaning and purpose of the speech.

#         Speech to Improvise:
#         {speech_text}

#         Return the improved version of the speech.
#         """
#     response = client.chat.complete(
#         model=model,
#         messages=[{"role": "user", "content": improviser_prompt}],
#         response_format={"type": "text"},
#     )
#     return response.choices[0].message.content
