import os
from mistralai import Mistral
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json


SEGMENT_FOLDER = "segments"

CHUNKING_PROMPT = """
Task Description
You are an expert in natural language understanding. Your task is to analyze a provided document and organize its content into logical sections based on context. The sections should group related ideas, themes, or recurring topics into meaningful categories while maintaining the original order of the text. Follow these rules:

Top-Level Context: Use the provided top_of_hierarchy value as the overarching key for text that does not belong to any specific inferred section.
Section Identification: Based on your understanding of the document, infer meaningful section names to group text that shares a common context, theme, or topic.
Text Allocation and Order:
Divide the document content into sections without altering the original order of the text.
Allocate each sentence or group of sentences to the relevant inferred section or the top_of_hierarchy key as appropriate.
Fallback Option: If no meaningful sections can be inferred, include all content under the top_of_hierarchy key only.
Valid JSON Output: Ensure the output is well-structured JSON with no duplication of text, and section names must reflect the context accurately.
Text under the section must be as provided in document content, and in string, do not make internal data structures in page content.

### Input
Top of Hierarchy: "{top_of_hierarchy}"
Document Content: "{document_content}"

### Output
Return a JSON response in the following format:
{{
  "Value of Top of Hierarchy": "Text that doesn't belong to a specific inferred section.",
  "Inferred Section Name 1": "Text under this section based on its context.",
  "Inferred Section Name 2": "Text under this section based on its context.",
  ...
}}
"""

CHUNKING_PROMPT2 = """
Task Description:
You are a natural language processing expert skilled in understanding, organizing, and categorizing text. Your task is to analyze the provided document and systematically divide it into logically coherent sections based on thematic or contextual relevance. Ensure the resulting organization makes the text easier to understand while preserving its original flow. Follow these rules carefully:

### Guidelines:
1. **Top-Level Context**:
   - Use the provided top_of_hierarchy value as the overarching key for content that does not clearly fit into inferred sections.
   - Ensure this key acts as a fallback, but prioritize meaningful categorization when possible.

2. **Section Identification**:
   - Analyze the document for recurring topics, themes, or contextual connections.
   - Assign intuitive and descriptive section names that accurately reflect the grouped content (e.g., "Background Information," "Key Findings," "Conclusion").
   - Avoid vague or overly generic section names.

3. **Text Allocation and Order**:
   - Maintain the original order of the document’s text.
   - Divide the text into sections such that each section contains related sentences or ideas.
   - Sentences or text fragments must not be split across multiple sections.

4. **Fallback Option**:
   - If no clear section can be inferred for a specific part of the text, include it under the top_of_hierarchy key.
   - Do not attempt to infer a section where one does not naturally exist.

5. **JSON Output Format**:
   - Ensure the output is a valid JSON object.
   - Avoid duplication of text across sections.
   - Section names should be unique and contextually relevant.

6. **Preserve Original Text**:
   - Do not paraphrase or modify the original text.
   - Include the text exactly as it appears in the document content.

7. **Section Priority**:
   - Sections should group content in the following logical order (if applicable): Introduction, Background, Main Content, Analysis, Conclusion, and Miscellaneous.
   - You can infer additional sections based on your analysis.

### Input:
Top of Hierarchy: "{top_of_hierarchy}"
Document Content: "{document_content}"

### Output
Return a JSON response in the following format:
{{
  "Value of Top of Hierarchy": "Text that doesn't belong to a specific inferred section.",
  "Inferred Section Name 1": "Text under this section based on its context.",
  "Inferred Section Name 2": "Text under this section based on its context.",
  ...
}}
"""


CHUNKING_PROMPT3 = """
Task Description:
You are an expert in natural language understanding. Your task is to analyze the provided document and organize its content into logical sections based on context. The sections should group related ideas, themes, or recurring topics into meaningful categories while maintaining the original order of the text. Follow these rules:

1. **Top-Level Context**: Use the provided `top_of_hierarchy` value as the overarching key for text that does not belong to any specific inferred section.
2. **Section Identification**: Based on your understanding of the document, infer meaningful section names to group text that shares a common context, theme, or topic.
3. **Text Allocation and Order**: 
   - Divide the document content into sections without altering the original order of the text.
   - Allocate each sentence or group of sentences to the relevant inferred section or the `top_of_hierarchy` key as appropriate.
4. **Fallback Option**: If no meaningful sections can be inferred, include all content under the `top_of_hierarchy` key only.
5. **Valid JSON Output**: Ensure the output is well-structured JSON with no duplication of text, and section names must reflect the context accurately.
   - Text under each section must be exactly as provided in the document content.
   - Do not create internal data structures within the page content.

### Input
- **Top of Hierarchy**: "{top_of_hierarchy}"
- **Document Content**: "{document_content}"

### Output
Return a JSON response in the following format:
{{
  "Value of Top of Hierarchy": "Text that doesn't belong to a specific inferred section.",
  "Inferred Section Name 1": "Text under this section based on its context.",
  "Inferred Section Name 2": "Text under this section based on its context.",
  ...
}}

"""

# Initialize the text splitter with improved chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4096,  # Adjust chunk size for LLM compatibility
    chunk_overlap=200,  # Overlap to maintain context across chunks
    length_function=len,
    is_separator_regex=True,  # Allows breaking at semantic boundaries
    separators=["\n\n", "\n", ".", " "],  # Priority order for splitting
)

# Initialize the Mistral API client
api_key = os.getenv("MISTRAL_API_KEY")
model = "pixtral-12b-2409"
client = Mistral(api_key=api_key)


def generate_speech(content):
    """Generate speech for structured content using the LLM."""
    try:
        speech_segments = []
        for section, text in content.items():
            response = client.chat.complete(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": f"Take the provided section and text, and transform it into a natural, conversational, and engaging speech. The speech should flow smoothly and captivate the listener while preserving the essence and details of the content. Ensure the tone is appropriate for the context, whether it’s informative, persuasive, or entertaining.\n\nSection: {section}\n\n{text}",
                    }
                ],
                response_format={"type": "text"},  # Get the plain text response
            )
            speech_segments.append(response.choices[0].message.content)
        return "\n\n".join(speech_segments)
    except Exception as e:
        print(f"Error generating speech: {e}")
        return ""


def process_chunk(client, model, chunk, top_of_hierarchy):
    """Process a single text chunk using the Mistral API."""
    try:
        response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": CHUNKING_PROMPT.format(
                        top_of_hierarchy=top_of_hierarchy,
                        document_content=chunk,
                    ),
                }
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return {}


def chunk_transcription(transcribed_file):
    try:
        # Read the transcribed text
        with open(transcribed_file, "r") as f:
            full_text = f.read()

        # Split the text into manageable chunks
        chunks = text_splitter.split_text(full_text)

        # Initialize content and hierarchy
        content = {}
        top_of_hierarchy = "Introduction"

        for chunk in chunks:
            # Process each chunk and merge responses
            chat_response = process_chunk(client, model, chunk, top_of_hierarchy)
            for section, text in chat_response.items():
                if section not in content:
                    content[section] = text
                else:
                    content[section] += "\n\n" + text

        generated_speech = generate_speech(content)
        speech_file = os.path.join(
            os.path.dirname(transcribed_file),
            os.path.basename(transcribed_file).split(".")[0] + "_speech.txt",
        )
        if os.path.exists(transcribed_file):
            os.remove(transcribed_file)
            with open(
                speech_file,
                "w",
            ) as f:
                f.write(generated_speech)
        return speech_file
    except Exception as e:
        print(f"Error processing transcription to speech: {e}")
        return ""
