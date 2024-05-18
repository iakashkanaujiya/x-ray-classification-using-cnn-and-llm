import os
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from google.cloud import aiplatform

load_dotenv()
aiplatform.init(project=os.getenv("GOOGLE_CLOUD_PROJECT_ID"))

# Create a chat model
chat = ChatVertexAI(
    model_name="gemini-pro",
    google_api_key=os.getenv("GOOGLE_GEMINI_API_KEY")
)

prompt = PromptTemplate(
    template="""As a knowledgeable medical assistant, your role is to provide precise information about the disease {disease_name}.
        Your responses should be clear, accurate, and tailored to assist individuals seeking information on {disease_name}.
    """,
    input_variables=["disease_name"]
)

system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)


def generate_disease_summary(disease):
    """
    Generate diease summary
    """
    formatted_message = system_message_prompt.format(disease_name=disease)
    messages = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(content=str(formatted_message.content)),
    ]
    response = chat.invoke(messages)
    return response.content


def generate_detailed_overview(disease, question):
    prompt = PromptTemplate(
        template="""As a knowledgeable medical assistant, your role is to provide precise information about the disease {disease_name}.
        Your responses should be clear, accurate, and tailored to assist individuals seeking information specifically about {disease_name}.
        Please ensure that your answers are related only to {disease_name} and avoid providing information about any other disease.
        Here is the question: {question}""",
        input_variables=["disease_name", "question"]
    )
    system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
    formatted_message = system_message_prompt.format(
        disease_name=disease, question=question)
    messages = [
        SystemMessage(content="You're a helpful assistant"),
        HumanMessage(content=str(formatted_message.content)),
    ]
    response = chat.invoke(messages)
    return response.content
