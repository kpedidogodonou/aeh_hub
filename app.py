from dotenv import load_dotenv
import os 
import streamlit as st 
import qdrant_client
from qdrant_client import QdrantClient
from InstructorEmbedding import INSTRUCTOR
from llama_index.llms.mistralai import MistralAI
from llama_index.core.llms import ChatMessage

# Settup Qdrant Client 
q_client = qdrant_client.QdrantClient(
    os.getenv("QDRANT_HOST"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Settup Instructor for embedding 
embedding_model = INSTRUCTOR('hkunlp/instructor-large')

# Settup MistalAI as llm 
mistral_api_key = "ttKEyFeBQ04BblNHWv3AorCn87hrf7QR"
model = "mistral-medium-latest"
llm = MistralAI(api_key=mistral_api_key, model=model)


def aeh_chatbot(query):
  #Embedd the query
  embedded_query = embedding_model.encode(query)

  # Search on qdrant for similarities 
  search_result = q_client.search(
      collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
      query_vector=embedded_query
  )
  #Create context and metaprompt 
  context = "\n".join(r.payload["page_content"] for r in search_result)
  metaprompt = f"""
    You are an expert in African Economy History. 
    Answer the following question using the provided context. 
    If you can't find the answer, do not pretend you know it, but answer "I don't know".
    Also provide the source(references) of you information
    
    Question: {query.strip()}
    
    Context: 
    {context.strip()}
    
    Answer:
    """
    
  # Setup for Mistral
  messages = [
      ChatMessage(role="system", content="You are a well-knwon Historian that work on Africa Economics History"),
      ChatMessage(role="user", content= metaprompt),
  ]
    
  # Ask Question to mistral
  response = llm.chat(messages)
  return response.message.content

def main():
    load_dotenv()
    st.set_page_config(page_title="AEH Knwoledge Hub")
    st.header("Ask your remote database")

    # show user imput 
    user_question = st.text_input("Ask a question about AEH:")

    if user_question:
        st.write(f"Question: {user_question}")
        answser = aeh_chatbot(user_question)
        st.write(f"Answer: {answser}")

if __name__ == "__main__":
    main()