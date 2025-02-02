import streamlit as st
import speech_recognition as sr
import pyttsx3
import os
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv

load_dotenv()
st.secrets["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
st.secrets["LANGCHAIN_TRACING_V2"] = "true"
st.secrets["LANGCHAIN_PROJECT"] = "Voice Assistance"
st.secrets["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Initialize the embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Configure the voice properties
def configure_voice():
    """Set the speaking speed and change the voice to female."""
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate)  # Reduce speed by 30 units
    voices = engine.getProperty('voices')
    female_voice_found = False
    for voice in voices:
        if "female" in voice.name.lower() or "en-gb" == voice.id:
            engine.setProperty('voice', voice.id)
            female_voice_found = True
            break
    

configure_voice()

def speak(text):
    """Speak out the given text."""
    engine.say(text)
    engine.runAndWait()
    time.sleep(15)
    

def listen_from_microphone():
    """Listen to input from the microphone and return the recognized text."""
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("Listening... Speak something:")
        try:
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=40)
            text = recognizer.recognize_google(audio)
            time.sleep(2)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand what you said."
        except sr.RequestError as e:
            return f"Error with the speech recognition service: {e}"
        except Exception as e:
            return f"An error occurred: {e}"

st.title("Conversational Rag with PDF Upload and Chat History")
st.write("Upload your PDF and chat with their content")

api_key = st.text_input("Enter your GROQ API Key: ", type="password")

if api_key:
    llm = ChatGroq(model_name="Llama3-8b-8192", groq_api_key=api_key)
    session_id = st.text_input("session ID", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Choose A PDF file", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        documents = []
        temppdf = "./temp.pdf"
        
        # Check if temp.pdf exists and delete if necessary
        if os.path.exists(temppdf):
            os.remove(temppdf)
        
        # Write new temp.pdf file
        for uploaded_file in uploaded_files:
            with open(temppdf, "wb") as file:
                file.write(uploaded_file.getvalue())
                file_name = uploaded_file.name
            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
        retriever = vectorstore.as_retriever()

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question, which might reference context in the chat history,"
            " formulate a standalone question which can be understood without the chat history. Do not answer the question,"
            " just reformulate it if needed and otherwise return it as it is"
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. Use three sentences maximum and keep the answer concise."
            "\n\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        def get_session_history(session: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
        
        

        if st.button("Start Listening"):
            st.write("Click 'Allow' if the browser requests microphone access.")
            session_history = get_session_history(session_id)
        
            st.session_state.stop_listening = False  # Reset the flag
        
            while not st.session_state.stop_listening:
                user_input = listen_from_microphone()
        
                # Handle special commands like "exit"
                if user_input.lower() == "exit":
                    st.session_state.stop_listening = True
                    st.write("Stopping listening.")
                    st.write("Chat History:", session_history.messages)
                    break
        
                elif "error" in user_input.lower():
                    st.warning(f"Error: {user_input}")
                    break  # Stop the loop on critical errors
        
                elif user_input:
                    st.write(f"You said: {user_input}")
                    response = conversational_rag_chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": session_id}}
                    )
                    speak(response['answer'])  # Speak the response

        
                    # Ensure that speaking completes before continuing
                    st.write("Assistance:", response['answer'])
        
                    
                else:
                    st.warning("No input detected. Please speak clearly.")
                    st.write("Chat History:", session_history.messages)

else:
    st.write("Please enter your GROQ API Key")
