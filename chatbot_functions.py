# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 16:02:35 2024

@author: Br1CM
"""

import numpy as np
import os
import nest_asyncio
import pickle
import faiss
import ollama 
from sentence_transformers import SentenceTransformer
from typing_extensions import TypedDict
from typing import List
from tavily import TavilyClient
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langgraph.graph import StateGraph, END

workpath = os.getcwd()
nest_asyncio.apply()
from dotenv import load_dotenv


LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:11434")
local_llm = 'llama3.1'
    
#------ FUNCTIONS FOR CHATBOT MANAGEMENT AND USAGE OF LLM--------

class GraphState(TypedDict):
    """
    Represents the State of the conversation Graph

    Attributes:
        message: user's question
        documents: retreived documents for answering
        generation: LLM generation
        web_search: wether to search with tavily
    """
    message: str
    documents: List[str]
    generation: str
    web_search: str

def start_workflow(state):
    """
    Starting node for introducing the message into the workflow.

    Args:
        state (dict): A dictionary containing the user's message.

    Returns:
        dict: A dictionary containing the user's message.
    """
    return {"message": state["message"]}

def create_vector_store(docspath, vs_path):
    """
    Creates a vector store based on the docs already chunked

    Args:
        docspath: str
        Path to find the pickle docs
        vs_path: str
        Path to generate the vector store

    Returns:
        NoneType
    """
    # Load your documents
    with open(docspath, 'rb') as file:
        all_docs = pickle.load(file)
    
    # Set up the embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Create the FAISS vector store
    vectorstore = FAISS.from_texts(all_docs, embedding=embeddings)
    
    vectorstore.save_local(vs_path)
    print('Vector store created and saved at: ', vs_path)
    

def load_vector_store(folder_path, index_name):
    """
    load the vector store to

    Args:
        folder_path: str
        Path to find the vector store
        index_name: str
        name of the index in the vector store (without the .faiss)

    Returns:
        dict: A dictionary containing the user's message.
    """
    hf_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = FAISS.load_local(
        folder_path=folder_path,
        index_name=index_name,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})

def message_classifier(state):
    """
    Classifies whether the user's message falls under the purpose of the llm to answer it

    Args:
        state (dict): A dictionary containing the user's message.

    Returns:
        str: whether the message is related or not
    """
    print('-----message Classification-----')
    print(state)
    llm = ChatOllama(model=local_llm, base_url=LLAMA_SERVER_URL, format="json", temperature=0)
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a classifier. The goal of your answer is to determine if the user's message is related to sports training,
        or a completely different topic. Answer 'yes' if it could be interpreted as related to sports, 'no' otherwise.
         Provide the binary score as a JSON with a single key 'score' with a binary score 'yes' or 'no'.  Give no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is my message: {message}
        Is it sports-related? or am I talking about other topic?
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        json: 
        <|eot_id|>""",
        input_variables=["message"]
    )
    classifier_chain = prompt | llm | JsonOutputParser()
    message = state['message']
    classification_score = classifier_chain.invoke({"message": message})
    print(f'cl_grade = {classification_score}')
    classification_grade = classification_score['score']
    print(f'cl_score = {classification_grade}')
    if classification_grade.lower() == 'yes':
        return 'related'
    else:
        return 'not_related'



def retrieve(state):
    """
    Retrieves documents from the vector store based on the user's message.

    Args:
        state (dict): A dictionary containing the user's message.

    Returns:
        dict: A dictionary containing the retrieved documents and the user's message.
    """
    message = state['message']
    retriever = load_vector_store('Models/RAG/vs', 'index')
    documents = retriever.invoke(message)
    return {"documents": documents, "message": message}


def grade_document_relevance(state):
    """
    Grades the documents retrieved previously to check relevance to anser the user's message.

    Args:
        state (dict): A dictionary containing the user's message and the retrieved documents.

    Returns:
        dict: A dictionary containing the retrieved documents, the user's message and web_search need.
    """
    llm = ChatOllama(model=local_llm, base_url=LLAMA_SERVER_URL, format="json", temperature=0)
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Give a binary answer to the question provided in the end of the text
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        I have some documents. I am looking for an answer to my question in these documents.\n
        Here are the documents: \n\n {document} \n\n
        Here is the question I want to answer with these documents: {message} \n
        Can I find the solution to my question in the documents?\n
        Provide the binary score as a JSON with a single key 'score' with a binary score 'yes' or 'no'.  Give no preamble or explanation.
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        One word answer:
        <|eot_id|>""",
        input_variables=["message", "document"]
    )
    retrieval_grade = prompt | llm | JsonOutputParser()

    message = state["message"]
    documents = state["documents"]

    # filter relevant documents
    filtered_docs = []
    web_search = "No"

    for doc in documents:
        score = retrieval_grade.invoke({"message": message, "document": doc.page_content})
        grade = score["score"]
        #document is relevant
        if grade.lower() == 'yes':
            filtered_docs.append(doc)
    
    # Check if filtered_docs is empty
    if not filtered_docs:
        web_search = "Yes"
    else:
        print(f"Found {len(filtered_docs)} relevant documents.")
        web_search = "No"
    
    return {"documents": filtered_docs, "message": message, "web_search": web_search} 


def create_answer(state):
    """
    Creates an LLM generated answer for the user's message based on the filtered documents.

    Args:
        state (dict): A dictionary containing the user's message, the filtered documents and web_search need.

    Returns:
        dict: A dictionary containing the retrieved documents, the user's message and the LLM generated answer.
    """
    llm = ChatOllama(model=local_llm, base_url=LLAMA_SERVER_URL, temperature=0)
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are an assistant for question-answering tasks. Use the context given to answer the question. If you don't get the answer with the context, just say that you don't know. Use three sentences maximum and keep the answer concise.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {message} \n
        Context: {documents} \n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        Answer:
        <|eot_id|>""",
        input_variables=["message", "documents"]
    )
    rag_chain = prompt | llm | StrOutputParser()
    message = state["message"]
    documents = state["documents"]
    # Merge all documents strings into a single string
    merged_documents = "\n".join([doc.page_content for doc in documents])
    answer = rag_chain.invoke({"message": message, "documents": merged_documents})
    
    return {"documents": documents, "message": message, "generation": answer}


def check_generation_against_message_and_docs(state):
    """
    Checks whether the LLM generation has hallucinations compared to the documents used, and if the answer is helpful for the user's message

    Args:
        state (dict): A dictionary containing the user's message, the filtered documents and the LLM generated answer.

    Returns:
        str: information regarding the usefulness of the LLM answer
    """
    llm = ChatOllama(model=local_llm, base_url=LLAMA_SERVER_URL, format="json", temperature=0)
    hallucination_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a grader assessing whether an answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a single key 'score' with a binary score 'yes' or 'no'.  Give no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here I have documents: {documents} \n
        Here is the answer I am giving from those facts: {answer} \n
        Is my answer based in the facts of the document?
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        One word answer:
        <|eot_id|>""",
        input_variables=["documents", "answer"]
    )

    hallucination_chain = hallucination_prompt | llm | JsonOutputParser()
    
    answer_prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a grader assessing whether an answer is helpful for a previous message.  Provide the binary score as a JSON with a single key 'score' with a binary score 'yes' or 'no'.  Give no preamble or explanation
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here I have my message: {message} \n
        Here is the answer I am receiving: {answer} \n
        Is this answer actually helpful for my message?
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        One word score:
        <|eot_id|>""",
        input_variables=["message", "answer"]
    )
    check_qa_chain = answer_prompt | llm | JsonOutputParser()

    hallucination_score = hallucination_chain.invoke({"documents": state["documents"], "answer": state["generation"]})
    hallucination_grade = hallucination_score["score"]
    #check hallucination
    if hallucination_grade.lower() == 'yes':
        check_qa_score = check_qa_chain.invoke({"message": state["message"], "answer": state["generation"]})
        check_qa_grade = check_qa_score['score']
        if check_qa_grade.lower() == 'yes':
            return "useful"
        else:
            return "not useful"
    else:
        return "not supported"

def decide_to_generate(state):
    """
    derive the workflow to websearch if needed. If not, generate 

    Args:
        state (dict): A dictionary containing the user's message, the filtered documents, the websearch need and the LLM generated answer.

    Returns:
        str: name of the node to go to.
    """
    web_search = state['web_search']
    if web_search == 'Yes':
        return "web_search"
    else:
        return "generate"


def tavily_doc_retrieval(message):
    """
    websearch tool to retrieve information found on internet based on user's message

    Args:
        message: str
        User's message

    Returns:
        str: documentation found on internet
    """
    print('--------GETTING TAVILY DOC SAMPLE---------')
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    web_answer = tavily.search(query=message)
    doc = ''
    for web_index in range(len(web_answer['results'])):
        doc += '-------------START WEB PAGE------------------'
        doc += '# WEB PAGE ' + str(web_index + 1) + ' \n'
        doc += '## URL: ' + web_answer['results'][web_index]['url'] + ' \n'
        doc += '### TITLE: ' + web_answer['results'][web_index]['title'] + ' \n'
        doc += '#### CONTENT: \n '+ web_answer['results'][web_index]['content'] + ' \n'
        doc += '-------------END WEB PAGE------------------'
    print('--------TAVILY DOC SAMPLE DONE---------')
    return doc

    
def web_answering(state):
    """
    websearch tool to answer user's message based on internet information.
    Avoids errors for length by summarizing user's message if needed for the tavily client.

    Args:
        state (dict): A dictionary containing the user's message, the filtered documents, the websearch need and the LLM generated answer.

    Returns:
        dict: A dictionary containing the user's message and the LLM generated answer.
    """
    message = state["message"]
    print('--------SEARCHING ONLINE ANSWERS---------')
    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    llm = ChatOllama(model=local_llm, base_url=LLAMA_SERVER_URL, temperature=0)
    if len(message) < 140:
        doc = tavily_doc_retrieval(message)
        prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a content summarizer. You get documents based on webpages search, your goal is to extract the key information from the webpages' document.
            Give an answer to the question based in the webpages' document. 
            Provide the URLS in the answer so the user can search more information.
            It is important to state in the answer that this is an answer based in web search, so there could be some mistakes in the given information.
            Encourage the user to verify the information.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Question: {message} \n
            Webpages' documents: {documents} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            Answer:
            <|eot_id|>""",
            input_variables=["message", "documents"]
        )
        web_chain = prompt | llm | StrOutputParser()
        answer = web_chain.invoke({"message": message, "documents": [doc]})
        return {"message": message, "generation": answer}
    else:
        # Summarize the question first
        summarize_prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a question summarizer. You get a message from the user, your goal is to extract the key information to create a short question that sums up what the user is looking for.
            The question will be used to search on the internet for an answer. Make the question efficient for a proper google search
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Question: {message} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            Summarized One sentence question:
            <|eot_id|>""",
            input_variables=["message"]
        )
        summarize_chain = summarize_prompt | llm | StrOutputParser()
        summarized = summarize_chain.invoke({"message": message})
        summarized_question = summarized["message"]
        doc = tavily_doc_retrieval(summarized_question)
        prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a content summarizer. You get documents based on webpages search, your goal is to extract the key information from the webpages' document.
            Give an answer to the question based in the webpages' document. 
            Provide the URLS in the answer so the user can search more information.
            It is important to state in the answer that this is an answer based in web search, so there could be some mistakes in the given information.
            Encourage the user to verify the information.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Question: {message} \n
            Webpages' documents: {documents} \n
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            Answer:
            <|eot_id|>""",
            input_variables=["message", "documents"]
        )
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"message": message, "documents": [doc]})
        return {"message": message, "generation": response}


def wrong_topic_message(state):
    """
    Creates an LLM answer stating that the user's message is off-topic

    Args:
        state (dict): A dictionary containing the user's message

    Returns:
        dict: A dictionary containing the user's message and the LLM generated answer.
    """
    message = state["message"]
    llm = ChatOllama(model=local_llm, base_url=LLAMA_SERVER_URL, temperature=0)
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a sports chatbot. You have been asked for a topic that is not under the scope of sports. In this case, your main goal is to answer politely to the user saying that you can not provide information related to that topic. Suggest asking for something related to calisthenics, saying that you can help in that matter.
        <|eot_id|><|start_header_id|>user<|end_header_id|>
        {message} \n
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        Polite rejection of answer:
        <|eot_id|>""",
        input_variables=["message"]
    )
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"message": message})
    return {"message": message, "generation": response}




def compile_workflow():
    """
    Orchestrates the workflow creation for its use in the app

    Returns:
        func: the compilation of the workflow
    """
    workflow = StateGraph(GraphState)

    # nodes
    workflow.add_node("start", start_workflow)
    workflow.add_node("wrong_topic", wrong_topic_message)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", create_answer)
    workflow.add_node("websearch", web_answering)
    workflow.add_node("grade_documents", grade_document_relevance)
    # create the graph
    workflow.set_entry_point("start")
    workflow.add_conditional_edges(
        "start", 
        message_classifier,
        {
            "related": "retrieve", 
            "not_related": "wrong_topic"
        },
    )
    # If off-topic question, then explain and finish
    workflow.add_edge("wrong_topic", END)

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents", 
        decide_to_generate,
        {
            "web_search": "websearch", 
            "generate": "generate"
        },
    )
    
    workflow.add_conditional_edges(
        "generate", 
        check_generation_against_message_and_docs,
        {
            "not supported": "generate",
            "useful": END, 
            "not useful": "websearch"
        },
    )
    workflow.add_edge("websearch", END)
    
    return workflow.compile()
