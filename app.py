from flask import Flask
import json

app = Flask(__name__)

from langchain.indexes.vectorstore import VectorStoreIndexWrapper # For wrapping the vectors in a package
from langchain.vectorstores import FAISS # For Storing the Vectors 
from langchain.llms import OpenAI # LLM Model
from langchain.embeddings import OpenAIEmbeddings # For converting text into vectors 
from PyPDF2 import PdfReader # For reading a PDF Document
# from typing_extensions import Concatenate # For Concatenating Strings
from langchain.text_splitter import CharacterTextSplitter # for splitting words into characters


OPENAI_API_KEY = os.environ['API_KEY']


llm = OpenAI(openai_api_key = OPENAI_API_KEY)
embedding = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)



@app.route('/<question>')
def Model(question):
  ques = requests.get_data('question')

  print(ques)
  if question == "":
    return {"":""}
  
  vector_store = FAISS.load_local("vector", embeddings=embedding)
  vector_index = VectorStoreIndexWrapper(vectorstore=vector_store)

  query_text = question
  answer = vector_index.query(query_text, llm=llm).strip()
  

  str = "ans : {}".format(answer)

  str = '{"' + str.replace(':', '":"') +'"}'

  # print(json.loads(str))
  return json.loads(str)

if __name__ == "__main__":
    app.run(debug=True)  
