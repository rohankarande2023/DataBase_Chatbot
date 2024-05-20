from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
#from langchain_experimental.sql import SQLDatabaseChain
#from langchain_community.llms import OpenAI
from langchain_openai import OpenAI

import streamlit as st
import os

# Set your OpenAI API key
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt Template

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)
## streamlit framework

st.title('Langchain Demo With OPEN_AI API')
input_text=st.text_input("Search the topic u want")

# OPENAI LLm 
llm=ChatOpenAI(model="gpt-3.5-turbo")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser



# Connect to MySQL database from Jupyter notebook
db = SQLDatabase.from_uri(
    database_uri=os.getenv("database_uri")
)


# Create SQLDatabse chain Object
# from langchain.chains import create_sql_query_chain
# db_chain=create_sql_query_chain(llm,db)
# response = db_chain.invoke({"question": "How many total t-shirts we have?"})
# print(response)


from langchain_experimental.sql import SQLDatabaseChain
db_chain=SQLDatabaseChain.from_llm(llm,db,verbose=True)
#q1=db_chain("How many total t-shirts we have?")

from langchain_experimental.sql import SQLDatabaseChain

# Create the SQLDatabaseChain instance
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# Check if there's any input text
import os
import streamlit as st
from langchain_experimental.sql import SQLDatabaseChain
from io import StringIO
import sys

# Assume 'llm' and 'db' are already defined
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# Example input text
input_text = "Which brand offers maximum discount and how much"

if input_text:
    # Capture the intermediate outputs
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    # Execute the query using the SQLDatabaseChain instance
    q1 = db_chain(input_text)
    
    # Get the intermediate results from stdout
    intermediate_output = sys.stdout.getvalue()
    
    # Reset stdout
    sys.stdout = old_stdout
    

    # Find and display the SQL query part from the intermediate output
    sql_query_start = intermediate_output.find("SQLQuery:")
    sql_query_end = intermediate_output.find("SQLResult:")
    import re
    if sql_query_start != -1 and sql_query_end != -1:
        sql_query = intermediate_output[sql_query_start + len("SQLQuery:"):sql_query_end].strip()
        # Remove ANSI escape sequences
        sql_query_clean = re.sub(r'\x1b\[[0-9;]*m', '', sql_query)
        st.write("SQLQuery:")
        st.code(sql_query_clean, language='sql')


    # Display the final answer
    st.write(q1['result'])




# if input_text:
#     q1=db_chain(input_text)
#     st.write(q1)
    #st.write(db_chain.invoke({"query":input_text}))
    #st.write(q1)

    #st.write(db_chain(input_text))
