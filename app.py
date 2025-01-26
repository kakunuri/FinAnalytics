import streamlit as st
import pandas as pd
import json
from llama_index.core import Settings
from llama_index.llms.fireworks import Fireworks
from llama_index.experimental.query_engine import PandasQueryEngine
import os

# Page config
st.set_page_config(
    page_title="Financial Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

class FinancialAnalyzer:
    def __init__(self):
        # Set Fireworks API key
        os.environ["FIREWORKS_API_KEY"] = "YOUR_KEY"  # Replace with your key
        
        # Configure LlamaIndex to use Fireworks
        Settings.llm = Fireworks(
            model="accounts/fireworks/models/mixtral-8x7b-instruct",
            max_tokens=512,
            temperature=0.1
        )
        
        self.load_data()
        self.setup_query_engine()
    
    def load_data(self):
        # Read JSON files
        with open('income-2024.json', 'r') as file:
            data_2024 = json.load(file)
        with open('income-2025.json', 'r') as file:
            data_2025 = json.load(file)
        
        # Convert to DataFrame
        self.df = pd.json_normalize([data_2024, data_2025])
    
    def setup_query_engine(self):
        # Get DataFrame schema
        df_schema = "\n".join([
            f"{col} ({self.df[col].dtype}): {self.df[col].iloc[0]}" 
            for col in self.df.columns
        ])
        
        instruction_str = f"""You are a financial data analysis assistant working with a pandas DataFrame. Your goal is to generate accurate Python pandas code to answer user queries.

DataFrame Schema:
{df_schema}

IMPORTANT RESPONSE FORMAT RULES:
1. Provide ONLY the Python code, no explanations or markdown
2. Do not include backticks (```) or language indicators
3. Do not include any text before or after the code
4. The response should be a single line of Python code that can be directly evaluated

Examples of CORRECT responses:
Query: "What is the total operating expenses in 2024?"
Response: df[(df['Year'] == 2024)].filter(like='Cost of Sales.Operating Expenses.', axis=1).sum(axis=1).sum()

Query: "What is the total revenue in 2025?"
Response: df[(df['Year'] == 2025)]['Revenue.Total Revenue'].sum()

Remember:
- Output ONLY the Python code
- No explanatory text
- No markdown formatting
- No backticks
- Single line of code only
"""
        
        self.query_engine = PandasQueryEngine(
            df=self.df,
            verbose=True,
            instruction_str=instruction_str
        )
    
    def execute_query(self, query):
        try:
            response = self.query_engine.query(query)
            return str(response), None
        except Exception as e:
            return None, str(e)

def main():
    st.title("📊 Financial Data Analysis")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = FinancialAnalyzer()
    
    # Sidebar
    with st.sidebar:
        st.header("Options")
        
        # Model selection
        model = st.selectbox(
            "Select Model",
            [
                "mixtral-8x7b-instruct",
                "llama-v2-70b-chat",
                "llama-v2-13b-chat",
                "codellama-34b-instruct"
            ],
            index=0
        )
        
        # Temperature slider
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1
        )
        
        # Clear history button
        if st.button("Clear History"):
            st.session_state.query_history = []
    
    # Main area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Query Data")
        
        # Query input
        query = st.text_area("Enter your query:", height=100)
        
        if st.button("Execute Query"):
            if query:
                result, error = st.session_state.analyzer.execute_query(query)
                
                # Add to history
                st.session_state.query_history.append({
                    "query": query,
                    "result": result,
                    "error": error
                })
    
    with col2:
        st.header("Query History")
        
        for i, item in enumerate(reversed(st.session_state.query_history)):
            with st.expander(f"Query {len(st.session_state.query_history) - i}"):
                st.write("**Query:**")
                st.code(item["query"])
                
                if item["error"]:
                    st.error(f"Error: {item['error']}")
                else:
                    st.write("**Result:**")
                    st.code(item["result"])
                    
                    # Execute button for history items
                    if st.button("Re-run", key=f"rerun_{i}"):
                        result, error = st.session_state.analyzer.execute_query(item["query"])
                        if error:
                            st.error(f"Error: {error}")
                        else:
                            st.code(result)
    
    # Display DataFrame
    with st.expander("View Raw Data"):
        st.dataframe(st.session_state.analyzer.df)

if __name__ == "__main__":
    main()

