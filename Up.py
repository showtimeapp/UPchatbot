import pandas as pd
import os
import sqlite3
import json
import re
import streamlit as st
import requests
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables if using a .env file
load_dotenv()

class ElectoralDataAnalyzer:
    def __init__(self, csv_path, db_path=None):
        """Initialize the analyzer with the CSV file path and SQLite database path."""
        self.csv_path = csv_path
        
        # If no database path provided, use a default based on CSV name
        if db_path is None:
            csv_filename = Path(csv_path).stem
            self.db_path = f"{csv_filename}.db"
        else:
            self.db_path = db_path
        
        # SQLite connection
        self.conn = None
        self.table_name = "electoral_data"
        
        # Get API key from environment or Streamlit secrets
        self.api_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
        
        # Store original column names and their lowercase versions for mapping
        self.column_mapping = {}
        
        # Load data and create table in SQLite
        st.info(f"Loading data from {self.csv_path}...")
        self.df = self.load_csv_data()
        if self.df is not None:
            st.success(f"Successfully loaded data with {len(self.df)} rows.")
            self.create_sqlite_database()
    
    def load_csv_data(self):
        """Load CSV data into a pandas DataFrame for preprocessing."""
        try:
            # Check if file exists
            if not os.path.exists(self.csv_path):
                st.error(f"CSV file not found: {self.csv_path}")
                return None
            
            # Detect file type (TSV or CSV) based on content
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                sample = f.read(1024)
                if '\t' in sample:
                    separator = '\t'  # Tab-separated file
                else:
                    separator = ','   # Comma-separated file
            
            # Load the file with the detected separator
            df = pd.read_csv(self.csv_path, sep=separator)
            
            # Store original column names before cleaning
            self.original_columns = df.columns.tolist()
            
            # Clean column names - remove any trailing whitespace and replace spaces with underscores
            # But KEEP the original capitalization
            df.columns = [col.strip().replace(' ', '_') for col in df.columns]
            
            # Create a mapping between cleaned column names and their lowercase versions
            self.column_mapping = {col.lower(): col for col in df.columns}
            
            # Convert specific columns to appropriate types
            type_conversions = {
                'Year': 'int',
                'AC_No': 'int', 
                'PC_No': 'int',
                'CandID': 'int',
                'Votes': 'int',
                'Rank': 'int',
                'Valid_Votes': 'int',
                'Vote_Share_Percentage': 'float',
                'Margin': 'float',
                'Margin_Percentage': 'float',
                'ENOP': 'float'
            }
            
            # Apply type conversions
            for col, dtype in type_conversions.items():
                # Look for column case-insensitively
                matching_cols = [actual_col for actual_col in df.columns 
                                if actual_col.lower() == col.lower()]
                
                if matching_cols:
                    actual_col = matching_cols[0]
                    try:
                        df[actual_col] = df[actual_col].astype(dtype)
                    except:
                        st.warning(f"Could not convert column {actual_col} to {dtype}. Keeping original type.")
            
            return df
        except Exception as e:
            st.error(f"Error loading CSV data: {str(e)}")
            return None
    
    def create_sqlite_connection(self):
        """Create a connection to the SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except Exception as e:
            st.error(f"Error connecting to SQLite database: {str(e)}")
            return None
    
    def create_sqlite_database(self):
        """Create the SQLite database and load data."""
        try:
            # Create connection
            self.conn = self.create_sqlite_connection()
            if self.conn is None:
                return False
            
            # Create table
            self.create_table()
            
            # Load data into table
            self.load_data_to_sqlite()
            
            st.success(f"Database created at {self.db_path}")
            return True
        except Exception as e:
            st.error(f"Error creating SQLite database: {str(e)}")
            return False
    
    def create_table(self):
        """Create the electoral_data table in SQLite."""
        cursor = self.conn.cursor()
        
        # Drop table if exists
        cursor.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        
        # Generate CREATE TABLE statement
        create_table_sql = self.generate_create_table_sql()
        
        # Create table
        cursor.execute(create_table_sql)
        self.conn.commit()
    
    def generate_create_table_sql(self):
        """Generate SQLite CREATE TABLE statement based on DataFrame schema."""
        create_table_sql = f"CREATE TABLE {self.table_name} (\n"
        
        # Define explicit column types based on the electoral data structure
        column_types = {
            'district': 'TEXT',
            'region': 'TEXT',
            'ac_name': 'TEXT',
            'pc_name': 'TEXT',
            'year': 'INTEGER',
            'ac_no': 'INTEGER',
            'pc_no': 'INTEGER',
            'constituency_type': 'TEXT',
            'candid': 'INTEGER',
            'candidate': 'TEXT',
            'party': 'TEXT',
            'votes': 'INTEGER',
            'rank': 'INTEGER',
            'valid_votes': 'INTEGER',
            'vote_share_percentage': 'REAL',
            'margin': 'REAL',
            'margin_percentage': 'REAL',
            'enop': 'REAL',
            'candidate_type': 'TEXT'
        }
        
        for col in self.df.columns:
            # Store lowercase column name for matching
            col_lower = col.lower()
            
            # Use our predefined type if available, otherwise infer from data
            if col_lower in column_types:
                data_type = column_types[col_lower]
            elif pd.api.types.is_integer_dtype(self.df[col]):
                data_type = "INTEGER"
            elif pd.api.types.is_float_dtype(self.df[col]):
                data_type = "REAL"
            elif pd.api.types.is_bool_dtype(self.df[col]):
                data_type = "INTEGER"  # SQLite doesn't have a boolean type
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                data_type = "TEXT"  # Store dates as text in SQLite
            else:
                data_type = "TEXT"
            
            # Escape column name with double quotes for SQLite
            create_table_sql += f'    "{col}" {data_type},\n'
        
        # Remove trailing comma and close the statement
        create_table_sql = create_table_sql.rstrip(",\n") + "\n);"
        
        return create_table_sql
        
    def load_data_to_sqlite(self):
        """Load DataFrame data into SQLite table."""
        # Insert data into SQLite
        self.df.to_sql(self.table_name, self.conn, if_exists='replace', index=False)
        self.conn.commit()
    
    def get_table_schema(self):
        """Get the actual schema of the table from SQLite."""
        try:
            cursor = self.conn.cursor()
            
            # Get table info
            cursor.execute(f"PRAGMA table_info({self.table_name});")
            columns = cursor.fetchall()
            
            schema_str = f"CREATE TABLE {self.table_name} (\n"
            
            for col in columns:
                # SQLite PRAGMA table_info returns: 
                # (id, name, type, notnull, default_value, primary_key)
                col_id, col_name, data_type, not_null, default_val, primary_key = col
                
                schema_line = f'    "{col_name}" {data_type}'
                
                if not_null == 1:
                    schema_line += " NOT NULL"
                    
                if primary_key == 1:
                    schema_line += " PRIMARY KEY"
                    
                if default_val is not None:
                    schema_line += f" DEFAULT {default_val}"
                    
                schema_str += schema_line + ",\n"
            
            schema_str = schema_str.rstrip(",\n") + "\n);"
            
            return schema_str
        except Exception as e:
            return f"Error retrieving schema: {str(e)}"
    
    def get_column_names(self):
        """Get the actual column names from SQLite table."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"PRAGMA table_info({self.table_name});")
            columns = cursor.fetchall()
            # Column name is at index 1 in each row
            return [col[1] for col in columns]
        except Exception as e:
            st.error(f"Error retrieving column names: {str(e)}")
            return []
    
    def get_data_sample(self, rows=5):
        """Get a sample of data from the SQLite table."""
        try:
            # Create a pandas DataFrame from a sample query
            sample_df = pd.read_sql_query(f"SELECT * FROM {self.table_name} LIMIT {rows}", self.conn)
            
            # Convert to list of dictionaries
            sample_data = sample_df.to_dict(orient='records')
            
            return json.dumps(sample_data, indent=2, default=str)
        except Exception as e:
            return f"Error retrieving sample data: {str(e)}"
    
    def execute_query_with_llm(self, user_query):
        """Use Groq LLM API to generate SQL query based on user's question and execute it against SQLite."""
        # Get the actual schema and sample data
        schema = self.get_table_schema()
        sample_data = self.get_data_sample(3)
        
        # Get the actual column names to provide to the LLM
        actual_columns = self.get_column_names()
        columns_info = "\n".join([f"- {col}" for col in actual_columns])
        
        # Prepare the prompt for the LLM
        prompt = f"""
You are an expert SQL analyst working with SQLite. I have a dataset with the following schema:

{schema}

Here's a sample of the data:
{sample_data}

IMPORTANT: The exact column names in the database are case-sensitive and are as follows:
{columns_info}

The user's question is: "{user_query}"

Please:
1. Generate a SQLite compatible query that would answer this question.
2. Make sure your SQL query is properly formatted and follows SQLite syntax.
3. CRITICAL: Use EXACTLY the column names listed above - they are case-sensitive.
4. Only use SQL functions and features that are available in SQLite.
5. Explain your approach to answering the question.
6. The query should be ready to execute directly against the database.

Your response should be formatted as:
```sql
-- SQL Query for SQLite
<the SQL query>
```

EXPLANATION: <explanation of your approach>
"""

        # Check if API key is available
        if not self.api_key:
            return "Error: Groq API key not found. Please set the GROQ_API_KEY environment variable or provide it in Streamlit secrets."
        
        # Make API call to Groq
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama3-70b-8192",  # Using llama3-70b model
            "messages": [
                {"role": "system", "content": "You are a helpful assistant specialized in SQLite queries and data analysis."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 2048
        }
        
        try:
            with st.spinner("Generating SQL query with Groq LLM..."):
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                
                llm_response = response.json()["choices"][0]["message"]["content"]
                
                # Extract the SQL query from the response
                sql_match = re.search(r'```sql\n(.*?)```', llm_response, re.DOTALL)
                if sql_match:
                    sql_query = sql_match.group(1).strip()
                    if "-- SQL Query for SQLite" in sql_query:
                        sql_query = sql_query.replace("-- SQL Query for SQLite", "").strip()
                    
                    # Execute the query on SQLite
                    with st.spinner("Executing SQL query..."):
                        result = self.execute_sql_query(sql_query)
                    
                    # Get explanation part
                    explanation = ""
                    if 'EXPLANATION:' in llm_response:
                        explanation = llm_response.split('EXPLANATION:')[-1].strip()
                    
                    return {
                        "query": sql_query,
                        "result": result,
                        "explanation": explanation
                    }
                else:
                    return {
                        "error": f"Could not extract SQL query from LLM response. Full response:\n\n{llm_response}"
                    }
                
        except Exception as e:
            return {"error": f"Error calling Groq API: {str(e)}"}
    
    def execute_sql_query(self, sql_query):
        """Execute an SQL query on SQLite and return the results as a DataFrame."""
        try:
            # Execute query and return as DataFrame
            result_df = pd.read_sql_query(sql_query, self.conn)
            return result_df
        except Exception as e:
            return f"Error executing SQL query: {str(e)}"
    
    def get_column_map_info(self):
        """Returns information about the column mapping for debugging."""
        return {
            "original_columns": self.original_columns if hasattr(self, 'original_columns') else [],
            "current_columns": self.df.columns.tolist() if hasattr(self, 'df') else [],
            "column_mapping": self.column_mapping if hasattr(self, 'column_mapping') else {}
        }

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Electoral Data Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Electoral Data Analyzer")
    st.subheader("Analyze electoral data using natural language questions")
    
    # Add description based on the sample data
    st.markdown("""
    This app analyzes electoral data with information about candidates, constituencies, votes, and more.
    The data includes fields such as:
    - **District & Region**: Geographic information
    - **AC Name & PC Name**: Assembly and Parliamentary Constituency names
    - **Year, AC No, PC No**: Election year and constituency numbers
    - **Candidate details**: Name, Party, Votes received, Rank
    - **Electoral metrics**: Vote Share, Margin, ENOP (Effective Number of Parties)
    
    Upload your electoral data or provide a path to your CSV file to get started.
    """)
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # File uploader for CSV
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv", "tsv", "txt"])
        
        # # API Key input
        # api_key = st.text_input("Groq API Key", 
        #                        value=os.environ.get("GROQ_API_KEY", ""), 
        #                        type="password",
        #                        help="Enter your Groq API key. It will be saved to session state.")
        
        # if api_key:
        #     os.environ["GROQ_API_KEY"] = api_key
        
        st.divider()
        st.markdown("### Example Questions")
        example_questions = [
            "Show candidates who won with highest vote share percentage",
            "What was the average margin of victory by party?",
            "Compare performance of BJP vs INC across all constituencies",
            "Show districts where BSP candidates ranked 3rd"
        ]   
        
        for question in example_questions:
            if st.button(question):
                st.session_state.user_query = question
    
    # Initialize session state
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = None
    
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    
    # Process uploaded file
    if uploaded_file is not None:
        # Save uploaded file to disk temporarily
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize analyzer with the uploaded file
        st.session_state.analyzer = ElectoralDataAnalyzer(temp_file_path)
    elif "analyzer" not in st.session_state or st.session_state.analyzer is None:
        # Default path prompt
        default_path = "electoral_data.csv"
        csv_path = st.text_input("Enter path to CSV/TSV file:", default_path, 
                                help="Enter the path to your tab-separated or comma-separated electoral data file")
        
        file_type = st.radio(
            "Select file type:",
            ["CSV (comma-separated)", "TSV (tab-separated)"],
            index=1,  # Default to TSV since your sample is tab-separated
            horizontal=True
        )
        
        if st.button("Load Data"):
            if os.path.exists(csv_path):
                st.session_state.analyzer = ElectoralDataAnalyzer(csv_path)
            else:
                st.error(f"File not found: {csv_path}")
                st.info("Please make sure the file exists in the correct location and that you've entered the path correctly.")
                st.markdown("**Tip**: If you're running this in a notebook or IDE, make sure the working directory is set correctly.")
    
    # Display dataframe if analyzer is initialized
    if st.session_state.analyzer is not None and hasattr(st.session_state.analyzer, 'df'):
        with st.expander("Preview of loaded data"):
            st.dataframe(st.session_state.analyzer.df.head(10))
            
            # Debug option to show column information
            if st.checkbox("Show column information (for debugging)"):
                col_info = st.session_state.analyzer.get_column_map_info()
                st.json(col_info)
    
        # Input for question
        query_input = st.text_input("Ask a question about the electoral data:",
                                   value=st.session_state.user_query,
                                   key="query_input")
        
        # Button to submit question
        if st.button("Analyze") or st.session_state.user_query:
            if query_input:
                # Store query in session state
                current_query = query_input
                st.session_state.user_query = ""  # Reset for next query
                
                # Execute query
                result = st.session_state.analyzer.execute_query_with_llm(current_query)
                
                # Display results
                if isinstance(result, dict):
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        # Show SQL Query
                        st.subheader("SQL Query")
                        st.code(result["query"], language="sql")
                        
                        # Show Results
                        st.subheader("Query Results")
                        if isinstance(result["result"], pd.DataFrame):
                            st.dataframe(result["result"])
                            
                            # Add download button for results
                            csv = result["result"].to_csv(index=False)
                            st.download_button(
                                label="Download results as CSV",
                                data=csv,
                                file_name="query_results.csv",
                                mime="text/csv",
                            )
                        else:
                            st.text(result["result"])
                        
                        # Show Explanation
                        if result["explanation"]:
                            st.subheader("Explanation")
                            st.markdown(result["explanation"])
                else:
                    st.error("An error occurred processing your query.")

if __name__ == "__main__":
    main()