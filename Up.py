import pandas as pd
import os
import sqlite3
import json
import re
import streamlit as st
import requests
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import wikipedia
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables if using a .env file
load_dotenv()

# Configure API keys
GROQ_API_KEY = "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"
GEMINI_API_KEY = "AIzaSyBHBFb7iJ4VNSazi_oNZ50pwSo8suH7Y4M"

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

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
        self.groq_api_key = GROQ_API_KEY
        self.gemini_api_key = GEMINI_API_KEY
        
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
                'Voteshare_Percentage': 'float',
                'Margin': 'float',
                'Margin_Percentage': 'float',
                'Margin_Per': 'float',
                'ENOP': 'float',
                'Turnout_Percentage': 'float'
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
            'ac_type': 'TEXT',
            'candid': 'INTEGER',
            'candidate': 'TEXT',
            'candidate_name': 'TEXT',
            'party': 'TEXT',
            'votes': 'INTEGER',
            'rank': 'INTEGER',
            'valid_votes': 'INTEGER',
            'electors': 'INTEGER',
            'vote_share_percentage': 'REAL',
            'voteshare_percentage': 'REAL',
            'margin': 'REAL',
            'margin_percentage': 'REAL',
            'margin_per': 'REAL',
            'enop': 'REAL',
            'candidate_type': 'TEXT',
            'turnout_percentage': 'REAL'
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
        if not self.groq_api_key:
            return "Error: Groq API key not found. Please set the GROQ_API_KEY environment variable or provide it in Streamlit secrets."
        
        # Make API call to Groq
        headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
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
    
    def generate_gemini_analysis(self, user_query, sql_result_df):
        """Generate additional analysis using Google's Gemini API."""
        try:
            # Convert DataFrame to a string representation
            if isinstance(sql_result_df, pd.DataFrame):
                data_str = sql_result_df.to_string()
                if len(data_str) > 10000:  # Limit data size for API
                    data_str = sql_result_df.head(50).to_string() + "\n[...truncated for brevity...]"
            else:
                data_str = str(sql_result_df)
            
            # Create the prompt for Gemini
            prompt = f"""
You are an expert in Uttar Pradesh political data analyst. I have a query and its results from an electoral database:

Query: "{user_query}"

Results:
{data_str}

Please provide a detailed analysis of this data. Include:
1. Key insights from the data
2. Political trends or patterns
3. Significance of these results in the electoral context
4. Any recommendations for further analysis
5. if you think the above data in result section is wrong then give your own analysis to the question. 

Make your analysis concise but comprehensive.
"""

            # Set up Gemini model
            generation_config = {
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            # Get Gemini model
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash-002",
                generation_config=generation_config
            )
            
            # Generate response
            with st.spinner("Generating additional analysis with Gemini..."):
                response = model.generate_content(prompt)
                
            return response.text
        except Exception as e:
            return f"Error generating Gemini analysis: {str(e)}"
    
    def generate_visualization(self, df, user_query):
        """Generate appropriate visualizations based on the data and query."""
        try:
            if not isinstance(df, pd.DataFrame) or df.empty:
                return None
            
            # Determine visualization type based on data and query
            viz_type = None
            
            # Check for comparison queries (typically need bar charts)
            comparison_patterns = ["compare", "versus", "vs", "difference", "ranking", "rank"]
            if any(pattern in user_query.lower() for pattern in comparison_patterns):
                viz_type = "bar"
                
            # Check for trend queries (typically need line charts)
            trend_patterns = ["trend", "over time", "year", "years", "period"]
            if any(pattern in user_query.lower() for pattern in trend_patterns):
                viz_type = "line"
                
            # Check for distribution queries (pie charts or histograms)
            distribution_patterns = ["distribution", "share", "percentage", "proportion"]
            if any(pattern in user_query.lower() for pattern in distribution_patterns):
                if df.shape[0] <= 10:  # Small number of categories
                    viz_type = "pie"
                else:
                    viz_type = "histogram"
            
            # Default to bar chart for small datasets with categorical data
            if viz_type is None:
                if df.shape[0] <= 20:
                    viz_type = "bar"
                else:
                    viz_type = "table"  # Just show the table for large datasets
            
            # Create visualization based on type
            fig = None
            
            if viz_type == "bar":
                # Try to intelligently determine x and y axes
                categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
                numerical_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                
                if categorical_columns and numerical_columns:
                    # Choose the first categorical column for x-axis
                    x_col = categorical_columns[0]
                    
                    # For y-axis, prefer columns with 'vote', 'margin', or 'percentage'
                    y_col = None
                    for col in numerical_columns:
                        if any(term in col.lower() for term in ["vote", "margin", "percentage", "share"]):
                            y_col = col
                            break
                    
                    if y_col is None and numerical_columns:
                        y_col = numerical_columns[0]
                    
                    if x_col and y_col:
                        # Sort data by y_col for better visualization
                        sorted_df = df.sort_values(by=y_col, ascending=False)
                        
                        # Create bar chart with Plotly
                        fig = px.bar(
                            sorted_df, 
                            x=x_col, 
                            y=y_col, 
                            color=categorical_columns[0] if len(categorical_columns) > 1 else None,
                            title=f"{y_col} by {x_col}",
                            labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()}
                        )
            
            elif viz_type == "line":
                # Try to find a date/year column for x-axis
                time_cols = [col for col in df.columns if "year" in col.lower() or "date" in col.lower()]
                numerical_columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                
                if time_cols and numerical_columns:
                    x_col = time_cols[0]
                    y_col = None
                    
                    # Find an appropriate y column (votes, percentage, etc.)
                    for col in numerical_columns:
                        if any(term in col.lower() for term in ["vote", "margin", "percentage", "share"]):
                            y_col = col
                            break
                    
                    if y_col is None and numerical_columns:
                        y_col = numerical_columns[0]
                    
                    if x_col and y_col:
                        # Create line chart with Plotly
                        fig = px.line(
                            df, 
                            x=x_col, 
                            y=y_col,
                            color=df.columns[0] if len(df.columns) > 2 else None,
                            title=f"{y_col} over {x_col}",
                            labels={x_col: x_col.replace('_', ' ').title(), y_col: y_col.replace('_', ' ').title()}
                        )
            
            elif viz_type == "pie":
                # Choose columns for pie chart
                categorical_cols = [col for col in df.columns if df[col].dtype == 'object']
                numerical_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                
                if categorical_cols and numerical_cols:
                    # Use the first categorical column for names
                    names_col = categorical_cols[0]
                    
                    # Use the first numerical column for values
                    value_col = None
                    for col in numerical_cols:
                        if any(term in col.lower() for term in ["vote", "margin", "percentage", "share"]):
                            value_col = col
                            break
                    
                    if value_col is None and numerical_cols:
                        value_col = numerical_cols[0]
                    
                    if names_col and value_col:
                        # Create pie chart with Plotly
                        fig = px.pie(
                            df, 
                            names=names_col, 
                            values=value_col,
                            title=f"Distribution of {value_col} by {names_col}"
                        )
            
            elif viz_type == "histogram":
                # Choose column for histogram
                numerical_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
                
                if numerical_cols:
                    # Choose a numerical column, preferring vote-related ones
                    hist_col = None
                    for col in numerical_cols:
                        if any(term in col.lower() for term in ["vote", "margin", "percentage", "share"]):
                            hist_col = col
                            break
                    
                    if hist_col is None:
                        hist_col = numerical_cols[0]
                    
                    # Create histogram with Plotly
                    fig = px.histogram(
                        df, 
                        x=hist_col,
                        title=f"Distribution of {hist_col}",
                        labels={hist_col: hist_col.replace('_', ' ').title()}
                    )
            
            return fig
            
        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")
            return None

    def get_candidate_info(self, candidate_name):
        """Get information about a candidate from Wikipedia."""
        try:
            # Search for the candidate on Wikipedia
            search_results = wikipedia.search(f"{candidate_name} politician India")
            
            if not search_results:
                return f"No Wikipedia information found for {candidate_name}."
            
            # Try to get a page about the candidate
            try:
                page = wikipedia.page(search_results[0], auto_suggest=False)
            except wikipedia.exceptions.DisambiguationError as e:
                # If disambiguation page, try the first option
                page = wikipedia.page(e.options[0], auto_suggest=False)
            except Exception:
                return f"Could not retrieve detailed information for {candidate_name}."
            
            # Get a summary of the page
            summary = wikipedia.summary(page.title, sentences=5)
            
            return {
                "name": page.title,
                "summary": summary,
                "url": page.url
            }
        except Exception as e:
            return f"Error retrieving information: {str(e)}"


def main():
    st.set_page_config(
        page_title="Electoral Data Analyzer",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Showtime Uttar Pradesh Electoral Data Analyzer")
    st.subheader("Analyze using natural language questions")
    
    # Add description based on the sample data
    st.markdown("""
    This app analyzes electoral data with information about candidates, constituencies, votes, and more.
    The data includes fields such as:
    - **District & Region**: Geographic information
    - **AC Name & PC Name**: Assembly and Parliamentary Constituency names
    - **Year, AC No, PC No**: Election year and constituency numbers
    - **Candidate details**: Name, Party, Votes received, Rank
    - **Electoral metrics**: Vote Share, Margin, ENOP (Effective Number of Parties)
    
    Select from available datasets or upload your own electoral data CSV file.
    """)
    
    # Initialize session state
    if "analyzer" not in st.session_state:
        st.session_state.analyzer = None
    
    if "user_query" not in st.session_state:
        st.session_state.user_query = ""
    
    if "active_dataset" not in st.session_state:
        st.session_state.active_dataset = None
    
    # Define available datasets with correct paths
    available_datasets = {
        "General Election Data": "General ELection UP 2014_19_24 .csv",
        "State Election Data": "Assembly Election UP 2012_17_22 .csv"
    }
    
    # Dataset selection in main area
    st.header("Choose a dataset to analyze:")
    dataset_cols = st.columns(2)
    
    with dataset_cols[0]:
        if st.button("General Election Data", use_container_width=True):
            selected_path = available_datasets["General Election Data"]
            if os.path.exists(selected_path):
                st.session_state.analyzer = ElectoralDataAnalyzer(selected_path)
                st.session_state.active_dataset = "General Election Data"
            else:
                st.error(f"Dataset file not found: {selected_path}")
    
    with dataset_cols[1]:
        if st.button("State Election Data", use_container_width=True):
            selected_path = available_datasets["State Election Data"]
            if os.path.exists(selected_path):
                st.session_state.analyzer = ElectoralDataAnalyzer(selected_path)
                st.session_state.active_dataset = "State Election Data"
            else:
                st.error(f"Dataset file not found: {selected_path}")
    
    # Display example questions in the main area
    if st.session_state.analyzer is not None:
        st.markdown("### Example Questions")
        
        # Create a 2x2 grid of example questions
        col1, col2 = st.columns(2)
        
        example_questions = [
            "Show candidates who won with highest vote share percentage",
            "What was the average margin of victory by party?",
            "Compare performance of BJP vs INC across all constituencies",
            "Show districts where candidates ranked 3rd"
        ]
        
        with col1:
            if st.button(example_questions[0]):
                st.session_state.user_query = example_questions[0]
            if st.button(example_questions[2]):
                st.session_state.user_query = example_questions[2]
        
        with col2:
            if st.button(example_questions[1]):
                st.session_state.user_query = example_questions[1]
            if st.button(example_questions[3]):
                st.session_state.user_query = example_questions[3]
    
    # Display dataframe if analyzer is initialized
    if st.session_state.analyzer is not None and hasattr(st.session_state.analyzer, 'df'):
        # Display active dataset info
        st.info(f"Active Dataset: {st.session_state.active_dataset}")
        
        with st.expander("Preview of loaded data"):
            st.dataframe(st.session_state.analyzer.df.head(10))
            
            # Debug option to show column information
            if st.checkbox("Show column information (for debugging)"):
                col_info = st.session_state.analyzer.get_column_map_info()
                st.json(col_info)
    
        # Input for question
        query_input = st.text_input("Ask a question about the electoral data:",
                                   value=st.session_state.user_query,
                                   key="query_input",
                                   placeholder="Example: Who won with the highest margin?")
        
        # Button to submit question
        if st.button("Analyze") or st.session_state.user_query:
            if query_input:
                # Store query in session state
                current_query = query_input
                st.session_state.user_query = ""  # Reset for next query
                
                # Create tabs for different display formats
                result_tabs = st.tabs(["SQL Results", "Visualization", "Gemini Analysis", "Candidate Info"])
                
                # Execute query
                result = st.session_state.analyzer.execute_query_with_llm(current_query)

                # Handle result in the first tab (SQL Results)
                with result_tabs[0]:
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
                
                # Handle visualization in the second tab
                with result_tabs[1]:
                    if isinstance(result, dict) and "result" in result and isinstance(result["result"], pd.DataFrame):
                        st.subheader("Visualization")
                        
                        # Generate appropriate visualization based on data
                        fig = st.session_state.analyzer.generate_visualization(result["result"], current_query)
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("No suitable visualization could be generated for this query result.")
                            
                            # Show a table as fallback
                            st.dataframe(result["result"])
                    else:
                        st.info("No data available for visualization.")
                
                # Handle Gemini Analysis in the third tab
                with result_tabs[2]:
                    st.subheader("Gemini AI Analysis")
                    
                    if isinstance(result, dict) and "result" in result and isinstance(result["result"], pd.DataFrame):
                        # Generate analysis with Gemini
                        gemini_analysis = st.session_state.analyzer.generate_gemini_analysis(
                            current_query, result["result"]
                        )
                        
                        if gemini_analysis:
                            st.markdown(gemini_analysis)
                        else:
                            st.info("Gemini AI analysis could not be generated.")
                    else:
                        st.info("No data available for Gemini AI analysis.")
                
                # Handle Candidate Information in the fourth tab
                with result_tabs[3]:
                    st.subheader("Candidate Information")
                    
                    if isinstance(result, dict) and "result" in result and isinstance(result["result"], pd.DataFrame):
                        # Check if there are candidate names in the result
                        candidate_cols = [col for col in result["result"].columns 
                                         if "candidate" in col.lower() or "name" in col.lower()]
                        
                        if candidate_cols:
                            candidate_col = candidate_cols[0]
                            candidates = result["result"][candidate_col].unique()
                            
                            if len(candidates) > 0:
                                # Let user choose a candidate to get info about
                                selected_candidate = st.selectbox(
                                    "Select a candidate to view information:",
                                    candidates
                                )
                                
                                if selected_candidate:
                                    with st.spinner(f"Fetching information about {selected_candidate}..."):
                                        candidate_info = st.session_state.analyzer.get_candidate_info(selected_candidate)
                                        
                                        if isinstance(candidate_info, dict):
                                            st.subheader(candidate_info["name"])
                                            st.markdown(candidate_info["summary"])
                                            st.markdown(f"[Read more on Wikipedia]({candidate_info['url']})")
                                        else:
                                            st.info(candidate_info)
                            else:
                                st.info("No candidate names found in the result.")
                        else:
                            st.info("No candidate names found in the query result.")
                    else:
                        st.info("No candidate data available.")

if __name__ == "__main__":
    main()
