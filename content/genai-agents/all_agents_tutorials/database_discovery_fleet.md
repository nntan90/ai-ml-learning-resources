# Notebook: database_discovery_fleet

> Source: https://github.com/NirDiamant/GenAI_Agents/blob/HEAD/all_agents_tutorials/database_discovery_fleet.ipynb

---

# DataScribe: AI-Powered Schema Explorer

## Disclaimer
DO NOT use this on production data, it is untested and has no safety rails as to what the agent will do, it is possible for it to attempt to do INSERT, UPDATE and DELETE.

If you are planning on testing it on your own data, ensure you have setup a READONLY user.

You have been warned. :)

## 🚀 **What’s the Idea?**

This tutorial focuses on exploring an **AI Agent system** designed to assist users in exploring, querying, and analysing relational databases.

The system looks to simplify database query tasks by enabling intuitive operations, such as:

- **Schema Discovery and Inference**: Helping users uncover and understand the structure of databases.
- **Complex Query Execution**: Supporting intricate operations to extract insights effectively.

The system has a **stateful Supervisor Agent** that oversees and coordinates multiple specialised, stateless sub-agents. These sub-agents will focus on tasks like:

- **Planning**
- **Discovery**
- **Inference**

---

## 🛠 **Tech Stack**

Here’s the stack we’re leveraging to bring this project to life:

- **Python**: The backbone of our agent system, powering logic and workflows.
- **LangChain/LangGraph**: Managing interactions and workflows for AI components.
- **SQLite**: The primary relational database for development and testing.
- **GraphDB**: NetworkX, Visualising relationships within the database.

---

## 🧩 **Use Case**

This tutorial is to explore interacting with relational databases for accessibility, especially for non-expert users.

The **AI agents** look to simplify:

- **Database Discovery**: Exploring and understanding schemas and relationships.
- **Pattern Inference**: Querying and analysing data.

---

## 🎯 **What’s the Goal?**

To create a robust and versatile **AI agent system** that:

- **Simplifies database tasks** for non-expert users.
- Handles operations like discovery, inference, and insight generation autonomously.
- Bridges the gap between technical data management and intuitive user interaction.

Here is the State Graph for how the agent fleet processes input from a user.

![Screenshot 2024-11-24 at 9.00.09 am.png](attachment:4a02c7ba-ad35-4759-bab5-5d46ef3ef5ed.png)

Here is the architecture of the application.

![Screenshot 2024-11-24 at 9.05.40 am.png](attachment:eaeb7f62-a929-4f56-93c8-58b7fdca383d.png)

# Packages needed to run the tutorial

```python
!pip install langgraph langchain langchain-openai langchain_community python-dotenv networkx matplotlib pydot networkx python-dotenv
```

# Import required libraries and export environment variables

```python
import json
import logging
import os
import sys
from typing import Annotated, TypedDict, List, Optional
from typing_extensions import NotRequired

import matplotlib.pyplot as plt
import networkx as nx
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import Tool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from networkx.drawing.nx_pydot import graphviz_layout
from networkx.drawing.nx_agraph import graphviz_layout
import pickle

```

```python
# Create environment variables for OpenAI and the Database

os.environ["OPENAI_API_KEY"] = "put your own key here"
os.environ["DATABASE"] = "data/chinook.db"
```

# Test the connection to the sample database


```python
import sqlite3
from pathlib import Path

def test_db_connection():
    try:
        # Set the path to the database and create a connection
        db_path = os.getenv("DATABASE")
        conn = sqlite3.connect(db_path)
        
        # Simple test query
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM tracks")
        track_count = cursor.fetchone()[0]

        # Close the database connection and notify the user of a successful test
        conn.close()
        print(f"Database connection successful. Found {track_count} tracks.")
        return True
        
    except Exception as e:
        # Notify users there was an error
        print(f"Database connection failed: {e}")
        return False

if __name__ == "__main__":
    test_db_connection()
```

# Setup logging


```python
# Configure the main logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Adjust logging levels for specific libraries to reduce noise
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

```

# Centralised config

Setup centralised config object to keep our code [DRY](https://www.getdbt.com/blog/guide-to-dry).

```python
class Config:
    def __init__(self):
        # Load required environment variables
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.db = os.getenv("DATABASE")

        # Ensure all required variables are set, otherwise raise an error
        if not all([self.openai_api_key, self.db]):
            raise ValueError("Missing required environment variables: OPENAI_API_KEY, DATABASE")

        # Configure database connection
        self.db_engine = SQLDatabase.from_uri(f"sqlite:///{self.db}")

        # Set up language models with specific configurations
        self.llm = ChatOpenAI(temperature=0)  # Default model (e.g., GPT-3.5)
        self.llm_gpt4 = ChatOpenAI(temperature=0, model_name="gpt-4")  # Explicitly use GPT-4

```

# Discovery Agent

DiscoverAgent looks at a database, identifies tables, columns and foreign keys, then returns a Graph object of the relations.

The graph that is created is then used by the InterfaceAgent to help it get more accurate and effiencnt answers by injected relevant context about the database based on the users request.

```python
class DiscoveryAgent:
    def __init__(self):
        # Initialize configuration and toolkit
        self.config = Config()
        self.toolkit = SQLDatabaseToolkit(db=self.config.db_engine, llm=self.config.llm_gpt4)
        self.tools = self.toolkit.get_tools()

        # Add custom tool for schema visualization
        self.tools.extend([
            Tool(
                name="VISUALISE_SCHEMA",
                func=self.discover,
                description="Creates a visual graph representation of the database schema showing tables, columns, and their relationships."
            )
        ])

        # Set up the chat prompt and OpenAI-based agent
        self.chat_prompt = self.create_chat_prompt()
        self.agent = create_openai_functions_agent(
            llm=self.config.llm_gpt4,
            prompt=self.chat_prompt,
            tools=self.tools
        )

        # Configure agent executor for query handling
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=15
        )

    def run_query(self, q):
        # Execute a SQL query using the configured database engine
        return self.config.db_engine.run(q)

    def create_chat_prompt(self):
        # Create the system message template for generating SQL responses
        system_message = SystemMessagePromptTemplate.from_template(
            """
            You are an AI assistant for querying a SQLLite database named {db_name}.
            Your responses should be formatted as json only.
            Always strive for clarity, terseness and conciseness in your responses.
            Return a json array with all the tables, using the example below:

            Example output:
            ```json
            [
                {{
                    tableName: [NAME OF TABLE RETURNED],
                    columns: [
                        {{
                            "columnName": [COLUMN 1 NAME],
                            "columnType": [COLUMN 1 TYPE],
                            "isOptional": [true OR false],
                            "foreignKeyReference": {{
                                "table": [REFERENCE TABLE NAME],
                                "column": [REFERENCE COLUMN NAME]
                            }}
                        }},
                        {{
                            "columnName": [COLUMN 2 NAME],
                            "columnType": [COLUMN 2 TYPE],
                            "isOptional": [true OR false],
                            "foreignKeyReference": {{
                                "table": [REFERENCE TABLE NAME],
                                "column": [REFERENCE COLUMN NAME]
                            }}
                        }}
                    ]
                }}
            ]
            ```

            ## mandatory
            only output json
            do not put any extra commentary
            """
        )

        # Define the human message template
        human_message = HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}")

        # Combine the system and human templates into a chat prompt
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def discover(self) -> nx.Graph:
        """Perform schema discovery and return a graph representation."""
        logger.info("Performing discovery...")
        prompt = "For all tables in this database, show the table name, column name, column type, if its optional. Also show Foreign key references to other columns. Do not show examples. Output only as json."
        
        # Invoke the agent executor with the discovery prompt
        response = self.agent_executor.invoke({"input": prompt, "db_name": self.config.db})

        # Convert the JSON response into a graph representation
        graph = self.jsonToGraph(response)
        return graph

    def jsonToGraph(self, response):
        # Parse the JSON response into a format suitable for graph generation
        output_ = response['output']
        return self.parseJson(output_)

    def parseJson(self, output_):
        # Parse JSON output and construct a graph of the database schema
        j = output_[output_.find('\n') + 1:output_.rfind('\n')]
        data = json.loads(j)

        G = nx.Graph()  # Initialize a new graph
        nodeIds = 0  # Track table nodes
        columnIds = len(data) + 1  # Track column nodes
        labeldict = {}  # Store node labels for visualization
        canonicalColumns = {}  # Map table-column pairs to column node IDs

        # Add tables and columns as nodes in the graph
        for table in data:
            nodeIds += 1
            G.add_node(nodeIds)
            G.nodes[nodeIds]['tableName'] = table["tableName"]
            labeldict[nodeIds] = table["tableName"]

            for column in table["columns"]:
                columnIds += 1
                G.add_node(columnIds)
                G.nodes[columnIds]['columnName'] = column["columnName"]
                G.nodes[columnIds]['columnType'] = column["columnType"]
                G.nodes[columnIds]['isOptional'] = column["isOptional"]
                labeldict[columnIds] = column["columnName"]
                canonicalColumns[table["tableName"] + column["columnName"]] = columnIds
                G.add_edge(nodeIds, columnIds)

        # Add edges for foreign key relationships
        for table in data:
            for column in table["columns"]:
                if column["foreignKeyReference"] is not None:
                    this_column = table["tableName"] + column["columnName"]
                    reference_column_ = column["foreignKeyReference"]["table"] + column["foreignKeyReference"]["column"]
                    G.add_edge(canonicalColumns[this_column], canonicalColumns[reference_column_])

        return G

```

# Discovery test

This enables you to test the DiscoverAgent discovery process in isolation.

This method takes some time (2-3 minutes), using OpenAI. Other llms untested.

Verbose is True for the agent so you can see its workings.

```python
# Create a DiscoveryAgent
agent = DiscoveryAgent()

# Generate the relationship graph
G = agent.discover()
```

# Visualising the database relationship graph

This visualises the relationship graph and highlights the tables (in BLUE) and the fields (in GREEN).

Ensure you have run the discovery step above.

```python
import matplotlib.pyplot as plt
import networkx as nx

def plot_graph(G, title="Graph Visualization"):
    """Plot a NetworkX graph with specific colors for tables and fields."""
    # Create a dictionary for node labels (tables and columns)
    labels = {}
    for node in G.nodes():
        if 'tableName' in G.nodes[node]:  # Label nodes with table names
            labels[node] = G.nodes[node]['tableName']
        elif 'columnName' in G.nodes[node]:  # Label nodes with column names
            labels[node] = G.nodes[node]['columnName']
        else:
            labels[node] = str(node)  # Default to the node ID as label

    # Set node colors based on their type
    color_map = []
    for node in G.nodes():
        if 'tableName' in G.nodes[node]:  # Tables are red
            color_map.append('red')
        elif 'columnName' in G.nodes[node]:  # Columns are green
            color_map.append('green')
        else:  # Other nodes are blue
            color_map.append('blue')

    # Define graph layout using Graphviz's 'neato' algorithm
    pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
    plt.rcParams['figure.figsize'] = [20, 20]  # Set figure size
    nx.draw(G, pos, labels=labels, node_color=color_map, with_labels=True)  # Draw the graph with labels and colors
    plt.title(title)  # Add a title to the plot
    plt.show()  # Display the plot

# Call the function to visualize the graph (ensure G is defined elsewhere)
plot_graph(G)
```

# InferenceAgent

This defines the inference agent. This agent's role is to answer any database analysis questions, "how many employees do we have", "what is the total value of all the invoices for the year" etc.

This agent uses `analyze_question_with_graph` to query the db_graph to get contextual information about the database relationships related to the user query. Because this tutorial uses NetworkX, it doesn't have a native query language. For production use you would use something like 4neoj which has its own query language which would enable for much more sophisticated retrieval.

```python
class InferenceAgent:
    def __init__(self):
        # Initialize configuration, toolkit, and tools
        self.config = Config()
        self.toolkit = SQLDatabaseToolkit(db=self.config.db_engine, llm=self.config.llm)
        self.tools = self.toolkit.get_tools()
        self.chat_prompt = self.create_chat_prompt()

        # Create an OpenAI-based agent with tools and prompt
        self.agent = create_openai_functions_agent(
            llm=self.config.llm,
            prompt=self.chat_prompt,
            tools=self.tools
        )

        # Configure the agent executor with runtime settings
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=False,
            handle_parsing_errors=True,
            max_iterations=15
        )

        # Test the database connection
        self.test_connection()

    def test_connection(self):
        # Verify the database connection by running a test query
        try:
            self.show_tables()
            logger.info("Database connection successful")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            raise

    def show_tables(self) -> str:
        # Query to list all tables and views in the database
        q = '''
            SELECT
                name,
                type
            FROM sqlite_master
            WHERE type IN ("table","view");
            '''
        return self.run_query(q)

    def run_query(self, q: str) -> str:
        # Execute a SQL query and handle errors if they occur
        try:
            return self.config.db_engine.run(q)
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            return f"Error executing query: {str(e)}"

    def create_chat_prompt(self) -> ChatPromptTemplate:
        # Create a system prompt to guide the LLM's behavior and response format
        system_message = SystemMessagePromptTemplate.from_template(
            """You are a database inference expert for a SQLite database named {db_name}.
            Your job is to answer questions by querying the database and providing clear, accurate results.

            Rules:
            1. ONLY execute queries that retrieve data
            2. DO NOT provide analysis or recommendations
            3. Format responses as:
               Query Executed: [the SQL query used]
               Results: [the query results]
               Summary: [brief factual summary of the findings]
            4. Keep responses focused on the data only
            """
        )

        # Create a template for user-provided input
        human_message = HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}")

        # Combine system and human message templates into a chat prompt
        return ChatPromptTemplate.from_messages([system_message, human_message])

    def analyze_question_with_graph(self, db_graph: nx.Graph, question: str) -> dict:
        # Analyze the user question in the context of the database graph
        print(f"\n🔎 Starting graph analysis for: '{question}'")
        question_lower = question.lower()

        # Structure to store analysis results
        analysis = {
            'tables': [],
            'relationships': [],
            'columns': [],
            'possible_paths': []
        }

        # Scan graph nodes to identify relevant tables and columns
        for node in db_graph.nodes():
            node_data = db_graph.nodes[node]

            if 'tableName' not in node_data:
                continue

            table_name = node_data['tableName'].lower()
            if not (table_name in question_lower or
                    table_name.rstrip('s') in question_lower or
                    f"{table_name}s" in question_lower):
                continue

            print(f"  📦 Found relevant table: {node_data['tableName']}")
            table_info = {'name': node_data['tableName'], 'columns': []}

            # Find matching columns connected to the table
            for neighbor in db_graph.neighbors(node):
                col_data = db_graph.nodes[neighbor]
                if 'columnName' in col_data and col_data['columnName'].lower() in question_lower:
                    table_info['columns'].append({
                        'name': col_data['columnName'],
                        'type': col_data['columnType'],
                        'table': node_data['tableName']
                    })
                    print(f"    📎 Found relevant column: {col_data['columnName']}")

            analysis['tables'].append(table_info)

        return analysis

    def query(self, text: str, db_graph) -> str:
        # Execute a query using graph-based analysis or standard prompt
        try:
            if db_graph:
                print(f"\n🔍 Analyzing query with graph: '{text}'")
                
                # Analyze the question with the database graph
                graph_analysis = self.analyze_question_with_graph(db_graph, text)
                print(f"\n📊 Graph Analysis Results:")
                print(json.dumps(graph_analysis, indent=2))

                # Enhance the prompt with graph analysis context
                enhanced_prompt = f"""
                Database Structure Analysis:
                - Available Tables: {[t['name'] for t in graph_analysis['tables']]}
                - Table Relationships: {graph_analysis['possible_paths']}

                User Question: {text}

                Use this structural information to form an accurate query.
                """
                print(f"\n📝 Enhanced prompt created with graph context")
                return self.agent_executor.invoke({"input": enhanced_prompt, "db_name": self.config.db})['output']

            print(f"\n⚡ No graph available, executing standard query: '{text}'")
            return self.agent_executor.invoke({"input": text, "db_name": self.config.db})['output']

        except Exception as e:
            # Handle errors during query processing
            print(f"\n❌ Error in inference query: {str(e)}")
            return f"Error processing query: {str(e)}"

```

# Planning Agent

This agents role is to help the Supervisor agent to plan and delegate the steps to the other agents.

This agent breaks the users request down into steps then tags actions with the agent it needs to be delegated to, then hands the plan back to the Supervisor to execute and delegate.

```python
class PlannerAgent:
    def __init__(self):
        # Initialize configuration and planner prompt
        self.config = Config()
        self.planner_prompt = self.create_planner_prompt()

    def create_planner_prompt(self):
        # Define the system template for planning instructions
        system_template = """You are a friendly planning agent that creates specific plans to answer questions about THIS database only.

        Available actions:
        1. Inference: [query] - Use this prefix for database queries
        2. General: [response] - Use this prefix for friendly responses

        Create a SINGLE, SEQUENTIAL plan where:
        - Each step should be exactly ONE line
        - Each step must start with either 'Inference:' or 'General:'
        - Steps must be in logical order
        - DO NOT repeat steps
        - Keep the plan minimal and focused

        Example format:
        Inference: Get all artists from the database
        Inference: Count tracks per artist
        General: Provide the results in a friendly way
        """

        # Define the human message template for user input
        human_template = "Question: {question}\n\nCreate a focused plan with appropriate action steps."

        # Combine system and human message templates into a chat prompt
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])

    def create_plan(self, question: str) -> list:
        # Generate a step-by-step plan to answer the given question
        try:
            logger.info(f"Creating plan for question: {question}")
            response = self.config.llm.invoke(self.planner_prompt.format(
                question=question
            ))

            # Extract and clean valid steps from the response
            steps = [step.strip() for step in response.content.split('\n')
                     if step.strip() and not step.lower() == 'plan:']

            # Provide a fallback message if no steps are returned
            if not steps:
                return ["General: I'd love to help you explore the database! What would you like to know?"]

            return steps

        except Exception as e:
            # Log and handle errors during plan creation
            logger.error(f"Error creating plan: {str(e)}", exc_info=True)
            return ["General: Error occurred while creating plan"]

```

# Agent State

The custom State object helps us keep track of:

- the question
- the type of input from the user
- the plan that was generated by the Planner agent
- aggregated results from the inference agent
- the Supervisors response
- the graph database that was generated by the discovery

There are a few custom reducers which are used to appropriately update the state object attributes, this helps simplify this process by using built-in features of the LangGraph framework.

```python
def db_graph_reducer():
    # Reducer function for handling database graph updates
    def _reducer(previous_value: Optional[nx.Graph], new_value: nx.Graph) -> nx.Graph:
        if previous_value is None:  # If no previous graph exists, use the new graph
            return new_value
        return previous_value  # Otherwise, retain the existing graph
    return _reducer

def plan_reducer():
    # Reducer function for updating plans
    def _reducer(previous_value: Optional[List[str]], new_value: List[str]) -> List[str]:
        return new_value if new_value is not None else previous_value  # Use the new plan if available
    return _reducer

def classify_input_reducer():
    # Reducer function for input classification
    def _reducer(previous_value: Optional[str], new_value: str) -> str:
        return new_value  # Always replace with the latest classification
    return _reducer

class ConversationState(TypedDict):
    # Defines the conversation state structure and associated reducers
    question: str  # Current user question
    input_type: Annotated[str, classify_input_reducer()]  # Classification of the input type
    plan: Annotated[List[str], plan_reducer()]  # Step-by-step plan to respond to the question
    db_results: NotRequired[str]  # Optional field for database query results
    response: NotRequired[str]  # Optional field for generated response
    db_graph: Annotated[Optional[nx.Graph], db_graph_reducer()] = None  # Optional field for database graph

```

# Classify user input

This step in the StateGraph helps the Supervisor decide how to appropriately respond.

If the user input is conversational it will simply respond itself without the need for a plan.

If the input is database related, it triggers the planning and delegation processes.

```python
def classify_user_input(state: ConversationState) -> ConversationState:
    """Classifies user input to determine if it requires database access."""
    
    # Define a system prompt for classifying input into predefined categories
    system_prompt = """You are an input classifier. Classify the user's input into one of these categories:
    - DATABASE_QUERY: Questions about data, requiring database access
    - GREETING: General greetings, how are you, etc.
    - CHITCHAT: General conversation not requiring database
    - FAREWELL: Goodbye messages

    Respond with ONLY the category name."""

    # Prepare messages for the LLM, including the system prompt and user's input
    messages = [
        ("system", system_prompt),  # Instructions for the LLM
        ("user", state['question'])  # User's question for classification
    ]

    # Invoke the LLM with a zero-temperature setting for deterministic output
    llm = ChatOpenAI(temperature=0)
    response = llm.invoke(messages)
    classification = response.content.strip()  # Extract the category from the response

    # Log the classification result
    logger.info(f"Input classified as: {classification}")

    # Update the conversation state with the input classification
    return {
        **state,
        "input_type": classification
    }
```

# Supervisor Agent

The role of the Supervisor agent is to over see the coordination of the other agents to respond to the requests from the user.

When a request comes into the Supervisor, it classifies the request and decides if it can simply respond (ie to a greeting or a conversational input) or if it needs to do something different. If the request is database related, the StateGraph triggers a database discovery process which builds a relationshop graph of the tables and fields.

Then the Supervisor then delegates to the Planning agent which creates a plan, breaking the users input into discreet steps so it can accurately delegate to the right agent. That plan is returned to the Supervisor.

The Supervisor then executes the plan.

```python
class SupervisorAgent:
    def __init__(self):
        # Initialize configuration and agents
        self.config = Config()
        self.inference_agent = InferenceAgent()
        self.planner_agent = PlannerAgent()
        self.discovery_agent = DiscoveryAgent()

        # Prompts for different types of responses
        self.db_response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a response coordinator that creates final responses based on:
            Original Question: {question}
            Database Results: {db_results}

            Rules:
            1. ALWAYS include ALL results from database queries in your response
            2. Format the response clearly with each piece of information on its own line
            3. Use bullet points or numbers for multiple pieces of information
            """)
        ])

        self.chat_response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly AI assistant.
            Respond naturally to the user's message.
            Keep responses brief and friendly.
            Don't make up information about weather, traffic, or other external data.
            """)
        ])

    def create_plan(self, state: ConversationState) -> ConversationState:
        # Generate a plan using the PlannerAgent
        plan = self.planner_agent.create_plan(
            question=state['question']
        )

        # Log the plan, separating inference and general steps
        logger.info("Generated plan:")
        inference_steps = [step for step in plan if step.startswith('Inference:')]
        general_steps = [step for step in plan if step.startswith('General:')]

        if inference_steps:
            logger.info("Inference Steps:")
            for i, step in enumerate(inference_steps, 1):
                logger.info(f"  {i}. {step}")
        if general_steps:
            logger.info("General Steps:")
            for i, step in enumerate(general_steps, 1):
                logger.info(f"  {i}. {step}")

        return {
            **state,
            "plan": plan
        }

    def execute_plan(self, state: ConversationState) -> ConversationState:
        # Execute the generated plan step by step
        results = []

        try:
            for step in state['plan']:
                if ':' not in step:
                    continue

                step_type, content = step.split(':', 1)
                content = content.strip()

                if step_type.lower().strip() == 'inference':
                    # Handle inference steps using the InferenceAgent
                    try:
                        result = self.inference_agent.query(content, state.get('db_graph'))
                        results.append(f"Step: {step}\nResult: {result}")
                    except Exception as e:
                        logger.error(f"Error in inference step: {str(e)}", exc_info=True)
                        results.append(f"Step: {step}\nError: Query failed - {str(e)}")
                else:
                    # Handle general steps
                    results.append(f"Step: {step}\nResult: {content}")

            # Return state with results
            return {
                **state,
                "db_results": "\n\n".join(results) if results else "No results were generated."
            }

        except Exception as e:
            logger.error(f"Error in execute_plan: {str(e)}", exc_info=True)
            return {**state, "db_results": f"Error executing steps: {str(e)}"}

    def generate_response(self, state: ConversationState) -> ConversationState:
        # Generate the final response based on the input type
        logger.info("Generating final response")
        is_chat = state.get("input_type") in ["GREETING", "CHITCHAT", "FAREWELL"]
        prompt = self.chat_response_prompt if is_chat else self.db_response_prompt

        # Invoke the LLM to generate the response
        response = self.config.llm.invoke(prompt.format(
            question=state['question'],
            db_results=state.get('db_results', '')
        ))

        # Update state with the response and clear the plan
        return {**state, "response": response.content, "plan": []}

```

# Discover Database

This method is what is used in the StateGraph when triggered to do the database discovery, it returns an NetworkX graph object which we use to update the ConversationState object (see the Examples section below).

```python
def discover_database(state: ConversationState) -> ConversationState:
    # Check if the database graph is already present in the state
    if state.get('db_graph') is None:
        logger.info("Performing one-time database schema discovery...")
        
        # Use the DiscoveryAgent to generate the database graph
        discovery_agent = DiscoveryAgent()
        graph = discovery_agent.discover()
        
        logger.info("Database schema discovery complete - this will be reused for future queries")
        
        # Update the state with the discovered database graph
        return {**state, "db_graph": graph}
    
    # Return the existing state if the database graph already exists
    return state
```

# StateGraph

This defines the Agent StateGraph and how the agent is to "behave" and make decisions as to what its actions are going to be.

```python
def create_graph():
    # Initialize the supervisor agent and state graph builder
    supervisor = SupervisorAgent()
    builder = StateGraph(ConversationState)

    # Add nodes representing processing steps in the flow
    builder.add_node("classify_input", classify_user_input)  # Classify the user input
    builder.add_node("discover_database", discover_database)  # Perform database discovery
    builder.add_node("create_plan", supervisor.create_plan)  # Create a plan based on input
    builder.add_node("execute_plan", supervisor.execute_plan)  # Execute the generated plan
    builder.add_node("generate_response", supervisor.generate_response)  # Generate the final response

    # Define the flow of states
    builder.add_edge(START, "classify_input")  # Start with input classification

    # Conditionally proceed to database discovery or directly to response generation
    builder.add_conditional_edges(
        "classify_input",
        lambda x: "discover_database" if x.get("input_type") == "DATABASE_QUERY" else "generate_response"
    )

    # Connect discovery to plan creation
    builder.add_edge("discover_database", "create_plan")

    # Conditionally execute the plan or generate a response if no plan exists
    builder.add_conditional_edges(
        "create_plan",
        lambda x: "execute_plan" if x.get("plan") is not None else "generate_response"
    )

    # Connect execution to response generation
    builder.add_edge("execute_plan", "generate_response")

    # End the process after generating the response
    builder.add_edge("generate_response", END)

    # Compile and return the state graph
    return builder.compile()

# Create the graph for processing
graph = create_graph()
```

# Example 1

This example demonstrates the StateGraph conversationally replying before ending the request, it doesn't trigger the database discovery or delegate to any other agents.

```python
state = graph.invoke({
    "question": "Hi there, how goes it?"
})
print(f"State after first invoke: {state}")
print(f"Response 1: {state['response']}\n")
```

# Example 2

This example shows the StateGraph node "database_discovery" step is triggered because the db_graph attribute of the ConversationState state object is not set. This then performs the database discovery and creates the relationshop graph and updates the global state object.

It then proceeds to processing the users request.

```python
state = graph.invoke({
    **state,
    "question": "Who are the top 3 artists by number of tracks?"
})
print(f"State after second invoke: {state}")
print(f"Response 2: {state['response']}\n")
```

# Example 3

This example is important because it shows that we are now using the db_graph attribute as context for when the user requests information, meaning the database discovery process is only run once per session (ie between restarts). The discovery process can be very intense and time consuming so we only want to run it once.

```python
state = graph.invoke({
    **state,
    "question": "What genres do they make?"
})
print(f"State after third invoke: {state}")
print(f"Response 3: {state['response']}\n")
```