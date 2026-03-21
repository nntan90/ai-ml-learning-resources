# Notebook: 10-expense_claim-demo

> Source: https://github.com/microsoft/ai-agents-for-beginners/blob/HEAD/10-ai-agents-production/code_samples/10-expense_claim-demo.ipynb

---

# Expense Claim Analysis

This notebook demonstrates how to create agents that use plugins to process travel expenses from local receipt images, generate an expense claim email, and visualize expense data using a pie chart. Agents dynamically choose functions based on the task context.

Steps:
1. OCR Agent processes the local receipt image and extracts travel expense data.
2. Email Agent generates an expense claim email.

### Example of a travel expense scenario:
Imagine you're an employee traveling for a business meeting in another city. Your company has a policy to reimburse all reasonable travel-related expenses. Here's a breakdown of potential travel expenses:
- Transportation:
Airfare for a round trip from your home city to the destination city.
Taxi or ride-hailing services to and from the airport.
Local transportation in the destination city (like public transit, rental cars, or taxis).

- Accommodation:
Hotel stay for three nights at a mid-range business hotel close to the meeting venue.

- Meals:
Daily meal allowance for breakfast, lunch, and dinner, based on the company's per diem policy.

- Miscellaneous Expenses:
Parking fees at the airport.
Internet access charges at the hotel.
Tips or small service charges.

- Documentation:
You submit all receipts (flights, taxis, hotel, meals, etc.) and a completed expense report for reimbursement.

## Import required libraries

Import the necessary libraries and modules for the notebook.

```python
import logging
logging.getLogger("agent_framework.azure").setLevel(logging.ERROR)

import os
import base64
from typing import Annotated, List

from pydantic import BaseModel, Field

from agent_framework import tool, AgentResponseUpdate, WorkflowBuilder
from agent_framework.azure import AzureAIProjectAgentProvider
from azure.identity import AzureCliCredential
```

```python
provider = AzureAIProjectAgentProvider(credential=AzureCliCredential())
```

 ## Define Expense Models

 Create a Pydantic model for individual expenses and an ExpenseFormatter class to convert a user query into structured expense data.

 Each expense will be represented in the format:
 `{'date': '07-Mar-2025', 'description': 'flight to destination', 'amount': 675.99, 'category': 'Transportation'}`


```python
class Expense(BaseModel):
    date: str = Field(..., description="Date of expense in dd-MMM-yyyy format")
    description: str = Field(..., description="Expense description")
    amount: float = Field(..., description="Expense amount")
    category: str = Field(..., description="Expense category (e.g., Transportation, Meals, Accommodation, Miscellaneous)")

class ExpenseFormatter(BaseModel):
    raw_query: str = Field(..., description="Raw query input containing expense details")
    
    def parse_expenses(self) -> List[Expense]:
        """
        Parses the raw query into a list of Expense objects.
        Expected format: "date|description|amount|category" separated by semicolons.
        """
        expense_list = []
        for expense_str in self.raw_query.split(";"):
            if expense_str.strip():
                parts = expense_str.strip().split("|")
                if len(parts) == 4:
                    date, description, amount, category = parts
                    try:
                        expense = Expense(
                            date=date.strip(),
                            description=description.strip(),
                            amount=float(amount.strip()),
                            category=category.strip()
                        )
                        expense_list.append(expense)
                    except ValueError as e:
                        print(f"[LOG] Parse Error: Invalid data in '{expense_str}': {e}")
        return expense_list
```

## Defining Tools - Generating the Email

Create a tool function to generate an email for submitting an expense claim.
- This tool uses the `@tool` decorator from the Microsoft Agent Framework.
- It calculates the total amount of the expenses and formats the details into an email body.

```python
@tool(approval_mode="never_require")
def generate_expense_email(
    expense_data: Annotated[str, "Semicolon-separated expense entries in 'date|description|amount|category' format"]
) -> str:
    """Generate an email to submit an expense claim to the Finance Team."""
    formatter = ExpenseFormatter(raw_query=expense_data)
    expenses = formatter.parse_expenses()
    if not expenses:
        return "No valid expenses found to include in the email."
    total_amount = sum(e.amount for e in expenses)
    email_body = "Dear Finance Team,\n\n"
    email_body += "Please find below the details of my expense claim:\n\n"
    for e in expenses:
        email_body += f"- {e.date} | {e.description}: ${e.amount:.2f} ({e.category})\n"
    email_body += f"\nTotal Amount: ${total_amount:.2f}\n\n"
    email_body += "Receipts for all expenses are attached for your reference.\n\n"
    email_body += "Thank you,\n[Your Name]"
    return email_body
```

# Tool for Extracting Travel Expenses from Receipt Images

Create a tool function to extract travel expenses from receipt images.
- This tool uses the `@tool` decorator from the Microsoft Agent Framework.
- It reads the receipt image, encodes it as base64, and returns the data URI for the agent to analyze.

```python
@tool(approval_mode="never_require")
def load_receipt_image(
    image_path: Annotated[str, "Path to the receipt image file"] = "receipt.jpg"
) -> str:
    """Load a receipt image and return its base64-encoded data URI for OCR extraction."""
    try:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{image_data}"
    except Exception as e:
        error_msg = f"[LOG] Error loading image '{image_path}': {str(e)}"
        print(error_msg)
        return error_msg
```

## Processing Expenses

Define the agents and wire them into a sequential workflow using `WorkflowBuilder`.
- The OCR agent extracts structured expense data from the receipt image using the `load_receipt_image` tool.
- The Email agent takes the extracted data and generates a professional expense claim email using the `generate_expense_email` tool.
- `WorkflowBuilder` with `add_edge` creates a sequential pipeline: OCR Agent → Email Agent.

```python
ocr_agent = await provider.create_agent(
    tools=[load_receipt_image],
    name="OCRAgent",
    instructions=(
        "You are an expert OCR assistant specialized in extracting structured data from receipt images. "
        "Use the 'load_receipt_image' tool to load the receipt image, then analyze it and extract "
        "travel-related expense details in the format: 'date|description|amount|category' separated by semicolons. "
        "Follow these rules: "
        "- Date: Convert dates (e.g., '4/4/22') to 'dd-MMM-yyyy' (e.g., '04-Apr-2022'). "
        "- Description: Extract item names. "
        "- Amount: Use numeric values (e.g., '4.50' from '$4.50'). "
        "- Category: Infer from context (e.g., 'Meals' for food, 'Transportation' for travel, "
        "'Accommodation' for lodging, 'Miscellaneous' otherwise). "
        "Ignore totals, subtotals, or service charges unless they are itemized expenses. "
        "If no expenses are found, return 'No expenses detected'. "
        "Return only the structured data, no additional text."
    ),
)

email_agent = await provider.create_agent(
    name="EmailAgent",
    instructions=(
        "You are an expense claim email generator. Take the travel expense data from the previous agent "
        "(in 'date|description|amount|category' format separated by semicolons) and use the "
        "'generate_expense_email' tool to produce a professional expense claim email. "
        "Pass the semicolon-separated expense data directly to the tool."
    ),
)
```

## Main function

Build the sequential workflow and run it to process the receipt image and generate the expense claim email.

```python
workflow = WorkflowBuilder(start_executor=ocr_agent) \
    .add_edge(ocr_agent, email_agent) \
    .build()

prompt = (
    "Please extract the raw text from the receipt image at 'receipt.jpg', "
    "focusing on travel expenses like dates, descriptions, amounts, and categories "
    "(e.g., Transportation, Accommodation, Meals, Miscellaneous). "
    "Then generate a professional expense claim email."
)

last_author = None
events = workflow.run(
    prompt,
    stream=True,
)
async for event in events:
    if event.type == "output" and isinstance(event.data, AgentResponseUpdate):
        update = event.data
        author = update.author_name
        if author != last_author:
            if last_author is not None:
                print()
            print(f"\n{'='*50}")
            print(f"# Agent - {author}:")
            print(f"{'='*50}")
            last_author = author
        print(update.text, end="", flush=True)
```