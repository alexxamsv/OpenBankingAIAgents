import csv
from pydantic import BaseModel, Field
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

CHUNK_SIZE = 16
FILE_PATH = "C:\\Users\\lenovo\\OneDrive\\Работен плот\\openbanking\\Open Bank Transaction Data.csv"
API_KEY = ""
MODEL_ID = "gemini-2.0-flash"

class Transaction(BaseModel):
    description: str = Field(description="The description of the transaction")
    debitAmount: float = Field(description="The debit amount of the transaction")
    creditAmount: float = Field(description="The credit amount of the transaction")
    balance: float = Field(description="The balance after the transaction")
    category: str = Field(description="The predicted category for the transaction")

class TransactionGroup(BaseModel):
    transactions: List[Transaction] = Field(description="An array of transactions")

model = ChatGoogleGenerativeAI(
    model=MODEL_ID,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=1,
    google_api_key=API_KEY)

def stream_csv_batch(file_path):
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        next(reader, None) # skip header
        batch = []
        for row in reader:
            batch.append(row)
            if len(batch) == CHUNK_SIZE:
                yield batch
                batch = []

        if batch:
            yield batch

def main():
    transaction_parser = JsonOutputParser(pydantic_object=TransactionGroup)

    classification_prompt = PromptTemplate(
        template="Classify the transactions based on the transaction's description, debit and credit amount.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": transaction_parser.get_format_instructions()})

    classification_chain = classification_prompt | model | transaction_parser

    test = True
    for batch in stream_csv_batch(FILE_PATH):
        if test:
            formatted_string = ""
            for transaction in batch:
                formatted_string += "Transaction Description: %s Debit Amount: %s Credit Amount: %s Balance: %s \n" % (transaction["Transaction Description"], transaction["Debit Amount"], transaction["Credit Amount"], transaction["Balance"])

            test = False
            print("Input:")
            print(formatted_string)

            result = classification_chain.invoke({"query": formatted_string})

            print("Result:")
            print(result)
        else:
            break

if __name__ == "__main__":
    main()
