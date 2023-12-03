import os
from dotenv import load_dotenv
import openai  # Importing the openai module
import PyPDF2
import time

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise ValueError("API key not found in .env file")

# Setting the API key for the openai module
openai.api_key = api_key

# Calling the embeddings API
my_vector = openai.Embedding.create(
    model="text-embedding-ada-002",
    input="The food was amazing and the waiter was intoxicated."
)

# Let's see how many embeddings are in this vector....
# Extracting the embeddings vector
embeddings = my_vector["data"][0]["embedding"]
the_object = my_vector["object"]

print(f"the_object is (should be list): {the_object}")

# Counting the number of embeddings
num_embeddings = len(embeddings)

print(f"Number of embeddings in my_vector is: {num_embeddings}")
print(f"\n input string is: 'The food was amazing and the waiter was intoxicated.'")
print(embeddings)

print(type(my_vector))

print(my_vector)


# Now, let's cut up a pdf and vectorize it!

# Extract the text from a PDF.

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text


# Chunk the text.

def chunk_text(text, max_chunk_size):
    chunks = []
    words = text.split()
    current_chunk = ""
    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_chunk_size:
            current_chunk += word + " "
        else:
            chunks.append(current_chunk)
            current_chunk = word + " "
    chunks.append(current_chunk)  # Add the last chunk
    return chunks


# Vectorize each chunk.
def vectorize_chunks(chunks, model_name="text-embedding-ada-002"):
    embeddings02 = []
    for chunk in chunks:
        response = openai.Embedding.create(
            model=model_name,
            input=chunk
        )
        embeddings02.append(response['data'][0]['embedding'])
    return embeddings02


start_time = time.time()  # Start the timer
# C://aiChatBotDev//samplePDFs//smallUnitLeadership//howTheArmyRuns.pdf
my_pdf_path = "C://aiChatBotDev//samplePDFs//smallUnitLeadership//howTheArmyRuns.pdf"
pdf_text = extract_text_from_pdf(my_pdf_path)

# print(pdf_text)
text_chunks = chunk_text(pdf_text, 1024)  # Adjust chunk size as needed
# print(text_chunks)

embeddings03 = vectorize_chunks(text_chunks)

# Counting the number of embeddings
num_embeddings = len(embeddings03)

end_time = time.time()  # End the timer
duration = end_time - start_time  # Calculate total duration

print(f"\n\nTotal time to create embeddings: {duration} seconds\n\n")


# print(f"Number of embeddings is: {num_embeddings}")
# print(embeddings03)
