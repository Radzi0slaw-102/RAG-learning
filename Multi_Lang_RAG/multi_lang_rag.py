import ollama
from sklearn.metrics.pairwise import cosine_similarity

dataset = []
with open('cat-facts.txt', 'r', encoding='utf-8') as f:
    dataset = f.readlines()
    print(f'Loaded {len(dataset)} entries')

EMB_MODEL = "bge-m3"
LANG_MODEL = "llama3"

# custom vector (tuple) database
VECTOR_DB = []

def add_chunk_to_database(chunk):
    embedding = ollama.embed(model=EMB_MODEL, input=chunk)['embeddings'][0]
    VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
    add_chunk_to_database(chunk)
    print(f'Added chunk {i+1}/{len(dataset)} to the database')


def retrieve(query, top_n=3, min_similarity=0.4):
    query_embedding = ollama.embed(model=EMB_MODEL, input=query)['embeddings'][0]
    
    similarities = []
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity([query_embedding], [embedding])[0][0]
        similarities.append((chunk, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [(c, s) for c, s in similarities[:top_n] if s >= min_similarity]

input_query = input('Ask me a question: ')
retrieved_knowledge = retrieve(input_query)

print('Retrieved knowledge:')
for chunk, similarity in retrieved_knowledge:
    print(f' - {chunk} (sim: {similarity:.2f})')

# context_block construction prevents earlier python versions error:
# "SyntaxError: f-string expression part cannot include a backslash"
context_block = '\n'.join([f' - {chunk}' for chunk, _similarity in retrieved_knowledge])

# custom instruction prompt, to remind LLM of it's prior task
instruction_prompt = f'''You are a helpful chatbot.
Use only the following pieces of context to answer the question. Don't make up any new information:
{context_block}

If there is no given context, say that there was no suitable information and you just don't know.
Respond in the same language as the user's question, even if the context above is in a different language.'''

# ensure LLM will respond in the same language as given question
user_prompt = f'''{input_query}

(Remember: answer in the same language as this question.)'''

# message construction
stream = ollama.chat(
    model=LANG_MODEL,
    messages=[
        {'role': 'system', 'content': instruction_prompt},
        {'role': 'user', 'content': user_prompt}
    ],
    stream=True,
)

print("Response:")
for chunk in stream:
    print(chunk.message.content, end='', flush=True)