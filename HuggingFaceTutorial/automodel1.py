from transformers import RagRetriever, RagSequenceForGeneration, RagTokenizer

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq", dataset="wiki_dpr", index_name="compressed"
)

model = RagSequenceForGeneration.from_pretrained(
    "facebook/rag-sequence-nq",
    retriever=retriever,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
inputs = tokenizer("How many people live in Paris?", return_tensors="pt").to(model.device)
generated = model.generate(input_ids=inputs["input_ids"])
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])

# Depricated tutorial & methodology, due to:
#  - Deprecated & removed classes: The RagRetriever, RagSequenceForGeneration, and RagTokenizer modules were completely deprecated and removed from mainstream transformers releases (v4.40.0+). They are no longer maintained by Hugging Face.
#  - Security risks with legacy datasets: The tutorial depends on wiki_dpr, which utilizes arbitrary external Python execution scripts (.py files). Modern datasets libraries permanently block these scripts for security and remote code execution prevention.
#  - Breaking NumPy 2.x compatibility: The older framework versions required to run this code are fundamentally incompatible with NumPy 2.x and newer Python releases (like Python 3.12+), leading to critical low-level binary conflicts (_ARRAY_API errors).
#  - Lack of flexibility: Original RAG models tightly coupled the retriever component with specific, hardcoded Meta models from 2020 (trained on 2018 Wikipedia dumps).
#  - Shift to modular architecture: Modern AI engineering has completely abandoned "all-in-one" RAG classes in favor of a decoupled, modular pipeline approach (e.g., using LangChain, LlamaIndex, or direct FAISS integrations) (sic!) paired with state-of-the-art LLMs.