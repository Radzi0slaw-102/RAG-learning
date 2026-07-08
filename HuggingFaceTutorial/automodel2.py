import torch
from transformers import BitsAndBytesConfig, RagRetriever, RagSequenceForGeneration, RagTokenizer

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq", dataset="wiki_dpr", index_name="compressed"
)

model = RagSequenceForGeneration.from_pretrained(
    "facebook/rag-sequence-nq",
    retriever=retriever,
    quantization_config=bnb,
    device_map="auto",
)
inputs = tokenizer("How many people live in Paris?", return_tensors="pt").to(model.device)
generated = model.generate(input_ids=inputs["input_ids"])
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])