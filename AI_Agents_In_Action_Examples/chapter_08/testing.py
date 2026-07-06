from langchain_community.document_loaders import WebBaseLoader

url = "https://www.archives.gov/founding-docs/constitution-transcript"
loader = WebBaseLoader(url)

data = loader.load()

print(f"The document was successfully downloaded: {data[0].metadata['title']}\n")

print("First 500 chars:")
print(data[0].page_content[:500].strip())