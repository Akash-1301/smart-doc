import os
import asyncio
import aiohttp
import wikipedia
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

PDF_FOLDER = "pdfs"
SCRAPE_QUERY = "theory of computation"
MAX_ARTICLES = 3

def load_and_split_pdfs(folder_path):
    all_docs = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder_path, filename))
            all_docs.extend(loader.load_and_split())
    return all_docs

def scrape_wikipedia(query):
    try:
        page = wikipedia.page(query)
        return [Document(page_content=page.content, metadata={"source": "Wikipedia", "title": page.title})]
    except Exception as e:
        print(f"[Wikipedia]  {e}")
        return []

async def scrape_medium(query):
    base_url = f"https://api.rss2json.com/v1/api.json?rss_url=https://medium.com/feed/{query}"
    articles = []
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url) as resp:
                data = await resp.json()
                for entry in data.get("items", [])[:MAX_ARTICLES]:
                    articles.append(Document(page_content=entry["content"], metadata={"source": entry["link"]}))
    except Exception as e:
        print(f"[Medium]  {e}")
    return articles

async def scrape_geeksforgeeks(query):
    headers = {"User-Agent": "Mozilla/5.0"}
    search_url = f"https://www.geeksforgeeks.org/?s={query.replace(' ', '+')}"
    articles = []

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(search_url, headers=headers) as resp:
                soup = BeautifulSoup(await resp.text(), "html.parser")
                links = list({a['href'] for a in soup.select("a") if a.get('href', '').startswith("https://www.geeksforgeeks.org")})[:MAX_ARTICLES]
            
            for url in links:
                async with session.get(url, headers=headers) as page:
                    soup = BeautifulSoup(await page.text(), "html.parser")
                    text = " ".join(p.get_text() for p in soup.find_all("p"))
                    if len(text.strip()) > 300:
                        articles.append(Document(page_content=text, metadata={"source": url}))
        except Exception as e:
            print(f"[GFG]  {e}")
    return articles

async def main():
    print("Loading PDFs...")
    pdf_docs = load_and_split_pdfs(PDF_FOLDER)

    print("Scraping web...")
    gfg_docs, medium_docs = await asyncio.gather(
        scrape_geeksforgeeks(SCRAPE_QUERY),
        scrape_medium(SCRAPE_QUERY)
    )
    wiki_docs = scrape_wikipedia(SCRAPE_QUERY)

    all_docs = pdf_docs + gfg_docs + medium_docs + wiki_docs
    print(f" Total raw documents: {len(all_docs)}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(all_docs)
    print(f" Total chunks: {len(split_docs)}")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(split_docs, embeddings)
    vectordb.save_local("vector_store")
    print("Vector store saved to './vector_store/'")

if __name__ == "__main__":
    asyncio.run(main())
