from selenium import webdriver
from selenium.webdriver.common.by import By
from collections import deque
from urllib.parse import urlparse
from time import time
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import dotenv
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

dotenv.load_dotenv()


def convert_html_to_text(docs):
    html2text = Html2TextTransformer()
    return html2text.transform_documents(docs)


def chunk_split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    return text_splitter.split_documents(docs)


def is_valid_url_to_crawl(url):
    """Check whether the url fits our filtering scheme."""
    parsed = urlparse(url)
    return (
        parsed.scheme in ["http", "https"]
        and parsed.netloc in ["support.madkudu.com", "www.madkudu.com"]
        and not parsed.query
        and not parsed.fragment
    )


def is_valid_url_to_store_content(url):
    """Check whether we want to store the contents of the page based off the url."""
    parsed = urlparse(url)
    if parsed.netloc == "support.madkudu.com":
        return "articles" in parsed.path
    else:
        return True


def store_html_recursive(
    start_url,
    max_depth,
    valid_url_to_crawl=is_valid_url_to_crawl,
    valid_url_to_store=is_valid_url_to_store_content,
):
    """Crawl webpages recursively until a certain depth."""
    start_time = time()
    driver = webdriver.Chrome()
    visited = set()
    queue = deque([(start_url, 0)])
    html_contents = []

    while queue:
        url, depth = queue.popleft()
        if depth > max_depth:
            break

        if url not in visited:
            try:
                driver.get(url)
                # Check for redirected urls
                if driver.current_url not in visited:
                    visited.add(driver.current_url)
                    if valid_url_to_store(url):
                        html_contents.append(
                            Document(
                                page_content=driver.page_source,
                                metadata={"source": url, "title": driver.title},
                            )
                        )
                visited.add(url)
                print(
                    f"Elapsed: {time() - start_time:.2f}s, Visited: {url}, Depth: {depth}"
                )

                if depth < max_depth:
                    # Add all links of interest to the queue
                    links = driver.find_elements(by=By.XPATH, value="//a[@href]")
                    for link in links:
                        href = link.get_attribute("href")
                        if valid_url_to_crawl(href):
                            queue.append((href, depth + 1))

            except Exception as e:
                print(f"Error accessing {url}: {e}")

    driver.quit()
    return html_contents


start_url = "https://support.madkudu.com/hc/en-us"
max_depth = 2
db_directory = "./document_db"

print("Begin webscraping for content...")
html_contents = store_html_recursive(start_url, max_depth)
print(len(html_contents))

print("Converting pages html source code to text...")
docs = convert_html_to_text(html_contents)

print("Splitting documents into chunks...")
splits = chunk_split_documents(docs)

print("Storing documents into file database...")
faiss_db = FAISS.from_documents(docs, embedding=OpenAIEmbeddings())
faiss_db.save_local(db_directory)

print("All done!")
