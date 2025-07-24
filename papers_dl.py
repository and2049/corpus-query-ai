import arxiv
import os

output_dir = "papers"
os.makedirs(output_dir, exist_ok=True)

# Construct a search for recent papers in the cs.AI category
search = arxiv.Search(
  query = "cat:cs.AI",
  max_results = 50,
  sort_by = arxiv.SortCriterion.SubmittedDate
)

# Use the client to fetch the results from the search object
client = arxiv.Client()
results = client.results(search)

# Download the PDF for each result
for result in results:
    try:
        pdf_path = result.download_pdf(dirpath=output_dir)
        print(f"Downloaded '{result.title}' to {pdf_path}")
    except Exception as e:
        print(f"Failed to download {result.title}: {e}")

print("\nDownload complete.")