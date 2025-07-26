import fitz
import os

pdf_path = "papers/2507.17309v1.Confounded_Causal_Imitation_Learning_with_Instrumental_Variables.pdf"
output_image_dir = "extracted_images"
os.makedirs(output_image_dir, exist_ok=True)

doc = fitz.open(pdf_path)

print(f"--- Processing {pdf_path} ---")
print(f"Number of pages: {len(doc)}")

page_one = doc.load_page(0)
text = page_one.get_text()
print("\n--- Text from Page 1 (first 300 chars) ---")
print(text[:300] + "...")

for i in range(len(doc)):
    page = doc.load_page(i)

    zoom = 2
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    image_filename = f"{output_image_dir}/page_{i + 1}.png"
    pix.save(image_filename)

print(f"\n--- Image Rendering ---")
print(f"Successfully rendered {len(doc)} pages as images in the '{output_image_dir}' folder.")

doc.close()