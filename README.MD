How the classifier processes evidence

Image-based input to the LLM – Only preview images are sent for classification. All documents are rasterized into images, with limits on both pages converted and images sent (max_pages default = 2, max_images default = 4).

Document-to-image conversion – PDFs are rendered with pdf2image; DOCX files are converted to PDF then rendered, or, if that fails, their text is rendered as an image. Emails, plain text, and similar sources are also converted to images, ensuring the model always receives a visual representation.

Embedding taxonomy data – Before classification, the app extracts all quality statements and evidence categories from the taxonomy using build_qs_brief and build_evidence_category_brief. Each quality statement includes its ID, domain, title, “we” statement, “what it means” text, I-statements, subtopics, and source URL. This structured list is embedded in the LLM prompt.

Selection process – The model compares the preview images against the taxonomy entries and returns the IDs of the best-matching quality statements, along with:

Confidence scores

Matching I-statements

Matching subtopics

Matching “we” statements

Matching “what it means” text

These matches act as justifications for the selection rather than the sole determinant.

The app enumerates every quality statement from the taxonomy with its ID, domain, title, “we” statement, “what it means” text, I‑statements, subtopics, and source URL via build_qs_brief, then embeds this list in the prompt schema sent to the LLM.

When preview images of the evidence are supplied, the model selects the statements whose verbatim descriptions best match the visible content and returns their IDs, confidence scores, and matching text, yielding the suggested quality statements for classification.