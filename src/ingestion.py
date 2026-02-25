import fitz  # PyMuPDF
import re

class PDFCleaner:
    @staticmethod
    def fix_kerning(text):
        def collapse(match): return match.group(0).replace(' ', '')
        return re.sub(r'(?:\b[A-Za-z]\s+){3,}[A-Za-z]\b', collapse, text)

    @staticmethod
    def clean_block(text):
        text = PDFCleaner.fix_kerning(text)
        lines = []
        for line in text.split('\n'):
            line = line.strip()
            if re.search(r'\.{3,}\s*\d+$', line): continue 
            if re.match(r'^\d+$', line): continue
            if line: lines.append(line)
        return " ".join(lines)

class IngestionSystem:
    def __init__(self, chunk_size=500):
        self.chunk_size = chunk_size

    def load_pdf(self, file_path):
        doc = fitz.open(file_path)
        chunks_with_metadata = []
        
        print(f"Loading {file_path}...")
        
        # Process each page individually to keep page numbers accurate
        for page_num, page in enumerate(doc):
            # 1. Extract text from this specific page
            page_text = ""
            blocks = page.get_text("blocks")
            blocks.sort(key=lambda b: (b[1], b[0]))
            
            for b in blocks:
                cleaned = PDFCleaner.clean_block(b[4])
                if cleaned: page_text += cleaned + "\n\n"
            
            # 2. Chunk this page immediately
            # We pass the page_text to a helper function
            page_chunks = self._chunk_text(page_text)
            
            # 3. Tag these chunks with the page number (start at 1, not 0)
            for chunk_text in page_chunks:
                chunks_with_metadata.append({
                    "text": chunk_text,
                    "page": page_num + 1 
                })
                
        return chunks_with_metadata

    def _chunk_text(self, text):
        """Internal helper to split text into chunks"""
        sentences = re.split(r'(?<=[.?!])\s+', text)
        chunks = []
        curr_chunk = []
        curr_len = 0
        
        for s in sentences:
            s = s.strip()
            if not s: continue
            w_count = len(s.split())
            
            if curr_len + w_count > self.chunk_size:
                chunks.append(" ".join(curr_chunk))
                overlap = curr_chunk[-3:] if len(curr_chunk) > 3 else curr_chunk
                curr_chunk = list(overlap) + [s]
                curr_len = sum(len(x.split()) for x in curr_chunk)
            else:
                curr_chunk.append(s)
                curr_len += w_count
                
        if curr_chunk: chunks.append(" ".join(curr_chunk))
        return chunks