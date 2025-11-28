
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.schema import Document
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFLoader:
    """Handles loading PDF documents from knowledge base"""
    
    def __init__(self, pdf_directory: str):
        self.pdf_directory = Path(pdf_directory)
        
    def load_all_pdfs(self) -> List[Document]:
        """Load all PDF files from the directory"""
        all_documents = []
        
        if not self.pdf_directory.exists():
            logger.error(f"Directory not found: {self.pdf_directory}")
            return all_documents
        
        # Find all PDF files
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            
            try:
                loader = PyPDFLoader(str(pdf_file))
                documents = loader.load()
                
                # Add enhanced metadata
                for idx, doc in enumerate(documents):
                    doc.metadata['source_file'] = pdf_file.name
                    doc.metadata['file_type'] = 'pdf'
                    doc.metadata['page'] = idx + 1
                    doc.metadata['full_path'] = str(pdf_file)
                
                all_documents.extend(documents)
                logger.info(f"  ✓ Loaded {len(documents)} pages from {pdf_file.name}")
                
            except Exception as e:
                logger.error(f"  ✗ Error loading {pdf_file.name}: {e}")
        
        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def load_directory_lazy(self):
        """Alternative: Load using DirectoryLoader (lazy loading)"""
        dir_loader = DirectoryLoader(
            str(self.pdf_directory),
            glob="*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        return dir_loader.load()