# RAG System Analysis and Optimization

This project implements a Retrieval-Augmented Generation (RAG) pipeline and analyzes its performance from a systems perspective.
It was developed as the final project for Cornell CS 4414 (Fall 2025), built entirely from scratch without any provided starter code.

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/cristinarubylee/rag-project.git
cd rag-project
```

2. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Download datasets:**
```bash
wget -c https://cornell.box.com/shared/static/987z1jm86dzqhdddkxzxpkhsjngcv4cc.json -O data/documents.json
wget -c https://cornell.box.com/shared/static/lcjm8ipmds5p9cpwnil1krrjzgwac2td.json -O data/queries.json
```

## Usage