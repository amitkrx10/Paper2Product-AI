# 🚀 Paper2Product AI
### Transforming Research Papers into Working Software Prototypes

Paper2Product AI is an AI-powered research-to-code engineering platform that transforms research papers into functional software prototypes using the **Google Gemini API**.

It simplifies the journey from understanding a research paper to generating implementation plans, writing code, executing tests, identifying errors, and producing reproducible project reports.

## 💡 Problem Statement

Research papers contain valuable innovations, algorithms, and technical solutions. However, converting these ideas into working software prototypes requires significant time, coding expertise, testing, and debugging.

Paper2Product AI aims to bridge this gap by automating the research-to-development workflow through AI-powered analysis, code generation, testing, and automated repair.

## 🎯 Key Features

- **Research Paper Analysis:** Upload PDF research papers and extract their technical concepts.
- **AI-Powered Analysis:** Uses Google Gemini API to understand research content and generate structured specifications.
- **Requirement Extraction:** Converts research concepts into implementation requirements.
- **Implementation Planning:** Creates structured development plans with user approval.
- **AI Code Generation:** Generates software prototypes using templates or AI.
- **Automated Testing:** Executes generated project tests.
- **Automatic Error Detection:** Identifies errors and attempts bounded automated repairs.
- **Performance Evaluation:** Independently calculates metrics and compares results with reported research findings.
- **Project Reports:** Generates reports, reproducibility manifests, and downloadable ZIP files.
- **Project Management:** Provides project history, logs, retry, cancellation, and deletion features.

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| AI Model | Google Gemini API |
| Backend | Python, FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Database | SQLite |
| Code Execution | Docker |
| PDF Processing | PyMuPDF, pypdf |
| Data Science | NumPy, Pandas, Scikit-learn |
| Testing | Python unittest |
| Version Control | Git and GitHub |

## 🏗️ System Architecture

Research Paper (PDF)  
↓  
PDF Text Extraction  
↓  
Gemini AI Research Analysis  
↓  
Specification Generation and Approval  
↓  
Implementation Planning  
↓  
Dataset Resolution  
↓  
AI-Based Code Generation  
↓  
Security and Static Code Checks  
↓  
Docker-Based Code Execution  
↓  
Automated Testing and Error Repair  
↓  
Performance Evaluation  
↓  
Final Report and Downloadable Prototype

## 📂 Project Structure

```text
Paper2Product-AI/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── llm/
│   │   ├── integrations/
│   │   ├── main.py
│   │   ├── services.py
│   │   ├── planner.py
│   │   ├── codegen.py
│   │   ├── sandbox.py
│   │   ├── repair.py
│   │   └── evaluation.py
│   ├── tests/
│   └── requirements.txt
│
├── frontend/
│   └── index.html
│
├── sandbox/
│   └── Dockerfile
│
├── scripts/
│   ├── setup.py
│   ├── verify_setup.py
│   ├── run_tests.py
│   ├── doctor.py
│   └── run_demo.py
│
├── sample_papers/
├── docs/
├── experiments/
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── LICENSE
└── README.md
```

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.11 or newer
- Docker Desktop
- Google Gemini API key
- Git
- Visual Studio Code (recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Paper2Product-AI.git
cd Paper2Product-AI
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2: Install Dependencies

```bash
python scripts/setup.py
```

This setup script creates a virtual environment, installs dependencies, and helps configure the Gemini API key.

### Step 3: Configure Gemini API

Create a `.env` file using `.env.example` and add your API key:

```env
GEMINI_API_KEY=your_gemini_api_key
```

Get your API key from:

https://aistudio.google.com/apikey

Never upload your `.env` file or expose your API key publicly.

### Step 4: Start Docker Desktop

Open Docker Desktop and wait until the Docker Engine is running.

Build the sandbox image:

```bash
docker build -t paper2product-sandbox:1 sandbox
```

### Step 5: Run the Backend

Windows:

```powershell
venv\Scripts\activate
cd backend
python -m uvicorn app.main:app --reload --reload-dir app
```

### Step 6: Open the Application

Open your browser:

http://127.0.0.1:8000

## 🧪 Testing

Run the complete test suite:

```bash
python scripts/run_tests.py
```

Run workflow-specific tests:

```bash
python scripts/run_tests.py -k workflow
```

Verify the installation:

```bash
python scripts/verify_setup.py
```

Run the demonstration:

```bash
python scripts/run_demo.py
```

**Note:** Full end-to-end execution requires Docker Engine to be running. Tests requiring live Gemini API access also require a valid API key.

## 🔐 Security

Paper2Product AI uses Docker-based isolation for generated-code execution.

Security measures include:

- Network-disabled containers.
- Read-only container filesystem.
- Restricted Linux capabilities.
- CPU and memory limits.
- Execution timeouts.
- Non-root container execution.
- No API keys or environment secrets passed into generated-code containers.
- Static code policy checks.
- Restricted execution commands.

Generated code is not silently executed directly on the host machine when Docker is unavailable.

## ⚠️ Limitations

- Arbitrary research-paper implementation depends on Gemini API availability and model output quality.
- Without a Gemini API key, the application supports a limited DEMO MODE.
- Docker is required for secure generated-code execution.
- Generated prototypes may require manual review and improvement.
- Reproduction results depend on dataset availability, implementation assumptions, and experimental conditions.

## 🔮 Future Enhancements

- Support for additional AI models.
- More research-paper formats.
- Advanced dataset discovery.
- Improved automated debugging.
- Multi-language code generation.
- Cloud-based prototype deployment.
- Collaborative project development.

## 👨‍💻 Project Information

**Project Name:** Paper2Product AI  
**Category:** AI and Developer Tools  
**Core AI Integration:** Google Gemini API  
**Backend:** Python FastAPI  
**Execution Environment:** Docker  

## 📜 License

This project is licensed under the BSD 3-Clause License. See the `LICENSE` file for details.

---

**Paper2Product AI — From Research to Reality with AI.**
