# EU AI Bot
An *on-prem* version of a helpful bot which can help answer questions about the EU AI Act. This version of the bot is meant to run locally on an M1 Mac.

## Getting started.

1. Install [Ollama](https://ollama.com/)
2. From a terminal window, run the command:
    `ollama run llama3.2:1b`\
    This will start up llama3.2 model (1b parameter variant) locally on your mac.
3. Ensure python 3.9 is installed on your mac (the version I tested with). If not, it is recommended you use miniconda or pyenv to set it up.
4. Create a venv for your experiments (highly recommended).
5. Install dependencies:
    `pip install -r requirements.txt`
6. Run the application as: `uvicorn app.server:app --host 0.0.0.0 --port 8080`