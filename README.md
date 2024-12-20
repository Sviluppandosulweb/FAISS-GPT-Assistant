# FAISS-GPT-Assistant

**FAISS-GPT Assistant** is an interactive graphical application built with Python and Tkinter that integrates OpenAI GPT with FAISS (Facebook AI Similarity Search). This tool enables advanced search and conversational capabilities by indexing and monitoring your documents.

## Key Features

- **File and Directory Indexing**: Index files and directories with FAISS.
- **Automatic Monitoring**: Track file changes using Watchdog.
- **Advanced Conversational Queries**: Use OpenAI GPT to answer questions based on indexed documents.
- **Persistent Conversation History**: Responses are invalidated for outdated data when files change.
- **User-Friendly Interface**: Configure and utilize AI models with an intuitive GUI.

---

## API Configuration Guide

To use **FAISS-GPT Assistant**, you must configure your OpenAI API key. Follow these steps based on your operating system.

### On Linux/Mac

1. Open your `~/.bashrc` or `~/.zshrc` file with a text editor:
   ```bash
   nano ~/.bashrc

or:

nano ~/.zshrc

    Add the following line at the end of the file:

export OPENAI_API_KEY="your_api_key_here"

Save the file and close the editor.

Apply the changes to your terminal:

source ~/.bashrc

or:

    source ~/.zshrc

On Windows

    Go to Control Panel > System > Advanced System Settings.
    Click Environment Variables.
    Add a new system variable:
        Name: OPENAI_API_KEY
        Value: your_api_key_here
    Save and close.

Optional: Set the Key Dynamically

You can also set the API key directly in Python for quick testing:

import os
os.environ['OPENAI_API_KEY'] = 'your_api_key_here'

Verifying the Configuration

After configuring the API key, run the program. If the API key is not set correctly, you will see an error message. You can use this snippet to explicitly check the setup:

import os

if 'OPENAI_API_KEY' not in os.environ:
    raise ValueError("The OpenAI API key is not set. Please configure the OPENAI_API_KEY environment variable.")
else:
    print("API key configured successfully.")
