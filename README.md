# Language Learning Bot

A sophisticated AI-powered language learning assistant built with Google's Gemini model and LangChain framework.

## Overview

The Language Learning Bot is an interactive CLI application designed to help users practice foreign languages through natural conversations. It leverages Google's Gemini AI to provide realistic dialogue practice, error correction, and progress tracking.

## Features

- **Personalized Learning**: Configure sessions based on target language, native language, proficiency level, and conversation scenario.
- **Real-time Error Correction**: Identifies and explains language mistakes as you practice.
- **Progress Tracking**: Analyzes your strengths and areas for improvement.
- **Persistent Storage**: Saves all conversations and corrections for future reference.
- **Natural Conversations**: Engages in realistic dialogues based on practical scenarios.

## Requirements

- Python 3.8+
- Google API key with access to Gemini models

## Installation

1. Clone this repository or download the script.
2. Install required dependencies:
   ```sh
   pip install google-generativeai langchain langchain-google-genai pydantic
   ```
3. Set up your Google API key:
   - Create an API key in the [Google AI Studio](https://makersuite.google.com/).
   - Either set it as an environment variable `GOOGLE_API_KEY` or provide it when prompted.

## Usage

Run the script from the command line:

```sh
python language_learning_bot.py
```

Follow the interactive prompts to:
1. Enter your Google API key (if not set as an environment variable).
2. Select the language you want to learn.
3. Select your native language.
4. Choose your proficiency level.
5. Select a conversation scenario.

### Available Commands

During a conversation, you can use these special commands:

- `mistakes` - View your language mistakes.
- `progress` - View your learning progress analysis.
- `help` - Show available commands.
- `exit`, `quit`, or `bye` - End the conversation.

## How It Works

The bot uses a multi-component architecture:

1. **LangChain Framework**: Structures interactions with Google's Gemini AI.
2. **Pydantic Models**: Define structured outputs for error analysis and progress tracking.
3. **SQLite Database**: Stores conversation history, user messages, and language mistakes.
4. **Conversation Memory**: Maintains context throughout the learning session.

## Database Schema

The application uses SQLite with three main tables:

- **chats**: Stores session information (languages, proficiency level, scenario).
- **messages**: Records all conversation messages.
- **mistakes**: Tracks identified language mistakes and corrections.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Feel free to fork the repository, open an issue, or submit a pull request.

## Contact

For questions or feedback, please reach out via [GitHub Issues](https://github.com/marupakasrija).
