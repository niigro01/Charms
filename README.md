```markdown name=README.md
# Discord Bot for Charms Selection and Analysis

This Discord bot allows users to manage their character stats, select charms, and analyze hunt data to optimize charm assignments for various creatures. The bot uses several libraries for optimization and matrix calculations.

## Features

1. **Character Stats Management**: Users can input their character stats, including level, max hitpoints, and max mana.
2. **Charm Selection**: Users can select from a variety of charms to optimize their performance.
3. **Hunt Analyzer**: Users can paste their hunt analyzer data to extract information about killed creatures and get optimal charm assignments.

## Installation

To run this bot, you need to have Python installed. Follow the steps below to set up and run the bot.

### Requirements

The required Python packages are listed in `requirements.txt`. You can install them using `pip`.

```plaintext name=requirements.txt
python-dotenv
discord.py
ortools
numpy
scipy
```

### Installation Steps

1. Clone the repository or download the code.
2. Navigate to the project directory.
3. Create a virtual environment (optional but recommended).
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. Install the required packages.
   ```bash
   pip install -r requirements.txt
   ```
5. Create a `.env` file in the project directory and add your Discord bot token.
   ```plaintext
   DISCORD_TOKEN=your_discord_bot_token
   ```

## Usage

Run the bot using the following command:

```bash
python bot.py
```

## Commands

### /mystats

Allows the user to input their character stats and select charms.

```plaintext
/mystats
```

### /analyzer

Analyzes hunt data to provide optimal charm assignments for the killed creatures.

```plaintext
/analyzer
```

## Code Overview

### Main Bot File (`bot.py`)

The main bot file contains the logic for handling Discord interactions, managing user data, and performing optimization calculations.

Key sections of the code:

- **Environment Setup**: Loading environment variables using `dotenv`.
- **Bot Configuration**: Setting up the bot with the required intents and commands.
- **Charm Manager**: Instance to manage user charms.
- **Matrix Functions**: Functions to check matrix feasibility and solve assignment problems with constraints.
- **Murty's Algorithm**: Implementation to generate multiple assignment solutions.
- **Text Splitting**: Function to split long texts into smaller parts for Discord messages.
- **Event Handlers**: Handling the bot's readiness and user commands.

### Environment Variables

The bot uses a `.env` file to store sensitive information like the Discord token. Ensure you create this file and add your token before running the bot.

```plaintext name=.env
DISCORD_TOKEN=your_discord_bot_token
```

## Contributing

Feel free to fork this repository and submit pull requests. For major changes, please open an issue to discuss what you would like to change.

## License

This project is licensed under the MIT License.
```
