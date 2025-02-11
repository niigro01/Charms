```markdown name=README.md
# Discord Charms Bot

This Discord bot helps players manage their character stats and charms in a game, and analyze their Hunt Analyzer data
to optimize charm assignments for maximum damage. The bot is built using `discord.py` and has the following features:

## Features

1. **/mystats Command**:
   - Allows users to enter their character stats such as level, max hitpoints, and max mana.
   - Users can select their charms from a dropdown menu.
   - Saves the user's stats and selected charms.

2. **/analyzer Command**:
   - Analyzes the user's Hunt Analyzer data to extract information about the killed creatures.
   - Calculates the optimal charm assignments to maximize damage.
   - Provides detailed analysis and sends the results in multiple embedded messages.

## Setup and Configuration

### Prerequisites

- Python 3.8 or higher
- `discord.py` library
- `python-dotenv` library
- `numpy` library

### Installation

1. Clone this repository:


2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory of the project and add your Discord bot token:
   ```env
   DISCORD_TOKEN=your_discord_bot_token
   ```

5. Run the bot:
   ```bash
   python bot.py
   ```

## Bot Commands

### /mystats

- **Description**: Enter your stats and select your charms.
- **Usage**: `/mystats`
- **Flow**:
  1. The user is prompted with a modal to enter their character level, max hitpoints, and max mana.
  2. After submitting the stats, the user is presented with a dropdown menu to select their charms.
  3. The bot saves the user's stats and selected charms.

### /analyzer

- **Description**: Analyze a Hunt Analyzer to extract information about the killed creatures and provide optimal charm assignments.
- **Usage**: `/analyzer`
- **Flow**:
  1. The user is prompted with a modal to paste their Hunt Analyzer data.
  2. The bot extracts the "Killed Monsters" section from the provided data.
  3. The bot processes the frequencies of the killed creatures and sorts them by frequency.
  4. The bot calculates the optimal charm assignments for maximum damage.
  5. The bot sends the results in embedded messages, each containing details of one optimal combination.

### Detailed Charm Analysis Process

1. **Data Extraction**:
   - The bot extracts the "Killed Monsters" section from the Hunt Analyzer data provided by the user.
   - It processes the frequencies of the killed creatures and sorts them by frequency.

2. **User Data Retrieval**:
   - The bot retrieves the user's saved stats and selected charms.
   - If no charms are selected, the bot prompts the user to use `/mystats` to select their charms.

3. **Damage Calculation**:
   - For each creature, the bot calculates the damage type damage factor, which is the product of the creature's health and the number of times it was killed.
   - The bot builds matrices representing the potential damage for each charm against each creature.

4. **Optimization**:
   - The bot uses the Branch and Bound algorithm to find the optimal charm assignments that maximize total damage.
   - The algorithm iteratively assigns charms to creatures and calculates the total damage for each assignment.

5. **Result Generation**:
   - For each optimal combination, the bot generates a detailed summary including:
     - The charm assigned to each creature.
     - The base damage, vulnerability percentage, and mitigation percentage.
     - The final damage and damage type efficiency.
   - The bot sends each combination in a separate embedded message to the user.

## Example

### /mystats Command

1. User runs `/mystats`.
2. User enters their character level, max hitpoints, and max mana in the modal.
3. User selects their charms from the dropdown menu.
4. Bot saves the stats and charms and confirms with a message.

### /analyzer Command

1. User runs `/analyzer`.
2. User pastes their Hunt Analyzer data in the modal.
3. Bot analyzes the data and calculates optimal charm assignments.
4. Bot sends the results in embedded messages, each containing details of one optimal combination.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

```plaintext name=requirements.txt
discord.py
python-dotenv
numpy
```
