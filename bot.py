import os
from dotenv import load_dotenv
import discord
from discord.ext import commands
from charms_selector import CharmManager, charms, load_creatures_database, find_creature_by_name
import numpy as np

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

# Bot configuration
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)

# Instance of the charms manager
charm_manager = CharmManager()

# Emojis for the charms
charm_emojis = {
    "wound": ":crossed_swords:",
    "curse": ":skull_crossbones:",
    "enflame": ":fire:",
    "freeze": ":snowflake:",
    "poison": ":deciduous_tree:",
    "zap": ":cloud_lightning:",
    "divine-wrath": ":church:",
    "overpower": ":drop_of_blood:",
    "overflux": ":droplet:"
}

# Finite value for unassignable cells
NEGATIVE_FILL = -1e12

# Function that checks if each row has at least one assignable value
def is_matrix_feasible(matrix):
    """
    Checks if the matrix is feasible for solving the assignment problem.
    :param matrix: Cost matrix.
    :return: True if feasible, False otherwise.
    """
    return all(any(cell > NEGATIVE_FILL for cell in row) for row in matrix)

# Function to split text into parts of up to 1024 characters (Discord's field value limit)
def split_text(text, limit=1024):
    parts = []
    while len(text) > limit:
        split_index = text.rfind("\n", 0, limit)
        if split_index == -1:
            split_index = limit
        parts.append(text[:split_index].strip())
        text = text[split_index:].strip()
    parts.append(text.strip())
    return parts

# Event when the bot is ready
@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f"✅ Bot connected and commands synced as {bot.user}.")

# Command /mystats
@bot.tree.command(name="mystats", description="Enter your stats and select your charms.")
async def mystats(interaction: discord.Interaction):
    class StatsModal(discord.ui.Modal, title="Stats"):
        level = discord.ui.TextInput(
            label="Character Level",
            placeholder="Enter your character level (minimum 8)",
            required=True,
            min_length=1,
            max_length=4
        )
        maxhp = discord.ui.TextInput(
            label="Max Hitpoints",
            placeholder="Enter your maximum Hitpoints",
            required=True,
            min_length=1,
            max_length=6
        )
        maxmana = discord.ui.TextInput(
            label="Max Mana",
            placeholder="Enter your maximum Mana",
            required=True,
            min_length=1,
            max_length=6
        )

        async def on_submit(self, interaction: discord.Interaction):
            try:
                level_value = int(self.level.value)
                if level_value < 8:
                    raise ValueError("Level must be at least 8.")

                stats_data = {
                    "level": level_value,
                    "maxhp": int(self.maxhp.value),
                    "maxmana": int(self.maxmana.value)
                }

                class CharmDropdown(discord.ui.Select):
                    def __init__(self):
                        super().__init__(
                            placeholder="Select your charms...",
                            min_values=1,
                            max_values=len(charms),
                            options=[discord.SelectOption(label=charm, description=f"Type: {details['type']}") for
                                     charm, details in charms.items()]
                        )

                    async def callback(self, interaction: discord.Interaction):
                        selected_charms = self.values
                        user_id = interaction.user.id
                        data = charm_manager.get_charms(user_id) or {}
                        data.update(stats_data)
                        data["charms"] = selected_charms
                        charm_manager.save_charms(user_id, data)
                        await interaction.response.send_message(
                            f"✅ Data saved:\nLevel: {stats_data['level']}\nHP: {stats_data['maxhp']}\nMana: {stats_data['maxmana']}\nCharms: {', '.join(selected_charms)}",
                            ephemeral=True
                        )

                view = discord.ui.View()
                view.add_item(CharmDropdown())
                await interaction.response.send_message("Select your charms from the menu:", view=view, ephemeral=True)
            except ValueError as e:
                await interaction.response.send_message(f"❌ Error: {e}", ephemeral=True)

    await interaction.response.send_modal(StatsModal())

# Command /analyzer
@bot.tree.command(name="analyzer", description="Analyze a Hunt Analyzer to extract information about the killed creatures.")
async def analyzer(interaction: discord.Interaction):
    class AnalyzerModal(discord.ui.Modal, title="Analyzer"):
        text_input = discord.ui.TextInput(
            label="Paste your Hunt Analyzer",
            style=discord.TextStyle.long,
            placeholder="Your Hunt Analyzer here"
        )

        async def on_submit(self, interaction: discord.Interaction):
            hunt_data = self.text_input.value
            try:
                # Extract "Killed Monsters" section
                hunt_data_lower = hunt_data.lower()
                if "killed monsters" not in hunt_data_lower:
                    raise ValueError("The Hunt Analyzer format is incorrect. 'Killed Monsters' not found.")

                start_index = hunt_data_lower.index("killed monsters") + len("killed monsters")
                end_index = hunt_data_lower.find("looted items", start_index)
                killed_monsters = (
                    hunt_data[start_index:end_index].strip() if end_index != -1 else hunt_data[start_index:].strip()
                )

                # Process frequencies
                creature_counts = {}
                for line in killed_monsters.split("\n"):
                    if "x" in line:
                        count, name = line.strip().split("x", 1)
                        creature_counts[name.strip().lower()] = int(count)

                # Sort creatures by frequency (highest to lowest)
                creatures_sorted = sorted(creature_counts.items(), key=lambda x: x[1], reverse=True)
                creatures = [name for name, _ in creatures_sorted]
                creature_data = [find_creature_by_name(name) for name in creatures if find_creature_by_name(name)]

                if not creature_data:
                    await interaction.response.send_message("❌ No data found for the killed creatures.", ephemeral=True)
                    return

                # User data
                user_id = interaction.user.id
                user_data = charm_manager.get_charms(user_id)
                selected_charms = user_data.get("charms", [])
                player_level = user_data.get("level", 808)
                max_hp = user_data.get("maxhp", 9351)
                max_mana = user_data.get("maxmana", 12990)

                if not selected_charms:
                    await interaction.response.send_message(
                        "❌ You have not selected any charms. Use `/mystats` to select them.", ephemeral=True)
                    return

                # Calculate damage type damage factor for each creature
                damage_type_damage_factors = []
                for creature in creature_data:
                    creature_health = float(creature.get("health", "1000"))
                    kills_per_session = creature_counts[creature["name"].lower()]
                    damage_type_damage_factor = creature_health * kills_per_session
                    damage_type_damage_factors.append(damage_type_damage_factor)

                sum_of_damage_type_damage_factors = sum(damage_type_damage_factors)

                # Build matrices
                opt_matrix = []
                breakdown_matrix = []

                for idx, creature in enumerate(creature_data):
                    creature_health = float(creature.get("health", "1000"))
                    mitigation = float(creature.get("mitigation", "1.0"))
                    damage_taken = creature.get("damage_taken_from_elements", {})
                    row_opt = []
                    row_break = []

                    for charm in selected_charms:
                        charm_type = charms[charm]["type"]
                        charm_emoji = charm_emojis.get(charm, "")
                        vulnerability_percentage = damage_taken.get(charm_type.capitalize(), "100%")
                        try:
                            vulnerability = float(vulnerability_percentage.strip('%')) / 100
                        except ValueError:
                            vulnerability = 1.0

                        if charm == "overpower":
                            base_damage = min(0.05 * max_hp, player_level * 2)
                            damage_cap = 0.08 * creature_health
                            if base_damage > damage_cap:
                                base_damage = damage_cap
                                damage_cap_message = f" (cap due to creature's health: {damage_cap:.2f})"
                            else:
                                damage_cap_message = ""
                            final_damage = base_damage * vulnerability  # Ignore mitigation
                            mitigation_percentage_formatted = "0.00"  # Ignore mitigation
                        elif charm == "overflux":
                            base_damage = min(0.025 * max_mana, player_level * 2)
                            damage_cap = 0.08 * creature_health
                            if base_damage > damage_cap:
                                base_damage = damage_cap
                                damage_cap_message = f" (cap due to creature's health: {damage_cap:.2f})"
                            else:
                                damage_cap_message = ""
                            final_damage = base_damage * vulnerability  # Ignore mitigation
                            mitigation_percentage_formatted = "0.00"  # Ignore mitigation
                        else:
                            base_damage = min(0.05 * creature_health, player_level * 2)
                            final_damage = base_damage * vulnerability * (1 - mitigation / 100)
                            damage_cap_message = ""
                            mitigation_percentage_formatted = "{:.2f}".format(mitigation)

                        damage_type_efficiency = (damage_type_damage_factors[idx] * vulnerability) / sum_of_damage_type_damage_factors * 100

                        row_opt.append(final_damage * creature_counts[creature["name"].lower()])
                        row_break.append({
                            "base_damage": base_damage,
                            "vulnerability_percentage": vulnerability_percentage,
                            "mitigation_percentage": mitigation_percentage_formatted,
                            "final_damage": final_damage,
                            "charm_type": charm,
                            "charm_emoji": charm_emoji,
                            "damage_cap_message": damage_cap_message,
                            "damage_type_efficiency": damage_type_efficiency
                        })

                    opt_matrix.append(row_opt)
                    breakdown_matrix.append(row_break)

                if not is_matrix_feasible(opt_matrix):
                    await interaction.response.send_message(
                        "❌ The cost matrix is not valid. Check your charms and the database.", ephemeral=True)
                    return

                # Convert to numpy array
                M = np.array(opt_matrix)

                def branch_and_bound(M, creature_counts, num_combinations=3):
                    num_creatures, num_charms = M.shape
                    best_combinations = []

                    def calculate_total_damage(assignment):
                        return sum(M[i, assignment[i]] for i in range(num_creatures))

                    def branch(assignment, remaining_charms):
                        if len(assignment) == num_creatures:
                            total_damage = calculate_total_damage(assignment)
                            best_combinations.append((assignment, total_damage))
                            best_combinations.sort(key=lambda x: x[1], reverse=True)
                            if len(best_combinations) > num_combinations:
                                best_combinations.pop()
                            return

                        creature_idx = len(assignment)
                        for charm_idx in remaining_charms:
                            new_assignment = assignment + [charm_idx]
                            new_remaining_charms = remaining_charms - {charm_idx}
                            branch(new_assignment, new_remaining_charms)

                    branch([], set(range(num_charms)))
                    return best_combinations

                # Use Branch and Bound to get the best assignments
                best_combinations = branch_and_bound(M, creature_counts)

                # Build the summary for the best assignments
                summaries = []
                for combination_index, (assignment, total_damage) in enumerate(best_combinations, start=1):
                    details = []
                    for i in range(len(creature_data)):
                        j = assignment[i]
                        eff = breakdown_matrix[i][j]["final_damage"]
                        base_dmg = breakdown_matrix[i][j]["base_damage"]
                        vuln_percentage = breakdown_matrix[i][j]["vulnerability_percentage"]
                        mit_percentage = breakdown_matrix[i][j]["mitigation_percentage"]
                        charm = breakdown_matrix[i][j]["charm_type"]
                        charm_emoji = breakdown_matrix[i][j]["charm_emoji"]
                        damage_cap_message = breakdown_matrix[i][j]["damage_cap_message"]
                        damage_type_efficiency = breakdown_matrix[i][j]["damage_type_efficiency"]
                        creature_count = creature_counts[creature_data[i]["name"].lower()]
                        details.append(
                            f"> {creature_data[i]['name']}: {charm.capitalize()} {charm_emoji}\n"
                            f"{creature_data[i]['health']} hitpoints × 5% base charm damage = {base_dmg:.2f} maximum base charm damage{damage_cap_message}\n"
                            f"{base_dmg:.2f} base charm damage × {vuln_percentage} charm efficiency - {mit_percentage}% mitigation = {eff:.2f} damage per proc\n"
                            f"Damage Type Efficiency: {damage_type_efficiency:.2f}%"
                        )
                    summary = (
                        f"> Best Combination #{combination_index}\n"
                        f"Total Damage: {total_damage:.2f}\n"
                        + "\n".join(details)
                    )
                    summaries.append(summary)

                # Create a single string with double newlines separating combinations
                response_text = "\n\n".join(summaries).strip()

                # Create a list of embed fields to send the response
                fields = split_text(response_text, limit=1024)
                embeds = []
                for field in fields:
                    # Create a new embed if necessary
                    if not embeds or len(embeds[-1].fields) >= 25:
                        embed = discord.Embed(title="Optimal Charms Assignment Analysis", color=0x00ff00)
                        embeds.append(embed)
                    embeds[-1].add_field(name="\u200b", value=field, inline=False)

                await interaction.response.send_message(embed=embeds[0], ephemeral=True)
                for embed in embeds[1:]:
                    await interaction.followup.send(embed=embed, ephemeral=True)

            except ValueError as e:
                await interaction.response.send_message(f"❌ Error: {e}", ephemeral=True)

    modal = AnalyzerModal()
    await interaction.response.send_modal(modal)

# Run the bot
bot.run(TOKEN)