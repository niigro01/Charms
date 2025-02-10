import os
from dotenv import load_dotenv
import discord
from discord.ext import commands
from charms_selector import CharmManager, charms, load_creatures_database, find_creature_by_name
from ortools.linear_solver import pywraplp
import copy
import numpy as np
import heapq
from scipy.optimize import linear_sum_assignment
import itertools

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

# Function to solve the assignment problem with constraints
def solve_assignment_with_constraints(C, fixed, forbidden, huge_value=np.inf):
    """
    Solves the linear assignment problem with fixed and forbidden constraints.
    :param C: Cost matrix (numpy array, minimization).
    :param fixed: Dictionary of fixed assignments {row: column}.
    :param forbidden: Dictionary of forbidden assignments {row: set of forbidden columns}.
    :param huge_value: High value to mark unassignable cells.
    :return: Tuple (assignment, total cost) or None if no valid solution.
    """
    C_mod = np.array(C, copy=True)
    m, n = C_mod.shape

    # Apply fixed constraints
    for r, col in fixed.items():
        for j in range(n):
            if j != col:
                C_mod[r, j] = huge_value

    # Apply forbidden constraints
    for r, forb in forbidden.items():
        for j in forb:
            C_mod[r, j] = huge_value

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(C_mod)
    total_cost = C_mod[row_ind, col_ind].sum()

    # Check if any assignment has a cost of "huge_value"
    if any(C_mod[i, j] >= huge_value for i, j in zip(row_ind, col_ind)):
        return None

    assignment = [None] * m
    for i, j in zip(row_ind, col_ind):
        assignment[i] = j

    return assignment, total_cost

# Basic implementation of Murty’s algorithm
def murty(C, k):
    """
    C: Cost matrix (numpy array) for minimization.
       In our case, C = - (weighted_damage) for maximization.
    k: Number of desired solutions.
    Returns a list of (assignment, cost).
    """
    m, n = C.shape
    init = solve_assignment_with_constraints(C, {}, {})
    if init is None:
        return []

    best_assignment, best_cost = init
    Q = []
    counter = itertools.count()  # Counter to generate unique IDs
    heapq.heappush(Q, (best_cost, next(counter), {}, {}, best_assignment))
    solutions = []
    seen_combinations = set()

    while Q and len(solutions) < k:
        cost, _, fixed, forbidden, assignment = heapq.heappop(Q)
        assignment_tuple = tuple(assignment)
        if assignment_tuple not in seen_combinations:
            solutions.append((assignment, cost))
            seen_combinations.add(assignment_tuple)

        # Generate subproblems
        for r in range(m):
            new_fixed = {i: assignment[i] for i in range(r)}
            new_forbidden = copy.deepcopy(forbidden)
            if r not in new_forbidden:
                new_forbidden[r] = set()
            new_forbidden[r].add(assignment[r])

            sol = solve_assignment_with_constraints(C, new_fixed, new_forbidden)
            if sol is not None:
                new_assignment, new_cost = sol
                heapq.heappush(Q, (new_cost, next(counter), new_fixed, new_forbidden, new_assignment))

    return solutions

# Function to split text into parts of up to 1024 characters
def split_text(text, limit=1024):
    parts = []
    while len(text) > limit:
        split_index = text.rfind("\n", 0, limit)
        if split_index == -1:
            split_index = limit
        parts.append(text[:split_index])
        text = text[split_index:].lstrip()
    parts.append(text)
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
                            f"✅ Data saved:\nLevel: {stats_data['level']}\nHP: {stats_data['maxhp']}\nMana: {stats_data['maxmana']}\nCharms: {', '.join(selected_charms)}"
                        )

                view = discord.ui.View()
                view.add_item(CharmDropdown())
                await interaction.response.send_message("Select your charms from the menu:", view=view)
            except ValueError as e:
                await interaction.response.send_message(f"❌ Error: {e}", ephemeral=True)

    await interaction.response.send_modal(StatsModal())

# Command /analyzer
@bot.tree.command(name="analyzer",
                  description="Analyze a Hunt Analyzer to extract information about the killed creatures.")
async def analyzer(interaction: discord.Interaction):
    class AnalyzerModal(discord.ui.Modal, title="Analyzer"):
        text_input = discord.ui.TextInput(
            label="Paste your Hunt Analyzer here",
            style=discord.TextStyle.long,
            placeholder="Example: Session data: From 2025-01-06... Killed Monsters: 1028x choking fear..."
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
                    await interaction.response.send_message("❌ No data found for the killed creatures.",
                                                            ephemeral=True)
                    return

                # User data
                user_id = interaction.user.id
                user_data = charm_manager.get_charms(user_id)
                selected_charms = user_data.get("charms", [])
                player_level = user_data.get("level", 8)
                max_hp = user_data.get("maxhp", 1000)
                max_mana = user_data.get("maxmana", 500)

                if not selected_charms:
                    await interaction.response.send_message(
                        "❌ You have not selected any charms. Use `/mystats` to select them.", ephemeral=True)
                    return

                # Build matrices
                opt_matrix = []
                breakdown_matrix = []

                for creature in creature_data:
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

                        row_opt.append(final_damage)
                        row_break.append({
                            "base_damage": base_damage,
                            "vulnerability_percentage": vulnerability_percentage,
                            "mitigation_percentage": mitigation_percentage_formatted,
                            "final_damage": final_damage,
                            "charm_type": charm,
                            "charm_emoji": charm_emoji,
                            "damage_cap_message": damage_cap_message
                        })

                    opt_matrix.append(row_opt)
                    breakdown_matrix.append(row_break)

                if not is_matrix_feasible(opt_matrix):
                    await interaction.response.send_message(
                        "❌ The cost matrix is not valid. Check your charms and the database.", ephemeral=True)
                    return

                # Convert to numpy array and transform to minimization (multiply by -1)
                M = np.array(opt_matrix)
                cost_matrix = -M

                # Apply Murty’s algorithm to get the top 5 assignments
                murty_solutions = murty(cost_matrix, k=5)
                if not murty_solutions:
                    await interaction.response.send_message("❌ Cannot generate more valid combinations.",
                                                            ephemeral=True)
                    return

                # Build the summary
                solution_texts = []
                m = len(creature_data)
                seen_combinations = set()

                def get_combination_key(assignment, breakdown_matrix):
                    return tuple(
                        (breakdown_matrix[i][j]["charm_type"], breakdown_matrix[i][j]["base_damage"]) for i, j in
                        enumerate(assignment))

                for idx, (assignment, cost) in enumerate(murty_solutions, start=1):
                    combination_key = get_combination_key(assignment, breakdown_matrix)
                    if combination_key in seen_combinations:
                        continue
                    seen_combinations.add(combination_key)

                    details = []
                    total_effective = 0
                    for i in range(m):
                        j = assignment[i]
                        eff = breakdown_matrix[i][j]["final_damage"]
                        base_dmg = breakdown_matrix[i][j]["base_damage"]
                        vuln_percentage = breakdown_matrix[i][j]["vulnerability_percentage"]
                        mit_percentage = breakdown_matrix[i][j]["mitigation_percentage"]
                        charm = breakdown_matrix[i][j]["charm_type"]
                        charm_emoji = breakdown_matrix[i][j]["charm_emoji"]
                        damage_cap_message = breakdown_matrix[i][j]["damage_cap_message"]
                        total_effective += eff
                        details.append(
                            f"**{creature_data[i]['name']}: {charm.capitalize()} {charm_emoji}**\n"
                            f"{creature_data[i]['health']} hitpoints × 5% base charm damage = {base_dmg:.2f} maximum base charm damage{damage_cap_message}\n"
                            f"{base_dmg:.2f} base charm damage × {vuln_percentage} charm efficiency - {mit_percentage}% mitigation = {eff:.2f} damage per proc"
                        )
                    sol_text = (
                            f"**Combination #{idx}**\n"
                            f"Total Damage: {total_effective:.2f}\n" + "\n".join(details)
                    )
                    solution_texts.append(sol_text)

                # Create a list of embeds to send the response
                embeds = []
                embed_description = "Priority: Most frequent creatures"
                embed_title = "Optimal Charms Assignment Analysis"
                current_embed = discord.Embed(title=embed_title, description=embed_description, color=0x00ff00)

                for sol_text in solution_texts:
                    fields = split_text(sol_text, limit=1024)
                    for field in fields:
                        if len(current_embed) + len(field) > 6000 or len(current_embed.fields) == 25:
                            embeds.append(current_embed)
                            current_embed = discord.Embed(title=embed_title, description=embed_description,
                                                          color=0x00ff00)
                        current_embed.add_field(name="\u200b", value=field, inline=False)

                embeds.append(current_embed)

                await interaction.response.send_message(embed=embeds[0])
                for embed in embeds[1:]:
                    await interaction.followup.send(embed=embed)

            except ValueError as e:
                await interaction.response.send_message(f"❌ Error: {e}", ephemeral=True)

    modal = AnalyzerModal()
    await interaction.response.send_modal(modal)


# Run the bot
bot.run(TOKEN)