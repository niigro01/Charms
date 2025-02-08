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

# Cargar variables de entorno
load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")

# Configuración del bot
intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
bot = commands.Bot(command_prefix="/", intents=intents)

# Instancia del gestor de charms
charm_manager = CharmManager()

# Valor finito para celdas no asignables
NEGATIVE_FILL = -1e12

# Función que verifica que cada fila tenga al menos un valor asignable
def is_matrix_feasible(matrix):
    """
    Verifica si la matriz es factible para resolver el problema de asignación.
    :param matrix: Matriz de costos.
    :return: True si es factible, False en caso contrario.
    """
    return all(any(cell > NEGATIVE_FILL for cell in row) for row in matrix)

# Función para resolver el problema de asignación con restricciones
def solve_assignment_with_constraints(C, fixed, forbidden, huge_value=np.inf):
    """
    Resuelve el problema de asignación lineal con restricciones fijas y prohibidas.
    :param C: Matriz de costos (numpy array, minimización).
    :param fixed: Diccionario de asignaciones fijas {fila: columna}.
    :param forbidden: Diccionario de asignaciones prohibidas {fila: conjunto de columnas prohibidas}.
    :param huge_value: Valor alto para marcar celdas no asignables.
    :return: Tupla (asignación, costo total) o None si no hay solución válida.
    """
    C_mod = np.array(C, copy=True)
    m, n = C_mod.shape

    # Aplicar restricciones fijas
    for r, col in fixed.items():
        for j in range(n):
            if j != col:
                C_mod[r, j] = huge_value

    # Aplicar restricciones prohibidas
    for r, forb in forbidden.items():
        for j in forb:
            C_mod[r, j] = huge_value

    # Resolver el problema de asignación
    row_ind, col_ind = linear_sum_assignment(C_mod)
    total_cost = C_mod[row_ind, col_ind].sum()

    # Verificar si alguna asignación tiene costo "huge_value"
    if any(C_mod[i, j] >= huge_value for i, j in zip(row_ind, col_ind)):
        return None

    assignment = [None] * m
    for i, j in zip(row_ind, col_ind):
        assignment[i] = j

    return assignment, total_cost

# Implementación básica de Murty’s algorithm
import itertools


def murty(C, k):
    """
    C: matriz de costos (numpy array) para minimización.
       En nuestro caso, C = - (weighted_damage) para maximización.
    k: número de soluciones deseadas.
    Devuelve una lista de (assignment, cost).
    """
    m, n = C.shape
    init = solve_assignment_with_constraints(C, {}, {})
    if init is None:
        return []

    best_assignment, best_cost = init
    Q = []
    counter = itertools.count()  # Contador para generar IDs únicos
    heapq.heappush(Q, (best_cost, next(counter), {}, {}, best_assignment))
    solutions = []

    while Q and len(solutions) < k:
        cost, _, fixed, forbidden, assignment = heapq.heappop(Q)
        solutions.append((assignment, cost))

        # Generar subproblemas
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

# Evento cuando el bot está listo
@bot.event
async def on_ready():
    await bot.tree.sync()
    print(f"✅ Bot conectado y comandos sincronizados como {bot.user}.")

# Comando /mystats
@bot.tree.command(name="mystats", description="Ingresa tus estadísticas y selecciona tus charms.")
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
                    raise ValueError("El nivel debe ser al menos 8.")

                stats_data = {
                    "level": level_value,
                    "maxhp": int(self.maxhp.value),
                    "maxmana": int(self.maxmana.value)
                }

                class CharmDropdown(discord.ui.Select):
                    def __init__(self):
                        super().__init__(
                            placeholder="Selecciona tus charms...",
                            min_values=1,
                            max_values=len(charms),
                            options=[discord.SelectOption(label=charm, description=f"Tipo: {details['type']}") for charm, details in charms.items()]
                        )

                    async def callback(self, interaction: discord.Interaction):
                        selected_charms = self.values
                        user_id = interaction.user.id
                        data = charm_manager.get_charms(user_id) or {}
                        data.update(stats_data)
                        data["charms"] = selected_charms
                        charm_manager.save_charms(user_id, data)
                        await interaction.response.send_message(
                            f"✅ Datos guardados:\nNivel: {stats_data['level']}\nHP: {stats_data['maxhp']}\nMana: {stats_data['maxmana']}\nCharms: {', '.join(selected_charms)}"
                        )

                view = discord.ui.View()
                view.add_item(CharmDropdown())
                await interaction.response.send_message("Selecciona tus charms del menú:", view=view)
            except ValueError as e:
                await interaction.response.send_message(f"❌ Error: {e}", ephemeral=True)

    await interaction.response.send_modal(StatsModal())

# Comando /analyzer
@bot.tree.command(name="analyzer", description="Analiza un Hunt Analyzer para extraer información sobre las criaturas matadas.")
async def analyzer(interaction: discord.Interaction):
    class AnalyzerModal(discord.ui.Modal, title="Analyzer"):
        text_input = discord.ui.TextInput(
            label="Pega tu Hunt Analyzer aquí",
            style=discord.TextStyle.long,
            placeholder="Ejemplo: Session data: From 2025-01-06... Killed Monsters: 1028x choking fear..."
        )

        async def on_submit(self, interaction: discord.Interaction):
            hunt_data = self.text_input.value
            try:
                # Extraer sección "Killed Monsters"
                hunt_data_lower = hunt_data.lower()
                if "killed monsters" not in hunt_data_lower:
                    raise ValueError("El formato del Hunt Analyzer es incorrecto. No se encontró 'Killed Monsters'.")

                start_index = hunt_data_lower.index("killed monsters") + len("killed monsters")
                end_index = hunt_data_lower.find("looted items", start_index)
                killed_monsters = (
                    hunt_data[start_index:end_index].strip() if end_index != -1 else hunt_data[start_index:].strip()
                )

                # Procesar frecuencias
                creature_counts = {}
                for line in killed_monsters.split("\n"):
                    if "x" in line:
                        count, name = line.strip().split("x", 1)
                        creature_counts[name.strip()] = int(count)

                # Ordenar criaturas por frecuencia (de mayor a menor)
                creatures_sorted = sorted(creature_counts.items(), key=lambda x: x[1], reverse=True)
                creatures = [name for name, _ in creatures_sorted]
                creature_data = [find_creature_by_name(name) for name in creatures if find_creature_by_name(name)]

                if not creature_data:
                    await interaction.response.send_message("❌ No se encontraron datos de las criaturas matadas.", ephemeral=True)
                    return

                # Datos del usuario
                user_id = interaction.user.id
                user_data = charm_manager.get_charms(user_id)
                selected_charms = user_data.get("charms", [])
                player_level = user_data.get("level", 8)
                max_hp = user_data.get("maxhp", 1000)
                max_mana = user_data.get("maxmana", 500)

                if not selected_charms:
                    await interaction.response.send_message("❌ No has seleccionado ningún charm. Usa `/mystats` para seleccionarlos.", ephemeral=True)
                    return

                # Construir matrices
                opt_matrix = []
                breakdown_matrix = []

                for creature in creature_data:
                    creature_health = float(creature.get("health", "1000"))
                    damage_taken = creature.get("damage_taken_from_elements", {})
                    row_opt = []
                    row_break = []

                    for charm in selected_charms:
                        charm_type = charms[charm]["type"]
                        vulnerability = 1.0
                        if charm_type in damage_taken:
                            try:
                                vulnerability = float(damage_taken[charm_type].strip('%')) / 100
                            except ValueError:
                                pass

                        if charm == "overpower":
                            base_damage = 0.05 * max_hp
                        elif charm == "overflux":
                            base_damage = 0.025 * max_mana
                        else:
                            base_damage = 0.05 * creature_health * vulnerability

                        damage_cap = player_level * 2
                        effective_damage = min(base_damage, damage_cap)
                        if effective_damage <= 0:
                            effective_damage = 1

                        frequency = creature_counts.get(creature["name"], 1)
                        weighted_damage = effective_damage * frequency
                        row_opt.append(weighted_damage)
                        row_break.append({"effective_damage": effective_damage})

                    opt_matrix.append(row_opt)
                    breakdown_matrix.append(row_break)

                if not is_matrix_feasible(opt_matrix):
                    await interaction.response.send_message("❌ La matriz de costos no es válida. Revisa tus charms y la base de datos.", ephemeral=True)
                    return

                # Convertir a numpy array y transformar a minimización (multiplicar por -1)
                M = np.array(opt_matrix)
                cost_matrix = -M

                # Aplicar Murty’s algorithm para obtener las 5 mejores asignaciones
                murty_solutions = murty(cost_matrix, k=5)
                if not murty_solutions:
                    await interaction.response.send_message("❌ No se pueden generar más combinaciones válidas.", ephemeral=True)
                    return

                # Construir el resumen
                solution_texts = []
                m = len(creature_data)

                for assignment, cost in murty_solutions:
                    details = []
                    total_effective = 0
                    for i in range(m):
                        j = assignment[i]
                        eff = breakdown_matrix[i][j]["effective_damage"]
                        total_effective += eff
                        details.append(f"{creature_data[i]['name']}: {selected_charms[j].capitalize()} | Daño: {eff:.2f}")
                    sol_text = f"Daño Total: {total_effective:.2f}\n" + "\n".join(details)
                    solution_texts.append(sol_text)

                embed = discord.Embed(
                    title="Análisis de Asignación Óptima de Charms",
                    description="Prioridad: Criaturas más frecuentes",
                    color=discord.Color.green()
                )

                for idx, sol_text in enumerate(solution_texts, start=1):
                    if len(sol_text) > 1024:
                        sol_text = sol_text[:1020] + "..."
                    embed.add_field(name=f"Combinación #{idx}", value=sol_text, inline=False)

                await interaction.response.send_message(embed=embed)
            except ValueError as e:
                await interaction.response.send_message(f"❌ Error: {e}", ephemeral=True)

    modal = AnalyzerModal()
    await interaction.response.send_modal(modal)

# Ejecutar el bot
bot.run(TOKEN)