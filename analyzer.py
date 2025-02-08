import discord
from discord.ext import commands
from charms_selector import load_creatures_database, find_creature_by_name

# Cargar la base de datos de criaturas
creatures_database = load_creatures_database()

class Analyzer(commands.Cog):
    """Comandos relacionados con el Analyzer."""
    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_ready(self):
        print("Analyzer cog cargado.")

    @commands.hybrid_command(name="analyzer", description="Analiza un Hunt Analyzer para extraer información sobre las criaturas matadas.")
    async def analyzer(self, ctx):
        class AnalyzerModal(discord.ui.Modal, title="Analyzer"):
            text_input = discord.ui.TextInput(
                label="Pega tu Hunt Analyzer aquí",
                style=discord.TextStyle.long,
                placeholder="Ejemplo: killed monsters: Dragon x3 looted items: Gold"
            )

            async def on_submit(self, interaction: discord.Interaction):
                hunt_data = self.text_input.value
                try:
                    # Extraer la sección de "killed monsters"
                    start_index = hunt_data.lower().index("killed monsters") + len("killed monsters")
                    end_index = hunt_data.lower().index("looted items")
                    killed_monsters = hunt_data[start_index:end_index].strip()

                    # Procesar las criaturas matadas
                    creatures = [creature.strip().split(" x")[0] for creature in killed_monsters.split(",")]
                    creature_data = []

                    for creature_name in creatures:
                        creature = find_creature_by_name(creature_name)
                        if creature:
                            creature_data.append(creature)

                    # Crear un mensaje incrustado con la información de las criaturas
                    if creature_data:
                        embed = discord.Embed(
                            title="Análisis de Criaturas Matadas",
                            description="Información sobre las criaturas matadas:",
                            color=discord.Color.green()
                        )
                        for creature in creature_data:
                            health = creature.get("health", "Desconocido")
                            damage_taken = creature.get("damage_taken_from_elements", {})
                            damage_info = "\n".join(
                                [f"- {element}: {damage}" for element, damage in damage_taken.items()]
                            )
                            embed.add_field(
                                name=f"{creature['name']} (Salud: {health})",
                                value=damage_info or "No disponible",
                                inline=False
                            )
                        await interaction.response.send_message(embed=embed)
                    else:
                        await interaction.response.send_message("❌ No se encontraron datos de las criaturas matadas.", ephemeral=True)
                except ValueError:
                    await interaction.response.send_message("❌ El formato del Hunt Analyzer es incorrecto.", ephemeral=True)

        # Enviar el formulario modal al usuario
        modal = AnalyzerModal()
        await ctx.interaction.response.send_modal(modal)