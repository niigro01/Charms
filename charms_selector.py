import json
import os

# Definición de los charms
charms = {
    "curse": {"type": "death"},
    "divine-wrath": {"type": "holy"},
    "enflame": {"type": "fire"},
    "freeze": {"type": "ice"},
    "overpower": {"type": "truedamage"},
    "overflux": {"type": "truedamage"},
    "poison": {"type": "earth"},
    "wound": {"type": "physical"},
    "zap": {"type": "energy"}
}

class CharmManager:
    def __init__(self, filename="charms_data.json"):
        self.filename = filename
        self._load_data()
        # Si el archivo no existe, crear uno nuevo con una estructura predeterminada
        if not os.path.exists(self.filename):
            self._save_data()

    def _load_data(self):
        """Carga los datos de charms desde el archivo JSON."""
        if os.path.exists(self.filename):
            with open(self.filename, "r", encoding="utf-8") as file:
                try:
                    data = json.load(file)
                    if not isinstance(data, dict):
                        raise ValueError("El archivo JSON debe contener un diccionario.")
                    self.data = data
                except json.JSONDecodeError:
                    print("Error al decodificar el archivo JSON. Creando uno nuevo.")
                    self.data = {}
        else:
            self.data = {}

    def _save_data(self):
        """Guarda los datos de charms en el archivo JSON."""
        with open(self.filename, "w", encoding="utf-8") as file:
            json.dump(self.data, file, indent=4)

    def save_charms(self, user_id, data):
        """Guarda los datos completos para un usuario (estadísticas + charms)."""
        if not isinstance(data, dict):
            raise ValueError("Los datos deben ser un diccionario.")
        self.data[str(user_id)] = data
        self._save_data()

    def get_charms(self, user_id):
        """Obtiene los datos guardados para un usuario (estadísticas + charms)."""
        return self.data.get(str(user_id), {})

# Función para cargar la base de datos de criaturas
def load_creatures_database():
    """Carga la base de datos de criaturas desde el archivo JSON."""
    try:
        with open("base_datos_criaturas.json", "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print("El archivo 'base_datos_criaturas.json' no se encontró.")
        return []
    except json.JSONDecodeError:
        print("Error al decodificar el archivo JSON.")
        return []

# Función para buscar una criatura por su nombre
def find_creature_by_name(name):
    """Busca una criatura en la base de datos por su nombre (insensible a mayúsculas/minúsculas)."""
    creatures_database = load_creatures_database()
    name_lower = name.lower()
    for creature in creatures_database:
        if creature["name"].lower() == name_lower:
            return creature
    return None